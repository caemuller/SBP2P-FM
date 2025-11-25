import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
from einops import rearrange

from models.modules import Attention
from models.pvcnn import PVCData, LinearAttention, SharedMLP, Swish
from models.unet_pvc import PVCNN2Unet

class PVCNN2UnetFM(PVCNN2Unet):
    """
    A wrapper around PVCNN2Unet that injects Semantic Embeddings 
    into the Time Embedding stream.
    """
    def __init__(self, cfg: Dict):
        super().__init__(cfg)
        
        self.semantic_proj = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x, t, x_cond=None, semantic_emb=None):
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        (B, C, N), device = x.shape, x.device
        
        coords = x[:, : self.input_dim, :].contiguous()
        features = x[:, self.input_dim :, :].contiguous()

        if self.embed_feats is not None:
            if self.extra_feature_channels == 0:
                features = self.embed_feats(coords)
            else:
                features = self.embed_feats(features)

        data = PVCData(coords=coords, features=coords)

        if self.global_pnet is not None:
            global_feature = self.global_pnet(data)
            data.cond = global_feature
        else:
            global_feature = None

        features = torch.cat([coords, features], dim=1)
        
        coords_list, in_features_list = [], []
        out_features_list = []
        in_features_list.append(features)

        # --- SEMANTIC INJECTION ---
        time_emb = None
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            
            t_emb_val = self.embedf(self.get_timestep_embedding(t, device))
            
            if semantic_emb is not None:
                s_emb_val = self.semantic_proj(semantic_emb)
                t_emb_val = t_emb_val + s_emb_val
            
            time_emb = t_emb_val[:, :, None].expand(-1, -1, N)
        # --------------------------

        data.features = features
        data.time_emb = time_emb

        for i, sa_blocks in enumerate(self.sa_layers):
            in_features_list.append(data.features)
            coords_list.append(data.coords)

            if i > 0 and data.time_emb is not None:
                data.features = torch.cat([data.features, data.time_emb], dim=1)
                data = sa_blocks(data)
            else:
                data = sa_blocks(data)

        in_features_list.pop(1)

        if self.global_att is not None:
            features = data.features
            if isinstance(self.global_att, LinearAttention):
                features = self.global_att(features)
            elif isinstance(self.global_att, Attention):
                features = rearrange(features, "b n c -> b c n")
                features = self.global_att(features)
                features = rearrange(features, "b c n -> b n c")
            data.features = features

        out_features_list.append(data.features)

        for fp_idx, fp_blocks in enumerate(self.fp_layers):
            data_fp = PVCData(
                features=in_features_list[-1 - fp_idx],
                coords=coords_list[-1 - fp_idx],
                lower_coords=data.coords,
                lower_features=(
                    torch.cat([data.features, data.time_emb], dim=1) if data.time_emb is not None else data.features
                ),
                time_emb=data.time_emb,
                cond=data.cond,
            )
            data = fp_blocks(data_fp)
            out_features_list.append(data.features)

        for l in self.classifier:
            if isinstance(l, SharedMLP):
                data.features = l(data).features
            else:
                data.features = l(data.features)

        return data.features


class ConditionalFlowMatching(nn.Module):
    def __init__(self, cfg: Dict, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.device = cfg.gpu if cfg.gpu is not None else torch.device("cuda")

    def compute_loss(self, x0, x1, x_cond=None, semantic_emb=None):
        """
        x0: Clean Data (Target)
        x1: Noisy Data (Source)
        """
        # --- FIX: Transpose inputs to (B, C, N) ---
        # Data comes in as (B, N, 3), network wants (B, 3, N)
        if x0.shape[-1] == 3: x0 = x0.transpose(1, 2)
        if x1.shape[-1] == 3: x1 = x1.transpose(1, 2)
        if x_cond is not None and x_cond.shape[-1] == 3: 
            x_cond = x_cond.transpose(1, 2)
        # ------------------------------------------

        b = x0.shape[0]
        
        t = torch.rand(b, device=self.device).type_as(x0)
        
        t_expand = t.view(b, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        target_v = x1 - x0
        
        pred_v = self.backbone(x_t, t, x_cond=x_cond, semantic_emb=semantic_emb)

        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x1, x_cond=None, semantic_emb=None, steps=50):
        # --- FIX: Transpose inputs to (B, C, N) ---
        if x1.shape[-1] == 3: x1 = x1.transpose(1, 2)
        if x_cond is not None and x_cond.shape[-1] == 3: 
            x_cond = x_cond.transpose(1, 2)
        # ------------------------------------------

        b = x1.shape[0]
        self.backbone.eval()
        
        x_t = x1.clone()
        times = torch.linspace(1.0, 0.0, steps, device=self.device)
        
        x_chain = [x_t.clone()]

        for i in tqdm(range(steps - 1), desc="Flow Matching Sampling"):
            t_curr = times[i]
            t_next = times[i+1]
            dt = t_next - t_curr 
            
            t_tensor = torch.full((b,), t_curr, device=self.device)
            
            v_pred = self.backbone(x_t, t_tensor, x_cond=x_cond, semantic_emb=semantic_emb)
            
            x_t = x_t + v_pred * dt
            x_chain.append(x_t.clone())

        self.backbone.train()
        
        # Transpose back to (B, N, 3) for saving
        result = torch.stack(x_chain, dim=1).transpose(-1, -2) # (B, T, N, 3)
        final = x_t.transpose(1, 2)
        start = x1.transpose(1, 2)

        return {
            "x_chain": result,
            "x_pred": final,
            "x_start": start
        }

    def forward(self, x0, x1, x_cond=None, semantic_emb=None):
        return self.compute_loss(x0, x1, x_cond, semantic_emb)