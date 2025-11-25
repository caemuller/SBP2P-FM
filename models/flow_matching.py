
import math
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
        
        # CLIP embedding dim is 512, Time Embed dim is usually 64
        # We project 512 -> 64 to match dimensions
        self.semantic_proj = nn.Sequential(
            nn.Linear(512, self.embed_dim),
            nn.SiLU(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

    def forward(self, x, t, x_cond=None, semantic_emb=None):
        # --- Logic copied & modified from PVCNN2Unet.forward ---
        
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

        # --- MODIFICATION STARTS HERE ---
        time_emb = None
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            
            # 1. Get standard Time Embedding
            # Shape: (B, EmbedDim)
            t_emb_val = self.embedf(self.get_timestep_embedding(t, device))
            
            # 2. Inject Semantic Embedding if present
            if semantic_emb is not None:
                # Project (B, 512) -> (B, EmbedDim)
                s_emb_val = self.semantic_proj(semantic_emb)
                # Add them together
                t_emb_val = t_emb_val + s_emb_val
            
            # 3. Expand to nodes (B, EmbedDim, N)
            time_emb = t_emb_val[:, :, None].expand(-1, -1, N)
        # --- MODIFICATION ENDS HERE ---

        data.features = features
        data.time_emb = time_emb

        # The rest is identical to the original forward pass
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
        b = x0.shape[0]
        
        # Sample t uniform [0, 1]
        t = torch.rand(b, device=self.device).type_as(x0)
        
        # Flow Matching Interpolation
        # x_t = (1 - t) * x0 + t * x1
        t_expand = t.view(b, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # Target Vector Field: u_t = x1 - x0
        # (Pointing towards noise)
        target_v = x1 - x0
        
        # Predict Vector Field
        # Pass semantic_emb to our new backbone subclass
        pred_v = self.backbone(x_t, t, x_cond=x_cond, semantic_emb=semantic_emb)

        # MSE Loss
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x1, x_cond=None, semantic_emb=None, steps=50):
        """
        Euler ODE Solver from t=1 (Noisy) to t=0 (Clean)
        """
        b = x1.shape[0]
        self.backbone.eval()
        
        x_t = x1.clone()
        times = torch.linspace(1.0, 0.0, steps, device=self.device)
        
        # dt is negative because we go 1.0 -> 0.0
        
        x_chain = [x_t.clone()]

        for i in tqdm(range(steps - 1), desc="Flow Matching Sampling"):
            t_curr = times[i]
            t_next = times[i+1]
            dt = t_next - t_curr 
            
            t_tensor = torch.full((b,), t_curr, device=self.device)
            
            # Predict v
            v_pred = self.backbone(x_t, t_tensor, x_cond=x_cond, semantic_emb=semantic_emb)
            
            # Euler step: x_{t+1} = x_t + v * dt
            # Since dt is negative, we move towards x0
            x_t = x_t + v_pred * dt
            
            x_chain.append(x_t.clone())

        self.backbone.train()
        return {
            "x_chain": torch.stack(x_chain, dim=1),
            "x_pred": x_t,
            "x_start": x1
        }

    def forward(self, x0, x1, x_cond=None, semantic_emb=None):
        return self.compute_loss(x0, x1, x_cond, semantic_emb)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ConditionalFlowMatching(nn.Module):
    def __init__(self, cfg: Dict, backbone: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.backbone = backbone
        self.device = cfg.gpu if cfg.gpu is not None else torch.device("cuda")
        
        # Dimensions
        self.time_dim = cfg.model.time_embed_dim # Usually 64 based on your config
        self.semantic_dim = 512 # CLIP ViT-B/32 output dimension
        
        # 1. Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(self.time_dim),
            nn.Linear(self.time_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim),
        )

        # 2. Semantic Projection MLP (Projects 512 -> 64)
        self.semantic_proj = nn.Sequential(
            nn.Linear(self.semantic_dim, self.time_dim * 4),
            nn.GELU(),
            nn.Linear(self.time_dim * 4, self.time_dim)
        )

    def prepare_batch(self, x, t, x_cond=None, semantic_emb=None):
        """
        Prepares the PVCData object with combined embeddings.
        """
        # 1. Get Time Embedding
        t_emb = self.time_mlp(t) # (B, time_dim)

        # 2. Get Semantic Embedding & Combine
        if semantic_emb is not None:
            s_emb = self.semantic_proj(semantic_emb) # (B, time_dim)
            # We add semantic info to time info. 
            # The ScaleShift layers in your PVCNN will now condition on (Time + Text)
            global_cond = t_emb + s_emb 
        else:
            global_cond = t_emb

        # 3. Create PVCData package expected by your backbone
        # Note: We pass global_cond as 'time_emb' because your ScaleShift layers read 'time_emb'
        data = PVCData(
            features=x, 
            coords=x, 
            time_emb=global_cond, 
            cond=x_cond # If you have extra local conditioning
        )
        return data

    def compute_loss(self, x0, x1, x_cond=None, semantic_emb=None):
        """
        x0: Clean Data (Target)
        x1: Noisy Data (Source)
        """
        b = x0.shape[0]
        
        # Sample t uniform [0, 1]
        t = torch.rand(b, device=self.device).type_as(x0)
        
        # Flow Matching Interpolation
        # x_t = (1 - t) * x0 + t * x1
        t_expand = t.view(b, 1, 1)
        x_t = (1 - t_expand) * x0 + t_expand * x1
        
        # Target Vector Field: u_t = x1 - x0
        target_v = x1 - x0
        
        # Predict Vector Field
        # We wrap inputs into PVCData
        data_input = self.prepare_batch(x_t, t, x_cond, semantic_emb)
        
        # Forward pass through backbone
        # The backbone returns PVCData, we extract features
        out_data = self.backbone(data_input)
        pred_v = out_data.features

        # MSE Loss
        loss = torch.mean((pred_v - target_v) ** 2)
        return loss

    @torch.no_grad()
    def sample(self, x1, x_cond=None, semantic_emb=None, steps=50):
        """
        Euler ODE Solver from t=1 (Noisy) to t=0 (Clean)
        """
        b = x1.shape[0]
        self.backbone.eval()
        
        x_t = x1.clone()
        times = torch.linspace(1.0, 0.0, steps, device=self.device)
        dt = times[0] - times[1] # Negative step size effectively

        x_chain = [x_t.clone()]

        for i in tqdm(range(steps - 1), desc="Flow Matching Sampling"):
            t_curr = times[i]
            t_tensor = torch.full((b,), t_curr, device=self.device)
            
            # Predict v
            data_input = self.prepare_batch(x_t, t_tensor, x_cond, semantic_emb)
            pred_v = self.backbone(data_input).features
            
            # Euler step: x_{t-1} = x_t - v * dt
            # Note: We defined vector field as x1 - x0 (pointing to noise)
            # So to go to clean, we subtract.
            x_t = x_t - pred_v * (times[i] - times[i+1])
            
            x_chain.append(x_t.clone())

        self.backbone.train()
        return {
            "x_chain": torch.stack(x_chain, dim=1),
            "x_pred": x_t,
            "x_start": x1
        }

    def forward(self, x0, x1, x_cond=None, semantic_emb=None):
        return self.compute_loss(x0, x1, x_cond, semantic_emb)