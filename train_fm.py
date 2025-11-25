import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data
from loguru import logger
from omegaconf import DictConfig, OmegaConf
import wandb

# --- Standard Imports from your existing codebase ---
from utils.args import parse_args
from models.train_utils import (
    get_data_batch, 
    getGradNorm, 
    set_seed, 
    setup_output_subdirs, 
    to_cuda,
    save_iter
)
from models.model_loader import load_optim_sched
from models.evaluation import evaluate
from metrics.emd_assignment import emd_module

# --- NEW IMPORTS for Flow Matching & Semantics ---
# 1. Import the specific FM dataloader
from dataloaders.punet_fm import (
    get_dataset, 
    create_collate_fn, 
    get_alignment_clean
)
# 2. Import the FM Logic and the Semantic-Aware Backbone
from models.flow_matching import ConditionalFlowMatching, PVCNN2UnetFM


def init_processes(rank: int | str, size: int, fn: callable, args: DictConfig) -> None:
    """Initialize the distributed environment."""
    torch.cuda.set_device(rank)
    args.local_rank = rank
    args.global_rank = rank
    args.global_size = size
    args.gpu = rank

    os.environ["MASTER_ADDR"] = args.master_address
    os.environ["MASTER_PORT"] = args.master_port
    dist.init_process_group(backend="nccl", init_method="env://", rank=rank, world_size=size)
    fn(args)
    dist.barrier()
    dist.destroy_process_group()


def train(cfg: DictConfig) -> None:
    is_main_process = cfg.local_rank == 0
    logger.remove()

    if is_main_process:
        (outf_syn,) = setup_output_subdirs(cfg.output_dir, "output")
        cfg.outf_syn = outf_syn
        fmt = (
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            + "<level>{level: <8}</level> | "
            + "<level>{message}</level>"
        )
        logger.add(sys.stdout, level="INFO", format=fmt)

    set_seed(cfg)
    torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # 1. SETUP DATALOADERS (Using punet_fm)
    # -------------------------------------------------------------------------
    logger.info("Setting up datasets...")
    
    # Train Dataset
    train_ds = get_dataset(
        dataset_root=cfg.data.data_dir,
        split="train",
        dataset=cfg.data.dataset,
        # Pass necessary config args explicitly if get_dataset signature requires them
        # or rely on defaults if your get_dataset handles it.
        # Assuming standard signature from your provided snippets:
        resolutions=cfg.data.get("resolutions", ["10000_poisson"]),
        noise_min=cfg.data.get("noise_min", 0.01),
        noise_max=cfg.data.get("noise_max", 0.02),
        aug_rotate=cfg.data.get("augment", True)
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds) if cfg.distribution_type == "multi" else None

    # Collate function (Need the one that handles semantic embeddings)
    if cfg.data.dataset == "PUNet":
        aligner = emd_module.emdModule()
        collate_fn = create_collate_fn(aligner) # Imports from punet_fm
    else:
        collate_fn = None

    train_loader = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=cfg.training.bs,
        sampler=train_sampler,
        collate_fn=collate_fn,
        shuffle=(train_sampler is None),
        num_workers=cfg.data.workers,
        pin_memory=True,
        drop_last=True,
    )

    # Validation Dataset
    val_loader = None
    # Assuming config has val_data_dir, or we use the same root
    val_root = cfg.data.get("val_data_dir", cfg.data.data_dir) 
    
    if val_root is not None:
        val_ds = get_dataset(
            dataset_root=val_root,
            split="test", # Usually validation is done on test split in Point Cloud tasks
            dataset=cfg.data.dataset,
            resolutions=cfg.data.get("resolutions", ["10000_poisson"]),
            noise_min=cfg.data.get("noise_min", 0.01),
            noise_max=cfg.data.get("noise_max", 0.02),
            aug_rotate=False # No augmentation for val
        )

        val_sampler = torch.utils.data.distributed.DistributedSampler(val_ds) if cfg.distribution_type == "multi" else None
        
        val_loader = torch.utils.data.DataLoader(
            val_ds,
            batch_size=cfg.training.bs,
            sampler=val_sampler,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=cfg.data.workers,
            pin_memory=True,
            drop_last=False,
        )

    # -------------------------------------------------------------------------
    # 2. SETUP MODEL (Flow Matching + Semantics)
    # -------------------------------------------------------------------------
    logger.info(f"Instantiating Semantic-Aware Backbone: PVCNN2UnetFM")
    
    # Instantiate the backbone that can accept semantic_emb
    backbone = PVCNN2UnetFM(cfg)
    
    # Wrap with ODE Flow Matching Logic
    model = ConditionalFlowMatching(cfg, backbone)
    
    # Resume / Load Checkpoint
    if cfg.resume_path:
        ckpt = torch.load(cfg.resume_path, map_location="cpu")
        cfg.start_step = ckpt["step"]
        # load_state_dict with strict=False in case we are loading a non-FM checkpoint (optional)
        model.load_state_dict(ckpt["model_state"], strict=False)
        logger.info("Resumed model from {}", cfg.resume_path)
    else:
        ckpt = None
        cfg.start_step = 0

    model = model.cuda()
    
    # Setup Optimizer
    optimizer, lr_scheduler = load_optim_sched(cfg, model, ckpt)
    logger.info("Training with config {}", cfg.config)

    # Setup Alignment Helper (for EMD loss calculation if needed, or visual alignment)
    if cfg.data.dataset == "PUNet":
        aligner = emd_module.emdModule()
        emd_align = get_alignment_clean(aligner)

        @torch.no_grad()
        def align_fn(noisy, clean):
            align_idxs = emd_align(noisy, clean).detach().long()
            align_idxs = align_idxs.unsqueeze(1).expand(-1, 3, -1)
            clean = torch.gather(clean, -1, align_idxs)
            return clean
    else:
        align_fn = None

    # Setup WandB
    if is_main_process:
        wandb.login()
        wandb.init(
            project=cfg.wandb_project,
            config=OmegaConf.to_container(cfg, resolve=True),
            entity=cfg.wandb_entity,
            name=f"FM_{cfg.data.dataset}"
        )
        try:
            wandb.watch(model, log="all", log_freq=cfg.training.log_interval * 10)
        except Exception as e:
            logger.warning("Could not watch model. Skipping.")

    ampscaler = torch.cuda.amp.GradScaler(enabled=cfg.training.amp)
    train_iter = save_iter(train_loader, train_sampler)
    torch.cuda.empty_cache()
    logger.info("Setup training and evaluation iterators.")

    # -------------------------------------------------------------------------
    # 3. TRAINING LOOP
    # -------------------------------------------------------------------------
    for step in range(cfg.start_step, cfg.training.steps):
        optimizer.zero_grad()

        if cfg.distribution_type == "multi":
            train_sampler.set_epoch(step // len(train_loader))

        loss_accum = torch.tensor(0.0, dtype=torch.float32, device=cfg.local_rank)

        for accum_iter in range(cfg.training.accumulation_steps):
            next_batch = next(train_iter)
            next_batch = to_cuda(next_batch, cfg.local_rank)

            data = next_batch
            
            # Helper to split data into clean/noisy/etc
            data_batch = get_data_batch(batch=data, cfg=cfg, align_fn=align_fn)
            
            x_gt = data_batch["x_gt"]       # Clean Data (x0)
            x_cond = data_batch["x_cond"]   # Conditioning Input
            x_start = data_batch["x_start"] # Noisy Source (x1)

            # --- NEW: Extract Semantic Embedding ---
            # Ensure your punet_fm.py collate_fn stacks these into the batch
            semantic_emb = data["semantic_emb"].cuda()
            # ---------------------------------------

            # Flow Matching Loss
            # Flow from x1 (Noisy) -> x0 (Clean)
            loss = model(x0=x_gt, x1=x_start, x_cond=x_cond, semantic_emb=semantic_emb)
            
            loss /= cfg.training.accumulation_steps
            loss_accum += loss.detach()

            ampscaler.scale(loss).backward()

        ampscaler.unscale_(optimizer)
        if cfg.training.grad_clip.enabled:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.grad_clip.value)

        ampscaler.step(optimizer)
        ampscaler.update()
        lr_scheduler.step()

        # NOTE: If you are using EMA, ensure your EMA class supports the new model wrapper
        # If model.ema is a custom class, it might need tweaks. 
        # If it's just updating parameters, it should be fine.
        if hasattr(model, 'ema') and model.ema is not None:
             model.ema.update()

        if cfg.distribution_type == "multi":
            dist.all_reduce(loss_accum)

        # Logging
        if step % cfg.training.log_interval == 0 and is_main_process:
            loss_accum /= cfg.global_size
            loss_accum = loss_accum.item()
            # getGradNorm might need adjustment if it expects a specific model structure, 
            # but usually works on standard modules
            netpNorm, netgradNorm = getGradNorm(model) 

            logger.info(
                "[{:>3d}/{:>3d}]\tloss: {:>10.6f},\t" "netpNorm: {:>10.2f},\tnetgradNorm: {:>10.4f}",
                step,
                cfg.training.steps,
                loss_accum,
                netpNorm,
                netgradNorm,
            )
            wandb.log(
                {
                    "loss": loss_accum,
                    "netpNorm": netpNorm,
                    "netgradNorm": netgradNorm,
                },
                step=step,
            )

        # Saving
        if (step + 1) % cfg.training.save_interval == 0:
            if is_main_process:
                save_dict = {
                    "step": step + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                }
                torch.save(save_dict, "%s/step_%d.pth" % (cfg.output_dir, step + 1))
                logger.info("Saved checkpoint to {}", cfg.output_dir)

            if cfg.distribution_type == "multi":
                dist.barrier()
                # Optional: Sync loading to verify save worked, but usually unnecessary overhead here
                pass

        # Visualization / Evaluation
        if (step + 1) % cfg.training.viz_interval == 0:
            if cfg.distribution_type == "multi":
                dist.barrier()

            model.eval()
            if is_main_process and val_loader is not None:
                try:
                    # evaluate function usually expects 'model.sample' which we implemented
                    evaluate(model, val_loader, cfg, step + 1)
                except Exception as e:
                    print(sys.exc_info())
                    logger.warning("Could not evaluate model. Skipping.")
                    logger.warning(e)

            torch.cuda.empty_cache()
            model.train()

    wandb.finish()


if __name__ == "__main__":
    opt = parse_args()

    # Save configuration
    os.makedirs(opt.output_dir, exist_ok=True)
    save_data = DictConfig({})
    save_data.data = opt.data
    save_data.diffusion = opt.diffusion
    save_data.model = opt.model
    save_data.sampling = opt.sampling
    save_data.training = opt.training
    OmegaConf.save(save_data, os.path.join(opt.output_dir, "opt.yaml"))

    opt.ngpus_per_node = torch.cuda.device_count()
    torch.set_float32_matmul_precision("high")

    if opt.distribution_type == "multi":
        opt.world_size = opt.ngpus_per_node * opt.world_size
        opt.training.bs = int(opt.training.bs / opt.ngpus_per_node)
        opt.sampling.bs = opt.training.bs
        mp.spawn(init_processes, nprocs=opt.world_size, args=(opt.world_size, train, opt))
    else:
        torch.cuda.set_device(0)
        opt.global_rank = 0
        opt.local_rank = 0
        opt.global_size = 1
        opt.gpu = 0
        train(opt)