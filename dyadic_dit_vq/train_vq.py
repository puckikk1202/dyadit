"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import argparse
import copy
import json
import logging
import os
import random
import sys
import warnings
from typing import Any, Dict


# import model.vqvae as vqvae # 1:1 mapping vqvae
# import model.vqvae_downsampled as vqvae # compressed vqvae
import numpy as np
import torch
import torch.optim as optim
import yaml
from omegaconf import OmegaConf
from datasets.vqvae_overfit_loader import OverfitLoader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not installed. Install with: pip install wandb")

warnings.filterwarnings("ignore")


def sum_flat(tensor: torch.Tensor) -> torch.Tensor:
    """
    Take the sum over all non-batch dimensions.
    """
    return tensor.sum(dim=list(range(1, len(tensor.shape))))


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility
    
    Args:
        seed: Random seed value
        deterministic: If True, enables fully deterministic behavior (slower but reproducible)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    
    # Set seed for Python hash
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    # For deterministic behavior (may impact performance)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Enable deterministic algorithms (PyTorch 1.8+)
        if hasattr(torch, 'use_deterministic_algorithms'):
            torch.use_deterministic_algorithms(True)
        print("Deterministic mode enabled - this may reduce performance")
    else:
        torch.backends.cudnn.benchmark = True
        print("⚡ Benchmark mode enabled - faster but not fully deterministic")


def setup_gpu(gpu_id: int = None):
    """Setup GPU device"""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU: {gpu_id}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs available: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    return device


def get_logger(out_dir: str):
    logger = logging.getLogger("Exp")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_path = os.path.join(out_dir, "run.log")
    file_hdlr = logging.FileHandler(file_path)
    file_hdlr.setFormatter(formatter)

    strm_hdlr = logging.StreamHandler(sys.stdout)
    strm_hdlr.setFormatter(formatter)

    logger.addHandler(file_hdlr)
    logger.addHandler(strm_hdlr)
    return logger


class ModelTrainer:
    def __init__(self, args, net, logger, writer, use_wandb=False):
        self.net = net
        self.warm_up_iter = args.warm_up_iter
        self.lr = args.lr
        self.optimizer = optim.AdamW(
            self.net.parameters(),
            lr=args.lr,
            betas=(0.9, 0.99),
            weight_decay=args.weight_decay,
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=args.lr_scheduler, gamma=args.gamma
        )
        self.data_format = args.data_format
        self.loss = torch.nn.SmoothL1Loss()
        self.loss_vel = args.loss_vel
        self.commit = args.commit
        self.logger = logger
        self.writer = writer
        self.use_wandb = use_wandb
        self.best_commit = float("inf")
        self.best_recons = float("inf")
        self.best_perplexity = float("inf")
        self.best_iter = 0
        self.out_dir = args.out_dir

    def _masked_l2(self, a, b, mask):
        loss = self._l2_loss(a, b)
        loss = sum_flat(loss * mask.float())
        n_entries = a.shape[1] * a.shape[2]
        non_zero_elements = sum_flat(mask) * n_entries
        mse_loss_val = loss / non_zero_elements
        return mse_loss_val

    def _l2_loss(self, motion_pred, motion_gt, mask=None):
        if mask is not None:
            return self._masked_l2(motion_pred, motion_gt, mask)
        else:
            return self.loss(motion_pred, motion_gt)

    def _vel_loss(self, motion_pred, motion_gt):
        model_results_vel = motion_pred[..., :-1] - motion_pred[..., 1:]
        model_targets_vel = motion_gt[..., :-1] - motion_gt[..., 1:]
        return self.loss(model_results_vel, model_targets_vel)

    def _update_lr_warm_up(self, nb_iter):
        current_lr = self.lr * (nb_iter + 1) / (self.warm_up_iter + 1)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
        return current_lr

    def run_warmup_steps(self, train_loader_iter, logger, print_iter, warm_up_iter):
        avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0
        for nb_iter in tqdm(range(1, warm_up_iter)):
            current_lr = self._update_lr_warm_up(nb_iter)
            batch = next(train_loader_iter)
            loss_dict = self.run_train_step(batch)

            avg_recons += loss_dict["loss_motion"]
            avg_perplexity += loss_dict["perplexity"]
            avg_commit += loss_dict["loss_commit"]

            if nb_iter % print_iter == 0:
                avg_recons /= print_iter
                avg_perplexity /= print_iter
                avg_commit /= print_iter

                logger.info(
                    f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
                )
                
                # Wandb logging
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.log({
                        "warmup/lr": current_lr,
                        "warmup/commit_loss": avg_commit,
                        "warmup/perplexity": avg_perplexity,
                        "warmup/recons_loss": avg_recons,
                    }, step=nb_iter)

                avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0

    def run_train_step(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, Any]:
        self.net.train()
        loss_dict = {}
        
        # Extract motion from batch
        # OverfitLoader returns: motion_latent [batch, seq_len, motion_dim]
        # VQ-VAE expects: [batch, seq_len, motion_dim]  <- Same format! No transpose needed!
        gt_motion = batch['motion_latent'].cuda().float()
        
        # Debug: Print shape
        if not hasattr(self, '_shape_printed'):
            self.logger.info(f"DEBUG: Input motion shape: {gt_motion.shape} [batch, seq_len, motion_dim]")
            self._shape_printed = True
        
        # Forward pass through VQ-VAE (no transpose needed!)
        pred_motion, loss_commit, perplexity = self.net(gt_motion, mask=None)
        
        # Calculate reconstruction loss
        loss_motion = self._l2_loss(pred_motion, gt_motion).mean()
        
        # Calculate velocity loss if enabled
        loss_vel = 0.0
        if self.loss_vel > 0:
            loss_vel = self._vel_loss(pred_motion, gt_motion)
        
        # Total loss
        loss = loss_motion + self.commit * loss_commit + self.loss_vel * loss_vel
        
        # Optimization step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Record losses
        if self.loss_vel > 0:
            loss_dict["vel"] = loss_vel.item()
        loss_dict["loss"] = loss.item()
        loss_dict["loss_motion"] = loss_motion.item()
        loss_dict["loss_commit"] = loss_commit.item()
        loss_dict["perplexity"] = perplexity.item()
        
        return loss_dict

    def save_model(self, save_path):
        torch.save(
            {
                "net": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler,
            },
            save_path,
        )

    def _save_predictions(self, name, unstd_pose, unstd_pred):
        curr_name = os.path.basename(name)
        path = os.path.join(self.out_dir, curr_name)
        for j in range(len(path.split("/")) - 1):
            if not os.path.exists("/".join(path.split("/")[: j + 1])):
                os.system("mkdir " + "/".join(path.split("/")[: j + 1]))
        np.save(os.path.join(self.out_dir, curr_name + "_gt.npy"), unstd_pose)
        np.save(os.path.join(self.out_dir, curr_name + "_pred.npy"), unstd_pred)

    def _log_losses(
        self,
        commit_loss: float,
        recons_loss: float,
        total_perplexity: float,
        nb_iter: int,
        nb_sample: int,
        draw: bool,
        save: bool,
    ) -> None:
        avg_commit = commit_loss / nb_sample
        avg_recons = recons_loss / nb_sample
        avg_perplexity = total_perplexity / nb_sample
        self.logger.info(
            f"Eval. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
        )

        if draw:
            self.writer.add_scalar("./Val/Perplexity", avg_perplexity, nb_iter)
            self.writer.add_scalar("./Val/Commit", avg_commit, nb_iter)
            self.writer.add_scalar("./Val/Recons", avg_recons, nb_iter)
        
        # Wandb logging
        if self.use_wandb and WANDB_AVAILABLE:
            wandb.log({
                "val/commit_loss": avg_commit,
                "val/perplexity": avg_perplexity,
                "val/recons_loss": avg_recons,
                "val/best_perplexity": self.best_perplexity,
                "val/best_commit": self.best_commit,
                "val/best_recons": self.best_recons,
            }, step=nb_iter)

        if avg_perplexity < self.best_perplexity:
            msg = f"--> --> \t Perplexity Improved from {self.best_perplexity:.5f} to {avg_perplexity:.5f} !!!"
            self.logger.info(msg)
            self.best_perplexity = avg_perplexity
            if save:
                print(f"saving checkpoint net_best.pth")
                self.save_model(os.path.join(self.out_dir, "net_best.pth"))
                if self.use_wandb and WANDB_AVAILABLE:
                    wandb.run.summary["best_perplexity"] = avg_perplexity
                    wandb.run.summary["best_perplexity_iter"] = nb_iter

        if avg_commit < self.best_commit:
            msg = f"--> --> \t Commit Improved from {self.best_commit:.5f} to {avg_commit:.5f} !!!"
            self.logger.info(msg)
            self.best_commit = avg_commit
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.run.summary["best_commit"] = avg_commit

        if avg_recons < self.best_recons:
            msg = f"--> --> \t Recons Improved from {self.best_recons:.5f} to {avg_recons:.5f} !!!"
            self.logger.info(msg)
            self.best_recons = avg_recons
            if self.use_wandb and WANDB_AVAILABLE:
                wandb.run.summary["best_recons"] = avg_recons

    @torch.no_grad()
    def evaluation_vqvae(
        self,
        val_loader,
        nb_iter: int,
        draw: bool = True,
        save: bool = True,
        savenpy: bool = False,
    ) -> None:
        self.net.eval()
        nb_sample = 0
        commit_loss = 0
        recons_loss = 0
        total_perplexity = 0
        
        for batch in tqdm(val_loader, desc="Evaluation"):
            # Extract motion from batch
            # VQ-VAE expects: [batch, seq_len, motion_dim] - no transpose needed!
            motion = batch['motion_latent'].cuda().float()  # [batch, seq_len, motion_dim]
            
            bs = motion.shape[0]
            
            # Forward pass
            pred, loss_commit_model, perplexity = self.net(motion)
            
            # Calculate commit loss manually for eval mode (model returns 0 in eval)
            # We need to compute it to track validation commit loss
            B, T = motion.shape[0], motion.shape[1]
            with torch.no_grad():
                # Get encoder output
                encoded = self.net.encoder(motion)  # [B, T//downsample, latent_dim]
                # Get quantized output
                # Handle both downsampled and non-downsampled models
                downsample_factor = getattr(self.net, 'downsample_factor', 1)
                quantized, _, _ = self.net.quantizer(encoded.view(-1, self.net.latent_dim), B, T // downsample_factor, None)
                quantized = quantized.view(B, -1, self.net.latent_dim)
                # Compute commit loss manually
                actual_commit_loss = torch.nn.functional.mse_loss(encoded, quantized)
            
            # Calculate losses
            l2_loss = self._l2_loss(pred, motion)
            recons_loss += l2_loss.mean().item() * bs
            commit_loss += actual_commit_loss.item() * bs  # Use manually computed commit loss
            total_perplexity += perplexity.item() * bs
            
            nb_sample += bs

        self._log_losses(
            commit_loss, recons_loss, total_perplexity, nb_iter, nb_sample, draw, save
        )
        if save:
            print(f"saving checkpoint net_last.pth")
            self.save_model(os.path.join(self.out_dir, "net_last.pth"))
            if nb_iter % 100000 == 0:
                print(f"saving checkpoint net_iter_x.pth")
                self.save_model(
                    os.path.join(self.out_dir, "net_iter" + str(nb_iter) + ".pth")
                )


def _load_data_info(cfg, logger, split_json_path=None):
    """Load data using OverfitLoader
    
    Args:
        cfg: Configuration object
        logger: Logger instance
        split_json_path: Optional path to JSON file containing train/val/test split info.
                        If provided, uses JSON-based split. Otherwise, uses automatic split.
    """
    logger.info("Loading data with OverfitLoader...")
    
    # Create dataset
    dataset = OverfitLoader(cfg=cfg)
    
    # Choose split method based on whether JSON file is provided
    if split_json_path and os.path.exists(split_json_path):
        # JSON-based split
        logger.info(f"Using JSON-based split from: {split_json_path}")
        from utils.data_split import create_splits_from_json
        
        train_dataset, val_dataset, test_dataset = create_splits_from_json(
            dataset,
            split_json_path=split_json_path,
            data_root=cfg.data.data_path
        )
    else:
        # Automatic split (80/10/10 split with fixed seed)
        # This ensures reproducible splits across runs
        if split_json_path:
            logger.warning(f"Split JSON not found at {split_json_path}, falling back to automatic split")
        logger.info("Using automatic split (80/10/10)")
        from utils.data_split import create_fixed_splits
        
        train_dataset, val_dataset, test_dataset = create_fixed_splits(
            dataset, 
            train_ratio=0.8, 
            val_ratio=0.1, 
            test_ratio=0.1, 
            seed=42  # Fixed seed for reproducibility
        )
    
    # Create data loaders with optimized settings
    # Important: If using preload_to_memory, set num_workers=0 for best performance
    use_preload = cfg.data.preload_to_memory if hasattr(cfg.data, 'preload_to_memory') else False
    actual_num_workers = 0 if use_preload else cfg.data.num_workers
    
    if use_preload and cfg.data.num_workers > 0:
        logger.info("preload_to_memory enabled: setting num_workers=0 for best performance")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=True,
        num_workers=actual_num_workers,
        pin_memory=cfg.data.pin_memory if not use_preload else False,
        drop_last=True,
        persistent_workers=True if actual_num_workers > 0 else False,
        prefetch_factor=2 if actual_num_workers > 0 else None,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.data.batch_size,
        shuffle=False,
        num_workers=actual_num_workers,
        pin_memory=cfg.data.pin_memory if not use_preload else False,
        drop_last=False,
        persistent_workers=True if actual_num_workers > 0 else False,
        prefetch_factor=2 if actual_num_workers > 0 else None,
    )
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")
    logger.info(f"Motion dimension: {cfg.model.pose_dim}, Sequence length: {cfg.model.pose_length}")
    
    train_loader_iter = cycle(train_loader)
    
    return train_loader_iter, val_loader


def _load_checkpoint(resume_pth, net, logger):
    """Load checkpoint from file"""
    logger.info(f"Loading checkpoint from {resume_pth}")
    ckpt = torch.load(resume_pth, map_location="cpu")
    net.load_state_dict(ckpt["net"], strict=True)
    logger.info("Checkpoint loaded successfully")
    return net


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train VQ-VAE for motion data")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/siyeol_dyadit_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="outputs/vqvae",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--resume_pth",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID to use (e.g., 0, 1, 2). If None, uses CUDA_VISIBLE_DEVICES or default GPU."
    )
    # VQ-VAE specific arguments
    parser.add_argument("--code_dim", type=int, default=1024, help="Codebook size")
    parser.add_argument("--output_emb_width", type=int, default=512, help="Latent dimension")
    parser.add_argument("--depth", type=int, default=3, help="Number of residual layers")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument("--commit", type=float, default=0.02, help="Commitment loss weight")
    parser.add_argument("--loss_vel", type=float, default=0.5, help="Velocity loss weight")
    parser.add_argument("--warm_up_iter", type=int, default=1000, help="Warmup iterations")
    parser.add_argument("--total_iter", type=int, default=100000, help="Total iterations")
    parser.add_argument("--print_iter", type=int, default=100, help="Print every N iterations")
    parser.add_argument("--eval_iter", type=int, default=5000, help="Evaluate every N iterations")
    parser.add_argument("--lr_scheduler", type=int, nargs="+", default=[50000, 80000], help="LR scheduler milestones")
    parser.add_argument("--gamma", type=float, default=0.1, help="LR scheduler gamma")
    parser.add_argument("--seed", type=int, default=222, help="Random seed for reproducibility")
    parser.add_argument("--downsample_factor", type=int, default=None, help="Downsample factor")
    parser.add_argument(
        "--prequant_norm",
        action="store_true",
        help="Apply LayerNorm before quantization for better codebook utilization"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for full reproducibility (may reduce performance)"
    )
    # Wandb arguments
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Use Weights & Biases for logging"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="vqvae_motion",
        help="Wandb project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Wandb entity (username or team name)"
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="Wandb run name (default: auto-generated)"
    )
    parser.add_argument(
        "--split_json",
        type=str,
        default=None,
        help="Path to JSON file containing train/val/test split information. "
             "If not provided, uses automatic 80/10/10 split with fixed seed."
    )
    
    return parser.parse_args()


def main(args):
    # Setup GPU device first
    device = setup_gpu(args.gpu_id)
    
    # Set random seed for reproducibility
    print(f"🎲 Setting random seed: {args.seed}")
    set_seed(args.seed, deterministic=args.deterministic)
    
    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)
    
    # Setup logger and tensorboard writer
    logger = get_logger(args.out_dir)
    writer = SummaryWriter(args.out_dir)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Random seed: {args.seed}")
    logger.info(f"Deterministic mode: {args.deterministic}")
    
    # Load config
    logger.info(f"Loading config from {args.config}")
    with open(args.config, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # Initialize Wandb
    use_wandb = args.use_wandb and WANDB_AVAILABLE
    if use_wandb:
        wandb_run_name = args.wandb_run_name or f"vqvae_c{args.code_dim}_d{args.depth}_lr{args.lr}_downsample{args.downsample_factor}_batch{cfg.data.batch_size}_pose{cfg.model.pose_length}"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=wandb_run_name,
            config={
                **vars(args),
                "pose_dim": cfg.model.pose_dim,
                "pose_length": cfg.model.pose_length,
                "pose_fps": cfg.model.pose_fps,
                "batch_size": cfg.data.batch_size,
                "use_6d_rotation": cfg.data.use_6d_rotation,
            },
            dir=args.out_dir,
        )
        logger.info(f"Wandb initialized: {wandb.run.name}")
        logger.info(f"Wandb URL: {wandb.run.get_url()}")
    else:
        if args.use_wandb:
            logger.warning("Wandb requested but not available. Install with: pip install wandb")
        logger.info("Running without Wandb logging")
    
    # Add missing attributes to args for ModelTrainer compatibility
    args.nb_joints = cfg.model.pose_dim  # motion dimension
    args.data_format = 'motion'
    args.data_root = cfg.data.data_path
    
    # Log arguments
    logger.info("Arguments:")
    logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
    
    # Save args to file
    args_path = os.path.join(args.out_dir, "args.json")
    with open(args_path, "w") as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    # Load data
    train_loader_iter, val_loader = _load_data_info(cfg, logger, split_json_path=args.split_json)
    
    # Create VQ-VAE model
    # n_vertices is the motion dimension (258 for 6D rotation)
    logger.info("Creating VQ-VAE model...")
    if args.downsample_factor is None:
        import model.vqvae as vqvae # compressed vqvae
        net = vqvae.TemporalVertexCodec(
            n_vertices=cfg.model.pose_dim,  # 258 for 6D rotation
            latent_dim=args.output_emb_width,
            categories=args.code_dim,
            residual_depth=args.depth,
        )
    else:
        import model.vqvae_downsampled as vqvae # compressed vqvae
        net = vqvae.TemporalVertexCodecDownsampled(
            n_vertices=cfg.model.pose_dim,  # 258 for 6D rotation
            latent_dim=args.output_emb_width,
            categories=args.code_dim,
            residual_depth=args.depth,
            downsample_factor=args.downsample_factor,
        )
    
    # Load checkpoint if provided
    if args.resume_pth:
        net = _load_checkpoint(args.resume_pth, net, logger)
    
    net.train()
    net.cuda()
    
    logger.info(f"VQ-VAE model created with {sum(p.numel() for p in net.parameters())/1e6:.2f}M parameters")
    
    # Log model to wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.watch(net, log="all", log_freq=args.print_iter)
    
    # Create trainer
    trainer = ModelTrainer(args, net, logger, writer, use_wandb=use_wandb)
    
    # Warmup phase
    if args.warm_up_iter > 0:
        logger.info(f"Starting warmup for {args.warm_up_iter} iterations...")
        trainer.run_warmup_steps(train_loader_iter, logger, args.print_iter, args.warm_up_iter)
    
    # Initial evaluation
    logger.info("Running initial evaluation...")
    with torch.no_grad():
        trainer.evaluation_vqvae(
            val_loader, 0, save=(args.total_iter > 0), savenpy=False
        )
    
    # Training loop
    logger.info(f"Starting training for {args.total_iter} iterations...")
    avg_recons, avg_perplexity, avg_commit = 0.0, 0.0, 0.0
    
    for nb_iter in range(1, args.total_iter + 1):
        batch = next(train_loader_iter)
        loss_dict = trainer.run_train_step(batch)
        trainer.scheduler.step()

        avg_recons += loss_dict["loss_motion"]
        avg_perplexity += loss_dict["perplexity"]
        avg_commit += loss_dict["loss_commit"]

        if nb_iter % args.print_iter == 0:
            avg_recons /= args.print_iter
            avg_perplexity /= args.print_iter
            avg_commit /= args.print_iter

            writer.add_scalar("./Train/L1", avg_recons, nb_iter)
            writer.add_scalar("./Train/PPL", avg_perplexity, nb_iter)
            writer.add_scalar("./Train/Commit", avg_commit, nb_iter)

            logger.info(
                f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f}"
            )
            
            # Wandb logging
            if use_wandb and WANDB_AVAILABLE:
                current_lr = trainer.optimizer.param_groups[0]['lr']
                global_step = args.warm_up_iter + nb_iter
                wandb.log({
                    "train/lr": current_lr,
                    "train/commit_loss": avg_commit,
                    "train/perplexity": avg_perplexity,
                    "train/recons_loss": avg_recons,
                }, step=global_step)

            avg_recons, avg_perplexity, avg_commit = (0.0, 0.0, 0.0)

        if nb_iter % args.eval_iter == 0:
            global_step = args.warm_up_iter + nb_iter
            with torch.no_grad():
                trainer.evaluation_vqvae(
                    val_loader, global_step, save=True, savenpy=False
                )
            net.train()  # Set back to train mode after evaluation
    
    logger.info("Training completed!")
    
    # Finish wandb run
    if use_wandb and WANDB_AVAILABLE:
        wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    main(args)