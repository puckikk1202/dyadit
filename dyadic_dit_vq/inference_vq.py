"""
VQ-VAE Inference Script for Motion Data
Encode motion to discrete tokens and decode back to motion
"""

import argparse
import json
import os
import sys
from typing import Any

import model.vqvae_downsampled as vqvae
import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from datasets.siyeol_overfit_loader import OverfitLoader
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.motion_utils import convert_rotation_6d_to_axis_angle


def setup_gpu(gpu_id: int = None):
    """Setup GPU device"""
    if gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        print(f"Using GPU: {gpu_id}")
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("CUDA is not available. Using CPU.")
    
    return device


def load_vqvae_model(checkpoint_path: str, device: torch.device):
    """Load VQ-VAE model from checkpoint"""
    print(f"Loading VQ-VAE model from: {checkpoint_path}")
    
    # Load args from checkpoint directory
    checkpoint_dir = os.path.dirname(checkpoint_path)
    args_path = os.path.join(checkpoint_dir, "args.json")
    
    if not os.path.exists(args_path):
        raise FileNotFoundError(f"args.json not found in {checkpoint_dir}")
    
    with open(args_path, 'r') as f:
        args = json.load(f)
    
    # Get prequant_norm setting (default to False for backward compatibility)
    prequant_norm = args.get('prequant_norm', False)
    
    # Create model based on whether downsample_factor exists
    if 'downsample_factor' in args and args['downsample_factor'] is not None:
        print(f"   Using downsampled VQ-VAE (downsample_factor={args['downsample_factor']})")
        import model.vqvae_downsampled as vqvae
        model = vqvae.TemporalVertexCodecDownsampled(
            n_vertices=args['nb_joints'],
            latent_dim=args['output_emb_width'],
            categories=args['code_dim'],
            residual_depth=args['depth'],
            downsample_factor=args['downsample_factor'],
            prequant_norm=prequant_norm,
        )
    else:
        import model.vqvae as vqvae
        print(f"   Using original VQ-VAE (no downsampling)")
        model = vqvae.TemporalVertexCodec(
            n_vertices=args['nb_joints'],
            latent_dim=args['output_emb_width'],
            categories=args['code_dim'],
            residual_depth=args['depth'],
            prequant_norm=prequant_norm,
        )
    
    # Load checkpoint
    # Note: Use weights_only=False because checkpoint contains optimizer/scheduler
    # This is safe since we trust our own checkpoints
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['net'], strict=True)
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"   - Motion dim: {args['nb_joints']}")
    print(f"   - Latent dim: {args['output_emb_width']}")
    print(f"   - Codebook size: {args['code_dim']}")
    print(f"   - Depth: {args['depth']}")
    
    return model, args


@torch.no_grad()
def encode_motion(model, motion):
    """
    Encode motion to discrete tokens
    
    Args:
        model: VQ-VAE model
        motion: Motion tensor [batch, seq_len, motion_dim]
    
    Returns:
        indices: Discrete token indices [batch, num_tokens]
        quantized: Quantized motion [batch, seq_len, motion_dim]
    """
    model.eval()
    
    # Encode
    motion_encoded = model.encoder(motion)
    
    # Quantize
    quantized, loss_commit, perplexity, encoding_indices = model.quantizer(motion_encoded)
    
    return encoding_indices, quantized


@torch.no_grad()
def decode_motion(model, quantized):
    """
    Decode quantized motion back to motion
    
    Args:
        model: VQ-VAE model
        quantized: Quantized motion [batch, seq_len, motion_dim]
    
    Returns:
        motion: Reconstructed motion [batch, seq_len, motion_dim]
    """
    model.eval()
    
    # Decode
    motion = model.decoder(quantized)
    
    return motion


@torch.no_grad()
def encode_decode_motion(model, motion):
    """
    Full encode-decode pipeline
    
    Args:
        model: VQ-VAE model
        motion: Motion tensor [batch, seq_len, motion_dim]
    
    Returns:
        reconstructed: Reconstructed motion [batch, seq_len, motion_dim]
        indices: Discrete token indices [batch, num_tokens]
        perplexity: Codebook perplexity
    """
    model.eval()
    
    # Full forward pass
    reconstructed, loss_commit, perplexity = model(motion, mask=None)
    
    # Also get indices (use encode method which works for both model types)
    indices = model.encode(motion)
    
    return reconstructed, indices, perplexity


def compute_reconstruction_error(gt_motion, pred_motion):
    """Compute reconstruction error metrics"""
    mse = torch.mean((gt_motion - pred_motion) ** 2).item()
    mae = torch.mean(torch.abs(gt_motion - pred_motion)).item()
    
    # Per-frame error
    per_frame_mse = torch.mean((gt_motion - pred_motion) ** 2, dim=(0, 1))
    
    return {
        'mse': mse,
        'mae': mae,
        'rmse': np.sqrt(mse),
        'per_frame_mse': per_frame_mse.cpu().numpy()
    }


def process_batch(model, batch, device, denormalize_fn=None):
    """Process a single batch"""
    # Extract motion
    # VQ-VAE expects: [batch, seq_len, motion_dim] - no transpose needed!
    motion = batch['motion_latent'].to(device).float()  # [batch, seq_len, motion_dim]
    
    # Encode and decode
    reconstructed, indices, perplexity = encode_decode_motion(model, motion)
    
    # Compute error
    errors = compute_reconstruction_error(motion, reconstructed)
    
    return {
        'motion_gt': motion.cpu(),  # Move to CPU to avoid CUDA OOM
        'motion_pred': reconstructed.cpu(),  # Move to CPU to avoid CUDA OOM
        'indices': indices.cpu(),  # Move to CPU to avoid CUDA OOM
        'perplexity': perplexity.item(),
        'errors': errors,
        'video_id': batch['video_id']
    }


def run_inference(args):
    """Run inference on dataset"""
    # Setup device
    device = setup_gpu(args.gpu_id)
    
    # Load model
    model, model_args = load_vqvae_model(args.checkpoint, device)
    
    # Load config
    print(f"\nLoading data config from: {args.config}")
    with open(args.config, 'r') as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # Create dataset
    print("\nCreating dataset...")
    dataset = OverfitLoader(cfg=cfg)
    
    # Select split based on args
    from utils.data_split import get_split_by_name
    
    split_dataset = get_split_by_name(
        dataset, 
        split_name=args.split,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42  # Same seed as training
    )
    
    print(f"\nUsing {args.split.upper()} split: {len(split_dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        split_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )


    
    print(f"Split dataset size: {len(split_dataset)}")
    print(f"Number of batches: {len(dataloader)}")
    
    # Setup output directory
    output_dir = args.output_dir or os.path.join(os.path.dirname(args.checkpoint), f"inference_results_{args.split}")
    if args.save_results:
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nOutput directory: {output_dir}")
    
    # Run inference
    print("\n" + "="*60)
    print("Running inference...")
    print("="*60)
    
    batch_summaries = []  # Light-weight summaries for all batches
    total_mse = 0.0
    total_perplexity = 0.0
    num_samples = 0
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        results = process_batch(model, batch, device)
        
        # Accumulate stats
        batch_size = results['motion_gt'].shape[0]
        total_mse += results['errors']['mse'] * batch_size
        total_perplexity += results['perplexity'] * batch_size
        num_samples += batch_size
        
        # Save batch summary (lightweight)
        batch_summaries.append({
            'batch_idx': batch_idx,
            'batch_size': batch_size,
            'perplexity': results['perplexity'],
            'mse': results['errors']['mse'],
            'rmse': results['errors']['rmse'],
            'mae': results['errors']['mae'],
            'video_ids': results['video_id']
        })
        
        # Save individual batch results to disk immediately (if requested)
        if args.save_results and args.save_individual:
            result_path = os.path.join(output_dir, f'batch_{batch_idx:04d}.npz')
            np.savez(
                result_path,
                motion_gt=results['motion_gt'].numpy(),
                motion_pred=results['motion_pred'].numpy(),
                indices=results['indices'].numpy(),
                perplexity=results['perplexity'],
                mse=results['errors']['mse'],
                per_frame_mse=results['errors']['per_frame_mse']
            )
        
        # Print sample results
        if batch_idx == 0:
            print(f"\nSample batch results:")
            print(f"   Motion shape: {results['motion_gt'].shape}")
            print(f"   Indices shape: {results['indices'].shape}")
            print(f"   Perplexity: {results['perplexity']:.2f}")
            print(f"   MSE: {results['errors']['mse']:.6f}")
            print(f"   RMSE: {results['errors']['rmse']:.6f}")
            print(f"   MAE: {results['errors']['mae']:.6f}")
        
        # Clear CUDA cache periodically to avoid OOM
        if batch_idx % 50 == 0 and batch_idx > 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Limit number of batches if specified
        if args.num_batches > 0 and batch_idx >= args.num_batches - 1:
            print(f"\nStopped after {args.num_batches} batches (as requested)")
            break
    
    # Compute overall stats
    avg_mse = total_mse / num_samples
    avg_perplexity = total_perplexity / num_samples
    
    print("\n" + "="*60)
    print("Overall Statistics")
    print("="*60)
    print(f"Total samples processed: {num_samples}")
    print(f"Average MSE: {avg_mse:.6f}")
    print(f"Average RMSE: {np.sqrt(avg_mse):.6f}")
    print(f"Average Perplexity: {avg_perplexity:.2f}")
    
    # Save results
    if args.save_results:
        print(f"\nSaving results to: {output_dir}")
        
        # Save overall summary
        summary = {
            'num_samples': num_samples,
            'num_batches': len(batch_summaries),
            'avg_mse': float(avg_mse),
            'avg_rmse': float(np.sqrt(avg_mse)),
            'avg_perplexity': float(avg_perplexity),
            'checkpoint': args.checkpoint,
            'config': args.config,
            'split': args.split,
            'batch_size': args.batch_size,
        }
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"   Saved summary: {summary_path}")
        
        # Save per-batch summary (lightweight - always save this)
        batch_summary_path = os.path.join(output_dir, 'batch_summary.json')
        with open(batch_summary_path, 'w') as f:
            json.dump({
                'batches': batch_summaries,
                'num_batches': len(batch_summaries),
                'total_samples': num_samples
            }, f, indent=2)
        print(f"   Saved batch summary: {batch_summary_path}")
        
        # Individual batch NPZ files were already saved during processing if save_individual was enabled
        if args.save_individual:
            print(f"   Individual batch NPZ files: {len(batch_summaries)} files saved")
    
    print("\nInference completed!")


def main():
    parser = argparse.ArgumentParser(description="VQ-VAE Inference for Motion Data")
    
    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to VQ-VAE checkpoint (e.g., outputs/vqvae/net_best.pth)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/siyeol_dyadit_config.yaml",
        help="Path to data config file"
    )
    
    # Optional arguments
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU device ID to use"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=-1,
        help="Number of batches to process (-1 for all)"
    )
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="Save inference results"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual batch results (can be large)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for results (default: checkpoint_dir/inference_results)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val", "test"],
        help="Which data split to use for inference (default: val)"
    )
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Run inference
    run_inference(args)


if __name__ == "__main__":
    main()

