import argparse
import os
import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from datasets.dyadic_loader import DyadicLoader
from model import Audio2PoseDiTDual_VQVAE
from tool import (
    apply_denormalization,
    save_motion_and_audio_outputs,
    visualize_audio_waveform,
    visualize_motion_comparison,
    visualize_motion_heatmap,
)

PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if PARENT_DIR not in sys.path:
    sys.path.insert(1, PARENT_DIR)

from train_wan_ours import (
    DiffusionConfig,
    LitAudio2PoseDiT,
    encode_with_vqvae,
    get_codebook_representation,
)
import dyadic_dit_vq.model.vqvae_downsampled as vqvae


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_config_obj(config_dict: Dict[str, Any]):
    """Create a simple attribute-style config object from dictionary."""

    class Config:
        def __init__(self, data: Dict[str, Any]):
            for key, value in data.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)

    return Config(config_dict)


def prepare_vqvae(config) -> vqvae.TemporalVertexCodecDownsampled:
    """Instantiate and load the VQ-VAE codec defined in the config."""
    codec = vqvae.TemporalVertexCodecDownsampled(
        n_vertices=config.vqvae.nb_joints,
        latent_dim=config.vqvae.output_emb_width,
        categories=config.vqvae.code_dim,
        residual_depth=config.vqvae.depth,
        downsample_factor=config.vqvae.downsample_factor,
    )
    ckpt_path = getattr(config.vqvae, "checkpoint_path", None)
    if ckpt_path is None or not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"VQ-VAE checkpoint not found at: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = state["net"] if isinstance(state, dict) and "net" in state else state
    codec.load_state_dict(state_dict)
    codec.eval()
    return codec


def to_codebook_representation(
    tensor: Optional[torch.Tensor], codec: vqvae.TemporalVertexCodecDownsampled
) -> Optional[torch.Tensor]:
    """Encode raw motion tensor with VQ-VAE and return continuous codebook representation."""
    if tensor is None:
        return None
    codes = encode_with_vqvae(tensor, codec)
    return get_codebook_representation(codes, codec)


def main():
    parser = argparse.ArgumentParser(description="Inference script for DyaDiT (ours, VQ-VAE-based)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML configuration file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--override", nargs="*", help="Override config values (format: key=value)")
    parser.add_argument("--pose_length", type=int, help="Override pose_length in config")
    parser.add_argument("--batch_idx", type=int, default=0, help="Select starting batch index to process (default: 0)")
    parser.add_argument("--num_batches", type=int, default=1, help="Number of batches to process sequentially (default: 1)")
    parser.add_argument("--win_length", type=int, default=384, help="Sliding window length for inference")
    parser.add_argument("--stride", type=int, default=384, help="Stride between windows")
    parser.add_argument("--guidance_scale", type=float, default=4.0, help="Guidance scale for CFG")
    parser.add_argument("--inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument(
        "--mode",
        type=str,
        default="quality",
        choices=["quality", "relat", "ps", "uncond", "random", "relat_random", "ps_random"],
        help="quality: use dataset relationship/personality; relat: sweep relationship options; ps: fix relationship to friend and sweep personality one-hots; uncond: drop all conditioning (audio, style, relat, personality, motion_other); random: randomly sample relationship and personality; relat_random: randomly sample relationship with personality set to zero; ps_random: randomly sample personality with relationship set to zero",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Random seed for random modes (random, relat_random, ps_random) (default: None, uses current time)",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="inference_outputs",
        help="Directory prefix to store batch outputs (default: inference_outputs)",
    )
    parser.add_argument(
        "--index_parity",
        type=str,
        default="even",
        choices=["even", "odd", "all"],
        help="Which dataloader indices to include when collecting batches. Use even/odd to avoid duplicate A-B/B-A pairs (default: even)",
    )
    parser.add_argument(
        "--save_visuals",
        action="store_true",
        help="Save waveform/motion visualizations for each processed batch (default: disabled)",
    )
    args = parser.parse_args()
    uncond_mode = args.mode == "uncond"

    # Set random seed for random modes if provided
    if args.mode in ["random", "relat_random", "ps_random"] and args.random_seed is not None:
        random.seed(args.random_seed)
        np.random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        print(f"Random seed set to: {args.random_seed}")

    config_dict = load_config(args.config)
    config = create_config_obj(config_dict)

    if args.pose_length is not None:
        if not hasattr(config, "model"):
            config.model = type("Model", (), {})()
        config.model.pose_length = args.pose_length
        print(f"Overriding pose_length to: {args.pose_length}")

    if args.override:
        for override in args.override:
            key, value = override.split("=", 1)
            try:
                if "." in value or "e" in value.lower():
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass

            target = config
            parts = key.split(".")
            for part in parts[:-1]:
                if hasattr(target, part):
                    target = getattr(target, part)
                else:
                    target = None
                    break
            if target is not None:
                setattr(target, parts[-1], value)

    device = getattr(config.paths, "device", "cuda:0")

    codec = prepare_vqvae(config)
    codec.to(device)

    model = Audio2PoseDiTDual_VQVAE(
        pose_dim=config.model.pose_dim,
        model_dim=config.model.model_dim,
        ffn_dim=config.model.ffn_dim,
        heads=config.model.heads,
        layers=config.model.layers,
        time_dim=config.model.time_dim,
        use_cross_attn=config.model.use_cross_attn,
        use_rope=config.model.use_rope,
        fusion_drop=config.model.fusion_drop,
        style_drop=config.model.style_drop,
        wav2vec_layer=config.model.wav2vec_layers,
        motion_bank_size=config.model.motion_bank_size,
        relat_drop=config.model.relat_drop,
        person_drop=config.model.person_drop,
        motion_other_drop=config.model.motion_other_drop,
        orthogonal_fusion=getattr(config.model, "orthogonal_fusion", False),
    )

    lit = LitAudio2PoseDiT(
        model=model,
        lr=config.training.lr,
        wd=config.training.weight_decay,
        diffusion=DiffusionConfig(
            num_train_timesteps=config.training.diffusion.num_train_timesteps,
            objective=config.training.diffusion.objective,
            scheduler_type=config.training.diffusion.scheduler_type,
            beta_start=config.training.diffusion.beta_start,
            beta_end=config.training.diffusion.beta_end,
            beta_schedule=config.training.diffusion.beta_schedule,
            variance_type=config.training.diffusion.variance_type,
            clip_sample=config.training.diffusion.clip_sample,
            prediction_type=config.training.diffusion.prediction_type,
        ),
        mask_length=config.model.mask_length,
        vqvae=codec,
    )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    lit.load_state_dict(checkpoint["state_dict"])
    lit.to(device)
    lit.eval()

    dataset = DyadicLoader(config)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(args.output_prefix, exist_ok=True)

    desired_mod = 0 if args.index_parity == "even" else 1
    parity_enabled = args.index_parity != "all"

    print(f"Processing batches starting at index {args.batch_idx} (target count: {args.num_batches})...")
    processed_batches = 0
    available_indices = len(dataloader)

    relat_dtype = torch.float32
    personality_dtype = torch.float32

    relation_names = ["ours_friend", "ours_stranger", "ours_family", "ours_dating"]
    relation_onehots = [
        torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=relat_dtype),
        torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=relat_dtype),
        torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=relat_dtype),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=relat_dtype),
    ]
    personality_names = ["extra", "agree", "cons", "neuro", "open"]
    personality_onehots = [
        torch.tensor([[5.0, 0.0, 0.0, 0.0, 0.0]], device=device, dtype=personality_dtype),
        torch.tensor([[0.0, 5.0, 0.0, 0.0, 0.0]], device=device, dtype=personality_dtype),
        torch.tensor([[0.0, 0.0, 5.0, 0.0, 0.0]], device=device, dtype=personality_dtype),
        torch.tensor([[0.0, 0.0, 0.0, 5.0, 0.0]], device=device, dtype=personality_dtype),
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 5.0]], device=device, dtype=personality_dtype),
    ]

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx < args.batch_idx:
            continue

        if parity_enabled and not (args.num_batches == 1 and batch_idx == args.batch_idx):
            if batch_idx % 2 != desired_mod:
                continue

        audio = batch["audio"].to(device)
        audio_other = batch["audio_other"].to(device)
        gt_motion_raw = batch["motion_latent"].to(device)
        motion_other_raw = batch["motion_latent_other"].to(device)
        style_raw = batch.get("style_latent", None)
        if style_raw is not None:
            style_raw = style_raw.to(device)
        relat = batch.get("relationship_onehot", None)
        if relat is not None:
            relat = relat.to(device)
            relat_dtype = relat.dtype
        personality = batch.get("personality_traits", None)
        if personality is not None:
            personality = personality.to(device)
            personality_dtype = personality.dtype

        print(f"\n=== Processing batch {batch_idx} ===")
        print(f"Audio shape: {audio.shape}")
        print(f"Audio_other shape: {audio_other.shape}")
        print(f"GT motion (raw) shape: {gt_motion_raw.shape}")
        print(f"GT motion_other (raw) shape: {motion_other_raw.shape}")

        with torch.no_grad():
            gt_motion_latent = to_codebook_representation(gt_motion_raw, codec)
            motion_other_latent = (
                None if uncond_mode else to_codebook_representation(motion_other_raw, codec)
            )
            style_latent = (
                None
                if uncond_mode
                else to_codebook_representation(style_raw, codec) if style_raw is not None else None
            )

            gt_motion_decoded_full = codec.decode_motion(gt_motion_latent)
            motion_other_decoded_full = (
                codec.decode_motion(motion_other_latent) if motion_other_latent is not None else None
            )

        style_window = None
        if style_latent is not None:
            style_window = style_latent[:, : args.win_length, :]
            style_window = torch.zeros_like(style_window).clone()

        if args.mode == "quality":
            scenario_specs = [
                {
                    "relat": relat,
                    "rel_name": "dataset_relat" if relat is not None else "no_relat",
                    "personality": personality,
                    "personality_name": "dataset_personality" if personality is not None else "no_personality",
                    "output_subdirs": [],
                }
            ]
        elif args.mode == "relat":
            scenario_specs = [
                {
                    "relat": relation_tensor,
                    "rel_name": rel_name,
                    "personality": personality,
                    "personality_name": "dataset_personality" if personality is not None else "no_personality",
                    "output_subdirs": [rel_name, "dataset_personality" if personality is not None else "no_personality"],
                }
                for rel_name, relation_tensor in zip(relation_names, relation_onehots)
            ]
        elif args.mode == "ps":
            friend_tensor = relation_onehots[0]
            scenario_specs = [
                {
                    "relat": friend_tensor,
                    "rel_name": f"ours_{personality_name}",
                    "personality": personality_tensor,
                    "personality_name": personality_name,
                    "output_subdirs": [f"ours_{personality_name}"],
                }
                for personality_name, personality_tensor in zip(personality_names, personality_onehots)
            ]
        elif args.mode == "random":
            # Randomly sample one relationship and one personality
            random_rel_idx = random.randint(0, len(relation_names) - 1)
            random_ps_idx = random.randint(0, len(personality_names) - 1)
            random_rel_tensor = relation_onehots[random_rel_idx]
            random_ps_tensor = personality_onehots[random_ps_idx]
            random_rel_name = relation_names[random_rel_idx]
            random_ps_name = personality_names[random_ps_idx]
            
            print(f"Randomly selected: relationship={random_rel_name}, personality={random_ps_name}")
            
            scenario_specs = [
                {
                    "relat": random_rel_tensor,
                    "rel_name": random_rel_name,
                    "personality": random_ps_tensor,
                    "personality_name": random_ps_name,
                    "output_subdirs": ["random_vqvae"],
                }
            ]
        elif args.mode == "relat_random":
            # Randomly sample one relationship with value 5, personality from dataloader
            # Create relationship tensors with value 5: [5,0,0,0], [0,5,0,0], [0,0,5,0], [0,0,0,5]
            relat_scaled_onehots = [
                torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=relat_dtype),
                torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device, dtype=relat_dtype),
                torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device, dtype=relat_dtype),
                torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=device, dtype=relat_dtype),
            ]
            
            random_rel_idx = random.randint(0, len(relation_names) - 1)
            random_rel_tensor = relat_scaled_onehots[random_rel_idx]
            random_rel_name = relation_names[random_rel_idx]
            
            # Use personality from dataloader
            scenario_personality = personality
            personality_name = "dataset_personality" if personality is not None else "no_personality"
            
            print(f"Randomly selected: relationship={random_rel_name} (scaled to 5), personality=dataset")
            
            scenario_specs = [
                {
                    "relat": random_rel_tensor,
                    "rel_name": random_rel_name,
                    "personality": scenario_personality,
                    "personality_name": personality_name,
                    "output_subdirs": [],
                }
            ]
        elif args.mode == "ps_random":
            # Randomly sample one personality, relationship set to zero
            random_ps_idx = random.randint(0, len(personality_names) - 1)
            random_ps_tensor = personality_onehots[random_ps_idx]
            random_ps_name = personality_names[random_ps_idx]
            
            # Create zero relationship tensor with same shape as dataset relationship
            zero_relat = None
            if relat is not None:
                zero_relat = torch.zeros_like(relat).clone()
            else:
                # If no relationship in dataset, create a zero tensor with shape (1, 4)
                zero_relat = torch.zeros((1, 4), device=device, dtype=relat_dtype)
            
            print(f"Randomly selected: relationship=zero, personality={random_ps_name}")
            
            scenario_specs = [
                {
                    "relat": zero_relat,
                    "rel_name": "zero",
                    "personality": random_ps_tensor,
                    "personality_name": random_ps_name,
                    "output_subdirs": [],
                }
            ]
        else:  # uncond_mode
            scenario_specs = [
                {
                    "relat": torch.zeros_like(relat).clone(),
                    "rel_name": "uncond",
                    "personality": torch.zeros_like(personality).clone(),
                    "personality_name": "uncond",
                    "output_subdirs": ["uncond"],
                }
            ]

        for scenario in scenario_specs:
            scenario_relat = scenario["relat"]
            if scenario_relat is not None:
                scenario_relat = scenario_relat.to(device=device, dtype=relat_dtype)
            scenario_personality = scenario["personality"]
            if scenario_personality is not None:
                scenario_personality = scenario_personality.to(device=device, dtype=personality_dtype)
            rel_name = scenario["rel_name"]
            personality_name = scenario["personality_name"]
            output_subdirs = scenario["output_subdirs"]

            print(
                f"\n\n======= Processing relationship: {rel_name} / personality: {personality_name} ======="
            )
            print(f"relationship tensor: {scenario_relat}")
            print(f"personality tensor: {scenario_personality}")

            audio_for_model = audio
            audio_other_for_model = audio_other
            style_for_model = style_window
            motion_other_for_model = motion_other_latent

            if uncond_mode:
                audio_for_model = torch.zeros_like(audio_for_model)
                audio_other_for_model = torch.zeros_like(audio_other_for_model)
                style_for_model = None
                motion_other_for_model = None
                print("Running in unconditional mode: using zero audio and dropping style/relationship/personality/motion_other.")

            # Prepare output directory for saving latent
            path_components = [args.output_prefix]
            path_components.extend(output_subdirs)
            path_components.append(f"batch_{batch_idx:05d}")
            scenario_output_dir = os.path.join(*path_components)
            os.makedirs(scenario_output_dir, exist_ok=True)
            latent_path = os.path.join(scenario_output_dir, "generated_motion_latent.npz")

            with torch.no_grad():
                motion = lit.inference(
                    audio_for_model,
                    audio_other_for_model,
                    cond=None,
                    gt_motion=gt_motion_latent,
                    style=style_for_model,
                    relat=scenario_relat,
                    personality=scenario_personality,
                    motion_other=motion_other_for_model,
                    win_length=args.win_length,
                    stride=args.stride,
                    inference_steps=args.inference_steps,
                    guidance_scale=args.guidance_scale,
                    save_latent_path=latent_path,
                )

            gt_motion_decoded = gt_motion_decoded_full.clone()
            motion_other_decoded = (
                motion_other_decoded_full.clone() if motion_other_decoded_full is not None else None
            )

            print(f"Generated motion shape: {motion.shape}")
            print(f"Decoded GT motion shape: {gt_motion_decoded.shape}")

            if motion.shape[1] != gt_motion_decoded.shape[1]:
                min_len = min(motion.shape[1], gt_motion_decoded.shape[1])
                print(
                    f"Motion length mismatch: generated={motion.shape[1]}, GT={gt_motion_decoded.shape[1]} -> cropping to {min_len}"
                )
                motion = motion[:, :min_len, :]
                gt_motion_decoded = gt_motion_decoded[:, :min_len, :]
                if motion_other_decoded is not None:
                    motion_other_decoded = motion_other_decoded[:, :min_len, :]

            mse = F.mse_loss(motion, gt_motion_decoded)
            print(f"MSE (generated vs GT decoded): {mse.item():.6f}")

            print("\nApplying denormalization...")
            motion_denorm, gt_motion_denorm, is_6d_rotation = apply_denormalization(
                motion, gt_motion_decoded, dataset
            )

            motion_other_denorm = None
            if motion_other_decoded is not None:
                print("\nApplying denormalization to motion_other...")
                _, motion_other_denorm, _ = apply_denormalization(
                    motion_other_decoded, motion_other_decoded, dataset
                )

            print("\nSaving outputs...")
            # scenario_output_dir already created above for latent saving

            save_motion_and_audio_outputs(
                motion_denorm,
                gt_motion_denorm,
                audio,
                batch,
                config,
                is_6d_rotation=is_6d_rotation,
                motion_other_denorm=motion_other_denorm,
                audio_other=audio_other,
                output_dir=scenario_output_dir,
            )

            # Save personality and relationship scores to npz
            metadata_path = os.path.join(scenario_output_dir, "metadata.npz")
            personality_np = scenario_personality.cpu().numpy() if scenario_personality is not None else None
            relat_np = scenario_relat.cpu().numpy() if scenario_relat is not None else None
            np.savez(
                metadata_path,
                personality=personality_np,
                relationship=relat_np,
            )
            print(f"Saved metadata (personality, relationship) to: {metadata_path}")

            if args.save_visuals:
                visual_dir = os.path.join(scenario_output_dir, "visualizations")
                os.makedirs(visual_dir, exist_ok=True)

                def save_pure_waveform(audio_tensor: torch.Tensor, save_path: str, color):
                    """Save minimal waveform visualization without axes or labels."""
                    if audio_tensor.dim() == 3:
                        audio_np = audio_tensor[0, 0].cpu().detach().numpy()
                    elif audio_tensor.dim() == 2:
                        audio_np = audio_tensor[0].cpu().detach().numpy()
                    else:
                        audio_np = audio_tensor.cpu().detach().numpy()

                    time_axis = np.linspace(0, len(audio_np) / 16000, len(audio_np))
                    fig, ax = plt.subplots(1, 1, figsize=(12, 4))
                    fig.patch.set_facecolor("white")
                    ax.plot(time_axis, audio_np, linewidth=0.9, alpha=1, color=color)
                    ax.set_xlim(time_axis[0], time_axis[-1])
                    ax.set_ylim(audio_np.min(), audio_np.max())
                    ax.axis("off")
                    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                    plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
                    plt.close()
                    print(f"Pure waveform saved to: {save_path}")

                print("\nCreating pure audio waveform visualizations...")
                save_pure_waveform(
                    audio,
                    save_path=os.path.join(visual_dir, "audio_waveform.png"),
                    color=(0.35, 0.55, 0.75),
                )
                save_pure_waveform(
                    audio_other,
                    save_path=os.path.join(visual_dir, "audio_other_waveform.png"),
                    color="gray",
                )

                visualize_audio_waveform(
                    audio,
                    save_path=os.path.join(visual_dir, "audio_waveform_detailed.png"),
                    title="Input Audio Waveform",
                )

                print("Creating motion visualizations...")
                visualize_motion_comparison(
                    motion_denorm,
                    gt_motion_denorm,
                    audio=audio,
                    save_path=os.path.join(visual_dir, "motion_comparison.png"),
                    title="Generated vs Ground Truth Motion (Denormalized, with Audio Background)",
                )

                visualize_motion_heatmap(
                    motion_denorm,
                    gt_motion_denorm,
                    save_path=os.path.join(visual_dir, "motion_heatmap.png"),
                    title="Motion Heatmap Comparison (Denormalized)",
                )

            print(f"Successfully processed relationship={rel_name}, personality={personality_name}")

        processed_batches += 1
        if processed_batches >= args.num_batches:
            break

    if processed_batches == 0:
        print(
            f"Error: No batches processed. Please check batch_idx ({args.batch_idx}), parity setting ({args.index_parity}), and dataset length ({available_indices})."
        )
        return

    if processed_batches < args.num_batches:
        print(
            f"Requested {args.num_batches} batches but only processed {processed_batches}. Dataset may be smaller than expected."
        )


if __name__ == "__main__":
    main()
