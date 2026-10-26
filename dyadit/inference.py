"""Dataset-conditioned inference with the same model construction as training."""

import argparse
from pathlib import Path

import pytorch_lightning as pl
import torch

from .checkpoint import load_system_weights
from .config import load_config
from .data import DyadicDataset
from .diffusion import DyaDiTSystem
from .export import save_outputs


def main():
    parser = argparse.ArgumentParser(
        description="Generate dyadic motion using the core DyaDiT model"
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "configs" / "core.yaml"),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument(
        "--data-root", help="Override the data directory used for inference"
    )
    parser.add_argument("--output-dir", default="inference_outputs")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument(
        "--pose-length", type=int, help="Dataset segment length in motion frames"
    )
    parser.add_argument(
        "--window-frames", type=int, help="Generation window in motion frames"
    )
    parser.add_argument(
        "--stride-frames", type=int, help="Generation stride in motion frames"
    )
    parser.add_argument("--inference-steps", type=int, default=50)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--relationship", nargs=4, type=float)
    parser.add_argument("--personality", nargs=5, type=float)
    parser.add_argument(
        "--no-reference-prefix",
        action="store_true",
        help="Generate without using the target motion prefix",
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render conversation.mp4: blue/left=condition B, orange/right=generated A; stereo audio",
    )
    parser.add_argument(
        "--smplh-model", help="SMPL-H model file/directory used with --render"
    )
    parser.add_argument(
        "--render-backend",
        choices=("auto", "egl", "osmesa", "software"),
        default="auto",
    )
    args = parser.parse_args()
    if args.render and not args.smplh_model:
        parser.error("--render requires --smplh-model")
    config = load_config(args.config, args.set)
    pl.seed_everything(config["seed"], workers=True)
    if args.num_samples < 1 or args.index < 0:
        parser.error("index must be non-negative and num-samples must be positive")
    length = args.pose_length or config["model"]["pose_length"]
    if length % config["vqvae"]["downsample_factor"]:
        parser.error("pose-length must be divisible by the codec downsample factor")
    dataset = DyadicDataset(
        config, data_root=args.data_root, pose_length=length, random_style=False
    )
    if args.index + args.num_samples > len(dataset):
        parser.error(f"Requested samples exceed dataset size ({len(dataset)})")
    system = DyaDiTSystem(config)
    load_system_weights(system, args.checkpoint)
    system.to(args.device).eval()
    print(f"orthogonal_fusion={system.model.orthogonal_fusion}")
    for index in range(args.index, args.index + args.num_samples):
        sample = dataset[index]
        for key, values in (
            ("relationship_onehot", args.relationship),
            ("personality_traits", args.personality),
        ):
            if values is not None:
                sample[key] = torch.tensor(values, dtype=torch.float32)
        batch = {
            key: value.unsqueeze(0).to(args.device)
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        generated, latent = system.generate(
            batch["audio"],
            batch["audio_other"],
            length,
            style_motion=batch["style_latent"],
            motion_other=batch["motion_latent_other"],
            relationship=batch["relationship_onehot"],
            personality=batch["personality_traits"],
            reference_motion=(
                None if args.no_reference_prefix else batch["motion_latent"]
            ),
            window_frames=args.window_frames,
            stride_frames=args.stride_frames,
            inference_steps=args.inference_steps,
            guidance_scale=args.guidance_scale,
        )
        destination = Path(args.output_dir) / f"sample_{index:05d}"
        save_outputs(destination, generated, latent, sample, dataset, config)
        print(f"Saved {tuple(generated.shape)} motion to {destination}")
        if args.render:
            from .render_cli import render_sample
            from .rendering import RenderSettings

            render_sample(
                destination,
                args.smplh_model,
                settings=RenderSettings(
                    fps=config["model"]["pose_fps"], backend=args.render_backend
                ),
            )


if __name__ == "__main__":
    main()
