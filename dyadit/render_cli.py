"""Command-line entry for a two-person SMPL-H conversation video."""

import argparse
import os
from pathlib import Path

from .rendering import RenderSettings, render_pair


def sample_render_inputs(sample_dir):
    """Model B is the blue condition actor; generated model A is orange.

    Renderer slots A/B mean blue/left and orange/right, not the model's A/B.
    Move the corresponding waveforms with the motions, not just their colors.
    """
    directory = Path(sample_dir)

    def audio(name):
        path = directory / name
        return path if path.is_file() else None

    return {
        "motion_a": directory / "target_motion_other_denorm.npz",
        "motion_b": directory / "generated_motion_denorm.npz",
        "audio_a": audio("input_audio_other.wav"),
        "audio_b": audio("input_audio.wav"),
    }


def render_sample(
    sample_dir, model_path, output=None, *, settings=None, overwrite=False
):
    directory = Path(sample_dir)
    return render_pair(
        **sample_render_inputs(directory),
        model_path=model_path,
        output=output or directory / "conversation.mp4",
        settings=settings,
        overwrite=overwrite,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Render two facing SMPL-H people with fixed lower bodies"
    )
    parser.add_argument(
        "--sample-dir",
        help="Core sample: blue/left=condition person B, orange/right=generated person A",
    )
    parser.add_argument(
        "--motion-a", help="Blue/left actor motion (an exported SMPL-H NPZ)"
    )
    parser.add_argument(
        "--motion-b", help="Orange/right actor motion (an exported SMPL-H NPZ)"
    )
    parser.add_argument(
        "--audio-a", help="Blue/left actor audio, routed to the left channel"
    )
    parser.add_argument(
        "--audio-b", help="Orange/right actor audio, routed to the right channel"
    )
    parser.add_argument(
        "--audio-layout",
        choices=("stereo", "mix"),
        default="stereo",
        help="stereo: blue on left, orange on right; mix: blend both voices",
    )
    parser.add_argument(
        "--no-audio-normalize",
        dest="audio_normalize",
        action="store_false",
        help="Keep original voice levels instead of normalizing conversation loudness",
    )
    parser.add_argument(
        "--model-path",
        default=os.environ.get("SMPLH_MODEL_DIR"),
        help="SMPL-H model file/directory; or set SMPLH_MODEL_DIR",
    )
    parser.add_argument(
        "--output", help="Output MP4; default sample-dir/conversation.mp4"
    )
    parser.add_argument(
        "--backend", choices=("auto", "egl", "osmesa", "software"), default="auto"
    )
    parser.add_argument(
        "--device", default="cpu", help="Device for SMPL-H mesh generation"
    )
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--fps", type=float, default=30, help="Source motion frame rate"
    )
    parser.add_argument(
        "--separation",
        type=float,
        default=1.6,
        help="Pelvis-to-pelvis distance in meters",
    )
    parser.add_argument(
        "--gender-a", choices=("male", "female", "neutral"), default="male"
    )
    parser.add_argument(
        "--gender-b", choices=("male", "female", "neutral"), default="male"
    )
    parser.add_argument(
        "--lower-body", choices=("neutral", "first-frame"), default="neutral"
    )
    parser.add_argument(
        "--hand-mean",
        choices=("flat", "model"),
        default="flat",
        help="flat: absolute hand axis-angle; model: add the model's MANO means",
    )
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Subsample frames; output FPS is source FPS / step",
    )
    parser.add_argument(
        "--max-frames", type=int, default=0, help="Limit rendered frames; 0 means all"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="SMPL-H mesh inference batch size"
    )
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--camera-azimuth", type=float, default=12)
    parser.add_argument("--camera-elevation", type=float, default=10)
    parser.add_argument("--save-frames", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not args.model_path:
        parser.error("Pass --model-path or set SMPLH_MODEL_DIR")
    if not args.sample_dir and not (args.motion_a and args.motion_b):
        parser.error("Pass --sample-dir, or both --motion-a and --motion-b")
    if not args.output and not args.sample_dir:
        parser.error("Pass --output when rendering explicit motion files")
    if args.cpu_threads < 1:
        parser.error("cpu-threads must be positive")
    import torch

    torch.set_num_threads(args.cpu_threads)
    settings = RenderSettings(
        **{name: getattr(args, name) for name in RenderSettings.__dataclass_fields__}
    )
    sample = Path(args.sample_dir) if args.sample_dir else None
    inputs = sample_render_inputs(sample) if sample else {}
    motion_a = args.motion_a or inputs.get("motion_a")
    motion_b = args.motion_b or inputs.get("motion_b")

    try:
        path = render_pair(
            motion_a,
            motion_b,
            args.model_path,
            args.output or sample / "conversation.mp4",
            audio_a=args.audio_a or inputs.get("audio_a"),
            audio_b=args.audio_b or inputs.get("audio_b"),
            settings=settings,
            overwrite=args.overwrite,
        )
    except (ValueError, FileNotFoundError, FileExistsError, ImportError) as error:
        parser.error(str(error))
    print(f"Video: {path}\nPreview: {path.with_suffix('.png')}")


if __name__ == "__main__":
    main()
