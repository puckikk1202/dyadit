"""Prepare normalization statistics and optional mmap-friendly motion/audio files."""

import argparse
from pathlib import Path

import numpy as np

from dyadit.data import load_motion, motion_directory, motion_name


def prepare(data_root, slim=False, stats_output=None):
    root = Path(data_root)
    # Prefer a complete legacy source when resuming a partially written slim directory.
    source = next(
        (root / name for name in ("motion_6d", "motion") if (root / name).is_dir()),
        motion_directory(root),
    )
    paths = sorted(p for p in source.iterdir() if p.suffix in (".npy", ".npz"))
    count = 0
    total, square_total = np.zeros(258), np.zeros(258)
    for path in paths:
        motion = load_motion(path).astype(np.float64)
        count += len(motion)
        total += motion.sum(axis=0)
        square_total += np.square(motion).sum(axis=0)
        if slim:
            destination = root / "motion_6d_slim" / (motion_name(path) + ".npy")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not destination.exists():
                np.save(destination, motion.astype(np.float32), allow_pickle=False)
    if not count:
        raise ValueError(f"No motion frames found under {root}")
    mean = total / count
    std = np.sqrt(np.maximum(square_total / count - np.square(mean), 0))
    std = np.where(std < 1e-6, 1.0, std)
    stats = Path(stats_output) if stats_output else root / "motion_6d_normalization.npz"
    stats.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stats,
        motion_mean=mean.astype(np.float32),
        motion_std=std.astype(np.float32),
        total_frames=count,
    )
    if slim:
        audio_dir = root / "audio_slim"
        audio_dir.mkdir(parents=True, exist_ok=True)
        for path in sorted((root / "audio_npz").glob("*.npz")):
            destination = audio_dir / (path.stem + ".npy")
            if destination.exists():
                continue
            with np.load(path, allow_pickle=True) as archive:
                key = next(
                    (
                        k
                        for k in ("audio", "audio_data", "signal", "waveform")
                        if k in archive
                    ),
                    None,
                )
                if key is None:
                    raise ValueError(f"No audio waveform found in {path}")
                np.save(
                    destination,
                    np.asarray(archive[key], dtype=np.float32).reshape(-1),
                    allow_pickle=False,
                )
    print(
        f"Prepared statistics from {len(paths)} motion files / {count} frames: {stats}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument(
        "--slim",
        action="store_true",
        help="Also extract raw .npy motion and audio arrays",
    )
    parser.add_argument(
        "--stats-output", help="Optional path for the training normalization statistics"
    )
    args = parser.parse_args()
    prepare(args.data_root, args.slim, args.stats_output)


if __name__ == "__main__":
    main()
