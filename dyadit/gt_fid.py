"""Ground-truth references for the exact protocol used by motion-FID.

The evaluation reference set stays fixed. Each real-real trial draws a disjoint
set of equally many validation clips. Its variation is conditional on that
reference set; neither it nor codec reconstruction is a strict generator bound.
This standalone evaluation needs only the codec, not the DiT or speech models.
"""

import argparse
import csv
import json
import os
import tempfile
from bisect import bisect_right
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .config import load_config
from .data import DyadicDataset, load_motion, motion_name
from .factory import build_codec
from .fid import (
    _tensor_digest,
    feature_statistics,
    frechet_distance,
    gaussian_frechet_distance,
    motion_features,
)


class MotionClips:
    """The existing dataset's target clips, without reading any audio or style.

    Cache each validation motion file once: legacy NPZ inputs otherwise require
    decompressing an entire conversation for every 96-frame clip.
    """

    def __init__(self, dataset):
        self.dataset = dataset
        self.cache = {}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        dataset = self.dataset
        if not 0 <= index < len(dataset):
            raise IndexError(index)
        segment, reverse = divmod(index, 2) if dataset.reverse_pairs else (index, 0)
        pair = bisect_right(dataset.cumulative, segment) - 1
        start = dataset.starts[pair][segment - dataset.cumulative[pair]]
        path = dataset.pairs[pair][2 if reverse else 0]
        if path not in self.cache:
            self.cache[path] = load_motion(path)
        motion = self.cache[path][start : start + dataset.pose_length]
        if dataset.normalize_motion:
            motion = (motion - dataset.motion_mean) / (dataset.motion_std + 1e-7)
        return {
            "motion": torch.from_numpy(np.array(motion, dtype=np.float32)),
            "video_id": f"{motion_name(path)}_{'B-A' if reverse else 'A-B'}_{start}",
        }


def gt_trials(reference, candidates, count, trials, seed):
    """Resample the candidate set without replacement within each trial."""
    reference, candidates = np.asarray(reference), np.asarray(candidates)
    if (
        reference.ndim != 2
        or candidates.ndim != 2
        or reference.shape[1] != candidates.shape[1]
        or count != len(reference)
        or count < 2
        or len(candidates) < count
        or trials < 2
    ):
        raise ValueError("Need at least two trials and disjoint equal-size GT sets")
    reference_statistics = feature_statistics(reference)
    feature_statistics(candidates)  # Reject nonfinite candidate features as well.
    rng = np.random.default_rng(seed)
    selections, scores = [], []
    for _ in range(trials):
        selected = np.sort(rng.choice(len(candidates), count, replace=False))
        candidate_statistics = feature_statistics(candidates[selected])
        scores.append(
            gaussian_frechet_distance(*reference_statistics, *candidate_statistics)
        )
        selections.append(selected)
    values = np.asarray(scores)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "median": float(np.median(values)),
        "p025": float(np.quantile(values, 0.025)),
        "p975": float(np.quantile(values, 0.975)),
        "min": float(values.min()),
        "max": float(values.max()),
        "first_split": float(values[0]),
        "scores": values.tolist(),
        "candidate_positions": np.stack(selections),
    }


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        json.dump(value, stream, indent=2)
        temporary = stream.name
    os.replace(temporary, path)


@torch.inference_mode()
def evaluate(run_dir, output_dir, trials=100, seed=223, batch_size=32, device="cpu"):
    run_dir, output_dir = Path(run_dir), Path(output_dir)
    config = load_config(run_dir / "config.yaml")
    metric_dir = run_dir / "metrics/fid_codec"
    protocol = json.loads((metric_dir / "protocol.json").read_text())
    if protocol["feature_extractor"] != "frozen_vqvae_encoder_prequant" or (
        protocol["feature_pooling"] != "temporal_mean_and_population_std"
    ):
        raise ValueError("Unsupported motion-FID feature protocol")
    codec = build_codec(config).to(device).eval()
    if _tensor_digest(codec.encoder.state_dict().items()) != protocol["encoder_sha256"]:
        raise ValueError("Pretrained codec encoder differs from the training evaluator")
    codec_digest = _tensor_digest(codec.state_dict().items())
    dataset = DyadicDataset(
        config,
        data_root=config["paths"]["val_data_root"],
        pose_length=protocol["pose_length"],
        random_style=False,
    )
    clips = MotionClips(dataset)
    indices = np.asarray(protocol["validation_indices"], dtype=np.int64)
    count = protocol["num_samples"]
    if len(indices) != count or len(np.unique(indices)) != count:
        raise ValueError("Invalid evaluation reference indices")
    candidates = np.setdiff1d(np.arange(len(clips)), indices)
    if len(candidates) < count:
        raise ValueError(
            f"Need {count} disjoint candidate clips, found {len(candidates)}"
        )
    video_ids = [clips[int(index)]["video_id"] for index in indices]
    if video_ids != protocol["video_ids"]:
        raise ValueError("Validation clip ordering differs from the training evaluator")
    with np.load(metric_dir / "reference_statistics.npz") as archive:
        reference = archive["features"].copy()
        reference_mean, reference_covariance = archive["mean"], archive["covariance"]
    if reference.shape != (count, protocol["feature_dim"]):
        raise ValueError("Cached reference feature shape differs from protocol")
    mean, covariance = feature_statistics(reference)
    np.testing.assert_allclose(mean, reference_mean)
    np.testing.assert_allclose(covariance, reference_covariance)
    excluded = protocol["excluded_prefix_frames"]
    features = []
    for start in range(0, len(clips), batch_size):
        motion = torch.stack(
            [
                clips[index]["motion"]
                for index in range(start, min(start + batch_size, len(clips)))
            ]
        ).to(device)
        features.append(motion_features(codec.encoder, motion, excluded))
        if start == 0 or (start // batch_size + 1) % 8 == 0:
            print(
                f"[GT FID] encoded {min(start + batch_size, len(clips))}/{len(clips)} GT clips",
                flush=True,
            )
    features = np.concatenate(features)
    # CUDA convolution (including TF32) and CPU FP32 need not be bitwise equal.
    # Check paired feature agreement AND the resulting FID: keep the exact
    # cached CUDA reference and refuse a material backend distribution shift.
    np.testing.assert_allclose(features[indices], reference, rtol=2e-3, atol=1e-2)
    reference_error = float(np.max(np.abs(features[indices] - reference)))
    alignment_fid = frechet_distance(reference, features[indices])
    print(
        f"[GT FID] CPU/cache alignment FID: {alignment_fid:.8f}; max feature difference: {reference_error:.8f}",
        flush=True,
    )
    if alignment_fid > 0.01:
        raise ValueError(
            f"CPU/cache alignment FID {alignment_fid} exceeds 0.01; use the training backend"
        )
    result = gt_trials(reference, features[candidates], count, trials, seed)
    selections = candidates[result.pop("candidate_positions")]
    reconstructed = []
    for start in range(0, count, batch_size):
        motion = torch.stack(
            [
                clips[int(index)]["motion"]
                for index in indices[start : start + batch_size]
            ]
        ).to(device)
        decoded = codec.decode(codec.encode(motion))
        reconstructed.append(motion_features(codec.encoder, decoded, excluded))
    reconstruction = np.concatenate(reconstructed)
    if _tensor_digest(codec.state_dict().items()) != codec_digest:
        raise RuntimeError("Codec weights changed during GT evaluation")
    result = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "protocol_hash": protocol["protocol_hash"],
        "encoder_sha256": protocol["encoder_sha256"],
        "codec_sha256": codec_digest,
        "run_dir": str(run_dir.resolve()),
        "num_samples_per_set": count,
        "validation_samples": len(clips),
        "candidate_samples": len(candidates),
        "feature_dim": reference.shape[1],
        "pose_length": protocol["pose_length"],
        "excluded_prefix_frames": excluded,
        "seed": seed,
        "trials": trials,
        "device": str(device),
        "reference_cpu_max_abs_error": reference_error,
        "reference_cpu_alignment_fid": alignment_fid,
        "gt_self_fid": frechet_distance(reference, reference),
        "gt_gt": result,
        "codec_reconstruction_fid": frechet_distance(reference, reconstruction),
        "reference_indices": indices.tolist(),
        "candidate_indices": candidates.tolist(),
        "trial_indices": selections.tolist(),
        "notes": [
            "Real-real reference uses the exact fixed GT set from training evaluation.",
            "Each trial uses equally many different validation clip indices; conversations/subjects may be shared.",
            "Mean/std/percentiles describe candidate resampling conditional on that fixed reference, not a confidence interval or strict model lower bound.",
            "The exact training reference features are retained. CPU/CUDA differences are checked pointwise and must yield an alignment FID <= 0.01.",
            "Codec reconstruction uses GT -> VQ encoder -> discrete codebook -> decoder -> same evaluation encoder.",
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "result.json", result)
    np.savez_compressed(
        output_dir / "features.npz",
        validation_features=features,
        reference_features=reference,
        reconstructed_features=reconstruction,
    )
    with (output_dir / "gt_gt_trials.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["trial", "fid_gt_gt"])
        writer.writerows(enumerate(result["gt_gt"]["scores"], 1))
    return result


def plot_comparison(result, run_dirs, output_dir):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    figure = Figure(figsize=(9, 5), tight_layout=True)
    FigureCanvasAgg(figure)
    axis = figure.subplots()
    plotted = []
    for directory in run_dirs:
        directory = Path(directory)
        protocol = json.loads(
            (directory / "metrics/fid_codec/protocol.json").read_text()
        )
        if protocol["protocol_hash"] != result["protocol_hash"]:
            raise ValueError("Cannot plot experiments with different FID protocols")
        config = load_config(directory / "config.yaml")
        label = (
            "Full model (ORCA)" if config["model"]["orthogonal_fusion"] else "No ORCA"
        )
        history = json.loads((directory / "metrics/fid_codec/history.json").read_text())
        axis.plot(
            [row["epoch"] for row in history],
            [row["fid_codec"] for row in history],
            label=label,
            linewidth=1.2,
        )
        plotted.append(
            {"label": label, "run_dir": str(directory.resolve()), "history": history}
        )
    gt = result["gt_gt"]
    axis.axhline(
        gt["mean"],
        color="tab:green",
        linestyle="--",
        label=f"GT-GT mean: {gt['mean']:.3f}",
    )
    axis.axhspan(
        gt["p025"],
        gt["p975"],
        color="tab:green",
        alpha=0.12,
        label="GT-GT 2.5-97.5% resampling range",
    )
    axis.axhline(
        result["codec_reconstruction_fid"],
        color="tab:purple",
        linestyle=":",
        label=f"Codec reconstruction: {result['codec_reconstruction_fid']:.3f}",
    )
    axis.set(
        xlabel="Completed epoch",
        ylabel="Codec motion-FID (lower is better)",
        title=f"Fixed reference: {result['num_samples_per_set']} clips | {result['trials']} GT-GT trials",
        ylim=(0, None),
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    destination = Path(output_dir) / "fid_with_gt_baseline.png"
    figure.savefig(destination, dpi=160)
    figure.clear()
    write_json(Path(output_dir) / "comparison_snapshot.json", plotted)
    return destination


def publish(result, output_dir, image_path):
    import wandb

    output_dir = Path(output_dir)
    snapshots = json.loads((output_dir / "comparison_snapshot.json").read_text())
    gt = result["gt_gt"]
    metrics = {
        "fid_reference/gt_self": result["gt_self_fid"],
        "fid_reference/gt_gt_mean": gt["mean"],
        "fid_reference/gt_gt_std": gt["std"],
        "fid_reference/gt_gt_p025": gt["p025"],
        "fid_reference/gt_gt_p975": gt["p975"],
        "fid_reference/codec_reconstruction": result["codec_reconstruction_fid"],
    }
    # Use a separate run: concurrent SDK writers must not resume the active
    # training run or reuse its native history step.
    run = wandb.init(
        id=wandb.util.generate_id(),
        resume="never",
        entity="ohicarip",
        project="wan_dit_audio2pose",
        name=f"GT FID baseline ({result['protocol_hash'][:8]})",
        job_type="ground_truth_fid_baseline",
        tags=["gt-baseline", "codec-fid"],
        dir=str(output_dir),
        config={
            key: result[key]
            for key in (
                "protocol_hash",
                "num_samples_per_set",
                "feature_dim",
                "pose_length",
                "excluded_prefix_frames",
                "seed",
                "trials",
            )
        },
    )
    try:
        run.log({**metrics, "fid_reference/comparison": wandb.Image(str(image_path))})
        table = wandb.Table(
            columns=["trial", "gt_gt_fid"],
            data=[[index, score] for index, score in enumerate(gt["scores"], 1)],
        )
        run.log({"fid_reference/gt_gt_trials": table})
        run.define_metric("fid/epoch")
        for name in (
            "fid/full_model",
            "fid/no_orca",
            "fid/gt_gt_baseline",
            "fid/codec_reconstruction",
        ):
            run.define_metric(name, step_metric="fid/epoch")
        rows = {}
        for snapshot in snapshots:
            key = (
                "fid/full_model"
                if snapshot["label"] == "Full model (ORCA)"
                else "fid/no_orca"
            )
            for row in snapshot["history"]:
                rows.setdefault(row["epoch"], {})[key] = row["fid_codec"]
        for epoch in sorted(rows):
            run.log(
                {
                    "fid/epoch": epoch,
                    **rows[epoch],
                    "fid/gt_gt_baseline": gt["mean"],
                    "fid/codec_reconstruction": result["codec_reconstruction_fid"],
                }
            )
        artifact = wandb.Artifact(
            f"gt-fid-{result['protocol_hash'][:8]}", type="fid_reference"
        )
        for name in (
            "result.json",
            "gt_gt_trials.csv",
            "fid_with_gt_baseline.png",
            "comparison_snapshot.json",
        ):
            artifact.add_file(str(output_dir / name))
        run.log_artifact(artifact)
        url = run.url
    finally:
        run.finish()
    write_json(
        output_dir / "wandb_publication.json",
        {"url": url, "run_id": run.id, "summary": metrics},
    )
    print(f"[GT FID] W&B: {url}", flush=True)
    return url


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--compare-run-dir", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=223)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--publish-only", action="store_true")
    args = parser.parse_args()
    if args.trials < 2 or args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("trials must be >= 2; batch-size and cpu-threads must be positive")
    if args.publish_only and not args.wandb:
        parser.error("--publish-only requires --wandb")
    output = args.output_dir or args.run_dir / "metrics/fid_codec/gt_baseline"
    torch.set_num_threads(args.cpu_threads)
    if args.publish_only:
        result = json.loads((output / "result.json").read_text())
    else:
        result = evaluate(
            args.run_dir, output, args.trials, args.seed, args.batch_size, args.device
        )
    image_path = plot_comparison(result, [args.run_dir, *args.compare_run_dir], output)
    print(
        json.dumps(
            {
                "gt_self_fid": result["gt_self_fid"],
                "gt_gt": {
                    key: value
                    for key, value in result["gt_gt"].items()
                    if key != "scores"
                },
                "codec_reconstruction_fid": result["codec_reconstruction_fid"],
                "plot": str(image_path),
            },
            indent=2,
        ),
        flush=True,
    )
    if args.wandb:
        publish(result, output, image_path)
