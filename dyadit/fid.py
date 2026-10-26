"""Per-epoch motion Fréchet distance using a fixed, pretrained codec encoder.

This is a codec-feature motion-FID, not Inception image-FID or a benchmark's
external gesture evaluator. Both decoded predictions and real motion pass
through the same frozen, pre-quantization encoder and temporal mean/std pooling.
"""

import csv
import hashlib
import json
import logging
import os
import random
import tempfile
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader, Subset

from .visualization import preview_context


_LOG = logging.getLogger(__name__)
INPUT_KEYS = (
    "audio",
    "audio_other",
    "motion_latent",
    "motion_latent_other",
    "style_latent",
    "relationship_onehot",
    "personality_traits",
)


def feature_statistics(features):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or features.shape[0] < 2 or features.shape[1] < 1:
        raise ValueError("FID features must have shape [at least 2 samples, features]")
    if not np.isfinite(features).all():
        raise ValueError("FID features must be finite")
    return features.mean(0), np.atleast_2d(np.cov(features, rowvar=False, ddof=1))


def gaussian_frechet_distance(
    mean_real, covariance_real, mean_generated, covariance_generated
):
    """Float64 Gaussian Fréchet distance, including singular PSD covariances.

    Use the symmetric PSD sandwich instead of sqrtm(C_real @ C_generated),
    avoiding spurious complex results from a nonsymmetric covariance product.
    """
    first, second = np.asarray(mean_real, dtype=np.float64), np.asarray(
        mean_generated, dtype=np.float64
    )
    real, generated = np.asarray(covariance_real, dtype=np.float64), np.asarray(
        covariance_generated, dtype=np.float64
    )
    if (
        first.ndim != 1
        or not len(first)
        or first.shape != second.shape
        or real.shape != (len(first), len(first))
        or generated.shape != real.shape
    ):
        raise ValueError("FID mean and covariance dimensions must match")
    if not all(np.isfinite(value).all() for value in (first, second, real, generated)):
        raise ValueError("FID statistics must be finite")
    real, generated = (real + real.T) / 2, (generated + generated.T) / 2
    eigenvalues, eigenvectors = np.linalg.eigh(real)
    root_real = (eigenvectors * np.sqrt(np.maximum(eigenvalues, 0))) @ eigenvectors.T
    sandwich = root_real @ generated @ root_real
    product_eigenvalues = np.linalg.eigvalsh((sandwich + sandwich.T) / 2)
    trace_root = np.sqrt(np.maximum(product_eigenvalues, 0)).sum()
    covariance_distance = max(
        float(np.trace(real) + np.trace(generated) - 2 * trace_root), 0.0
    )
    return float(np.square(first - second).sum() + covariance_distance)


def frechet_distance(real_features, generated_features):
    real = feature_statistics(real_features)
    generated = feature_statistics(generated_features)
    return gaussian_frechet_distance(*real, *generated)


def motion_features(encoder, motion, excluded_prefix_frames=0):
    """One feature vector per clip, always computed from motion in float32."""
    if motion.ndim != 3 or not 0 <= excluded_prefix_frames < motion.shape[1]:
        raise ValueError(
            "Expected [batch, frames, motion_features] with a valid excluded prefix"
        )
    with torch.autocast(device_type=motion.device.type, enabled=False):
        encoded = encoder(motion[:, excluded_prefix_frames:].float())
        features = torch.cat([encoded.mean(1), encoded.std(1, unbiased=False)], dim=-1)
    if not torch.isfinite(features).all():
        raise ValueError("Non-finite codec features during FID evaluation")
    return features.detach().double().cpu().numpy()


def _tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class MotionFIDCallback(Callback):
    """Fixed validation subset + fixed diffusion noise, evaluated once per epoch.

    Rank zero generates actual motions; other ranks wait and receive the same
    result/error. CPU validation inputs and real features are cached once. This
    callback does not render videos or change the train/validation-loss loaders.
    """

    metric_name = "val/fid_codec"

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings = config.get("fid", {})
        self.every_n_epochs = self.settings.get("every_n_epochs", 1)
        self.num_samples = self.settings.get("num_samples", 256)
        self.batch_size = self.settings.get("batch_size", 32)
        self.pose_length = self.settings.get(
            "pose_length", config["model"]["pose_length"]
        )
        self.inference_steps = self.settings.get("inference_steps", 50)
        self.cpu_threads = self.settings.get("cpu_threads", 4)
        self.seed = self.settings.get("seed", config["seed"])
        self.window_frames = self.settings.get(
            "window_frames", config["model"]["pose_length"]
        )
        factor = config["vqvae"]["downsample_factor"]
        prefix = config["model"]["mask_length"] * factor
        self.stride_frames = self.settings.get(
            "stride_frames", max(factor, self.window_frames - prefix)
        )
        self.reference_prefix = self.settings.get("reference_prefix", True)
        self.excluded_prefix_frames = (
            prefix
            if self.reference_prefix
            and self.settings.get("exclude_reference_prefix", True)
            else 0
        )
        self.num_workers = self.settings.get(
            "num_workers", min(config["data"].get("num_workers", 0), 4)
        )
        for name in (
            "every_n_epochs",
            "num_samples",
            "batch_size",
            "pose_length",
            "inference_steps",
            "cpu_threads",
            "window_frames",
            "stride_frames",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"fid.{name} must be a positive integer")
        if self.num_samples < 2:
            raise ValueError("fid.num_samples must be at least 2")
        if (
            not isinstance(self.num_workers, int)
            or isinstance(self.num_workers, bool)
            or self.num_workers < 0
        ):
            raise ValueError("fid.num_workers must be a non-negative integer")
        for name in ("pose_length", "window_frames", "stride_frames"):
            if getattr(self, name) % factor:
                raise ValueError(f"fid.{name} must be divisible by the codec factor")
        if self.pose_length - self.excluded_prefix_frames < factor:
            raise ValueError(
                "FID needs motion frames after the excluded reference prefix"
            )
        if (
            self.window_frames <= prefix
            or not 0 < self.stride_frames <= self.window_frames
        ):
            raise ValueError("Invalid FID sampling window/stride")
        if 0 < self.window_frames - self.stride_frames < prefix:
            raise ValueError(
                "FID windows must overlap by at least the reference prefix"
            )
        if (
            self.inference_steps
            > config["training"]["diffusion"]["num_train_timesteps"]
        ):
            raise ValueError("fid.inference_steps exceeds num_train_timesteps")
        self.directory = Path(config["paths"]["output_dir"]) / "metrics" / "fid_codec"
        self.last_evaluated_epoch = 0
        self.last_score = None
        self.protocol_hash = None
        self.restored_protocol_hash = None
        self.encoder = None
        self.batches = []
        self.indices = []
        self.history = []
        self.loggers = []

    def state_dict(self):
        return {
            "last_evaluated_epoch": self.last_evaluated_epoch,
            "last_score": self.last_score,
            "protocol_hash": self.protocol_hash,
        }

    def load_state_dict(self, state_dict):
        self.last_evaluated_epoch = state_dict.get("last_evaluated_epoch", 0)
        self.last_score = state_dict.get("last_score")
        self.restored_protocol_hash = state_dict.get("protocol_hash")

    def _synchronize(self, trainer, result):
        result = trainer.strategy.broadcast(result, src=0) or {}
        if result.get("error"):
            raise RuntimeError(f"Motion FID evaluation failed: {result['error']}")
        return result

    def on_fit_start(self, trainer, pl_module):
        result = None
        if trainer.is_global_zero:
            try:
                with preview_context(pl_module, self.seed, self.cpu_threads):
                    self._prepare(trainer, pl_module)
                result = {"error": None, "protocol_hash": self.protocol_hash}
            except Exception as error:
                _LOG.exception("Could not initialize motion FID")
                result = {"error": f"{type(error).__name__}: {error}"}
        result = self._synchronize(trainer, result)
        self.protocol_hash = result.get("protocol_hash", self.protocol_hash)

    def _prepare(self, trainer, module):
        # Imported here so disabling FID does not require the plotting dependency.
        from matplotlib.figure import Figure  # noqa: F401

        dataset = getattr(trainer.datamodule, "val_dataset", None)
        factory = getattr(trainer.datamodule, "get_visualization_dataset", None)
        if dataset is not None and factory is not None:
            dataset = factory(self.pose_length)
        if dataset is None or len(dataset) < 2:
            raise ValueError("FID requires at least two validation samples")
        if getattr(dataset, "random_style", False):
            raise ValueError("FID validation style sampling must be deterministic")
        if any(
            parameter.requires_grad for parameter in module.vqvae.encoder.parameters()
        ):
            raise ValueError("FID needs a frozen pretrained codec encoder")
        self.encoder = deepcopy(module.vqvae.encoder).requires_grad_(False).eval()
        count = min(self.num_samples, len(dataset))
        self.indices = sorted(
            random.Random(self.seed).sample(range(len(dataset)), count)
        )
        loader = DataLoader(
            Subset(dataset, self.indices),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            generator=torch.Generator().manual_seed(self.seed),
        )
        video_ids, real_features = [], []
        for batch in loader:
            if batch["motion_latent"].shape[1] != self.pose_length:
                raise ValueError(
                    f"FID dataset must provide {self.pose_length}-frame clips"
                )
            self.batches.append({key: batch[key] for key in INPUT_KEYS})
            video_ids.extend(batch["video_id"])
            real_features.append(
                motion_features(
                    self.encoder,
                    batch["motion_latent"].to(module.device),
                    self.excluded_prefix_frames,
                )
            )
        self.real_features = np.concatenate(real_features)
        self.real_mean, self.real_covariance = feature_statistics(self.real_features)
        if count <= self.real_features.shape[1]:
            _LOG.warning(
                "FID uses %s clips for %s-dimensional features; covariance is rank deficient. Use more clips for reliable comparisons.",
                count,
                self.real_features.shape[1],
            )
        self.protocol = {
            "metric": self.metric_name,
            "feature_extractor": "frozen_vqvae_encoder_prequant",
            "feature_pooling": "temporal_mean_and_population_std",
            "feature_dim": self.real_features.shape[1],
            "encoder_sha256": _tensor_digest(self.encoder.state_dict().items()),
            "input_sha256": _tensor_digest(
                (f"{index}/{key}", batch[key])
                for index, batch in enumerate(self.batches)
                for key in INPUT_KEYS
            ),
            "normalization": "same 258D motion normalization as training and the pretrained codec",
            "normalized_motion": bool(
                getattr(
                    dataset,
                    "normalize_motion",
                    self.config["data"].get("normalize_motion", True),
                )
            ),
            "validation_indices": self.indices,
            "video_ids": video_ids,
            "num_samples": count,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "pose_length": self.pose_length,
            "excluded_prefix_frames": self.excluded_prefix_frames,
            "window_frames": self.window_frames,
            "stride_frames": self.stride_frames,
            "inference_steps": self.inference_steps,
            "guidance_scale": self.settings.get("guidance_scale", 1.0),
            "reference_prefix": self.reference_prefix,
            "note": "Experiment-specific codec motion-FID; not directly comparable with other feature extractors or benchmark FID/FGD.",
        }
        self.protocol_hash = hashlib.sha256(
            json.dumps(self.protocol, sort_keys=True).encode()
        ).hexdigest()
        protocol_file = self.directory / "protocol.json"
        previous = (
            json.loads(protocol_file.read_text()) if protocol_file.exists() else None
        )
        if (
            self.restored_protocol_hash
            and self.restored_protocol_hash != self.protocol_hash
        ) or (previous and previous["protocol_hash"] != self.protocol_hash):
            raise ValueError(
                "FID protocol changed (features, samples or sampling settings). Use a separate output/run for a comparable curve."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
        self._write_json(
            "protocol.json", {"protocol_hash": self.protocol_hash, **self.protocol}
        )
        np.savez_compressed(
            self.directory / "reference_statistics.npz",
            mean=self.real_mean,
            covariance=self.real_covariance,
            features=self.real_features,
        )
        history_file = self.directory / "history.json"
        self.history = (
            json.loads(history_file.read_text()) if history_file.exists() else []
        )
        self.loggers = list(trainer.loggers)
        for logger in self.loggers:
            if isinstance(logger, WandbLogger):
                logger.experiment.define_metric("fid/epoch")
                logger.experiment.define_metric(
                    self.metric_name, step_metric="fid/epoch", summary="min"
                )
        print(
            f"[FID] Ready: {count} fixed validation clips, {self.real_features.shape[1]} features; exclude first {self.excluded_prefix_frames} frames; inputs cached on CPU",
            flush=True,
        )

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if (
            trainer.sanity_checking
            or trainer.fast_dev_run
            or epoch % self.every_n_epochs
            or epoch <= self.last_evaluated_epoch
        ):
            return
        result = None
        if trainer.is_global_zero:
            try:
                with preview_context(pl_module, self.seed, self.cpu_threads):
                    score = self._evaluate(trainer, pl_module, epoch)
                    self._record(trainer, epoch, score)
                result = {"error": None, "score": score}
            except Exception as error:
                _LOG.exception("Could not evaluate epoch %s motion FID", epoch)
                result = {"error": f"{type(error).__name__}: {error}"}
        result = self._synchronize(trainer, result)
        self.last_score = result.get("score")
        self.last_evaluated_epoch = epoch

    def _evaluate(self, trainer, module, epoch):
        generated_features = []
        for index, cached in enumerate(self.batches):
            batch = {key: value.to(module.device) for key, value in cached.items()}
            with trainer.precision_plugin.forward_context():
                generated, _ = module.generate(
                    batch["audio"],
                    batch["audio_other"],
                    self.pose_length,
                    style_motion=batch["style_latent"],
                    motion_other=batch["motion_latent_other"],
                    relationship=batch["relationship_onehot"],
                    personality=batch["personality_traits"],
                    reference_motion=(
                        batch["motion_latent"] if self.reference_prefix else None
                    ),
                    window_frames=self.window_frames,
                    stride_frames=self.stride_frames,
                    inference_steps=self.inference_steps,
                    guidance_scale=self.settings.get("guidance_scale", 1.0),
                )
            generated_features.append(
                motion_features(self.encoder, generated, self.excluded_prefix_frames)
            )
            print(
                f"[FID] epoch {epoch}: generated {min((index + 1) * self.batch_size, len(self.indices))}/{len(self.indices)} clips",
                flush=True,
            )
        mean, covariance = feature_statistics(np.concatenate(generated_features))
        return gaussian_frechet_distance(
            self.real_mean, self.real_covariance, mean, covariance
        )

    def _write_json(self, name, value):
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.directory, delete=False, suffix=".json"
        ) as stream:
            json.dump(value, stream, indent=2)
            temporary = stream.name
        os.replace(temporary, self.directory / name)

    def _record(self, trainer, epoch, score):
        row = {
            "epoch": epoch,
            "global_step": int(trainer.global_step),
            "fid_codec": score,
            "num_samples": len(self.indices),
            "feature_dim": self.real_features.shape[1],
            "protocol_hash": self.protocol_hash,
        }
        self.history = sorted(
            [previous for previous in self.history if previous["epoch"] != epoch]
            + [row],
            key=lambda value: value["epoch"],
        )
        self._write_json("history.json", self.history)
        with tempfile.NamedTemporaryFile(
            mode="w", dir=self.directory, delete=False, suffix=".csv", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(self.history)
            temporary = stream.name
        os.replace(temporary, self.directory / "fid_codec.csv")
        plot = self._plot()
        metrics = {
            self.metric_name: score,
            "fid/epoch": epoch,
            "fid/num_samples": len(self.indices),
            "fid/feature_dim": self.real_features.shape[1],
        }
        for logger in self.loggers:
            logger.log_metrics(metrics, step=trainer.global_step)
            if isinstance(logger, WandbLogger):
                logger.log_image(
                    "fid/curve",
                    images=[str(plot)],
                    step=trainer.global_step,
                    caption=[
                        "Codec motion-FID vs epoch (lower is better; fixed validation subset)"
                    ],
                )
        print(
            f"[FID] epoch {epoch}: {self.metric_name}={score:.6f}, n={len(self.indices)}",
            flush=True,
        )

    def _plot(self):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(7, 4), tight_layout=True)
        FigureCanvasAgg(figure)
        axis = figure.subplots()
        axis.plot(
            [row["epoch"] for row in self.history],
            [row["fid_codec"] for row in self.history],
            marker="o",
            markersize=3,
        )
        axis.set(
            xlabel="Epoch",
            ylabel="Codec motion-FID (lower is better)",
            title=f"Fixed VQ-VAE features | {len(self.indices)} validation clips",
        )
        axis.grid(alpha=0.3)
        with tempfile.NamedTemporaryFile(
            dir=self.directory, suffix=".png", delete=False
        ) as stream:
            temporary = stream.name
        figure.savefig(temporary, dpi=140)
        figure.clear()
        destination = self.directory / "fid_codec.png"
        os.replace(temporary, destination)
        return destination
