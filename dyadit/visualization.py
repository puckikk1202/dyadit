"""Periodic, reproducible validation samples with audible SMPL-H videos in W&B."""

import importlib.util
import json
import logging
import os
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger

from .export import save_outputs
from .render_cli import render_sample
from .rendering import RenderSettings
from .smplh import resolve_model_path
from .video import ffmpeg_executable


_LOG = logging.getLogger(__name__)


@contextmanager
def preview_context(module, seed, cpu_threads):
    """Keep preview sampling out of the training RNG and restore all train modes."""
    python_state, numpy_state = random.getstate(), np.random.get_state()
    modes = [(child, child.training) for child in module.modules()]
    original_threads = torch.get_num_threads()
    devices = []
    if module.device.type == "cuda":
        device = module.device.index
        devices = [torch.cuda.current_device() if device is None else device]
    try:
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            random.seed(seed)
            np.random.seed(seed % (2**32))
            # torch.manual_seed also changes other GPUs' RNGs. Only seed the
            # current rank's device, whose state fork_rng will restore.
            torch.random.default_generator.manual_seed(seed)
            for device in devices:
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(seed)
            torch.set_num_threads(cpu_threads)
            module.eval()
            yield
    finally:
        for child, mode in modes:
            child.training = mode
        torch.set_num_threads(original_threads)
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class ValidationRenderCallback(Callback):
    """Sample fixed validation clips after epochs 5, 10, ... on global rank zero.

    Call the unwrapped Lightning module's sampler, not the DDP forward wrapper.
    Other ranks wait in a collective, including on failure, before training resumes.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.settings = config.get("visualization", {})
        self.every_n_epochs = self.settings.get("every_n_epochs", 5)
        self.num_samples = self.settings.get("num_samples", 10)
        self.seed = self.settings.get("seed", config["seed"])
        self.inference_steps = self.settings.get("inference_steps", 50)
        self.cpu_threads = self.settings.get("cpu_threads", 4)
        self.pose_length = self.settings.get(
            "pose_length", config["model"]["pose_length"]
        )
        self.window_frames = self.settings.get(
            "window_frames", config["model"]["pose_length"]
        )
        factor = config["vqvae"]["downsample_factor"]
        prefix_frames = config["model"]["mask_length"] * factor
        self.stride_frames = self.settings.get(
            "stride_frames", max(factor, self.window_frames - prefix_frames)
        )
        for name in (
            "every_n_epochs",
            "num_samples",
            "inference_steps",
            "cpu_threads",
            "pose_length",
            "window_frames",
            "stride_frames",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"visualization.{name} must be a positive integer")
        for name in ("pose_length", "window_frames", "stride_frames"):
            if getattr(self, name) % factor:
                raise ValueError(
                    f"visualization.{name} must be divisible by the codec factor ({factor})"
                )
        if self.window_frames <= prefix_frames:
            raise ValueError(
                "visualization.window_frames must exceed the reference prefix length"
            )
        if self.stride_frames > self.window_frames:
            raise ValueError(
                "visualization.stride_frames must not exceed window_frames"
            )
        if 0 < self.window_frames - self.stride_frames < prefix_frames:
            raise ValueError(
                "Visualization windows must overlap by at least the reference prefix length"
            )
        if (
            self.inference_steps
            > config["training"]["diffusion"]["num_train_timesteps"]
        ):
            raise ValueError(
                "visualization.inference_steps exceeds num_train_timesteps"
            )
        self.render_settings = RenderSettings(
            **{**self.settings.get("render", {}), "fps": config["model"]["pose_fps"]}
        )
        self.render_settings.validate()
        self.model_path = self.settings.get("model_path") or os.environ.get(
            "SMPLH_MODEL_DIR"
        )
        self.output_dir = Path(config["paths"]["output_dir"]) / "visualizations"
        self.dataset = None
        self.indices = []
        self.wandb_loggers = []
        self.last_logged_epoch = 0

    def state_dict(self):
        return {"last_logged_epoch": self.last_logged_epoch}

    def load_state_dict(self, state_dict):
        self.last_logged_epoch = state_dict.get("last_logged_epoch", 0)

    def _validate_renderer(self):
        if not self.model_path:
            raise ValueError(
                "Set visualization.model_path or SMPLH_MODEL_DIR to an SMPL-H model"
            )
        for gender in (self.render_settings.gender_a, self.render_settings.gender_b):
            resolve_model_path(self.model_path, gender)
        dependencies = ["smplx", "scipy", "imageio", "cv2"]
        if self.render_settings.backend in ("egl", "osmesa"):
            dependencies += ["pyrender", "trimesh"]
        missing = [
            name for name in dependencies if importlib.util.find_spec(name) is None
        ]
        if missing:
            raise ImportError(
                f"Missing rendering dependencies: {', '.join(missing)}. "
                "Install requirements-render.txt, or set visualization.enabled=false."
            )
        ffmpeg_executable()

    def _prepare(self, trainer):
        self.wandb_loggers = [
            logger for logger in trainer.loggers if isinstance(logger, WandbLogger)
        ]
        if not self.wandb_loggers:
            raise ValueError(
                "Training video previews require logger.type=wandb. "
                "For training without them, set visualization.enabled=false."
            )
        self._validate_renderer()
        self.dataset = getattr(trainer.datamodule, "val_dataset", None)
        if self.dataset is None or not len(self.dataset):
            raise ValueError(
                "Training video previews require a non-empty validation dataset"
            )
        factory = getattr(trainer.datamodule, "get_visualization_dataset", None)
        if factory is not None:
            self.dataset = factory(self.pose_length)
        if not len(self.dataset):
            raise ValueError(
                f"No validation clips cover visualization.pose_length={self.pose_length}"
            )
        if getattr(self.dataset, "pose_length", self.pose_length) != self.pose_length:
            raise ValueError(
                f"Visualization dataset must provide {self.pose_length}-frame clips"
            )
        if getattr(self.dataset, "random_style", False):
            raise ValueError(
                "Visualization validation data must use random_style=False"
            )
        count = min(self.num_samples, len(self.dataset))
        if count < self.num_samples:
            _LOG.warning(
                "Only %s validation samples available; rendering all of them", count
            )
        self.indices = sorted(
            random.Random(self.seed).sample(range(len(self.dataset)), count)
        )

    def _check_all_ranks(self, trainer, error):
        # This broadcast also acts as the end-of-preview synchronization. Never
        # leave nonzero ranks waiting at a barrier if rendering/uploading fails.
        error = trainer.strategy.broadcast(error, src=0)
        if error:
            raise RuntimeError(f"Training visualization failed: {error}")

    def on_fit_start(self, trainer, pl_module):
        error = None
        if trainer.is_global_zero:
            try:
                self._prepare(trainer)
            except Exception as exception:
                error = f"{type(exception).__name__}: {exception}"
        self._check_all_ranks(trainer, error)

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1  # User-facing epochs start at 1.
        if (
            trainer.sanity_checking
            or trainer.fast_dev_run
            or epoch % self.every_n_epochs
            or epoch <= self.last_logged_epoch
        ):
            return
        error = None
        if trainer.is_global_zero:
            try:
                with preview_context(pl_module, self.seed, self.cpu_threads):
                    self._render_and_log(trainer, pl_module, epoch)
            except Exception as exception:
                _LOG.exception(
                    "Could not generate/render/upload epoch %s previews", epoch
                )
                error = f"{type(exception).__name__}: {exception}"
        self._check_all_ranks(trainer, error)
        self.last_logged_epoch = epoch

    def _render_one(self, trainer, module, index, destination):
        sample = self.dataset[index]
        if len(sample["motion_latent"]) != self.pose_length:
            raise ValueError(
                f"Visualization dataset returned {len(sample['motion_latent'])} frames; expected {self.pose_length}"
            )
        batch = {
            key: value.unsqueeze(0).to(module.device)
            for key, value in sample.items()
            if isinstance(value, torch.Tensor)
        }
        with trainer.precision_plugin.forward_context():
            generated, latent = module.generate(
                batch["audio"],
                batch["audio_other"],
                self.pose_length,
                style_motion=batch["style_latent"],
                motion_other=batch["motion_latent_other"],
                relationship=batch["relationship_onehot"],
                personality=batch["personality_traits"],
                reference_motion=(
                    batch["motion_latent"]
                    if self.settings.get("reference_prefix", True)
                    else None
                ),
                inference_steps=self.inference_steps,
                guidance_scale=self.settings.get("guidance_scale", 1.0),
                window_frames=self.window_frames,
                stride_frames=self.stride_frames,
            )
        save_outputs(destination, generated, latent, sample, self.dataset, self.config)
        video = render_sample(
            destination, self.model_path, settings=self.render_settings, overwrite=True
        )
        return video, sample["video_id"]

    def _render_and_log(self, trainer, module, epoch):
        destination = self.output_dir / f"epoch_{epoch:04d}"
        videos, captions = [], []
        audio_caption = (
            "audio L=blue, R=orange"
            if self.render_settings.audio_layout == "stereo"
            else "audio: mixed"
        )
        for number, index in enumerate(self.indices):
            print(
                f"[visualization] epoch {epoch}: sample {number + 1}/{len(self.indices)} (validation index {index})",
                flush=True,
            )
            with preview_context(module, self.seed + index, self.cpu_threads):
                video, video_id = self._render_one(
                    trainer, module, index, destination / f"sample_{index:05d}"
                )
            videos.append(str(video))
            captions.append(
                f"Epoch {epoch} | validation {index} | {video_id} | "
                f"{self.pose_length} frames / {self.pose_length / self.config['model']['pose_fps']:g}s | "
                f"Blue/left: condition (B); orange/right: generated (A) | {audio_caption}"
            )
        for logger in self.wandb_loggers:
            # Passing an existing MP4 preserves its embedded audio. Arrays/GIFs
            # would discard the corresponding conversation sound.
            logger.log_video(
                "visualization/samples",
                videos=videos,
                step=trainer.global_step,
                caption=captions,
                format=["mp4"] * len(videos),
            )
            logger.log_metrics(
                {
                    "visualization/epoch": epoch,
                    "visualization/pose_length": self.pose_length,
                    "visualization/duration_seconds": self.pose_length
                    / self.config["model"]["pose_fps"],
                },
                step=trainer.global_step,
            )
        with (destination / "manifest.json").open("w") as stream:
            json.dump(
                {
                    "epoch": epoch,
                    "global_step": trainer.global_step,
                    "validation_indices": self.indices,
                    "seed": self.seed,
                    "inference_steps": self.inference_steps,
                    "pose_length": self.pose_length,
                    "window_frames": self.window_frames,
                    "stride_frames": self.stride_frames,
                    "duration_seconds": self.pose_length
                    / self.config["model"]["pose_fps"],
                    "guidance_scale": self.settings.get("guidance_scale", 1.0),
                    "reference_prefix": self.settings.get("reference_prefix", True),
                    "display": {
                        "blue_left": "conditioning_person_B",
                        "orange_right": "generated_person_A",
                    },
                    "audio_layout": self.render_settings.audio_layout,
                    "videos": videos,
                },
                stream,
                indent=2,
            )
