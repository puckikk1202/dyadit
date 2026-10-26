"""Latent diffusion training and one sampler shared by all inference calls."""

from dataclasses import asdict, dataclass

import pytorch_lightning as pl
import torch
from diffusers import DDPMScheduler, DDIMScheduler, DPMSolverMultistepScheduler
from torch.nn import functional as F


@dataclass
class DiffusionConfig:
    num_train_timesteps: int = 1000
    objective: str = "eps"
    scheduler_type: str = "dpm_solver"
    beta_start: float = 0.0001
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    variance_type: str = "fixed_small"
    clip_sample: bool = False
    prediction_type: str = "epsilon"


def make_scheduler(settings, training=False):
    kwargs = asdict(settings)
    objective = kwargs.pop("objective")
    kind = "ddpm" if training else kwargs.pop("scheduler_type")
    kwargs.pop("scheduler_type", None)
    expected = {"eps": "epsilon", "x0": "sample", "v": "v_prediction"}
    if objective not in expected or kwargs["prediction_type"] != expected[objective]:
        raise ValueError("diffusion.objective and prediction_type must agree")
    if kind == "ddpm":
        return DDPMScheduler(**kwargs)
    kwargs.pop("variance_type")
    if kind == "ddim":
        return DDIMScheduler(**kwargs, set_alpha_to_one=False)
    if kind == "dpm_solver":
        kwargs.pop("clip_sample")
        return DPMSolverMultistepScheduler(
            **kwargs, algorithm_type="dpmsolver++", solver_order=2
        )
    raise ValueError(f"Unknown diffusion scheduler: {kind}")


class DyaDiTSystem(pl.LightningModule):
    """The original model.* and vqvae.* checkpoint namespace is preserved."""

    def __init__(self, config, model=None, codec=None):
        super().__init__()
        from .factory import build_codec, build_model

        self.save_hyperparameters({"config": config})
        self.config = config
        self.model = model if model is not None else build_model(config)
        self.vqvae = codec if codec is not None else build_codec(config)
        self.vqvae.requires_grad_(False).eval()
        self.mask_length = config["model"]["mask_length"]
        self.diff_cfg = DiffusionConfig(**config["training"]["diffusion"])
        self.train_noise_scheduler = make_scheduler(self.diff_cfg, training=True)

    def train(self, mode=True):
        super().train(mode)
        self.vqvae.eval()
        return self

    @torch.no_grad()
    def encode_motion(self, motion):
        if motion is None:
            return None
        codes = self.vqvae.encode(motion)
        batch, frames, depth = codes.shape
        return self.vqvae.quantizer.decode(codes.reshape(-1, depth)).reshape(
            batch, frames, self.vqvae.latent_dim
        )

    def _loss(self, batch):
        pose = self.encode_motion(batch["motion_latent"])
        style = self.encode_motion(batch["style_latent"])
        other = self.encode_motion(batch["motion_latent_other"])
        timesteps = torch.randint(
            self.diff_cfg.num_train_timesteps, (len(pose),), device=pose.device
        )
        noise = torch.randn_like(pose)
        noisy_pose = self.train_noise_scheduler.add_noise(pose, noise, timesteps)
        mask = pose.new_zeros(pose.shape[:2])
        mask[:, : self.mask_length] = 1
        prediction = self.model(
            noisy_pose,
            timesteps,
            batch["audio"],
            batch["audio_other"],
            style_motion=style,
            motion_other=other,
            relat_onehot=batch["relationship_onehot"],
            personality_feat=batch["personality_traits"],
            mask=mask,
        )
        target = noise
        if self.diff_cfg.objective == "x0":
            target = pose
        elif self.diff_cfg.objective == "v":
            target = self.train_noise_scheduler.get_velocity(pose, noise, timesteps)
        return F.mse_loss(prediction, target)

    def training_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch["audio"]),
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._loss(batch)
        self.log(
            "val/loss",
            loss,
            prog_bar=True,
            sync_dist=True,
            batch_size=len(batch["audio"]),
        )
        return loss

    def configure_optimizers(self):
        settings = self.config["training"]
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            parameters,
            lr=float(settings["lr"]),
            weight_decay=settings["weight_decay"],
            betas=tuple(settings.get("betas", (0.9, 0.95))),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config["trainer"]["max_epochs"]
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }

    @torch.no_grad()
    def sample_latents(
        self,
        audio,
        audio_other,
        latent_steps,
        *,
        style=None,
        motion_other=None,
        relationship=None,
        personality=None,
        prefix=None,
        inference_steps=50,
        guidance_scale=1.0,
    ):
        """Sample one latent window; CFG drops context like training fusion_drop."""
        self.eval()
        batch = audio.shape[0]
        x = torch.randn(batch, latent_steps, self.model.pose_dim, device=audio.device)
        scheduler = make_scheduler(self.diff_cfg)
        scheduler.set_timesteps(inference_steps, device=audio.device)
        x = x * scheduler.init_noise_sigma
        mask = x.new_zeros(batch, latent_steps)
        known = x.new_zeros(x.shape)
        if prefix is not None:
            count = min(prefix.shape[1], latent_steps)
            mask[:, :count] = 1
            known[:, :count] = prefix[:, :count]
        mask_3d = mask.unsqueeze(-1)
        prefix_noise = torch.randn_like(x)
        conditions = dict(
            style_motion=style,
            motion_other=motion_other,
            relat_onehot=relationship,
            personality_feat=personality,
            mask=mask,
        )
        for timestep in scheduler.timesteps:
            t = timestep.expand(batch)
            known_noisy = self.train_noise_scheduler.add_noise(known, prefix_noise, t)
            x = mask_3d * known_noisy + (1 - mask_3d) * x
            model_input = scheduler.scale_model_input(x, timestep)
            prediction = self.model(model_input, t, audio, audio_other, **conditions)
            if guidance_scale != 1:
                unconditional = self.model(
                    model_input, t, audio, audio_other, drop_context=True, **conditions
                )
                prediction = unconditional + guidance_scale * (
                    prediction - unconditional
                )
            x = scheduler.step(prediction, timestep, x).prev_sample
        return mask_3d * known + (1 - mask_3d) * x

    @torch.no_grad()
    def generate(
        self,
        audio,
        audio_other,
        target_frames,
        *,
        style_motion=None,
        motion_other=None,
        relationship=None,
        personality=None,
        reference_motion=None,
        window_frames=None,
        stride_frames=None,
        inference_steps=50,
        guidance_scale=1.0,
    ):
        """Generate raw motion. Window/stride are always measured in motion frames."""
        self.eval()
        factor = self.vqvae.downsample_factor
        window_frames = window_frames or self.config["model"]["pose_length"]
        stride_frames = stride_frames or max(
            factor, window_frames - self.mask_length * factor
        )
        if target_frames < factor or target_frames % factor:
            raise ValueError(
                "target_frames must be a positive multiple of the codec downsample factor"
            )
        if (
            window_frames % factor
            or stride_frames % factor
            or not 0 < stride_frames <= window_frames
        ):
            raise ValueError(
                "Window and stride must be positive multiples of the codec factor, with stride <= window"
            )
        if window_frames // factor <= self.mask_length:
            raise ValueError("Window must be longer than the configured latent prefix")
        overlap_steps = (window_frames - stride_frames) // factor
        if 0 < overlap_steps < self.mask_length:
            raise ValueError(
                "Overlapping windows must overlap by at least mask_length latent steps"
            )
        audio = audio.squeeze(1) if audio.ndim == 3 else audio
        audio_other = audio_other.squeeze(1) if audio_other.ndim == 3 else audio_other
        style = self.encode_motion(style_motion)
        other = self.encode_motion(motion_other)
        reference = self.encode_motion(reference_motion)
        latent_steps, window_steps = target_frames // factor, window_frames // factor
        result = audio.new_zeros(len(audio), latent_steps, self.model.pose_dim)
        fps, sample_rate = (
            self.config["model"]["pose_fps"],
            self.config["model"]["audio_sr"],
        )
        audio_size = round(window_frames * sample_rate / fps)

        def audio_window(wave, start):
            segment = wave[:, start : start + audio_size]
            return F.pad(segment, (0, audio_size - segment.shape[-1]))

        def latent_window(sequence, start):
            if sequence is None:
                return None
            segment = sequence[:, start : start + window_steps]
            if not segment.shape[1]:
                raise ValueError(
                    "Motion condition is shorter than the requested output"
                )
            return F.pad(
                segment.transpose(1, 2),
                (0, window_steps - segment.shape[1]),
                mode="replicate",
            ).transpose(1, 2)

        for frame_start in range(0, target_frames, stride_frames):
            start = frame_start // factor
            valid = min(window_steps, latent_steps - start)
            prefix = None
            if start == 0 and reference is not None:
                prefix = reference[:, : self.mask_length]
            elif start and overlap_steps:
                prefix = result[:, start : start + min(self.mask_length, valid)].clone()
            audio_start = int(frame_start * sample_rate / fps)
            chunk = self.sample_latents(
                audio_window(audio, audio_start),
                audio_window(audio_other, audio_start),
                window_steps,
                style=latent_window(style, 0),
                motion_other=latent_window(other, start),
                relationship=relationship,
                personality=personality,
                prefix=prefix,
                inference_steps=inference_steps,
                guidance_scale=guidance_scale,
            )[:, :valid]
            overlap = min(overlap_steps, valid) if start else 0
            if overlap:
                alpha = torch.linspace(0, 1, overlap, device=audio.device).view(
                    1, -1, 1
                )
                result[:, start : start + overlap] = (
                    result[:, start : start + overlap] * (1 - alpha)
                    + chunk[:, :overlap] * alpha
                )
            result[:, start + overlap : start + valid] = chunk[:, overlap:]
            if frame_start + window_frames >= target_frames:
                break
        return self.vqvae.decode_motion(result), result
