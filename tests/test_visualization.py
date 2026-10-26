"""Periodic training previews: cadence, reproducibility, DDP ownership and W&B media."""

import json
import random
import subprocess
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytorch_lightning as pl
import pytest
import torch
from pytorch_lightning.loggers import WandbLogger
from torch import nn

from dyadit import rendering
from dyadit.config import load_config
from dyadit.video import ffmpeg_executable
from dyadit.visualization import ValidationRenderCallback


class ValidationClips:
    random_style = False
    normalize_motion = False
    pose_length = 600

    def __len__(self):
        return 12

    def __getitem__(self, index):
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(
            self.pose_length, 43
        )
        times = torch.arange(round(self.pose_length * 16000 / 30)) / 16000
        return {
            "motion_latent": identity.clone(),
            "motion_latent_other": identity.clone(),
            "style_latent": identity.clone(),
            "audio": torch.sin(2 * torch.pi * (220 + index) * times) * 0.003,
            "audio_other": torch.sin(2 * torch.pi * 440 * times) * 0.002,
            "relationship_onehot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "personality_traits": torch.ones(5),
            "video_id": f"validation_{index}",
            "start_frame": index * self.pose_length,
        }


class PreviewSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.frozen = nn.Linear(1, 1).requires_grad_(False).eval()
        self.calls = []

    def generate(self, audio, audio_other, frames, **kwargs):
        assert not self.training and not self.frozen.training
        assert not torch.is_grad_enabled()
        assert torch.is_inference_mode_enabled()
        assert kwargs["motion_other"].shape == (1, frames, 258)
        assert kwargs["style_motion"].shape == (1, frames, 258)
        assert audio.shape[-1] == round(frames * 16000 / 30)
        assert audio_other.shape == audio.shape
        assert kwargs["window_frames"] == 96
        assert kwargs["stride_frames"] == 80
        self.calls.append((audio.clone(), kwargs))
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]).repeat(1, frames, 43)
        generated = identity + torch.randn_like(identity) * 0.01
        # A bf16-mixed Trainer may return bfloat16 tensors, which NumPy does not
        # directly support. Exporting previews must still work.
        return generated.bfloat16(), torch.randn(1, frames // 4, 64).bfloat16()


@pytest.fixture
def preview_config(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs" / "core.yaml")
    config["paths"]["output_dir"] = str(tmp_path / "outputs")
    config["visualization"]["cpu_threads"] = 1
    config["visualization"]["inference_steps"] = 3
    config["visualization"]["render"].update(
        width=160, height=120, step=8, backend="software"
    )
    return config


def make_trainer(tmp_path):
    pytest.importorskip("wandb")
    logger = WandbLogger(project="dyadit-test", save_dir=str(tmp_path), offline=True)
    return SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        global_step=123,
        sanity_checking=False,
        fast_dev_run=False,
        datamodule=SimpleNamespace(val_dataset=ValidationClips()),
        strategy=SimpleNamespace(broadcast=Mock(side_effect=lambda value, src: value)),
        precision_plugin=SimpleNamespace(forward_context=nullcontext),
        loggers=[logger],
    )


def stub_render(directory, model_path, **kwargs):
    directory = Path(directory)
    assert (directory / "input_audio.wav").is_file()
    assert (directory / "input_audio_other.wav").is_file()
    assert (directory / "generated_motion_denorm.npz").is_file()
    output = directory / "conversation.mp4"
    output.write_bytes(b"test-only video; log_video is mocked")
    return output


def test_default_config_has_requested_wandb_cadence(preview_config):
    assert preview_config["logger"]["type"] == "wandb"
    settings = preview_config["visualization"]
    assert settings["enabled"] is True
    assert settings["every_n_epochs"] == 5
    assert settings["num_samples"] == 10
    assert settings["pose_length"] == 600
    assert settings["window_frames"] == 96
    assert settings["stride_frames"] == 80
    assert settings["render"]["audio_normalize"] is True
    assert settings["render"]["audio_layout"] == "stereo"
    assert Path(settings["model_path"]).is_absolute()


def test_only_epochs_five_and_ten_render_ten_fixed_validation_samples(
    preview_config, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )
    monkeypatch.setattr("dyadit.visualization.render_sample", stub_render)
    trainer = make_trainer(tmp_path)
    logger = trainer.loggers[0]
    logger.log_video, logger.log_metrics = Mock(), Mock()
    system = PreviewSystem()
    callback = ValidationRenderCallback(preview_config)
    callback.on_fit_start(trainer, system)
    initial_modes = [child.training for child in system.modules()]
    random_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    threads_before = torch.get_num_threads()
    for epoch in range(10):
        trainer.current_epoch = epoch
        callback.on_train_epoch_end(trainer, system)
        assert len(system.calls) == 10 * ((epoch + 1) // 5)
    assert len(callback.indices) == len(set(callback.indices)) == 10
    assert logger.log_video.call_count == 2
    for call in logger.log_video.call_args_list:
        assert call.args == ("visualization/samples",)
        assert len(call.kwargs["videos"]) == 10
        assert call.kwargs["format"] == ["mp4"] * 10
        assert call.kwargs["step"] == 123
        assert all(
            "Blue/left: condition (B); orange/right: generated (A)" in caption
            for caption in call.kwargs["caption"]
        )
        assert all("600 frames / 20s" in caption for caption in call.kwargs["caption"])
        assert all(
            "audio L=blue, R=orange" in caption for caption in call.kwargs["caption"]
        )
    root = Path(preview_config["paths"]["output_dir"]) / "visualizations"
    assert sorted(path.name for path in root.iterdir()) == ["epoch_0005", "epoch_0010"]
    for index in callback.indices:
        first = root / "epoch_0005" / f"sample_{index:05d}"
        second = root / "epoch_0010" / f"sample_{index:05d}"
        with np.load(first / "generated_motion_denorm.npz") as a, np.load(
            second / "generated_motion_denorm.npz"
        ) as b:
            np.testing.assert_array_equal(a["smplh:body_pose"], b["smplh:body_pose"])
            assert a["smplh:body_pose"].shape == (600, 21, 3)
        with np.load(first / "generated_motion_latent.npz") as latent:
            assert latent["latent"].dtype == np.float32
    manifest = json.loads((root / "epoch_0005" / "manifest.json").read_text())
    assert manifest["validation_indices"] == callback.indices
    assert manifest["epoch"] == 5
    assert manifest["pose_length"] == 600
    assert manifest["window_frames"] == 96
    assert manifest["stride_frames"] == 80
    assert manifest["duration_seconds"] == 20
    assert manifest["audio_layout"] == "stereo"
    assert manifest["display"] == {
        "blue_left": "conditioning_person_B",
        "orange_right": "generated_person_A",
    }
    assert callback.state_dict() == {"last_logged_epoch": 10}
    assert [child.training for child in system.modules()] == initial_modes
    assert random.getstate() == random_before
    assert np.random.get_state()[0] == numpy_before[0]
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    assert torch.get_num_threads() == threads_before
    assert system.marker.item() == 0 and system.marker.grad is None


def test_sanity_fast_dev_run_and_restored_epoch_do_not_render(
    preview_config, tmp_path, monkeypatch
):
    callback = ValidationRenderCallback(preview_config)
    monkeypatch.setattr(callback, "_render_and_log", Mock())
    trainer = make_trainer(tmp_path)
    system = PreviewSystem()
    trainer.current_epoch = 4
    trainer.sanity_checking = True
    callback.on_train_epoch_end(trainer, system)
    trainer.sanity_checking = False
    trainer.fast_dev_run = True
    callback.on_train_epoch_end(trainer, system)
    trainer.fast_dev_run = False
    callback.load_state_dict({"last_logged_epoch": 5})
    callback.on_train_epoch_end(trainer, system)
    callback._render_and_log.assert_not_called()


def test_nonzero_rank_only_waits_for_previews(preview_config, tmp_path, monkeypatch):
    trainer = make_trainer(tmp_path)
    trainer.is_global_zero = False
    trainer.current_epoch = 4
    callback = ValidationRenderCallback(preview_config)
    monkeypatch.setattr(callback, "_prepare", Mock())
    monkeypatch.setattr(callback, "_render_and_log", Mock())
    callback.on_fit_start(trainer, PreviewSystem())
    callback.on_train_epoch_end(trainer, PreviewSystem())
    callback._prepare.assert_not_called()
    callback._render_and_log.assert_not_called()
    assert trainer.strategy.broadcast.call_count == 2
    assert callback.last_logged_epoch == 5


def test_preview_errors_are_synchronized_and_train_state_is_restored(
    preview_config, tmp_path, monkeypatch
):
    trainer = make_trainer(tmp_path)
    trainer.current_epoch = 4
    callback = ValidationRenderCallback(preview_config)
    system = PreviewSystem()
    before = torch.get_rng_state().clone()
    modes = [child.training for child in system.modules()]

    def broken(*args):
        random.random()
        torch.randn(5)
        raise ValueError("synthetic renderer failure")

    monkeypatch.setattr(callback, "_render_and_log", broken)
    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        callback.on_train_epoch_end(trainer, system)
    assert [child.training for child in system.modules()] == modes
    assert torch.equal(torch.get_rng_state(), before)
    assert callback.last_logged_epoch == 0
    assert "synthetic renderer failure" in trainer.strategy.broadcast.call_args.args[0]
    trainer.is_global_zero = False
    trainer.strategy.broadcast = Mock(
        return_value="ValueError: synthetic renderer failure"
    )
    with pytest.raises(RuntimeError, match="synthetic renderer failure"):
        callback.on_train_epoch_end(trainer, system)


@pytest.mark.parametrize(
    "name",
    [
        "every_n_epochs",
        "num_samples",
        "inference_steps",
        "cpu_threads",
        "pose_length",
        "window_frames",
        "stride_frames",
    ],
)
def test_preview_rejects_nonpositive_configuration(preview_config, name):
    preview_config["visualization"][name] = 0
    with pytest.raises(ValueError, match=name):
        ValidationRenderCallback(preview_config)


def test_missing_wandb_or_model_is_reported_at_fit_start(preview_config, tmp_path):
    trainer = make_trainer(tmp_path)
    callback = ValidationRenderCallback(preview_config)
    logger = trainer.loggers[0]
    trainer.loggers = []
    with pytest.raises(RuntimeError, match="logger.type=wandb"):
        callback.on_fit_start(trainer, PreviewSystem())
    trainer.loggers = [logger]
    callback.model_path = tmp_path / "missing_models"
    with pytest.raises(RuntimeError, match="No SMPL-H"):
        callback.on_fit_start(trainer, PreviewSystem())


@pytest.mark.parametrize(
    "name,value,message",
    [
        ("pose_length", 602, "divisible"),
        ("window_frames", 94, "divisible"),
        ("stride_frames", 79, "divisible"),
        ("window_frames", 16, "prefix"),
        ("stride_frames", 100, "exceed"),
        ("stride_frames", 84, "overlap"),
    ],
)
def test_invalid_long_sequence_geometry_is_rejected(
    preview_config, name, value, message
):
    preview_config["visualization"][name] = value
    with pytest.raises(ValueError, match=message):
        ValidationRenderCallback(preview_config)


def test_short_clips_are_not_stretched_to_long_previews(
    preview_config, tmp_path, monkeypatch
):
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )
    trainer = make_trainer(tmp_path)
    trainer.datamodule.val_dataset.pose_length = 96
    callback = ValidationRenderCallback(preview_config)
    with pytest.raises(RuntimeError, match="600-frame clips"):
        callback.on_fit_start(trainer, PreviewSystem())


def test_small_validation_set_is_not_duplicated(preview_config, tmp_path, monkeypatch):
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )
    trainer = make_trainer(tmp_path)
    trainer.datamodule.val_dataset = torch.utils.data.Subset(
        ValidationClips(), [0, 2, 4]
    )
    callback = ValidationRenderCallback(preview_config)
    callback.on_fit_start(trainer, PreviewSystem())
    assert callback.indices == [0, 1, 2]


def test_preview_context_restores_rng_even_when_logging_consumes_randomness(
    preview_config, tmp_path, monkeypatch
):
    callback = ValidationRenderCallback(preview_config)
    trainer = make_trainer(tmp_path)
    trainer.current_epoch = 4
    system = PreviewSystem()
    python_before = random.getstate()
    numpy_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()

    def logging(*args):
        random.random()
        np.random.normal(size=4)
        torch.randn(5)

    monkeypatch.setattr(callback, "_render_and_log", logging)
    callback.on_train_epoch_end(trainer, system)
    assert random.getstate() == python_before
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_before[1])
    assert np.random.get_state()[2:] == numpy_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


class TinyBodyModel:
    num_betas = 10
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    lbs_weights = torch.zeros(4, 52)
    lbs_weights[0, 0] = lbs_weights[1, 1] = 1
    lbs_weights[2, 16] = lbs_weights[3, 18] = 1

    def __call__(self, body_pose, **kwargs):
        vertices = body_pose.new_tensor(
            [[-0.2, -1.0, 0.1], [0.2, -1.0, 0.1], [-0.3, 0.7, 0.2], [0.3, 0.7, -0.2]]
        )
        return SimpleNamespace(
            vertices=vertices.unsqueeze(0).repeat(len(body_pose), 1, 1),
            joints=body_pose.new_zeros(len(body_pose), 1, 3),
        )


def test_wandb_offline_keeps_both_rendered_videos_and_audio(
    preview_config, tmp_path, monkeypatch
):
    pytest.importorskip("cv2")
    pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )
    monkeypatch.setattr(rendering, "load_body_model", lambda *args: TinyBodyModel())
    preview_config["visualization"]["num_samples"] = 2
    trainer = make_trainer(tmp_path)
    trainer.current_epoch = 4
    callback = ValidationRenderCallback(preview_config)
    system = PreviewSystem()
    run = trainer.loggers[0].experiment
    try:
        callback.on_fit_start(trainer, system)
        callback.on_train_epoch_end(trainer, system)
        media = list((Path(run.dir) / "media" / "videos").rglob("*.mp4"))
        assert len(media) == 2
        for path in media:
            decoded = subprocess.run(
                [
                    ffmpeg_executable(),
                    "-v",
                    "error",
                    "-i",
                    str(path),
                    "-map",
                    "0:a:0",
                    "-ar",
                    "16000",
                    "-f",
                    "f32le",
                    "pipe:1",
                ],
                capture_output=True,
                check=True,
            )
            audio = np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, 2)
            assert len(audio) / 16000 == pytest.approx(20, abs=0.03)
            assert np.sqrt(np.mean(audio**2)) > 0.01
            spectrum = np.abs(np.fft.rfft(audio, axis=0))
            frequencies = np.fft.rfftfreq(len(audio), 1 / 16000)
            # Conditioning/other voice is 440 Hz: it must be blue/left.
            # The generated person's voice is 220+index Hz: orange/right.
            left_peak = frequencies[spectrum[:, 0].argmax()]
            right_peak = frequencies[spectrum[:, 1].argmax()]
            assert left_peak == pytest.approx(440, abs=0.1)
            assert 220 <= right_peak <= 232
        assert run.offline
    finally:
        run.finish()
