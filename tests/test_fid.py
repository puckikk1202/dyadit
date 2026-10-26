"""Motion-FID mathematics, fixed-feature protocol, caching, plotting and DDP hooks."""

import csv
import json
import random
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

from dyadit.config import load_config
from dyadit.fid import (
    MotionFIDCallback,
    feature_statistics,
    frechet_distance,
    motion_features,
)


class FixedValidationClips:
    pose_length = 96
    random_style = False
    normalize_motion = True

    def __init__(self):
        self.reads = 0

    def __len__(self):
        return 12

    def __getitem__(self, index):
        self.reads += 1
        motion = (
            torch.arange(96).float().view(-1, 1) / 96
            + index / 10
            + torch.arange(258).float().view(1, -1) / 2580
        )
        return {
            "motion_latent": motion,
            "motion_latent_other": motion + 0.2,
            "style_latent": motion * 0.8,
            "audio": torch.full((128,), index / 10),
            "audio_other": torch.full((128,), index / 20),
            "relationship_onehot": torch.tensor([1.0, 0.0, 0.0, 0.0]),
            "personality_traits": torch.ones(5),
            "video_id": f"fid_clip_{index}",
        }


class FidSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.marker = nn.Parameter(torch.zeros(1))
        self.model = nn.Linear(1, 1)
        self.vqvae = nn.Module()
        self.vqvae.encoder = nn.Linear(258, 4)
        with torch.no_grad():
            self.vqvae.encoder.weight.copy_(
                torch.arange(4 * 258).reshape(4, 258) / (4 * 258 * 258)
            )
            self.vqvae.encoder.bias.zero_()
        self.vqvae.requires_grad_(False).eval()
        self.calls = 0
        self.offset = 0.1
        self.prefix_only = False

    def generate(self, audio, audio_other, frames, **kwargs):
        self.calls += 1
        assert not self.training and not self.vqvae.training
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        assert frames == 96
        assert kwargs["window_frames"] == 96 and kwargs["stride_frames"] == 80
        assert kwargs["inference_steps"] == 3
        assert kwargs["motion_other"].shape == (len(audio), 96, 258)
        # Sampling must not consume the subsequent training epoch's random state.
        random.random()
        np.random.normal()
        torch.randn(3)
        generated = kwargs["reference_motion"].clone()
        if self.prefix_only:
            generated[:, :16] += 100
        else:
            generated += self.offset
        return generated, generated.new_zeros(len(audio), 24, 4)


@pytest.fixture
def fid_config(tmp_path):
    config = load_config(Path(__file__).resolve().parents[1] / "configs/core.yaml")
    config["paths"]["output_dir"] = str(tmp_path / "outputs")
    config["fid"].update(
        num_samples=8, batch_size=4, num_workers=0, inference_steps=3, cpu_threads=1
    )
    return config


def fid_trainer(tmp_path):
    pytest.importorskip("wandb")
    logger = WandbLogger(experiment=Mock(), save_dir=str(tmp_path))
    logger.log_metrics, logger.log_image = Mock(), Mock()
    return SimpleNamespace(
        is_global_zero=True,
        current_epoch=0,
        global_step=10,
        sanity_checking=False,
        fast_dev_run=False,
        datamodule=SimpleNamespace(val_dataset=FixedValidationClips()),
        strategy=SimpleNamespace(broadcast=Mock(side_effect=lambda value, src: value)),
        precision_plugin=SimpleNamespace(forward_context=nullcontext),
        loggers=[logger],
    )


def test_frechet_identity_singular_covariance_and_known_scale_shift():
    random_features = np.random.default_rng(3).normal(size=(8, 128))
    assert frechet_distance(random_features, random_features) == pytest.approx(
        0, abs=1e-5
    )
    constant = np.ones((4, 8))
    assert frechet_distance(constant, constant) == 0
    first = np.array([[-1.0], [0.0], [1.0]])
    assert frechet_distance(first, first * 3 + 2) == pytest.approx(8)
    shifted = random_features + 0.1
    assert frechet_distance(random_features, shifted) == pytest.approx(
        128 * 0.1**2, abs=1e-5
    )


def test_frechet_matches_scipy_reference_for_full_rank_features():
    linalg = pytest.importorskip("scipy.linalg")
    rng = np.random.default_rng(8)
    first, second = rng.normal(size=(40, 8)), rng.normal(size=(50, 8)) * 1.5 + 0.4
    mean_a, cov_a = feature_statistics(first)
    mean_b, cov_b = feature_statistics(second)
    expected = np.square(mean_a - mean_b).sum() + np.trace(
        cov_a + cov_b - 2 * linalg.sqrtm(cov_a @ cov_b).real
    )
    assert frechet_distance(first, second) == pytest.approx(
        expected, rel=1e-10, abs=1e-10
    )


@pytest.mark.parametrize(
    "features",
    [
        np.zeros((1, 3)),
        np.zeros((3, 0)),
        np.zeros((3, 2, 2)),
        np.array([[0.0, np.nan], [1.0, 2.0]]),
    ],
)
def test_invalid_fid_features_are_rejected(features):
    with pytest.raises(ValueError, match="features"):
        feature_statistics(features)


def test_frozen_features_pool_mean_and_std_after_excluding_prefix():
    motion = torch.arange(24).float().reshape(1, 8, 3)
    features = motion_features(nn.Identity(), motion, excluded_prefix_frames=2)
    expected = torch.cat(
        [motion[:, 2:].mean(1), motion[:, 2:].std(1, unbiased=False)], dim=-1
    )
    np.testing.assert_allclose(features, expected.numpy())
    changed = motion.clone()
    changed[:, :2] += 10000
    np.testing.assert_array_equal(features, motion_features(nn.Identity(), changed, 2))


def test_each_epoch_generates_fixed_subset_logs_curve_and_preserves_training_state(
    fid_config, tmp_path
):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    callback = MotionFIDCallback(fid_config)
    modes = [child.training for child in module.modules()]
    python_state, numpy_state, torch_state = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    callback.on_fit_start(trainer, module)
    assert trainer.datamodule.val_dataset.reads == 8
    assert callback.encoder is not module.vqvae.encoder
    encoder_before = {
        key: value.clone() for key, value in callback.encoder.state_dict().items()
    }
    for epoch in range(3):
        trainer.current_epoch, trainer.global_step = epoch, (epoch + 1) * 10
        module.offset = 0.1 * (epoch + 1)
        callback.on_train_epoch_end(trainer, module)
    assert module.calls == 6  # 2 generated batches per epoch, not denoising loss.
    assert trainer.datamodule.val_dataset.reads == 8  # Cached, not re-read each epoch.
    assert len(callback.indices) == len(set(callback.indices)) == 8
    assert callback.last_evaluated_epoch == 3 and callback.last_score > 0
    for key, value in callback.encoder.state_dict().items():
        torch.testing.assert_close(value, encoder_before[key])
    assert [child.training for child in module.modules()] == modes
    assert random.getstate() == python_state
    np.testing.assert_array_equal(np.random.get_state()[1], numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]
    assert torch.equal(torch.get_rng_state(), torch_state)
    assert all(parameter.grad is None for parameter in module.parameters())
    logger = trainer.loggers[0]
    assert logger.log_metrics.call_count == logger.log_image.call_count == 3
    logger.experiment.define_metric.assert_any_call(
        "val/fid_codec", step_metric="fid/epoch", summary="min"
    )
    assert logger.log_metrics.call_args.kwargs["step"] == 30
    assert logger.log_metrics.call_args.args[0]["fid/epoch"] == 3
    rows = list(csv.DictReader((callback.directory / "fid_codec.csv").open()))
    assert [int(row["epoch"]) for row in rows] == [1, 2, 3]
    assert float(rows[-1]["fid_codec"]) > float(rows[0]["fid_codec"])
    assert (callback.directory / "fid_codec.png").stat().st_size > 1000
    protocol = json.loads((callback.directory / "protocol.json").read_text())
    assert protocol["feature_dim"] == 8 and protocol["num_samples"] == 8
    assert protocol["excluded_prefix_frames"] == 16 and protocol["pose_length"] == 96
    assert protocol["feature_extractor"] == "frozen_vqvae_encoder_prequant"
    assert len(protocol["encoder_sha256"]) == 64


def test_prefix_only_changes_do_not_improve_or_worsen_fid(fid_config, tmp_path):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    module.prefix_only = True
    callback = MotionFIDCallback(fid_config)
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_end(trainer, module)
    assert callback.last_score == pytest.approx(0, abs=1e-6)


def test_sanity_fast_dev_run_and_restored_epoch_skip_fid(fid_config, tmp_path):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    callback = MotionFIDCallback(fid_config)
    callback._evaluate = Mock()
    trainer.sanity_checking = True
    callback.on_train_epoch_end(trainer, module)
    trainer.sanity_checking, trainer.fast_dev_run = False, True
    callback.on_train_epoch_end(trainer, module)
    trainer.fast_dev_run = False
    callback.load_state_dict({"last_evaluated_epoch": 1, "last_score": 2.5})
    callback.on_train_epoch_end(trainer, module)
    callback._evaluate.assert_not_called()


def test_nonzero_rank_waits_and_receives_same_fid_state(fid_config, tmp_path):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    trainer.is_global_zero = False
    trainer.strategy.broadcast = Mock(
        side_effect=[
            {"error": None, "protocol_hash": "test"},
            {"error": None, "score": 0.25},
        ]
    )
    callback = MotionFIDCallback(fid_config)
    callback._prepare, callback._evaluate = Mock(), Mock()
    callback.on_fit_start(trainer, module)
    callback.on_train_epoch_end(trainer, module)
    callback._prepare.assert_not_called()
    callback._evaluate.assert_not_called()
    assert callback.last_score == 0.25 and callback.last_evaluated_epoch == 1
    assert callback.state_dict()["protocol_hash"] == "test"


def test_fid_failure_is_broadcast_and_does_not_change_rng(fid_config, tmp_path):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    callback = MotionFIDCallback(fid_config)
    callback.on_fit_start(trainer, module)
    before = torch.get_rng_state().clone()
    modes = [child.training for child in module.modules()]

    def fail(*args):
        torch.randn(3)
        raise ValueError("synthetic sampler failure")

    callback._evaluate = fail
    with pytest.raises(RuntimeError, match="synthetic sampler failure"):
        callback.on_train_epoch_end(trainer, module)
    assert torch.equal(torch.get_rng_state(), before)
    assert [child.training for child in module.modules()] == modes
    assert callback.last_evaluated_epoch == 0
    assert (
        "synthetic sampler failure"
        in trainer.strategy.broadcast.call_args.args[0]["error"]
    )
    trainer.is_global_zero = False
    trainer.strategy.broadcast = Mock(
        return_value={"error": "synthetic sampler failure"}
    )
    with pytest.raises(RuntimeError, match="synthetic sampler failure"):
        callback.on_train_epoch_end(trainer, module)


def test_resume_keeps_curve_and_rejects_changed_protocol(fid_config, tmp_path):
    trainer, module = fid_trainer(tmp_path), FidSystem()
    first = MotionFIDCallback(fid_config)
    first.on_fit_start(trainer, module)
    first.on_train_epoch_end(trainer, module)
    saved = first.state_dict()
    resumed = MotionFIDCallback(fid_config)
    resumed.load_state_dict(saved)
    resumed.on_fit_start(trainer, FidSystem())
    assert len(resumed.history) == 1
    resumed.on_train_epoch_end(trainer, module)
    assert module.calls == 2
    trainer.current_epoch = 1
    resumed.on_train_epoch_end(trainer, module)
    assert len(resumed.history) == 2
    fid_config["fid"]["inference_steps"] = 4
    changed = MotionFIDCallback(fid_config)
    changed.load_state_dict(saved)
    with pytest.raises(RuntimeError, match="protocol changed"):
        changed.on_fit_start(trainer, FidSystem())


@pytest.mark.parametrize(
    "name,value",
    [
        ("num_samples", 1),
        ("batch_size", 0),
        ("every_n_epochs", 0),
        ("pose_length", 95),
        ("num_workers", -1),
        ("inference_steps", 0),
    ],
)
def test_invalid_fid_configuration(fid_config, name, value):
    fid_config["fid"][name] = value
    with pytest.raises(ValueError):
        MotionFIDCallback(fid_config)


def test_wandb_offline_preserves_epoch_curve_images(fid_config, tmp_path):
    pytest.importorskip("wandb")
    trainer = fid_trainer(tmp_path)
    logger = WandbLogger(
        project="dyadit-fid-tests", save_dir=str(tmp_path), offline=True
    )
    trainer.loggers = [logger]
    run = logger.experiment
    try:
        callback, module = MotionFIDCallback(fid_config), FidSystem()
        callback.on_fit_start(trainer, module)
        for epoch in range(2):
            trainer.current_epoch, trainer.global_step = epoch, (epoch + 1) * 10
            callback.on_train_epoch_end(trainer, module)
        images = list((Path(run.dir) / "media" / "images").rglob("*.png"))
        assert len(images) == 2
        assert all(path.stat().st_size > 1000 for path in images)
        assert run.offline
    finally:
        run.finish()
