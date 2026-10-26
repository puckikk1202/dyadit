"""Offline checks for the extracted data/model/checkpoint/sampling contracts."""

import csv
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytorch_lightning as pl
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from dyadit import codec as codec_module
from dyadit import model as model_module
from dyadit import quantizer as quantizer_module
from dyadit.checkpoint import load_checkpoint, load_system_weights
from dyadit.config import load_config
from dyadit.data import DyadicDataset, UPPER_BODY_JOINTS, load_motion
from dyadit.diffusion import DyaDiTSystem
from dyadit.export import motion_to_smplh, save_outputs
from dyadit.factory import build_codec, build_model
from dyadit.fid import MotionFIDCallback
from dyadit.rotations import (
    convert_axis_angle_to_rotation_6d,
    convert_rotation_6d_to_axis_angle,
)
from dyadit.training import DyadicDataModule
from dyadit.visualization import ValidationRenderCallback
from preprocess import prepare


class AudioBatch(SimpleNamespace):
    def to(self, device):
        return AudioBatch(input_values=self.input_values.to(device))


class TinyProcessor:
    def __call__(self, waves, **kwargs):
        return AudioBatch(
            input_values=torch.tensor(np.stack(waves), dtype=torch.float32)
        )


class TinyAudioEncoder(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.feature_extractor = nn.Conv1d(1, 4, 3, padding=1)
        self.feature_projection = nn.Linear(4, 768)

    def forward(self, audio):
        pooled = F.adaptive_avg_pool1d(audio.unsqueeze(1), 8)
        features = self.feature_extractor(pooled).transpose(1, 2)
        return {"high_level": self.feature_projection(features)}


@pytest.fixture(autouse=True)
def offline_audio(monkeypatch):
    torch.set_num_threads(1)
    monkeypatch.setattr(model_module, "TANGOWrapedWav2Vec", TinyAudioEncoder)
    monkeypatch.setattr(
        model_module.Wav2Vec2Processor,
        "from_pretrained",
        lambda *args, **kwargs: TinyProcessor(),
    )


@pytest.fixture
def config():
    return load_config(
        Path(__file__).resolve().parents[1] / "configs" / "core.yaml",
        [
            "model.model_dim=16",
            "model.ffn_dim=32",
            "model.heads=2",
            "model.layers=1",
            "model.time_dim=16",
            "model.motion_bank_size=8",
            "data.batch_size=2",
            "data.num_workers=0",
        ],
    )


def make_system(config):
    codec = build_codec(config, load_weights=False)
    # A deterministic synthetic codebook supplies nonzero latent targets offline.
    with torch.no_grad():
        for layer in codec.quantizer.layers:
            layer._codebook.embed.normal_(std=0.1)
            layer._codebook.inited.fill_(True)
    return DyaDiTSystem(config, model=build_model(config), codec=codec)


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def data_root(tmp_path, config):
    root = tmp_path / "paired_data"
    (root / "motion_6d_slim").mkdir(parents=True)
    (root / "audio_slim").mkdir()
    for session, frames in (("0001", 192), ("0002", 288)):
        for participant, value in (("0001", 1.0), ("0002", 2.0)):
            name = f"V00_S{session}_I00000001_P{participant}"
            np.save(
                root / "motion_6d_slim" / (name + ".npy"),
                np.full((frames, 258), value, dtype=np.float32),
            )
            np.save(
                root / "audio_slim" / (name + ".npy"),
                np.full(round(frames * 16000 / 30), value, dtype=np.float32),
            )
    np.savez(
        root / "motion_6d_normalization.npz",
        motion_mean=np.zeros(258),
        motion_std=np.ones(258),
    )
    write_csv(
        root / "csv" / "relationships.csv",
        ["vendor_id", "session_id", "relationship_detail"],
        [
            {
                "vendor_id": "V00",
                "session_id": session,
                "relationship_detail": "friends",
            }
            for session in ("0001", "0002")
        ],
    )
    traits = [
        "extraversion_raw",
        "agreeableness_raw",
        "conscientiousness_raw",
        "neuroticism_raw",
        "openness_raw",
    ]
    write_csv(
        root / "csv" / "participants.csv",
        ["vendor_id", "participant_id"] + traits,
        [
            dict(
                vendor_id="00",
                participant_id=participant,
                **dict(zip(traits, [5, 4, 3, 2, 1])),
            )
            for participant in ("0001", "0002")
        ],
    )
    config["paths"]["data_root"] = str(root)
    config["paths"]["val_data_root"] = str(root)
    return root


def test_typed_configuration_and_portable_paths(config):
    path = Path(__file__).resolve().parents[1] / "configs" / "core.yaml"
    overridden = load_config(path, ["model.orthogonal_fusion=false"])
    assert overridden["model"]["orthogonal_fusion"] is False
    assert Path(config["paths"]["data_root"]).is_absolute()
    with pytest.raises(ValueError, match="divisible"):
        load_config(path, ["model.pose_length=95"])


def test_pair_indexing_uses_each_conversations_actual_length(config, data_root):
    dataset = DyadicDataset(config, random_style=False)
    assert len(dataset) == (2 + 3) * 2
    assert [dataset[index]["start_frame"] for index in range(0, len(dataset), 2)] == [
        0,
        96,
        0,
        96,
        192,
    ]
    a, b = dataset[0], dataset[1]
    torch.testing.assert_close(a["motion_latent"], b["motion_latent_other"])
    torch.testing.assert_close(a["audio"], b["audio_other"])
    assert a["audio"].shape == (51200,)
    torch.testing.assert_close(a["relationship_onehot"], torch.tensor([1.0, 0, 0, 0]))
    torch.testing.assert_close(
        a["personality_traits"], torch.tensor([1.0, 0.8, 0.6, 0.4, 0.2])
    )
    config["data"]["is_reversed"] = False
    assert len(DyadicDataset(config)) == 5


def test_legacy_axis_angle_loading_and_smplh_export(tmp_path):
    angles = torch.randn(1, 12, 129, dtype=torch.float64) * 0.1
    six_d = convert_axis_angle_to_rotation_6d(angles)
    torch.testing.assert_close(
        convert_rotation_6d_to_axis_angle(six_d), angles, atol=2e-5, rtol=2e-4
    )
    expected = motion_to_smplh(six_d[0].numpy())
    source = tmp_path / "raw.npz"
    raw = dict(expected)
    raw["smplh:body_pose"] = np.zeros((12, 21, 3))
    raw["smplh:body_pose"][:, UPPER_BODY_JOINTS, :] = (
        angles[0, :, :39].reshape(12, 13, 3).numpy()
    )
    raw["smplh:left_hand_pose"] = angles[0, :, 39:84].reshape(12, 15, 3).numpy()
    raw["smplh:right_hand_pose"] = angles[0, :, 84:129].reshape(12, 15, 3).numpy()
    np.savez(source, **raw)
    np.testing.assert_allclose(
        load_motion(source, 2, 4), six_d[0, 2:6].numpy(), atol=2e-5
    )
    assert expected["smplh:body_pose"].shape == (12, 21, 3)
    assert not expected["smplh:translation"].any()


@pytest.mark.parametrize("orthogonal_fusion", [True, False])
def test_codec_shapes_and_optimizer_only_updates_dit(
    config, data_root, orthogonal_fusion
):
    config["model"]["orthogonal_fusion"] = orthogonal_fusion
    system = make_system(config)
    batch = next(iter(DataLoader(DyadicDataset(config), batch_size=2)))
    assert system.encode_motion(batch["motion_latent"]).shape == (2, 24, 64)
    assert system.vqvae.decode_motion(
        system.encode_motion(batch["motion_latent"])
    ).shape == (2, 96, 258)
    system.train()
    assert not system.vqvae.training
    optimizer = system.configure_optimizers()["optimizer"]
    before = system.model.out_proj.weight.detach().clone()
    loss = system._loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert not torch.equal(before, system.model.out_proj.weight)
    assert all(p.grad is None for p in system.vqvae.parameters())


def test_model_initialization_preserves_both_pretrained_speech_encoders(
    config, monkeypatch
):
    loaded = []

    class PretrainedAudio(TinyAudioEncoder):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.encoder = nn.Sequential(nn.Linear(768, 768))
            with torch.no_grad():
                for parameter in self.parameters():
                    parameter.fill_(0.125 + len(loaded) * 0.25)
            loaded.append(
                {name: value.clone() for name, value in self.state_dict().items()}
            )

    monkeypatch.setattr(model_module, "TANGOWrapedWav2Vec", PretrainedAudio)
    model = build_model(config)
    assert len(loaded) == 2
    for encoder, expected in zip(
        (model.audio_encoder, model.audio_encoder_other), loaded
    ):
        for name, value in encoder.state_dict().items():
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        assert all(parameter.requires_grad for parameter in encoder.parameters())
    assert torch.count_nonzero(model.in_proj.weight) > 0
    assert torch.count_nonzero(model.in_proj.bias) == 0
    assert torch.count_nonzero(model.out_proj.weight) == 0


def test_orthogonal_fusion_reaches_the_attention_inputs(config):
    model = build_model(config).eval()
    captured, original_other = [], []
    handle = model.audio_attn.register_forward_pre_hook(
        lambda module, args: captured.append((args[0].detach(), args[1].detach()))
    )
    original_handle = model.audio_context_norm_other.register_forward_hook(
        lambda module, args, output: original_other.append(output.detach())
    )
    inputs = (
        torch.randn(2, 24, 64),
        torch.tensor([1, 2]),
        torch.randn(2, 5120),
        torch.randn(2, 5120),
    )
    with torch.no_grad():
        output = model(*inputs)
        model.orthogonal_fusion = False
        model(*inputs)
    handle.remove()
    original_handle.remove()
    assert output.shape == (2, 24, 64)
    main, orthogonal = captured[0]
    assert (main * orthogonal).sum(-1).abs().max() < 2e-4
    assert not torch.allclose(orthogonal, captured[1][1])
    torch.testing.assert_close(captured[1][1], original_other[1])


@pytest.mark.parametrize("kind", ["ddpm", "ddim", "dpm_solver"])
def test_schedulers_preserve_the_reference_prefix(config, kind):
    config["training"]["diffusion"]["scheduler_type"] = kind
    system = make_system(config)
    prefix = torch.randn(2, 4, 64)
    output = system.sample_latents(
        torch.randn(2, 5120), torch.randn(2, 5120), 24, prefix=prefix, inference_steps=3
    )
    assert output.shape == (2, 24, 64)
    assert torch.isfinite(output).all()
    torch.testing.assert_close(output[:, :4], prefix)


def test_cfg_is_used_with_dpm_solver(config, monkeypatch):
    system = make_system(config)
    calls = []
    original = system.model.forward

    def forward(*args, **kwargs):
        calls.append(kwargs.get("drop_context", False))
        return original(*args, **kwargs)

    monkeypatch.setattr(system.model, "forward", forward)
    system.sample_latents(
        torch.randn(1, 5120),
        torch.randn(1, 5120),
        24,
        inference_steps=3,
        guidance_scale=4,
    )
    assert calls == [False, True] * 3


def test_sliding_windows_align_audio_to_motion_frames(config, monkeypatch):
    system = make_system(config)
    calls = []

    def sample(audio, audio_other, latent_steps, **kwargs):
        calls.append(audio.clone())
        return torch.ones(len(audio), latent_steps, 64)

    monkeypatch.setattr(system, "sample_latents", sample)
    audio = torch.arange(128000, dtype=torch.float32).unsqueeze(0)
    motion, latent = system.generate(
        audio, audio, 240, window_frames=96, stride_frames=80
    )
    assert motion.shape == (1, 240, 258)
    assert latent.shape == (1, 60, 64)
    assert len(calls) == 3
    assert calls[0].shape == (1, 51200)
    assert calls[1][0, 0].item() == int(80 * 16000 / 30)
    assert torch.equal(latent, torch.ones_like(latent))


def test_legacy_checkpoint_metadata_loads_without_old_modules(
    config, tmp_path, monkeypatch
):
    system = make_system(config)
    old_codec = ModuleType("dyadic_dit_vq.model.vqvae_downsampled")
    old_quantizer = ModuleType("dyadic_dit_vq.model.vqvae")
    old_training = ModuleType("train_wan_ours")
    legacy_settings = type("DiffusionConfig", (), {"__module__": "train_wan_ours"})
    old_training.DiffusionConfig = legacy_settings
    path = tmp_path / "legacy.ckpt"
    with monkeypatch.context() as context:
        for name in ("dyadic_dit_vq", "dyadic_dit_vq.model"):
            context.setitem(sys.modules, name, ModuleType(name))
        for module, source, names in (
            (
                old_codec,
                codec_module,
                [
                    "TemporalVertexCodecDownsampled",
                    "TemporalVertexEncoderDownsampled",
                    "TemporalVertexDecoderUpsampled",
                ],
            ),
            (
                old_quantizer,
                quantizer_module,
                [
                    "ResidualVectorQuantization",
                    "VectorQuantization",
                    "EuclideanCodebook",
                ],
            ),
        ):
            context.setitem(sys.modules, module.__name__, module)
            for name in names:
                cls = getattr(source, name)
                setattr(module, name, cls)
                context.setattr(cls, "__module__", module.__name__)
        context.setitem(sys.modules, "train_wan_ours", old_training)
        torch.save(
            {
                "state_dict": system.state_dict(),
                "hyper_parameters": {
                    "diffusion": legacy_settings(),
                    "vqvae": system.vqvae,
                },
            },
            path,
        )
    loaded = load_checkpoint(path)
    assert isinstance(
        loaded["hyper_parameters"]["vqvae"], codec_module.TemporalVertexCodecDownsampled
    )
    restored = make_system(config)
    load_system_weights(restored, path)
    for key, expected in system.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], expected)


def test_full_generation_and_export(config, data_root, tmp_path):
    system = make_system(config)
    dataset = DyadicDataset(config, random_style=False)
    sample = dataset[0]
    motion, latent = system.generate(
        sample["audio"].unsqueeze(0),
        sample["audio_other"].unsqueeze(0),
        96,
        style_motion=sample["style_latent"].unsqueeze(0),
        motion_other=sample["motion_latent_other"].unsqueeze(0),
        relationship=sample["relationship_onehot"].unsqueeze(0),
        personality=sample["personality_traits"].unsqueeze(0),
        reference_motion=sample["motion_latent"].unsqueeze(0),
        inference_steps=3,
    )
    save_outputs(tmp_path / "outputs", motion, latent, sample, dataset, config)
    with np.load(tmp_path / "outputs" / "generated_motion_denorm.npz") as archive:
        assert archive["smplh:body_pose"].shape == (96, 21, 3)
    with np.load(tmp_path / "outputs" / "generated_motion_latent.npz") as archive:
        assert archive["latent"].shape == (1, 24, 64)
    assert (tmp_path / "outputs" / "input_audio.wav").is_file()


def test_lightning_training_validation_and_checkpoint(config, data_root, tmp_path):
    system = make_system(config)
    trainer = pl.Trainer(
        fast_dev_run=True,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(system, datamodule=DyadicDataModule(config))
    assert "train/loss" in trainer.callback_metrics
    assert "val/loss" in trainer.callback_metrics
    checkpoint = tmp_path / "core.ckpt"
    trainer.save_checkpoint(checkpoint)
    restored = make_system(config)
    load_system_weights(restored, checkpoint)
    for key, value in system.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_lightning_ten_epochs_generate_twenty_validation_previews(
    config, data_root, tmp_path, monkeypatch, precision
):
    pytest.importorskip("wandb")
    from pytorch_lightning.loggers import WandbLogger

    config["paths"]["output_dir"] = str(tmp_path / "preview_training")
    config["visualization"].update(inference_steps=3, cpu_threads=1, pose_length=96)
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )

    def render(directory, model_path, **kwargs):
        directory = Path(directory)
        assert (directory / "input_audio.wav").is_file()
        assert (directory / "input_audio_other.wav").is_file()
        path = directory / "conversation.mp4"
        path.write_bytes(b"test-only preview; W&B media is checked separately")
        return path

    monkeypatch.setattr("dyadit.visualization.render_sample", render)
    # Exercise a real Trainer and core diffusion model, without starting a W&B
    # service or sending training metrics/media to an external account.
    logger = WandbLogger(experiment=Mock(), save_dir=str(tmp_path))
    logger.log_video = Mock()
    callback = ValidationRenderCallback(config)
    system = make_system(config)
    trainer = pl.Trainer(
        max_epochs=10,
        precision=precision,
        accelerator="cpu",
        devices=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=1,
        callbacks=[callback],
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer.fit(system, datamodule=DyadicDataModule(config))
    assert logger.log_video.call_count == 2
    assert all(
        len(call.kwargs["videos"]) == 10 for call in logger.log_video.call_args_list
    )
    assert callback.last_logged_epoch == 10
    assert not system.vqvae.training
    assert torch.isfinite(trainer.callback_metrics["val/loss"])
    for epoch in (5, 10):
        directory = (
            Path(config["paths"]["output_dir"])
            / "visualizations"
            / f"epoch_{epoch:04d}"
        )
        assert len(list(directory.glob("sample_*/conversation.mp4"))) == 10
        with np.load(
            next(directory.glob("sample_*/generated_motion_denorm.npz"))
        ) as archive:
            assert archive["smplh:body_pose"].shape == (96, 21, 3)
    checkpoint = tmp_path / "preview_training.ckpt"
    trainer.save_checkpoint(checkpoint)
    saved = load_checkpoint(checkpoint)
    assert saved["callbacks"][callback.state_key]["last_logged_epoch"] == 10


def test_600_frame_preview_uses_sliding_inference_without_changing_training_length(
    config, data_root, tmp_path, monkeypatch
):
    pytest.importorskip("wandb")
    from pytorch_lightning.loggers import WandbLogger

    # Existing short validation clips cannot supply a twenty-second preview.
    # Add one full conversation; keep the train/val model length at 96 frames.
    for participant in ("0001", "0002"):
        name = f"V00_S0003_I00000001_P{participant}"
        np.save(
            data_root / "motion_6d_slim" / (name + ".npy"),
            np.ones((1200, 258), dtype=np.float32),
        )
        np.save(
            data_root / "audio_slim" / (name + ".npy"),
            np.sin(np.arange(640000, dtype=np.float32) * 0.03) * 0.1,
        )
    config["paths"]["output_dir"] = str(tmp_path / "long_previews")
    config["visualization"].update(
        num_samples=1, pose_length=600, inference_steps=3, cpu_threads=1
    )
    data = DyadicDataModule(config)
    data.setup("fit")
    assert data.train_dataset.pose_length == data.val_dataset.pose_length == 96
    assert data.get_visualization_dataset(96) is data.val_dataset
    previews = data.get_visualization_dataset(600)
    assert previews is not data.val_dataset
    assert previews.pose_length == 600 and previews.random_style is False
    assert len(previews) == 4
    assert previews[0]["audio"].shape == (320000,)
    assert previews[0]["motion_latent"].shape == (600, 258)

    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )

    def render(directory, model_path, **kwargs):
        path = Path(directory) / "conversation.mp4"
        path.write_bytes(b"test-only long preview; real media is tested separately")
        return path

    monkeypatch.setattr("dyadit.visualization.render_sample", render)
    logger = WandbLogger(experiment=Mock(), save_dir=str(tmp_path))
    logger.log_video = Mock()
    trainer = SimpleNamespace(
        is_global_zero=True,
        current_epoch=4,
        global_step=25,
        sanity_checking=False,
        fast_dev_run=False,
        datamodule=data,
        loggers=[logger],
        strategy=SimpleNamespace(broadcast=lambda value, src: value),
        precision_plugin=SimpleNamespace(
            forward_context=lambda: torch.autocast("cpu", dtype=torch.bfloat16)
        ),
    )
    system = make_system(config)
    original = system.sample_latents
    windows = []

    def sample(*args, **kwargs):
        windows.append(args[0].clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(system, "sample_latents", sample)
    callback = ValidationRenderCallback(config)
    callback.on_fit_start(trainer, system)
    callback.on_train_epoch_end(trainer, system)
    assert len(windows) == 8  # 600 target frames, 96-frame windows, 80-frame stride.
    assert all(wave.shape == (1, 51200) for wave in windows)
    directory = Path(logger.log_video.call_args.kwargs["videos"][0]).parent
    with np.load(directory / "generated_motion_denorm.npz") as archive:
        assert archive["smplh:body_pose"].shape == (600, 21, 3)
    with np.load(directory / "generated_motion_latent.npz") as archive:
        assert archive["latent"].shape == (1, 150, 64)
    assert system.training and not system.vqvae.training
    assert data.train_dataset.pose_length == data.val_dataset.pose_length == 96
    assert config["model"]["pose_length"] == 96


def test_checkpoint_resume_switches_only_previews_to_600_frames(
    config, data_root, tmp_path, monkeypatch
):
    pytest.importorskip("wandb")
    from pytorch_lightning.loggers import WandbLogger

    for participant in ("0001", "0002"):
        name = f"V00_S0003_I00000001_P{participant}"
        np.save(
            data_root / "motion_6d_slim" / (name + ".npy"),
            np.ones((1200, 258), dtype=np.float32),
        )
        np.save(
            data_root / "audio_slim" / (name + ".npy"),
            np.zeros(640000, dtype=np.float32),
        )
    config["paths"]["output_dir"] = str(tmp_path / "continued_run")
    config["visualization"].update(
        num_samples=1, pose_length=96, inference_steps=3, cpu_threads=1
    )
    monkeypatch.setattr(
        ValidationRenderCallback, "_validate_renderer", lambda self: None
    )

    def render(directory, model_path, **kwargs):
        path = Path(directory) / "conversation.mp4"
        path.write_bytes(b"test-only checkpoint-resume preview")
        return path

    monkeypatch.setattr("dyadit.visualization.render_sample", render)
    logger = WandbLogger(experiment=Mock(), save_dir=str(tmp_path))
    logger.log_video = Mock()
    settings = dict(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    old_callback = ValidationRenderCallback(config)
    first = pl.Trainer(max_epochs=5, callbacks=[old_callback], **settings)
    first.fit(make_system(config), datamodule=DyadicDataModule(config))
    checkpoint = tmp_path / "five_epochs.ckpt"
    first.save_checkpoint(checkpoint)
    assert first.global_step == 5
    assert old_callback.last_logged_epoch == 5
    assert logger.log_video.call_count == 1
    short = Path(logger.log_video.call_args.kwargs["videos"][0]).parent

    new_config = deepcopy(config)
    new_config["visualization"]["pose_length"] = 600
    logger.log_video.reset_mock()
    callback = ValidationRenderCallback(new_config)
    system = make_system(new_config)
    data = DyadicDataModule(new_config)
    resumed = pl.Trainer(max_epochs=10, callbacks=[callback], **settings)
    resumed.fit(system, datamodule=data, ckpt_path=checkpoint)
    assert resumed.global_step == 10
    assert callback.last_logged_epoch == 10
    assert logger.log_video.call_count == 1  # Epoch 5 is not regenerated.
    assert system.config["visualization"]["pose_length"] == 600
    assert system.config["model"]["pose_length"] == 96
    assert data.train_dataset.pose_length == data.val_dataset.pose_length == 96
    long = Path(logger.log_video.call_args.kwargs["videos"][0]).parent
    with np.load(short / "generated_motion_denorm.npz") as archive:
        assert archive["smplh:body_pose"].shape[0] == 96
    with np.load(long / "generated_motion_denorm.npz") as archive:
        assert archive["smplh:body_pose"].shape[0] == 600


def test_preview_dataset_rejects_conversations_shorter_than_requested(
    config, data_root
):
    data = DyadicDataModule(config)
    data.setup("fit")
    with pytest.raises(ValueError, match="at least 600 frames"):
        data.get_visualization_dataset(600)


def test_real_training_and_checkpoint_resume_evaluate_fid_every_epoch(
    config, data_root, tmp_path
):
    pytest.importorskip("wandb")
    from pytorch_lightning.loggers import WandbLogger

    config["paths"]["output_dir"] = str(tmp_path / "fid_training")
    config["fid"].update(
        num_samples=4, batch_size=2, num_workers=0, inference_steps=3, cpu_threads=1
    )
    logger = WandbLogger(experiment=Mock(), save_dir=str(tmp_path))
    logger.log_metrics, logger.log_image = Mock(), Mock()
    settings = dict(
        accelerator="cpu",
        devices=1,
        precision="bf16-mixed",
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    callback = MotionFIDCallback(config)
    system = make_system(config)
    before = {
        name: value.clone() for name, value in system.vqvae.encoder.state_dict().items()
    }
    first = pl.Trainer(max_epochs=2, callbacks=[callback], **settings)
    first.fit(system, datamodule=DyadicDataModule(config))
    assert first.global_step == 2 and callback.last_evaluated_epoch == 2
    assert len(callback.history) == 2
    assert all(
        np.isfinite(row["fid_codec"]) and row["fid_codec"] >= 0
        for row in callback.history
    )
    for name, value in system.vqvae.encoder.state_dict().items():
        torch.testing.assert_close(value, before[name])
    checkpoint = tmp_path / "fid_training.ckpt"
    first.save_checkpoint(checkpoint)
    restored = MotionFIDCallback(config)
    second = pl.Trainer(max_epochs=3, callbacks=[restored], **settings)
    second.fit(
        make_system(config), datamodule=DyadicDataModule(config), ckpt_path=checkpoint
    )
    assert second.global_step == 3 and restored.last_evaluated_epoch == 3
    assert restored.protocol_hash == callback.protocol_hash
    assert [row["epoch"] for row in restored.history] == [1, 2, 3]
    records = [
        call.args[0]
        for call in logger.log_metrics.call_args_list
        if call.args and "val/fid_codec" in call.args[0]
    ]
    assert [row["fid/epoch"] for row in records] == [1, 2, 3]
    assert logger.log_image.call_count == 3
    assert (restored.directory / "fid_codec.png").is_file()


def test_preprocessing_resumes_from_complete_legacy_source(tmp_path):
    root = tmp_path / "preprocessing"
    for directory in ("motion_6d", "motion_6d_slim", "audio_npz"):
        (root / directory).mkdir(parents=True)
    for participant, value in (("0001", 1.0), ("0002", 2.0)):
        name = f"V00_S0001_I00000001_P{participant}"
        motion = np.full((96, 258), value, dtype=np.float32)
        np.savez(root / "motion_6d" / (name + ".npz"), **{"smplh:motion_6d": motion})
        np.savez(
            root / "audio_npz" / (name + ".npz"), audio=np.ones(51200, dtype=np.float32)
        )
        if participant == "0001":
            np.save(root / "motion_6d_slim" / (name + ".npy"), motion)
    prepare(root, slim=True)
    with np.load(root / "motion_6d_normalization.npz") as archive:
        assert archive["total_frames"] == 192
        np.testing.assert_allclose(archive["motion_mean"], 1.5)
        np.testing.assert_allclose(archive["motion_std"], 0.5)
    assert len(list((root / "motion_6d_slim").glob("*.npy"))) == 2
    assert len(list((root / "audio_slim").glob("*.npy"))) == 2
