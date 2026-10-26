"""Shared YAML configuration and typed command-line overrides."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_config(path, overrides=()):
    path = Path(path).resolve()
    with path.open() as stream:
        config = yaml.safe_load(stream)
    for override in overrides:
        key, value = override.split("=", 1)
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = yaml.safe_load(value)
    return resolve_config(config, path.parent)


def resolve_config(config, base_dir):
    """Resolve paths relative to the YAML file, independently of the launch cwd."""
    config = deepcopy(config)
    base_dir = Path(base_dir)
    for section, key in (
        ("paths", "data_root"),
        ("paths", "val_data_root"),
        ("paths", "normalization_stats"),
        ("paths", "output_dir"),
        ("vqvae", "checkpoint_path"),
        ("data", "motion_path"),
        ("visualization", "model_path"),
    ):
        value = config.get(section, {}).get(key)
        if value:
            path = Path(value).expanduser()
            config[section][key] = str((base_dir / path).resolve())
    model, codec = config["model"], config["vqvae"]
    factor = codec["downsample_factor"]
    if factor not in (2, 4, 8):
        raise ValueError("vqvae.downsample_factor must be 2, 4 or 8")
    if model["pose_dim"] != codec["output_emb_width"]:
        raise ValueError("model.pose_dim must equal vqvae.output_emb_width")
    if model["pose_length"] % factor:
        raise ValueError(
            "model.pose_length must be divisible by the codec downsample factor"
        )
    if not 0 <= model["mask_length"] < model["pose_length"] // factor:
        raise ValueError(
            "model.mask_length is measured in latent steps and must be shorter than the sequence"
        )
    if not isinstance(model["orthogonal_fusion"], bool):
        raise ValueError("model.orthogonal_fusion must be a YAML boolean")
    if model["audio_sr"] != 16000:
        raise ValueError("The Wav2Vec2 audio encoder expects audio_sr=16000")
    return config
