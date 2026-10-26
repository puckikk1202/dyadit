"""One construction path for training and inference."""

from pathlib import Path

from .checkpoint import load_checkpoint
from .codec import TemporalVertexCodecDownsampled
from .model import Audio2PoseDiTDual_VQVAE


def build_model(config):
    source = config["model"]
    keys = (
        "pose_dim",
        "model_dim",
        "ffn_dim",
        "heads",
        "layers",
        "time_dim",
        "use_cross_attn",
        "use_rope",
        "fusion_drop",
        "style_drop",
        "relat_drop",
        "person_drop",
        "motion_other_drop",
        "motion_bank_size",
        "orthogonal_fusion",
    )
    kwargs = {key: source[key] for key in keys}
    kwargs["wav2vec_layer"] = source["wav2vec_layers"]
    kwargs["audio_model_name"] = source.get(
        "audio_model_name", "facebook/wav2vec2-base-960h"
    )
    kwargs["downsample_factor"] = config["vqvae"]["downsample_factor"]
    return Audio2PoseDiTDual_VQVAE(**kwargs)


def build_codec(config, load_weights=True):
    source = config["vqvae"]
    codec = TemporalVertexCodecDownsampled(
        n_vertices=source["nb_joints"],
        latent_dim=source["output_emb_width"],
        categories=source["code_dim"],
        residual_depth=source["depth"],
        downsample_factor=source["downsample_factor"],
    )
    if load_weights:
        path = Path(source["checkpoint_path"])
        if not path.is_file():
            raise FileNotFoundError(f"VQ-VAE checkpoint not found: {path}")
        checkpoint = load_checkpoint(path)
        codec.load_state_dict(checkpoint.get("net", checkpoint), strict=True)
    codec.requires_grad_(False)
    return codec.eval()
