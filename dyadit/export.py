"""Export denormalized upper-body/hand motion in the original SMPL-H schema."""

import json
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .data import UPPER_BODY_JOINTS
from .rotations import convert_rotation_6d_to_axis_angle


def motion_to_smplh(motion):
    """motion: [frames, 258], unnormalized 6D rotation."""
    angles = convert_rotation_6d_to_axis_angle(
        torch.as_tensor(motion).float().unsqueeze(0)
    )[0].numpy()
    frames = len(angles)
    body = np.zeros((frames, 21, 3), dtype=np.float32)
    body[:, UPPER_BODY_JOINTS, :] = angles[:, :39].reshape(frames, 13, 3)
    return {
        "smplh:body_pose": body,
        "smplh:left_hand_pose": angles[:, 39:84].reshape(frames, 15, 3),
        "smplh:right_hand_pose": angles[:, 84:129].reshape(frames, 15, 3),
        "smplh:global_orient": np.zeros((frames, 3), dtype=np.float32),
        "smplh:translation": np.zeros((frames, 3), dtype=np.float32),
        "smplh:is_valid": np.ones(frames, dtype=bool),
    }


def save_outputs(directory, generated, latent, sample, dataset, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)

    def denormalize(tensor):
        array = tensor.detach().float().cpu().numpy()
        if dataset.normalize_motion:
            array = array * dataset.motion_std + dataset.motion_mean
        return array

    arrays = {
        "generated_motion_denorm": denormalize(generated[0]),
        "target_motion_denorm": denormalize(sample["motion_latent"]),
        "target_motion_other_denorm": denormalize(sample["motion_latent_other"]),
    }
    for name, motion in arrays.items():
        np.savez_compressed(directory / (name + ".npz"), **motion_to_smplh(motion))
    np.savez_compressed(
        directory / "generated_motion_latent.npz",
        latent=latent.detach().float().cpu().numpy(),
    )
    for key, name in (
        ("audio", "input_audio.wav"),
        ("audio_other", "input_audio_other.wav"),
    ):
        sf.write(
            directory / name, sample[key].cpu().numpy(), config["model"]["audio_sr"]
        )
    metadata = {
        "video_id": sample["video_id"],
        "start_frame": sample["start_frame"],
        "relationship": sample["relationship_onehot"].tolist(),
        "personality": sample["personality_traits"].tolist(),
        "orthogonal_fusion": config["model"]["orthogonal_fusion"],
    }
    with (directory / "metadata.json").open("w") as stream:
        json.dump(metadata, stream, indent=2)
