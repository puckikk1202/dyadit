"""Paired conversation data: 258D motion, 16 kHz audio and social conditions."""

import csv
import random
import re
from bisect import bisect_right
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .rotations import convert_axis_angle_to_rotation_6d


UPPER_BODY_JOINTS = (2, 5, 8, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
PERSONALITY_COLUMNS = (
    "extraversion_raw",
    "agreeableness_raw",
    "conscientiousness_raw",
    "neuroticism_raw",
    "openness_raw",
)
RELATIONSHIP_CLASSES = {
    "friends": 0,
    "coworkers": 0,
    "stranger": 1,
    "family-generic": 2,
    "familiar-generic": 2,
    "dating/spouse/romantic_partner": 3,
}


def motion_name(path):
    name = Path(path).stem
    return name[:-3] if name.endswith("_6d") else name


def motion_directory(root, custom_path=None):
    if custom_path:
        return Path(custom_path)
    for name in ("motion_6d_slim", "motion_6d", "motion"):
        path = Path(root) / name
        if path.is_dir():
            return path
    raise FileNotFoundError(f"No motion directory found under {root}")


def load_motion(path, start=0, length=None):
    """Read 6D motion, converting legacy SMPL-H axis-angle data when needed."""
    path = Path(path)
    end = None if length is None else start + length
    if path.suffix == ".npy":
        motion = np.array(np.load(path, mmap_mode="r")[start:end], dtype=np.float32)
    else:
        with np.load(path, allow_pickle=True) as archive:
            if "smplh:motion_6d" in archive:
                motion = np.array(
                    archive["smplh:motion_6d"][start:end], dtype=np.float32
                )
            else:
                body = archive["smplh:body_pose"][start:end, UPPER_BODY_JOINTS, :]
                left = archive["smplh:left_hand_pose"][start:end]
                right = archive["smplh:right_hand_pose"][start:end]
                features = np.concatenate(
                    [x.reshape(len(x), -1) for x in (body, left, right)], axis=-1
                )
                motion = convert_axis_angle_to_rotation_6d(
                    torch.from_numpy(features).float().unsqueeze(0)
                )[0].numpy()
    if motion.ndim != 2 or motion.shape[-1] != 258:
        raise ValueError(f"Expected [frames, 258] motion in {path}, got {motion.shape}")
    if not np.isfinite(motion).all():
        raise ValueError(f"Non-finite motion values in {path}")
    return motion


def motion_length(path):
    if Path(path).suffix == ".npy":
        return np.load(path, mmap_mode="r").shape[0]
    with np.load(path, allow_pickle=True) as archive:
        key = "smplh:motion_6d" if "smplh:motion_6d" in archive else "smplh:body_pose"
        return archive[key].shape[0]


def find_audio(root, name):
    for directory, suffix in (
        ("audio_slim", ".npy"),
        ("audio_npz", ".npz"),
        ("audio", ".wav"),
        ("audio", ".mp3"),
    ):
        path = Path(root) / directory / (name + suffix)
        if path.is_file():
            return path
    raise FileNotFoundError(f"Audio not found for {name} under {root}")


def load_audio(path, start_frame, length_frames, fps=30, sample_rate=16000):
    path = Path(path)
    if path.suffix == ".npy":
        audio = np.load(path, mmap_mode="r")
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as archive:
            key = next(
                (
                    k
                    for k in ("audio", "audio_data", "signal", "waveform")
                    if k in archive
                ),
                None,
            )
            if key is None:
                raise ValueError(f"No waveform field found in {path}")
            audio = archive[key]
    else:
        import librosa

        audio, _ = librosa.load(path, sr=sample_rate, mono=True)
    audio = np.asarray(audio).reshape(-1)
    start = int(start_frame * sample_rate / fps)
    size = round(length_frames * sample_rate / fps)
    segment = np.array(audio[start : start + size], dtype=np.float32)
    if len(segment) < size:
        segment = np.pad(segment, (0, size - len(segment)), mode="constant")
    if not np.isfinite(segment).all():
        raise ValueError(f"Non-finite audio values in {path}")
    return torch.from_numpy(segment)


def _identifier(value):
    match = re.search(r"\d+", str(value))
    return str(int(match.group())) if match else str(value)


def _csv_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as stream:
        return list(csv.DictReader(stream))


def load_metadata(root):
    relationships, personalities = {}, {}
    for row in _csv_rows(Path(root) / "csv" / "relationships.csv"):
        value = np.zeros(4, dtype=np.float32)
        category = RELATIONSHIP_CLASSES.get(row["relationship_detail"])
        if category is not None:
            value[category] = 1
        relationships[
            (_identifier(row["vendor_id"]), _identifier(row["session_id"]))
        ] = value
    for row in _csv_rows(Path(root) / "csv" / "participants.csv"):
        values = []
        for column in PERSONALITY_COLUMNS:
            try:
                value = float(row[column])
                values.append(value / 5 if np.isfinite(value) else 0)
            except (ValueError, TypeError):
                values.append(0)
        personalities[
            (_identifier(row["vendor_id"]), _identifier(row["participant_id"]))
        ] = np.array(values, dtype=np.float32)
    return relationships, personalities


class DyadicDataset(Dataset):
    """Generate A/B and B/A samples with an explicit per-conversation segment index."""

    def __init__(self, config, data_root=None, pose_length=None, random_style=True):
        self.root = Path(data_root or config["paths"]["data_root"])
        self.pose_length = pose_length or config["model"]["pose_length"]
        self.fps = config["model"]["pose_fps"]
        self.sample_rate = config["model"]["audio_sr"]
        self.random_style = random_style
        self.normalize_motion = config["data"].get("normalize_motion", True)
        self.reverse_pairs = config["data"].get("is_reversed", True)
        self.relationships, self.personalities = load_metadata(self.root)
        self.motion_mean = np.zeros(258, dtype=np.float32)
        self.motion_std = np.ones(258, dtype=np.float32)
        if self.normalize_motion:
            stats = config["paths"].get("normalization_stats") or str(
                Path(config["paths"]["data_root"]) / "motion_6d_normalization.npz"
            )
            if not Path(stats).is_file():
                raise FileNotFoundError(
                    f"Normalization statistics not found: {stats}. Generate them with preprocess.py."
                )
            with np.load(stats) as archive:
                self.motion_mean = archive["motion_mean"].astype(np.float32)
                self.motion_std = archive["motion_std"].astype(np.float32)
            if self.motion_mean.shape != (258,) or self.motion_std.shape != (258,):
                raise ValueError(
                    "Normalization mean and std must each have shape [258]"
                )
            if (
                not np.isfinite(self.motion_mean).all()
                or not np.isfinite(self.motion_std).all()
                or (self.motion_std <= 0).any()
            ):
                raise ValueError(
                    "Normalization statistics must be finite and std must be positive"
                )
        grouped = defaultdict(list)
        directory = motion_directory(self.root, config["data"].get("motion_path"))
        for path in sorted(directory.iterdir()):
            if path.suffix in (".npy", ".npz"):
                grouped["_".join(motion_name(path).split("_")[:3])].append(path)
        self.pairs, self.starts, self.cumulative = [], [], [0]
        sample_count = config["data"].get("samples_per_conversation", -1)
        if sample_count != -1 and sample_count < 1:
            raise ValueError(
                "samples_per_conversation must be -1 or a positive integer"
            )
        for paths in grouped.values():
            if len(paths) != 2:
                continue
            first, second = paths
            frames = min(motion_length(first), motion_length(second))
            if frames < self.pose_length:
                continue
            pair = (
                first,
                find_audio(self.root, motion_name(first)),
                second,
                find_audio(self.root, motion_name(second)),
                frames,
            )
            starts = (
                list(range(0, frames - self.pose_length + 1, self.pose_length))
                if sample_count == -1
                else np.linspace(
                    0, frames - self.pose_length, sample_count, dtype=int
                ).tolist()
            )
            self.pairs.append(pair)
            self.starts.append(starts)
            self.cumulative.append(self.cumulative[-1] + len(starts))
        if not self.pairs:
            raise ValueError(
                f"No complete paired conversations of at least {self.pose_length} frames found in {self.root}"
            )

    def __len__(self):
        return self.cumulative[-1] * (2 if self.reverse_pairs else 1)

    def _motion(self, path, start):
        motion = load_motion(path, start, self.pose_length)
        if self.normalize_motion:
            motion = (motion - self.motion_mean) / (self.motion_std + 1e-7)
        return torch.from_numpy(np.asarray(motion, dtype=np.float32))

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        segment_index, reverse = divmod(index, 2) if self.reverse_pairs else (index, 0)
        pair_index = bisect_right(self.cumulative, segment_index) - 1
        start = self.starts[pair_index][segment_index - self.cumulative[pair_index]]
        motion, audio, other_motion, other_audio, frames = self.pairs[pair_index]
        if reverse:
            motion, other_motion, audio, other_audio = (
                other_motion,
                motion,
                other_audio,
                audio,
            )
        style_start = (
            random.randint(0, frames - self.pose_length) if self.random_style else 0
        )
        parts = motion_name(motion).split("_")
        vendor, session, participant = (
            _identifier(parts[0]),
            _identifier(parts[1]),
            _identifier(parts[3]),
        )
        return {
            "motion_latent": self._motion(motion, start),
            "motion_latent_other": self._motion(other_motion, start),
            "style_latent": self._motion(motion, style_start),
            "audio": load_audio(
                audio, start, self.pose_length, self.fps, self.sample_rate
            ),
            "audio_other": load_audio(
                other_audio, start, self.pose_length, self.fps, self.sample_rate
            ),
            "relationship_onehot": torch.from_numpy(
                self.relationships.get(
                    (vendor, session), np.zeros(4, dtype=np.float32)
                ).copy()
            ),
            "personality_traits": torch.from_numpy(
                self.personalities.get(
                    (vendor, participant), np.zeros(5, dtype=np.float32)
                ).copy()
            ),
            "video_id": f"{motion_name(motion)}_{'B-A' if reverse else 'A-B'}_{start}",
            "start_frame": start,
            "motion_path": str(motion),
            "motion_other_path": str(other_motion),
            "audio_path": str(audio),
            "audio_other_path": str(other_audio),
        }
