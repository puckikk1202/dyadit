"""Render two facing SMPL-H people with a fixed pelvis and lower body."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from .smplh import load_body_model


LOWER_BODY_JOINTS = (0, 1, 3, 4, 6, 7, 9, 10)  # body_pose indices; pelvis is separate.
FIXED_SKIN_JOINTS = (
    0,
    1,
    2,
    4,
    5,
    7,
    8,
    10,
    11,
)  # SMPL-H LBS indices, including pelvis.
COLORS = ((0.20, 0.48, 0.75), (0.88, 0.48, 0.26))


@dataclass
class RenderSettings:
    width: int = 1280
    height: int = 720
    fps: float = 30.0
    separation: float = 1.6
    backend: str = "auto"
    device: str = "cpu"
    batch_size: int = 16
    lower_body: str = "neutral"
    hand_mean: str = "flat"
    gender_a: str = "male"
    gender_b: str = "male"
    start_frame: int = 0
    step: int = 1
    max_frames: int = 0
    camera_azimuth: float = 12.0
    camera_elevation: float = 10.0
    save_frames: bool = False
    audio_normalize: bool = True
    audio_layout: str = "stereo"

    def validate(self):
        if self.width <= 0 or self.height <= 0 or self.width % 2 or self.height % 2:
            raise ValueError("Video width and height must be positive even integers")
        if self.fps <= 0 or self.separation <= 0 or self.batch_size < 1:
            raise ValueError("FPS, person separation and batch size must be positive")
        if self.step < 1 or self.start_frame < 0 or self.max_frames < 0:
            raise ValueError(
                "step must be positive; start-frame and max-frames must be non-negative"
            )
        if self.lower_body not in ("neutral", "first-frame"):
            raise ValueError("lower-body must be neutral or first-frame")
        if self.hand_mean not in ("flat", "model"):
            raise ValueError("hand-mean must be flat or model")
        if self.backend not in ("auto", "egl", "osmesa", "software"):
            raise ValueError("Unknown rendering backend")
        if self.audio_layout not in ("stereo", "mix"):
            raise ValueError("audio_layout must be stereo or mix")


@dataclass
class MotionSequence:
    body: np.ndarray
    left_hand: np.ndarray
    right_hand: np.ndarray
    betas: np.ndarray

    def __len__(self):
        return len(self.body)


def load_sequence(path):
    with np.load(path, allow_pickle=False) as archive:
        if "smplh:body_pose" not in archive:
            raise ValueError(
                f"{path} needs smplh:body_pose. Use exported *_motion*_denorm.npz files."
            )
        body = np.array(archive["smplh:body_pose"], dtype=np.float32).reshape(-1, 21, 3)
        count = len(body)
        if not count:
            raise ValueError(f"Empty motion sequence: {path}")

        def hand(key):
            value = (
                np.array(archive[key], dtype=np.float32).reshape(-1, 15, 3)
                if key in archive
                else np.zeros((count, 15, 3), dtype=np.float32)
            )
            if len(value) == 1:
                value = np.repeat(value, count, axis=0)
            if len(value) != count:
                raise ValueError(f"Motion and hand frame counts differ in {path}")
            return value

        beta_key = next(
            (key for key in ("smplh:betas", "betas") if key in archive), None
        )
        betas = (
            np.asarray(archive[beta_key], dtype=np.float32)
            if beta_key
            else np.zeros(10, dtype=np.float32)
        )
        betas = betas.reshape(-1, betas.shape[-1])[0].copy()
        result = MotionSequence(
            body, hand("smplh:left_hand_pose"), hand("smplh:right_hand_pose"), betas
        )
    if not all(
        np.isfinite(value).all()
        for value in (result.body, result.left_hand, result.right_hand, result.betas)
    ):
        raise ValueError(f"Non-finite SMPL-H parameters in {path}")
    return result


def freeze_lower_body(body, mode="neutral"):
    fixed = body.copy()
    if mode == "neutral":
        fixed[:, LOWER_BODY_JOINTS, :] = 0
    elif mode == "first-frame":
        fixed[:, LOWER_BODY_JOINTS, :] = body[:1, LOWER_BODY_JOINTS, :]
    else:
        raise ValueError(f"Unknown lower-body mode: {mode}")
    return fixed


def facing_rotation(person):
    """Native SMPL-H faces +Z; A faces +X and B faces -X, with Y up."""
    yaw = np.pi / 2 if person == 0 else -np.pi / 2
    return np.array(
        [[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]],
        dtype=np.float32,
    )


class SmplhActor:
    def __init__(self, model_path, sequence, settings, person):
        self.sequence = sequence
        self.device = settings.device
        gender = settings.gender_a if person == 0 else settings.gender_b
        self.model = load_body_model(
            model_path, gender, settings.device, settings.hand_mean == "flat"
        )
        self.body = freeze_lower_body(sequence.body, settings.lower_body)
        self.betas = np.zeros(self.model.num_betas, dtype=np.float32)
        self.betas[: min(len(sequence.betas), len(self.betas))] = sequence.betas[
            : len(self.betas)
        ]
        self.rotation = facing_rotation(person)
        self.position = np.array(
            [(-0.5 if person == 0 else 0.5) * settings.separation, 0, 0],
            dtype=np.float32,
        )
        reference_body = np.zeros((1, 21, 3), dtype=np.float32)
        reference_body[:, LOWER_BODY_JOINTS, :] = self.body[:1, LOWER_BODY_JOINTS, :]
        reference, pelvis = self._forward(
            reference_body,
            np.zeros((1, 15, 3), dtype=np.float32),
            np.zeros((1, 15, 3), dtype=np.float32),
        )
        self.reference = reference[0]
        self.pelvis = pelvis[0]
        self.position[1] = -(self.reference - self.pelvis)[:, 1].min()
        dominant_joint = self.model.lbs_weights.detach().cpu().numpy().argmax(axis=1)
        self.fixed_vertices = np.isin(dominant_joint, FIXED_SKIN_JOINTS)
        self.faces = self.model.faces
        self.reference_world = self._world(reference)[0]

    @torch.no_grad()
    def _forward(self, body, left, right):
        count = len(body)

        def tensor(value):
            return torch.as_tensor(value, dtype=torch.float32, device=self.device)

        output = self.model(
            betas=tensor(self.betas).unsqueeze(0).expand(count, -1),
            body_pose=tensor(body.reshape(count, 63)),
            left_hand_pose=tensor(left.reshape(count, 45)),
            right_hand_pose=tensor(right.reshape(count, 45)),
            global_orient=torch.zeros(count, 3, device=self.device),
            transl=torch.zeros(count, 3, device=self.device),
            return_verts=True,
        )
        return output.vertices.cpu().numpy(), output.joints[:, 0].cpu().numpy()

    def _world(self, vertices):
        vertices = vertices.copy()
        # Pose-corrective blendshapes can otherwise move leg/foot vertices slightly
        # even with constant lower-joint rotations. Pin those vertices explicitly.
        vertices[:, self.fixed_vertices] = self.reference[self.fixed_vertices]
        return (vertices - self.pelvis) @ self.rotation.T + self.position

    def vertices(self, indices):
        vertices, _ = self._forward(
            self.body[indices],
            self.sequence.left_hand[indices],
            self.sequence.right_hand[indices],
        )
        return self._world(vertices)


def render_pair(
    motion_a,
    motion_b,
    model_path,
    output,
    *,
    audio_a=None,
    audio_b=None,
    settings=None,
    overwrite=False,
):
    from .render_backends import make_renderer
    from .video import VideoWriter

    settings = settings or RenderSettings()
    settings.validate()
    output = Path(output).expanduser().resolve()
    if output.suffix.lower() != ".mp4":
        raise ValueError("Output video must use the .mp4 extension")
    for path in (output, output.with_suffix(".png"), output.with_suffix(".json")):
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass --overwrite to replace it."
            )
    for path in (audio_a, audio_b):
        if path is not None and not Path(path).is_file():
            raise FileNotFoundError(path)
    sequences = [load_sequence(path) for path in (motion_a, motion_b)]
    indices = np.arange(settings.start_frame, min(map(len, sequences)), settings.step)
    if settings.max_frames:
        indices = indices[: settings.max_frames]
    if not len(indices):
        raise ValueError(
            "No frames left after applying start-frame, step and max-frames"
        )
    actors = [
        SmplhActor(model_path, sequence, settings, person)
        for person, sequence in enumerate(sequences)
    ]
    reference = np.concatenate([actor.reference_world for actor in actors])
    bounds = np.stack([reference.min(0), reference.max(0)])
    bounds[0, 0] -= 0.6
    bounds[1, 0] += 0.6
    renderer = make_renderer(
        settings, bounds, [actor.faces for actor in actors], COLORS
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frames_dir = output.parent / (output.stem + "_frames")
    if settings.save_frames:
        frames_dir.mkdir(parents=True, exist_ok=True)
    fps = settings.fps / settings.step
    preview = None
    try:
        with VideoWriter(
            output,
            settings.width,
            settings.height,
            fps,
            len(indices) / fps,
            [audio_a, audio_b],
            settings.start_frame / settings.fps,
            normalize_audio=settings.audio_normalize,
            audio_layout=settings.audio_layout,
        ) as writer:
            for offset in range(0, len(indices), settings.batch_size):
                batch_indices = indices[offset : offset + settings.batch_size]
                meshes = [actor.vertices(batch_indices) for actor in actors]
                for index in range(len(batch_indices)):
                    image = renderer.render([mesh[index] for mesh in meshes])
                    writer.write(image)
                    if preview is None:
                        preview = image.copy()
                    if settings.save_frames:
                        import imageio.v3 as iio

                        iio.imwrite(
                            frames_dir / f"frame_{offset + index:06d}.png", image
                        )
                print(
                    f"Rendered {min(offset + settings.batch_size, len(indices))}/{len(indices)} frames",
                    flush=True,
                )
    finally:
        renderer.close()
    import imageio.v3 as iio

    iio.imwrite(output.with_suffix(".png"), preview)
    info = {
        "motion_a": str(Path(motion_a).resolve()),
        "motion_b": str(Path(motion_b).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "settings": asdict(settings),
        "backend": renderer.backend,
        "frames": len(indices),
        "output_fps": fps,
        "duration_seconds": len(indices) / fps,
        "audio": [str(Path(path).resolve()) for path in (audio_a, audio_b) if path],
        "layout": "Blue/left faces +X; orange/right faces -X; Y is up",
        "actors": {
            "blue": {
                "screen_position": "left",
                "motion": str(Path(motion_a).resolve()),
                "audio": str(Path(audio_a).resolve()) if audio_a is not None else None,
                "audio_channel": (
                    "left" if settings.audio_layout == "stereo" else "mixed"
                ),
            },
            "orange": {
                "screen_position": "right",
                "motion": str(Path(motion_b).resolve()),
                "audio": str(Path(audio_b).resolve()) if audio_b is not None else None,
                "audio_channel": (
                    "right" if settings.audio_layout == "stereo" else "mixed"
                ),
            },
        },
        "lower_body": "constant pelvis/leg/foot poses, fixed placement and pinned lower-body skin vertices",
        "camera": renderer.camera.pose.tolist(),
    }
    with output.with_suffix(".json").open("w") as stream:
        json.dump(info, stream, indent=2)
    return output
