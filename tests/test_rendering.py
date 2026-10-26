"""Rendering checks that do not require licensed SMPL-H model assets."""

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

from dyadit import rendering
from dyadit import render_cli
from dyadit.render_backends import Camera, SoftwareRenderer, look_at
from dyadit.rendering import (
    RenderSettings,
    SmplhActor,
    facing_rotation,
    freeze_lower_body,
    load_sequence,
    render_pair,
)
from dyadit.smplh import resolve_model_path
from dyadit.video import VideoWriter, ffmpeg_executable


def motion_file(path, frames=8):
    body = np.zeros((frames, 21, 3), dtype=np.float32)
    body[:, 2, 0] = np.linspace(0, 0.3, frames)
    body[:, 0, 0] = np.linspace(0, 1, frames)
    np.savez(
        path,
        **{
            "smplh:body_pose": body,
            "smplh:left_hand_pose": np.zeros((frames, 15, 3), dtype=np.float32),
            "smplh:right_hand_pose": np.zeros((frames, 15, 3), dtype=np.float32),
            "smplh:translation": np.ones((frames, 3), dtype=np.float32) * 100,
            "smplh:global_orient": np.ones((frames, 3), dtype=np.float32),
        }
    )
    return path


class TinyBodyModel:
    num_betas = 10
    faces = np.array([[0, 1, 2], [0, 3, 1], [0, 2, 3], [1, 3, 2]])
    lbs_weights = torch.zeros(4, 52)
    lbs_weights[0, 0] = lbs_weights[1, 1] = 1
    lbs_weights[2, 16] = lbs_weights[3, 18] = 1

    def __call__(self, body_pose, **kwargs):
        base = body_pose.new_tensor(
            [[-0.2, -1, 0.1], [0.2, -1, 0.1], [-0.3, 0.7, 0.2], [0.3, 0.7, -0.2]]
        )
        vertices = base.unsqueeze(0).repeat(len(body_pose), 1, 1)
        vertices += body_pose[:, 6:7].unsqueeze(-1)
        return SimpleNamespace(
            vertices=vertices, joints=body_pose.new_zeros(len(body_pose), 1, 3)
        )


def test_lower_body_is_constant_and_upper_body_preserved():
    original = np.random.default_rng(4).normal(size=(10, 21, 3)).astype(np.float32)
    fixed = freeze_lower_body(original)
    assert not fixed[:, rendering.LOWER_BODY_JOINTS].any()
    upper = [index for index in range(21) if index not in rendering.LOWER_BODY_JOINTS]
    np.testing.assert_array_equal(fixed[:, upper], original[:, upper])
    first = freeze_lower_body(original, "first-frame")
    np.testing.assert_array_equal(
        first[:, rendering.LOWER_BODY_JOINTS],
        np.repeat(original[:1, rendering.LOWER_BODY_JOINTS], len(original), axis=0),
    )
    assert original[:, rendering.LOWER_BODY_JOINTS].any()


def test_people_face_each_other():
    front = np.array([0, 0, 1])
    np.testing.assert_allclose(facing_rotation(0) @ front, [1, 0, 0], atol=1e-6)
    np.testing.assert_allclose(facing_rotation(1) @ front, [-1, 0, 0], atol=1e-6)


def test_model_resolution_accepts_easymocap_names(tmp_path):
    path = tmp_path / "smplh" / "SMPLH_male.pkl"
    path.parent.mkdir()
    path.touch()
    assert resolve_model_path(tmp_path, "male") == path
    with pytest.raises(FileNotFoundError):
        resolve_model_path(tmp_path, "female")


def test_world_placement_and_lower_mesh_vertices_are_fixed(tmp_path, monkeypatch):
    path = motion_file(tmp_path / "motion.npz")
    sequence = load_sequence(path)
    monkeypatch.setattr(rendering, "load_body_model", lambda *args: TinyBodyModel())
    actor = SmplhActor("unused", sequence, RenderSettings(), 0)
    vertices = actor.vertices(np.arange(len(sequence)))
    fixed = vertices[:, actor.fixed_vertices]
    np.testing.assert_allclose(fixed, np.repeat(fixed[:1], len(sequence), axis=0))
    assert (
        np.linalg.norm(
            vertices[-1, ~actor.fixed_vertices] - vertices[0, ~actor.fixed_vertices]
        )
        > 0.1
    )
    assert vertices[:, actor.fixed_vertices, 1].min() == pytest.approx(0)
    assert actor.position[0] == pytest.approx(-0.8)
    assert (
        np.abs(vertices).max() < 3
    )  # Input global translation/orientation are deliberately ignored.


def test_camera_is_fixed_and_orthonormal():
    pose = look_at([0, 2, 6], [0, 1, 0])
    np.testing.assert_allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-6)
    settings = RenderSettings(width=320, height=240)
    camera = Camera.fit(np.array([[-1.5, 0, -1], [1.5, 1.8, 1]]), settings)
    point = np.array([[0, 0.9, 0]])
    first, _ = camera.project(point)
    second, _ = camera.project(point)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_allclose(first[0], [160, 120], atol=1e-5)


def test_software_renderer_displays_both_people():
    pytest.importorskip("cv2")
    settings = RenderSettings(width=320, height=240, backend="software")
    vertices = np.array(
        [
            [-0.2, 0, -0.15],
            [0.2, 0, -0.15],
            [0.2, 1.7, -0.15],
            [-0.2, 1.7, -0.15],
            [-0.2, 0, 0.15],
            [0.2, 0, 0.15],
            [0.2, 1.7, 0.15],
            [-0.2, 1.7, 0.15],
        ]
    )
    faces = np.array(
        [
            [0, 3, 2],
            [0, 2, 1],
            [4, 5, 6],
            [4, 6, 7],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
            [0, 1, 5],
            [0, 5, 4],
            [3, 7, 6],
            [3, 6, 2],
        ]
    )
    renderer = SoftwareRenderer(
        settings,
        np.array([[-1.5, 0, -1], [1.5, 1.8, 1]]),
        [faces, faces],
        rendering.COLORS,
    )
    image = renderer.render([vertices + [-0.8, 0, 0], vertices + [0.8, 0, 0]])
    assert image.shape == (240, 320, 3)
    assert ((image[:, :, 2].astype(float) > image[:, :, 0] * 1.3)).sum() > 100
    assert ((image[:, :, 0].astype(float) > image[:, :, 2] * 1.3)).sum() > 100
    renderer.close()


def test_pair_video_contains_audio_and_preserves_duration(tmp_path, monkeypatch):
    pytest.importorskip("cv2")
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    monkeypatch.setattr(rendering, "load_body_model", lambda *args: TinyBodyModel())
    first = motion_file(tmp_path / "motion a.npz")
    second = motion_file(tmp_path / "motion b.npz")
    sample_rate = 16000
    times = np.arange(sample_rate) / sample_rate
    audio_a, audio_b = tmp_path / "audio a.wav", tmp_path / "audio b.wav"
    sf.write(audio_a, np.sin(2 * np.pi * 220 * times) * 0.1, sample_rate)
    sf.write(audio_b, np.sin(2 * np.pi * 440 * times) * 0.1, sample_rate)
    output = tmp_path / "conversation.mp4"
    settings = RenderSettings(
        width=320, height=240, backend="software", step=2, start_frame=1
    )
    render_pair(
        first,
        second,
        "unused",
        output,
        audio_a=audio_a,
        audio_b=audio_b,
        settings=settings,
    )
    assert output.is_file() and output.with_suffix(".png").is_file()
    info = json.loads(output.with_suffix(".json").read_text())
    assert info["frames"] == 4
    assert info["output_fps"] == 15
    assert info["duration_seconds"] == pytest.approx(4 / 15)
    assert info["settings"]["audio_layout"] == "stereo"
    assert info["actors"]["blue"]["motion"] == str(first)
    assert info["actors"]["blue"]["audio_channel"] == "left"
    assert info["actors"]["orange"]["motion"] == str(second)
    assert info["actors"]["orange"]["audio_channel"] == "right"
    frames = imageio_ffmpeg.read_frames(str(output))
    metadata = next(frames)
    assert metadata["size"] == (320, 240)
    assert metadata["fps"] == 15
    assert len(list(frames)) == 4
    check = subprocess.run(
        [
            ffmpeg_executable(),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
    )
    assert check.returncode == 0, check.stderr.decode()
    with pytest.raises(FileExistsError):
        render_pair(first, second, "unused", output, settings=settings)


def test_render_settings_reject_invalid_video_dimensions():
    with pytest.raises(ValueError, match="even"):
        RenderSettings(width=319).validate()


def test_quiet_corresponding_voices_are_audible_in_video(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    sample_rate = 16000
    times = np.arange(sample_rate * 2) / sample_rate
    first, second = tmp_path / "quiet_a.wav", tmp_path / "quiet_b.wav"
    sf.write(
        first, np.sin(2 * np.pi * 220 * times) * 0.001, sample_rate, subtype="FLOAT"
    )
    sf.write(
        second, np.sin(2 * np.pi * 440 * times) * 0.0005, sample_rate, subtype="FLOAT"
    )
    output = tmp_path / "audible.mp4"
    with VideoWriter(
        output, 64, 64, 10, 2, [first, second], audio_layout="mix"
    ) as writer:
        for _ in range(20):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    decoded = subprocess.run(
        [
            ffmpeg_executable(),
            "-v",
            "error",
            "-i",
            str(output),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-ac",
            "1",
            "-ar",
            str(sample_rate),
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    samples = np.frombuffer(decoded.stdout, dtype="<f4")
    assert np.sqrt(np.mean(samples**2)) > 0.02
    assert len(samples) / sample_rate == pytest.approx(2, abs=0.03)
    spectrum = np.abs(np.fft.rfft(samples))
    frequencies = np.fft.rfftfreq(len(samples), 1 / sample_rate)
    first_peak = spectrum[np.abs(frequencies - 220).argmin()]
    second_peak = spectrum[np.abs(frequencies - 440).argmin()]
    assert first_peak > second_peak
    assert first_peak / second_peak == pytest.approx(2, rel=0.3)


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_condition_and_generated_motions_move_with_their_audio(
    tmp_path, monkeypatch, entry
):
    for name in ("input_audio.wav", "input_audio_other.wav"):
        (tmp_path / name).touch()
    captured = {}

    def render(motion_a, motion_b, model_path, output, **kwargs):
        captured.update(
            motion_a=motion_a,
            motion_b=motion_b,
            model_path=model_path,
            output=output,
            **kwargs
        )
        return Path(output)

    monkeypatch.setattr(render_cli, "render_pair", render)
    if entry == "api":
        render_cli.render_sample(tmp_path, "unused")
    else:
        monkeypatch.setattr(
            "sys.argv",
            ["render.py", "--sample-dir", str(tmp_path), "--model-path", "unused"],
        )
        monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
        render_cli.main()
    assert captured["motion_a"] == tmp_path / "target_motion_other_denorm.npz"
    assert captured["motion_b"] == tmp_path / "generated_motion_denorm.npz"
    assert captured["audio_a"] == tmp_path / "input_audio_other.wav"
    assert captured["audio_b"] == tmp_path / "input_audio.wav"
    if entry == "cli":
        assert captured["settings"].audio_layout == "stereo"


def decode_native_audio(path, sample_rate=16000):
    result = subprocess.run(
        [
            ffmpeg_executable(),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-ar",
            str(sample_rate),
            "-f",
            "f32le",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return np.frombuffer(result.stdout, dtype="<f4").reshape(-1, 2)


@pytest.mark.parametrize("normalize", [False, True])
def test_stereo_audio_has_blue_only_left_and_orange_only_right(tmp_path, normalize):
    pytest.importorskip("imageio_ffmpeg")
    sample_rate = 16000
    times = np.arange(sample_rate * 2) / sample_rate
    blue, orange = tmp_path / "blue.wav", tmp_path / "orange.wav"
    sf.write(blue, np.sin(2 * np.pi * 220 * times) * 0.1, sample_rate)
    sf.write(orange, np.sin(2 * np.pi * 440 * times) * 0.05, sample_rate)
    output = tmp_path / "stereo.mp4"
    with VideoWriter(
        output, 64, 64, 10, 2, [blue, orange], normalize_audio=normalize
    ) as writer:
        for _ in range(20):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    audio = decode_native_audio(output, sample_rate)
    assert len(audio) / sample_rate == pytest.approx(2, abs=0.03)
    assert np.sqrt(np.mean(audio**2)) > 0.02
    spectrum = np.abs(np.fft.rfft(audio, axis=0))
    frequencies = np.fft.rfftfreq(len(audio), 1 / sample_rate)
    left_voice = spectrum[np.abs(frequencies - 220).argmin()]
    right_voice = spectrum[np.abs(frequencies - 440).argmin()]
    assert left_voice[0] > left_voice[1] * 100
    assert right_voice[1] > right_voice[0] * 100
    assert left_voice[0] / right_voice[1] == pytest.approx(2, rel=0.3)


@pytest.mark.parametrize("active_slot", [0, 1])
def test_missing_voice_leaves_its_actor_channel_silent(tmp_path, active_slot):
    pytest.importorskip("imageio_ffmpeg")
    sample_rate = 16000
    audio_path = tmp_path / "one_voice.wav"
    sf.write(
        audio_path,
        np.sin(2 * np.pi * 330 * np.arange(sample_rate) / sample_rate) * 0.1,
        sample_rate,
    )
    paths = [None, None]
    paths[active_slot] = audio_path
    output = tmp_path / "one_voice.mp4"
    with VideoWriter(output, 64, 64, 10, 1, paths, normalize_audio=False) as writer:
        for _ in range(10):
            writer.write(np.zeros((64, 64, 3), dtype=np.uint8))
    audio = decode_native_audio(output, sample_rate)
    assert len(audio) / sample_rate == pytest.approx(1, abs=0.03)
    rms = np.sqrt(np.mean(audio**2, axis=0))
    assert rms[active_slot] > 0.03
    assert rms[1 - active_slot] < max(1e-5, rms[active_slot] * 0.005)
