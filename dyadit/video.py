"""Stream RGB frames to FFmpeg with blue/left and orange/right conversation audio."""

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as error:
        raise RuntimeError("Install imageio-ffmpeg or put ffmpeg on PATH") from error


def mixed_audio_filter(track_count, duration, normalize_audio=True):
    """Mix corresponding voices together, preserving their relative levels."""
    if track_count < 1:
        raise ValueError("At least one audio track is required")
    sources = "".join(f"[{index + 1}:a]" for index in range(track_count))
    chain = f"{sources}amix=inputs={track_count}:duration=longest:normalize=0"
    return _finish_audio_filter(chain, duration, normalize_audio)


def stereo_audio_filter(channels, duration, normalize_audio=True):
    """Route each non-missing source exclusively to its actor's stereo channel.

    ``channels`` retains actor slots: 0 is blue/left, 1 is orange/right. If only
    the right actor has a waveform, the left channel must remain silent.
    """
    if (
        not channels
        or len(channels) > 2
        or any(channel not in (0, 1) for channel in channels)
    ):
        raise ValueError(
            "Stereo conversation audio needs one or two actor channels (0/1)"
        )
    filters, sources = [], []
    for index, channel in enumerate(channels, start=1):
        label = f"voice_{index}"
        filters.append(
            f"[{index}:a]aformat=channel_layouts=mono,"
            f"pan=stereo|c{channel}=c0|c{1 - channel}=0*c0[{label}]"
        )
        sources.append(f"[{label}]")
    chain = (
        "".join(sources) + f"amix=inputs={len(channels)}:duration=longest:normalize=0"
    )
    filters.append(_finish_audio_filter(chain, duration, normalize_audio))
    return ";".join(filters)


def _finish_audio_filter(chain, duration, normalize_audio):
    if normalize_audio:
        # Normalize the combined conversation, rather than separately amplifying
        # the quiet/listening person's background noise.
        chain += ",loudnorm=I=-18:TP=-1.5:LRA=11"
    return chain + f",apad,atrim=duration={duration},asetpts=PTS-STARTPTS[audio]"


class VideoWriter:
    def __init__(
        self,
        path,
        width,
        height,
        fps,
        duration,
        audio_paths=(),
        audio_start=0,
        *,
        normalize_audio=True,
        audio_layout="stereo",
    ):
        if audio_layout not in ("stereo", "mix"):
            raise ValueError("audio_layout must be stereo or mix")
        if audio_layout == "stereo" and len(audio_paths) > 2:
            raise ValueError("Stereo conversation audio supports at most two actors")
        audio_inputs = [
            (slot, path) for slot, path in enumerate(audio_paths) if path is not None
        ]
        self.path = Path(path)
        self.shape = (height, width, 3)
        self.temporary = tempfile.TemporaryDirectory(
            prefix="dyadit-render-", dir=self.path.parent
        )
        self.destination = Path(self.temporary.name) / "video.mp4"
        self.errors = tempfile.TemporaryFile()
        command = [
            ffmpeg_executable(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
        ]
        for _, audio in audio_inputs:
            command += ["-ss", str(audio_start), "-i", str(Path(audio).resolve())]
        command += ["-map", "0:v:0"]
        if audio_inputs:
            audio_filter = (
                stereo_audio_filter(
                    [slot for slot, _ in audio_inputs], duration, normalize_audio
                )
                if audio_layout == "stereo"
                else mixed_audio_filter(len(audio_inputs), duration, normalize_audio)
            )
            command += [
                "-filter_complex",
                audio_filter,
                "-map",
                "[audio]",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "48000",
            ]
        command += [
            "-c:v",
            "libx264",
            "-threads",
            "2",
            "-preset",
            "fast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-t",
            str(duration),
            "-movflags",
            "+faststart",
            str(self.destination),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.errors,
            )
        except Exception:
            self.errors.close()
            self.temporary.cleanup()
            raise

    def __enter__(self):
        return self

    def write(self, frame):
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise ValueError(
                f"Expected uint8 RGB frame of shape {self.shape}, got {frame.shape}, {frame.dtype}"
            )
        try:
            self.process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as error:
            self.process.wait()
            self.errors.seek(0)
            raise RuntimeError(self.errors.read().decode(errors="replace")) from error

    def __exit__(self, kind, error, traceback):
        try:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            code = self.process.wait()
            if kind is None:
                if code:
                    self.errors.seek(0)
                    raise RuntimeError(
                        f"FFmpeg failed: {self.errors.read().decode(errors='replace')}"
                    )
                os.replace(self.destination, self.path)
        finally:
            self.errors.close()
            self.temporary.cleanup()
