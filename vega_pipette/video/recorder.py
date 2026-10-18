"""Camera recordings: one H.264 mp4 per view, fed frame by frame (high quality: an intermediate for make_video)."""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class Writer:
    """One view's frames piped to ffmpeg."""

    def __init__(self, path: Path, w: int, h: int, fps: float) -> None:
        self.p = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
                                   "-r", f"{fps:g}", "-i", "-", "-c:v", "libx264", "-preset", "fast", "-crf", "12",
                                   "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)

    def add(self, frame: np.ndarray) -> None:
        self.p.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.p.stdin.close()
        if self.p.wait():
            raise RuntimeError("ffmpeg failed")


class Recorder:
    """<out_dir>/<view>.mp4 for every view handed to add(), one frame per call."""

    def __init__(self, out_dir: Path, fps: float) -> None:
        self.out_dir, self.fps = Path(out_dir), fps
        self._writers: dict[str, Writer] = {}

    def add(self, frames: dict[str, np.ndarray]) -> None:
        for view, fr in frames.items():
            if view not in self._writers:
                self._writers[view] = Writer(self.out_dir / f"{view}.mp4", fr.shape[1], fr.shape[0], self.fps)
            self._writers[view].add(fr)

    def close(self) -> None:
        for w in self._writers.values():
            w.close()
