"""The dashboard's local web server: the page (index.html) and one Server-Sent Events stream (/events).

"state" events carry the JSON of dashboard/state.py plus run information, pushed by the runner a few times a second.
"frame" events carry the camera sheet: all views in one JPEG, at most `fps` per second. The JPEG encoding runs in its
own thread on the newest frames only, so the simulation loop never waits for the page. Standard library only, plus
numpy and Pillow.

    dash = Dashboard(port=8765)
    dash.push_frames(session.frames)     # after every step
    dash.push_state({...})               # a few times a second
"""
from __future__ import annotations

import base64
import io
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image

from ..video.style import EGO_CROP, FONT_DIR

PAGE = Path(__file__).resolve().parent / "index.html"
FONT_FILES = ("IBMPlexSans-Regular.ttf", "IBMPlexSans-Medium.ttf", "IBMPlexSans-SemiBold.ttf", "IBMPlexSans-Bold.ttf")
TURN = {"left_wrist_view": -1, "right_wrist_view": 1}     # np.rot90 k, as the video frame turns the wrist views
SHEET_ROWS = (("side_view", "ego_view"), ("right_thumb_view", "left_wrist_view", "right_wrist_view"))


def _view(frames: dict, view: str) -> np.ndarray | None:
    """One camera as the frame shows it: the ego view's 4:3 centre, the wrist views turned upright."""
    fr = frames.get(view)
    if fr is None:
        return None
    if view == "ego_view":
        sc = fr.shape[1] / 424.0
        x0, y0, x1, y1 = (int(round(c * sc)) for c in EGO_CROP)
        fr = fr[y0:y1, x0:x1]
    if view in TURN:
        fr = np.rot90(fr, TURN[view])
    return fr


def camera_sheet(frames: dict) -> tuple[np.ndarray, dict[str, list[int]]]:
    """All views in one image (side + ego, then thumb + both wrists) and {view: [x, y, w, h]}."""
    tiles = {v: _view(frames, v) for row in SHEET_ROWS for v in row}
    rects, y, width = {}, 0, 0
    for row in SHEET_ROWS:
        x, h_row = 0, 0
        for v in row:
            t = tiles[v]
            if t is not None:
                rects[v] = [x, y, t.shape[1], t.shape[0]]
                x, h_row = x + t.shape[1], max(h_row, t.shape[0])
        y, width = y + h_row, max(width, x)
    sheet = np.zeros((y, width, 3), np.uint8)
    for v, (x0, y0, w, h) in rects.items():
        sheet[y0:y0 + h, x0:x0 + w] = tiles[v]
    return sheet, rects


class Dashboard:
    """The page server and the JPEG encoder thread (daemon threads: they end with the process)."""

    def __init__(self, port: int, host: str = "127.0.0.1", fps: float = 15.0, quality: int = 80) -> None:
        self.fps, self.quality = fps, quality
        self._cv = threading.Condition()
        self._frames: dict | None = None
        self._frames_ver = 0
        self._events = {"state": (0, b""), "frame": (0, b"")}     # name -> (version, the encoded SSE event)
        handler = type("Handler", (_Handler,), {"dash": self})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True
        self.url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{port}"
        for target in (self.httpd.serve_forever, self._encode_loop):
            threading.Thread(target=target, daemon=True).start()

    def push_frames(self, frames: dict) -> None:
        """The newest camera images {view: uint8 (H, W, 3)} (references only; the encoder picks them up)."""
        with self._cv:
            self._frames = dict(frames)
            self._frames_ver += 1
            self._cv.notify_all()

    def push_state(self, state: dict) -> None:
        self._publish("state", json.dumps(state))

    def close(self) -> None:
        self.httpd.shutdown()

    def _publish(self, name: str, data: str) -> None:
        with self._cv:
            ver = self._events[name][0] + 1
            self._events[name] = (ver, f"event: {name}\ndata: {data}\n\n".encode())
            self._cv.notify_all()

    def _encode_loop(self) -> None:
        seen, last = 0, 0.0
        while True:
            with self._cv:
                self._cv.wait_for(lambda: self._frames_ver != seen)
                frames, seen = self._frames, self._frames_ver
            wait = last + 1.0 / self.fps - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
            last = time.perf_counter()
            sheet, rects = camera_sheet(frames)
            buf = io.BytesIO()
            Image.fromarray(sheet).save(buf, "JPEG", quality=self.quality)
            self._publish("frame", json.dumps({"rects": rects, "jpeg": base64.b64encode(buf.getvalue()).decode()}))


class _Handler(BaseHTTPRequestHandler):
    dash: Dashboard
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:
        pass

    def _send(self, body: bytes, ctype: str, cache: bool = False) -> None:
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "max-age=3600" if cache else "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?")[0]
        if path == "/":
            self._send(PAGE.read_bytes(), "text/html; charset=utf-8")
        elif path.startswith("/fonts/") and path[len("/fonts/"):] in FONT_FILES:
            self._send((FONT_DIR / path[len("/fonts/"):]).read_bytes(), "font/ttf", cache=True)
        elif path == "/events":
            self._stream()
        else:
            self.send_error(404)

    def _stream(self) -> None:
        d = self.dash
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        seen = {name: 0 for name in d._events}
        try:
            while True:
                with d._cv:
                    d._cv.wait_for(lambda: any(d._events[n][0] != seen[n] for n in seen), timeout=2.0)
                    out = [ev for n, (ver, ev) in d._events.items() if ver != seen[n] and ev]
                    seen = {n: ver for n, (ver, _) in d._events.items()}
                self.wfile.write(b"".join(out) or b": keep-alive\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
