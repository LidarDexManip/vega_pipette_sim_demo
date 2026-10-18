"""Compose the evaluation video of one logged run: progress panel, side view, tip-vs-tube alignment, four cameras.

  python -m vega_pipette.video.make_video --objlog RUN/objlog.npz --cameras RUN/hd --out run.mp4 --subtitle b0911

--cameras holds one mp4 per view (side_view, ego_view, left_wrist_view, right_wrist_view, right_thumb_view), one frame
per logged control row: the run's own camera videos (standard) or those of video/render_hd.py (high resolution). The
milestones and the stop row come from the logged rows (progress/rollout.py). Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ..progress.rollout import run_state
from . import frame


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--objlog", type=Path, required=True, help="the run's objlog.npz")
    ap.add_argument("--cameras", type=Path, required=True, help="directory with one <view>.mp4 per camera")
    ap.add_argument("--out", type=Path, required=True, help="output mp4")
    ap.add_argument("--title", default="π0.5", help="the panel's title (default: π0.5)")
    ap.add_argument("--subtitle", default="", help="the line under the title, e.g. the layout name")
    ap.add_argument("--scale", type=int, default=2, help="frame scale: 2 = 3088 x 1596 (default)")
    ap.add_argument("--crf", type=int, default=20, help="encoder quality (lower = better; default 20)")
    ap.add_argument("--encoder", choices=("libx264", "nvenc"), default="libx264",
                    help="libx264 (default, portable) or nvenc (NVIDIA GPU encoder)")
    a = ap.parse_args()
    missing = [v for v in frame.VIEWS if not (a.cameras / f"{v}.mp4").is_file()]
    if missing:
        raise SystemExit(f"no camera video for {missing} in {a.cameras}")
    S = run_state(str(a.objlog))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame.compose(S, a.cameras, a.title, a.subtitle, a.out, k=a.scale, crf=a.crf, encoder=a.encoder)
    outcome = "success" if S["success"] else S["stop"]
    reached = " ".join(code for key, code, _ in frame.STAGES if S["fs"].get(key) is not None)
    print(f"{a.out}: {w} x {h}, {S['n']} rows ({S['n'] / 30.0:.1f} s); milestones {reached or 'none'}; {outcome}")


if __name__ == "__main__":
    main()
