"""Compose the evaluation video of one logged run: progress panel, side view, the task's alignment tile, the policy
cameras.

  python -m vega_pipette.video.make_video --objlog RUN/objlog.npz --cameras RUN/hd --out run.mp4 --subtitle b0911

--cameras holds one mp4 per view of the task's frame, one frame per logged control row: the run's own camera videos
(standard) or those of video/render_hd.py (high resolution, pipette task). The task (pipette or tube pick-and-place) is
read from the log; the milestones and the stop row come from its rows (progress/). Needs ffmpeg on PATH.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ..progress import pick_place, rollout
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
    with np.load(a.objlog) as z:
        tube_task = "hole_a" in z.files
    if tube_task:
        from .pick_place import PICK_PLACE as panel
        S = pick_place.run_state(str(a.objlog))
    else:
        panel, S = frame.PIPETTE, rollout.run_state(str(a.objlog))
    missing = [v for v in panel.views if not (a.cameras / f"{v}.mp4").is_file()]
    if missing:
        raise SystemExit(f"no camera video for {missing} in {a.cameras}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    w, h = frame.compose(S, a.cameras, a.title, a.subtitle, a.out, k=a.scale, crf=a.crf, encoder=a.encoder, panel=panel)
    last = S["n"] - 1
    reached = " ".join(m["code"] for m in panel.milestones(S, last) if m["row"] is not None)
    outcome = (pick_place.OUTCOME_TEXT[S["outcome"]] if tube_task else "success" if S["success"] else S["stop"])
    print(f"{a.out}: {w} x {h}, {S['n']} rows ({S['n'] / 30.0:.1f} s); milestones {reached or 'none'}; {outcome}")


if __name__ == "__main__":
    main()
