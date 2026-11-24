#!/usr/bin/env python3
"""make_hand_points.py: make assets/robot/wuji_left_hand_points.npz, the surface points of the left Wuji hand that the
tube pick-and-place runner uses to estimate how the tube sits in the hand (vega_pipette/control/tube_in_hand.py).

For each of the 16 links (proximal, middle and distal of the five fingers, and the palm l_wrist) it samples 150 points
on the link's collision mesh, in the link's frame, in metres. The sampling is seeded, so the file is the same on every
run. The meshes are those of wuji-description (see NOTICE), checked against the md5 list in tools/hand_meshes.md5.

  python3 tools/make_hand_points.py WUJI_DESCRIPTION/hand2/hand2_beta1/body/meshes/left [--out FILE]

Needs numpy and trimesh (both in the simulation environment).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
MD5_LIST = Path(__file__).resolve().parent / "hand_meshes.md5"
LINKS = tuple(f"l_{finger}_{segment}" for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
              for segment in ("proximal", "middle", "distal")) + ("l_wrist",)
POINTS_PER_LINK = 150
SEED = 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("meshes", type=Path, help="the left hand's mesh directory of wuji-description")
    ap.add_argument("--out", type=Path, default=REPO / "assets" / "robot" / "wuji_left_hand_points.npz")
    a = ap.parse_args()
    md5 = dict(line.split()[::-1] for line in MD5_LIST.read_text().splitlines() if line.strip())
    rng = np.random.default_rng(SEED)
    points = {}
    for link in LINKS:
        path = a.meshes / f"{link}.STL"
        if hashlib.md5(path.read_bytes()).hexdigest() != md5[path.name]:
            raise SystemExit(f"{path}: not the mesh this release was made with (md5 differs from {MD5_LIST.name})")
        p, _ = trimesh.sample.sample_surface(trimesh.load(path), POINTS_PER_LINK, seed=int(rng.integers(1 << 30)))
        points[link] = np.asarray(p, np.float64)
    if max(float(np.abs(p).max()) for p in points.values()) > 1.0:
        raise SystemExit("the meshes are not in metres")
    np.savez(a.out, **points)
    print(f"{a.out}: {len(points)} links x {POINTS_PER_LINK} points")


if __name__ == "__main__":
    main()
