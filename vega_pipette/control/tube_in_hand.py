"""Where the tube's bottom is in the left hand, from the hand's 20 joint angles and its own geometry (no camera, no
learned model).

The joints give every finger link's pose in the wrist frame (control/kinematics.py), and 150 points on each link's
surface (assets/robot/wuji_left_hand_points.npz, made by tools/make_hand_points.py) give the hand's shape. The tube's
axis is the line that a cylinder of the tube's radius can lie on while touching the links without entering them: per
link, the nearest surface point's distance to the cylinder; a link outside counts with its squared distance up to a cap
(OUTSIDE_CAP, so open fingers do not pull the line), a link entered counts PENETRATION times its squared depth. The
line is searched (Nelder-Mead) around a fixed start, the typical grasp's axis and bottom in the wrist frame (AXIS0,
BOTTOM0): from nine starts for a new grasp, then from the previous solution of the same grasp. The bottom is the point
at the along-axis coordinate BOTTOM_S on that line (sliding along the axis cannot be seen in the finger angles).
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import minimize

from .. import paths
from . import kinematics

POINTS = paths.ASSETS / "robot" / "wuji_left_hand_points.npz"
LINKS = tuple(f"l_{finger}_{segment}" for finger in ("thumb", "index_finger", "middle_finger", "ring_finger", "pinky")
              for segment in ("proximal", "middle", "distal")) + ("l_wrist",)
TUBE_R = 0.0075
OUTSIDE_CAP = 0.003
PENETRATION = 25.0
TOUCH = 0.0015
"""A link this close to the fitted cylinder's surface counts as touching it."""
AXIS0 = np.array([0.9738197116905546, 0.03228526496527192, -0.2250174010805552])
BOTTOM0 = np.array([-0.11640162081819097, -0.021271904360390385, -0.02303101779318157])
BOTTOM_S = -0.0007212460877302485
"""The tube bottom's coordinate along the fitted axis from the line's anchor point (m), the same for every grasp."""
STARTS = (np.zeros(4),) + tuple(np.array(v) for v in ((0.004, 0, 0, 0), (-0.004, 0, 0, 0), (0, 0.004, 0, 0),
                                                       (0, -0.004, 0, 0), (0, 0, 0.1, 0), (0, 0, -0.1, 0),
                                                       (0, 0, 0, 0.1), (0, 0, 0, -0.1)))
"""Cold starts of the search: (two offsets of the anchor point across the axis, two tilts of the axis)."""


class TubeInHand:
    """The tube bottom in the left wrist's frame for one grasp at a time (new_grasp() starts the next one)."""

    def __init__(self) -> None:
        with np.load(POINTS) as z:
            pts = [z[link] for link in LINKS]
        self.points = pts
        self.seg = np.concatenate([[0], np.cumsum([len(p) for p in pts])[:-1]])
        self.e1 = np.cross(AXIS0, [0.0, 0.0, 1.0])
        self.e1 /= np.linalg.norm(self.e1)
        self.e2 = np.cross(AXIS0, self.e1)
        self.last: np.ndarray | None = None

    def new_grasp(self) -> None:
        """Forget the previous solution: the next estimate searches from all starts."""
        self.last = None

    def cloud(self, q20: ArrayLike) -> np.ndarray:
        """All links' surface points in the wrist frame, one array (links in LINKS order)."""
        return np.concatenate([p @ R.T + t for p, (R, t) in zip(self.points, kinematics.hand_links(q20, LINKS))])

    def axis(self, q20: ArrayLike) -> tuple[np.ndarray, np.ndarray, int]:
        """The fitted axis (a point on it, its direction) and the number of links touching the cylinder."""
        P, seg = self.cloud(q20), self.seg
        e1, e2 = self.e1, self.e2

        def dists(x: np.ndarray) -> np.ndarray:
            a = AXIS0 + x[2] * e1 + x[3] * e2
            a /= np.linalg.norm(a)
            v = P - (BOTTOM0 + x[0] * e1 + x[1] * e2)
            w = v - np.outer(v @ a, a)
            return np.sqrt(np.minimum.reduceat(np.einsum("ij,ij->i", w, w), seg)) - TUBE_R

        def cost(x: np.ndarray) -> float:
            d = dists(x)
            return float(np.sum(np.where(d < 0, PENETRATION * d * d, np.minimum(d * d, OUTSIDE_CAP * OUTSIDE_CAP))))

        best = None
        for x0 in ([self.last] if self.last is not None else STARTS):
            s = minimize(cost, x0, method="Nelder-Mead", options=dict(xatol=1e-5, fatol=1e-10, maxiter=600))
            if best is None or s.fun < best.fun:
                best = s
        x = best.x
        a = AXIS0 + x[2] * e1 + x[3] * e2
        a /= np.linalg.norm(a)
        self.last = x
        return BOTTOM0 + x[0] * e1 + x[1] * e2, a, int((np.abs(dists(x)) < TOUCH).sum())

    def bottom(self, q20: ArrayLike) -> tuple[np.ndarray, int]:
        """The tube bottom (m) in the wrist frame and the number of links touching the tube."""
        p, a, touching = self.axis(q20)
        return p + BOTTOM_S * a, touching
