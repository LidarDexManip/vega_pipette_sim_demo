"""Geometry of the tube rack and the bevelled tube of the pick-and-place task, in the rack's frame (SI units, xyzw).

The rack is a plate of 26 pockets on a staggered grid: rows i = -2..1 at HOLE_PITCH[0] along the rack's x, columns j at
HOLE_PITCH[1] along its y, odd rows shifted by half a column (even rows j = 0..5, odd rows j = -1..5). Each pocket is a
16-sided bore of flat radius HOLE_R through two slabs (SLABS), with a lead-in bevel BEVEL at the top of both; the rack's
origin is the centre of its base, its top face TOP above it. The tube (radius TUBE_R, length 2 * TUBE_HALF, root at its
centre) has a bevelled bottom edge (TUBE_BEVEL).

The seat rule: a tube is seated in a pocket when its bottom is at most SEAT_DEPTH above the rack base (it stands on
the table through the pocket), its axis is within SEAT_TILT_DEG of the rack's vertical and within SEAT_LAT of the
pocket's axis at both bore heights SEAT_Z.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike

from .geometry import quat_matrix

HOLE_BASE = (0.0098, -0.058)       # pocket (0, 0)
HOLE_PITCH = (0.0205, 0.0230)
PLATE_HALF = (0.0453, 0.0843)
SLABS = ((0.0, 0.015), (0.0335, 0.050))
TOP = SLABS[-1][1]
HOLE_SIDES = 16
HOLE_R = 0.0088
BEVEL = (0.0025, 0.0040)           # width at the top face, depth
TUBE_R = 0.0075
TUBE_HALF = 0.080
TUBE_BEVEL = (0.003, 0.005)        # inset of the bottom disc, height up the wall
TUBE_SIDES = 20

POCKETS = tuple((i, j) for i in (-2, -1, 0, 1) for j in (range(0, 6) if i % 2 == 0 else range(-1, 6)))
"""All 26 pockets (i, j)."""
TUBE_HOLES = tuple((i, j) for i, j in POCKETS if j <= (5 if i % 2 == 0 else 4))
"""The 24 pockets the tube is taken from and put into (the rack's last odd-row pocket is never used)."""

SEAT_DEPTH = 0.0025
SEAT_TILT_DEG = 6.0
SEAT_LAT = 0.001815
"""The bore's free play at a polygon corner (8.972 - 7.408 mm) plus 0.25 mm of contact slop."""
SEAT_Z = (0.5 * (TUBE_BEVEL[1] + SLABS[0][1] - BEVEL[1]), 0.5 * (SLABS[1][0] + SLABS[1][1] - BEVEL[1]))
"""Heights of the two straight bore sections' middles (above the tube's own bevel in the lower slab), where the tube
axis is measured: 8.0 and 39.75 mm."""


def hole_xy(hole: tuple[int, int]) -> np.ndarray:
    """A pocket's centre (x, y) in the rack frame."""
    i, j = hole
    return np.array([HOLE_BASE[0] + i * HOLE_PITCH[0], HOLE_BASE[1] + (j + (abs(i) % 2) * 0.5) * HOLE_PITCH[1]])


def column_row(hole: tuple[int, int]) -> tuple[int, int]:
    """The (column, row) by which an instruction names a pocket."""
    i, j = hole
    return i + 3, (j + 1 if i % 2 == 0 else j + 2)


def nearest_pocket(xy: ArrayLike) -> tuple[tuple[int, int], float]:
    """The pocket nearest to a rack-frame point (x, y) and its distance."""
    d = [(math.hypot(float(xy[0]) - c[0], float(xy[1]) - c[1]), h) for h in POCKETS for c in (hole_xy(h),)]
    dist, hole = min(d)
    return hole, dist


def in_rack(tube_pos: ArrayLike, tube_quat: ArrayLike, rack_pos: ArrayLike,
            rack_quat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """The tube's bottom point and its axis (unit) in the rack frame."""
    R = quat_matrix(rack_quat)
    axis = R.T @ quat_matrix(tube_quat)[:, 2]
    root = R.T @ (np.asarray(tube_pos, np.float64) - np.asarray(rack_pos, np.float64))
    return root - TUBE_HALF * axis, axis


def seated(bottom: np.ndarray, axis: np.ndarray, hole: tuple[int, int]) -> bool:
    """The seat rule (module docstring) for a pocket, from in_rack()'s bottom and axis."""
    if bottom[2] > SEAT_DEPTH or math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2]))))) > SEAT_TILT_DEG:
        return False
    c = hole_xy(hole)
    for z in SEAT_Z:
        q = bottom + (z - bottom[2]) / max(float(axis[2]), 1e-6) * axis
        if math.hypot(float(q[0]) - c[0], float(q[1]) - c[1]) > SEAT_LAT:
            return False
    return True
