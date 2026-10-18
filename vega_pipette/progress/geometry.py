"""Tube / tip geometry shared by the milestones and the frame: quaternions, the tube mouth, and the press window."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

TUBE_HALF = 0.080                 # half the tube's height [m]: the mouth is this far along the tube axis from its root
TUBE_R_MM, WALL_MM = 7.5, 1.0     # the tube's outer radius (= the press window's radius) and its wall
HAND_DIST = 0.150                 # an object counts as held within this distance of the wrist [m]
R_MAX = 7.5                       # press window: the tip within this distance of the tube axis [mm]
H_LO, H_HI = 10.0, 40.0           # press window: the tip this high above the mouth [mm]
FIRST_STOP, RELEASED = -0.0060, -0.0010   # plunger positions [m]: the first stop, and "released"


def quat_z_axis(q: ArrayLike) -> np.ndarray:
    """Body +z of xyzw quaternions, (n, 4) or (4,), in the world frame."""
    q = np.asarray(q, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y)], axis=-1)


def quat_matrix(q: ArrayLike) -> np.ndarray:
    """Rotation matrices of xyzw quaternions, (..., 4) -> (..., 3, 3)."""
    x, y, z, w = np.moveaxis(np.asarray(q, np.float64), -1, 0)
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], -2)


def tip_r_h(tip: ArrayLike, tube_pos: ArrayLike, tube_quat: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Per row: the tip's distance r from the tube axis and its height h above the mouth along the axis [mm]."""
    a = quat_z_axis(tube_quat)
    v = np.asarray(tip, np.float64) - (np.asarray(tube_pos, np.float64) + TUBE_HALF * a)
    along = np.einsum("ij,ij->i", v, a)
    return 1e3 * np.linalg.norm(v - along[:, None] * a, axis=1), 1e3 * along
