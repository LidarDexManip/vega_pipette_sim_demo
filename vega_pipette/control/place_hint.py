"""The numbers in the tube pick-and-place policy's place instruction.

While placing, the instruction ends in "Pocket: bx by", the hand position (L_ee, robot root frame) that puts the tube
into hole B, in bins of about 0.7 mm (pocket_text). The policy server appends "Rel: dx dy", that position minus the
current hand position in 0.5 mm units (rel_text; the server's copy is policy_server/openpi_overlay/.../vega_policy.py).
The target is computed from the robot's own joints and the two hole numbers:
  grasp memory   from the first row the left hand's 20 joints sum to at least HAND_CLOSED rad, the hand position is
                 remembered on every row, until the hand has risen more than LIFTED above its height on that first row;
                 the memory then keeps the last row before that;
  target         the remembered hand position + the offset from hole A to hole B, from their numbers (hole_offset);
  correction     from the second place query on, the target moves against the tube's slip in the hand since the grasp:
                 the tube bottom relative to the hand, estimated from the finger joints (control/tube_in_hand.py),
                 now minus at the grasp. The grasp's estimate is made at that second query from the joints and link
                 poses saved with the memory.
If no grasp was remembered when placing starts, the target is hole A's position from the simulation (in 0.5 mm steps)
+ the typical hand offset at a grasp (GRASP_OFFSET) + the A -> B offset, without correction, and no_grasp is set.
"""
from __future__ import annotations

import re

import numpy as np
from numpy.typing import ArrayLike

from ..progress import rack
from ..progress.geometry import quat_matrix
from . import kinematics
from .tube_in_hand import TubeInHand

POCKET_ORIGIN = (0.20, 0.21)
POCKET_BIN = ((0.34 - 0.17) / 256, (0.55 - 0.37) / 256)
POCKET_BINS = (512, 456)
"""The Pocket grid: robot-frame x, y of bin 0's corner (m), bin sizes (m) and bin counts."""
REL_UNIT, REL_CLIP = 0.0005, 80
HAND_CLOSED = 17.96
LIFTED = 0.002
GRASP_OFFSET = np.array([-0.050590347377416384, 0.017031265518046013])
"""The hand (L_ee) minus hole A at the grasp in the training demonstrations, robot frame (m)."""
HOLE_ORIGIN, HOLE_STEP = (-0.60, 0.20), 0.0005
_POCKET = re.compile(r"Pocket: (\d+) (\d+)")


def pocket_text(xy: ArrayLike) -> str:
    """'Pocket: bx by' for a hand position (robot frame, m)."""
    b = [int(np.clip(np.floor((float(v) - o) / s), 0, n - 1)) for v, o, s, n in zip(xy, POCKET_ORIGIN, POCKET_BIN,
                                                                                    POCKET_BINS)]
    return f"Pocket: {b[0]} {b[1]}"


def pocket_xy(sentence: str) -> np.ndarray | None:
    """The hand position a sentence's Pocket stands for (its bins' centres), or None."""
    m = _POCKET.search(sentence)
    return None if m is None else np.array([o + (int(b) + 0.5) * s for b, o, s in zip(m.groups(), POCKET_ORIGIN,
                                                                                     POCKET_BIN)])


def rel_text(xy: ArrayLike, state: ArrayLike) -> str:
    """'Rel: dx dy': a hand position minus the hand position of a 54-D state, in REL_UNIT, clipped to +-REL_CLIP."""
    ee = kinematics.left_ee(np.asarray(state, np.float64)[:7])
    d = [int(np.clip(np.round((float(m) - float(e)) / REL_UNIT), -REL_CLIP, REL_CLIP)) for m, e in zip(xy, ee[:2])]
    return f"Rel: {d[0]} {d[1]}"


def as_read(sentence: str, state: ArrayLike) -> str:
    """The sentence as the policy reads it: with Rel appended when it carries a Pocket."""
    xy = pocket_xy(sentence)
    return sentence if xy is None else f"{sentence}. {rel_text(xy, state)}"


def hole_offset(hole_a: tuple[int, int], hole_b: tuple[int, int]) -> np.ndarray:
    """Hole B minus hole A in the robot frame (m), from the hole numbers (the rack stands square to the robot: robot x =
    rack y, robot y = -rack x)."""
    a, b = rack.hole_xy(hole_a), rack.hole_xy(hole_b)
    return np.array([b[1] - a[1], -(b[0] - a[0])])


def hand_move(xy: ArrayLike, state: ArrayLike, poses: dict) -> np.ndarray:
    """Where a hand position lies from the hand of a 54-D state, in the rack frame (mm; x right, y away from the
    robot)."""
    v = np.asarray(xy, np.float64) - kinematics.left_ee(np.asarray(state, np.float64)[:7])[:2]
    world = quat_matrix(poses["root_quat"]) @ np.array([v[0], v[1], 0.0])
    return 1e3 * (quat_matrix(poses["rack_quat"]).T @ world)[:2]


def to_root(p: np.ndarray, poses: dict) -> np.ndarray:
    """A world point in the robot's root frame (poses: root_pos, root_quat)."""
    return quat_matrix(poses["root_quat"]).T @ (p - poses["root_pos"])


class PlaceHint:
    """The place instruction's numbers for one run. Call observe() on every row before the stage update, sentence()
    at every place query."""

    def __init__(self, hole_a: tuple[int, int], hole_b: tuple[int, int]) -> None:
        self.hole_a, self.hole_b = tuple(hole_a), tuple(hole_b)
        self.row = -1
        self.grasp: np.ndarray | None = None          # remembered hand position (x, y)
        self.grasp_row: int | None = None
        self.no_grasp = False
        self.target: np.ndarray | None = None         # the uncorrected target (x, y)
        self.shift: np.ndarray | None = None          # the latest correction (x, y)
        self.corrections = 0
        self.first: str | None = None                 # the first place sentence
        self._z0: float | None = None
        self._lifted = False
        self._snap: dict | None = None                # joints and poses at the remembered row
        self._inhand0: np.ndarray | None = None       # tube bottom minus hand at the grasp (x, y)
        self._est = TubeInHand()

    def observe(self, stage: str, state: ArrayLike, poses: dict) -> None:
        """One row (state = the 54 joint positions, poses = the session's poses) while `stage` is the stage before
        this row's update."""
        self.row += 1
        if stage != "pick":
            return
        q = np.asarray(np.asarray(state, np.float32), np.float64)
        ee = kinematics.left_ee(q[:7])
        if self._z0 is None and float(q[7:27].sum()) >= HAND_CLOSED:
            self._z0 = float(ee[2])
        if self._z0 is not None and float(ee[2]) - self._z0 > LIFTED:
            self._lifted = True
        if self._z0 is not None and not self._lifted:
            self.grasp, self.grasp_row = ee[:2].copy(), self.row
            self._snap = {"q": q.copy(), **{k: np.asarray(poses[k], np.float64).copy()
                                            for k in ("lw_pos", "lw_quat", "root_pos", "root_quat")}}

    def sentence(self, base: str, state: ArrayLike, poses: dict) -> str:
        """The place sentence for a query: `base` with the Pocket of the (corrected) target."""
        if self.target is None:
            if self.grasp is None:
                self.no_grasp = True
                hole = poses["rack_pos"] + quat_matrix(poses["rack_quat"]) @ np.array([*rack.hole_xy(self.hole_a),
                                                                                       rack.TOP])
                h = to_root(hole, poses)
                start = np.array([int(round((h[0] - HOLE_ORIGIN[0]) / HOLE_STEP)) * HOLE_STEP + HOLE_ORIGIN[0],
                                  int(round((h[1] - HOLE_ORIGIN[1]) / HOLE_STEP)) * HOLE_STEP + HOLE_ORIGIN[1]])
                start = start + GRASP_OFFSET
            else:
                start = self.grasp
            self.target = start + hole_offset(self.hole_a, self.hole_b)
            self.first = f"{base.rstrip('.')}. {pocket_text(self.target)}"
            return self.first
        if self.grasp is None:
            return self.first
        q = np.asarray(np.asarray(state, np.float32), np.float64)
        if self._inhand0 is None:
            s = self._snap
            self._est.new_grasp()
            b, _ = self._est.bottom(s["q"][7:27])
            bottom = s["lw_pos"] + quat_matrix(s["lw_quat"]) @ b
            self._inhand0 = to_root(bottom, s)[:2] - self.grasp
        b, _ = self._est.bottom(q[7:27])
        bottom = np.asarray(poses["lw_pos"], np.float64) + quat_matrix(np.asarray(poses["lw_quat"], np.float64)) @ b
        self.shift = (to_root(bottom, poses)[:2] - kinematics.left_ee(q[:7])[:2]) - self._inhand0
        self.corrections += 1
        return f"{base.rstrip('.')}. {pocket_text(self.target - self.shift)}"

    def record(self) -> dict:
        """What run.json keeps of the hint."""
        f = (lambda v: None if v is None else [float(x) for x in v])
        return {"grasp_row": self.grasp_row, "grasp_xy": f(self.grasp), "target_xy": f(self.target),
                "no_grasp": self.no_grasp, "corrections": self.corrections,
                "last_shift_mm": None if self.shift is None else [1e3 * float(x) for x in self.shift]}
