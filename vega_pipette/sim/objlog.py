"""Per-step log of a run, saved as objlog.npz.

Keys: state, action (54-D per row); poses (the columns of pose_cols) and the same split by name; the extra columns;
infer_ms (query latency on the row a chunk was applied, else 0); the query records (query_step = observation step,
query_ms, query_apply_step, query_start_row); hold_steps; the run's start values; pose_cols; with prompts, the
instruction sent at each query (prompt_step, prompt_text).
Pipette task (the defaults): pose columns POSE_COLS (pip_pos, pip_quat, tube_pos, tube_quat, rw_pos, lw_pos, tip_pos,
plunger, rw_quat, lw_quat), extra rthumb_pos, stand_pos, rack_pos, start values p_hang0 / p_tube0 (pipette and tube
after the settle). Pick-and-place: PICK_PLACE_COLS, extra PICK_PLACE_EXTRA (the place hint as the frame draws it),
start values hole_a / hole_b, the prompts.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_COLS = ("pip_pos:3", "pip_quat:4", "tube_pos:3", "tube_quat:4", "rw_pos:3", "lw_pos:3", "tip_pos:3", "plunger:1",
             "rw_quat:4", "lw_quat:4")
"""The 32 pose columns of a pipette-task row: pipette, tube, wrists, pipette tip, plunger (quaternions xyzw)."""
EXTRA = ("rthumb_pos:3", "stand_pos:3", "rack_pos:3")
PICK_PLACE_COLS = ("tube_pos:3", "tube_quat:4", "rack_pos:3", "rack_quat:4", "lw_pos:3", "lw_quat:4", "root_pos:3",
                   "root_quat:4")
"""The 28 pose columns of a pick-and-place row: tube, rack, left wrist, robot root."""
PICK_PLACE_EXTRA = ("hint_move:2", "hint_shift:1")
"""While placing: where the place hint sends the hand from where it is (rack frame, mm) and the size of the hint's
in-hand correction (mm); NaN when there is none."""


def _split(pose: np.ndarray, cols: tuple[str, ...]) -> dict[str, np.ndarray]:
    """Pose columns (..., width) -> {name: (..., n)}; one-column names as (...,)."""
    out, c0 = {}, 0
    for col in cols:
        name, n = col.split(":")
        out[name] = pose[..., c0:c0 + int(n)] if int(n) > 1 else pose[..., c0]
        c0 += int(n)
    return out


class ObjLog:
    """Collects one row per control step and writes objlog.npz."""

    def __init__(self, start: dict[str, np.ndarray], pose_cols: tuple[str, ...] = POSE_COLS,
                 extra: tuple[str, ...] = EXTRA) -> None:
        self.start, self.pose_cols, self.extra = dict(start), tuple(pose_cols), tuple(extra)
        self.width = sum(int(c.split(":")[1]) for c in self.pose_cols)
        self.rows: dict[str, list] = {k: [] for k in ("state", "action", "pose", "infer_ms",
                                                      *(c.split(":")[0] for c in self.extra))}

    def __len__(self) -> int:
        return len(self.rows["pose"])

    def add(self, state: np.ndarray, action: np.ndarray, poses: dict, infer_ms: float = 0.0) -> None:
        """Log one step: the state the action was chosen from, the action, the poses of that step."""
        row = np.concatenate([np.atleast_1d(np.asarray(poses[c.split(":")[0]], np.float64)) for c in self.pose_cols])
        self.rows["state"].append(np.asarray(state, np.float32))
        self.rows["action"].append(np.asarray(action, np.float32).copy())
        self.rows["pose"].append(row.astype(np.float32))
        self.rows["infer_ms"].append(float(infer_ms))
        for c in self.extra:
            k = c.split(":")[0]
            self.rows[k].append(np.asarray(poses[k], np.float32))

    def row(self, i: int) -> dict[str, np.ndarray]:
        """Row i as saved: the state and the named pose columns (float32)."""
        return {"state": self.rows["state"][i], **_split(self.rows["pose"][i], self.pose_cols)}

    def arrays(self, records: list = (), hold_steps: list[int] = (),
               prompts: dict[int, str] | None = None) -> dict[str, np.ndarray]:
        """All logged arrays (see the module docstring), with the scheduler's query records, held steps and the
        prompt of each query by its observation step."""
        P = np.asarray(self.rows["pose"], np.float32).reshape(-1, self.width)
        out: dict[str, np.ndarray] = {"state": np.asarray(self.rows["state"], np.float32).reshape(-1, 54),
                                      "action": np.asarray(self.rows["action"], np.float32).reshape(-1, 54),
                                      "poses": P, **_split(P, self.pose_cols)}
        for c in self.extra:
            k, n = c.split(":")
            out[k] = np.asarray(self.rows[k], np.float32).reshape(-1, int(n)) if int(n) > 1 else \
                np.asarray(self.rows[k], np.float32).reshape(-1)
        out["infer_ms"] = np.asarray(self.rows["infer_ms"], np.float32)
        out["query_step"] = np.asarray([r.obs_step for r in records], np.int32)
        out["query_ms"] = np.asarray([r.latency_ms for r in records], np.float32)
        out["query_apply_step"] = np.asarray([r.apply_step for r in records], np.int32)
        out["query_start_row"] = np.asarray([r.start_row for r in records], np.int16)
        out["hold_steps"] = np.asarray(hold_steps, np.int32)
        if prompts is not None:
            out["prompt_step"] = np.asarray(sorted(prompts), np.int32)
            out["prompt_text"] = np.asarray([prompts[s] for s in sorted(prompts)], str)
        out.update(self.start)
        return out

    def save(self, path: Path, records: list, hold_steps: list[int], prompts: dict[int, str] | None = None) -> None:
        np.savez_compressed(path, pose_cols=np.array(self.pose_cols), **self.arrays(records, hold_steps, prompts))
