"""Per-step log of a run, saved as objlog.npz.

Keys: state, action (54-D per row); poses (the 32 columns of POSE_COLS) and the same split by name (pip_pos, pip_quat,
tube_pos, tube_quat, rw_pos, lw_pos, tip_pos, plunger, rw_quat, lw_quat); rthumb_pos, stand_pos, rack_pos; infer_ms
(query latency on the row a chunk was applied, else 0); the query records (query_step = observation step, query_ms,
query_apply_step, query_start_row); hold_steps; p_hang0 / p_tube0 (pipette and tube after the settle); pose_cols.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

POSE_COLS = ("pip_pos:3", "pip_quat:4", "tube_pos:3", "tube_quat:4", "rw_pos:3", "lw_pos:3", "tip_pos:3", "plunger:1",
             "rw_quat:4", "lw_quat:4")
"""The 32 pose columns of a logged row: pipette, tube, wrists, pipette tip, plunger (quaternions xyzw)."""
_EXTRA = ("rthumb_pos", "stand_pos", "rack_pos")


def _split(pose: np.ndarray) -> dict[str, np.ndarray]:
    """Pose columns (..., 32) -> {name: (..., width)}; the plunger as (...,)."""
    out, c0 = {}, 0
    for col in POSE_COLS:
        name, width = col.split(":")
        out[name] = pose[..., c0:c0 + int(width)]
        c0 += int(width)
    out["plunger"] = out["plunger"][..., 0]
    return out


class ObjLog:
    """Collects one row per control step and writes objlog.npz."""

    def __init__(self, p_hang0: np.ndarray, p_tube0: np.ndarray) -> None:
        self.p_hang0, self.p_tube0 = p_hang0, p_tube0
        self.rows: dict[str, list] = {k: [] for k in ("state", "action", "pose", "infer_ms", *_EXTRA)}

    def __len__(self) -> int:
        return len(self.rows["pose"])

    def add(self, state: np.ndarray, action: np.ndarray, poses: dict, infer_ms: float = 0.0) -> None:
        """Log one step: the state the action was chosen from, the action, the poses of that step."""
        row = np.concatenate([np.atleast_1d(np.asarray(poses[c.split(":")[0]], np.float64)) for c in POSE_COLS])
        self.rows["state"].append(np.asarray(state, np.float32))
        self.rows["action"].append(np.asarray(action, np.float32).copy())
        self.rows["pose"].append(row.astype(np.float32))
        self.rows["infer_ms"].append(float(infer_ms))
        for k in _EXTRA:
            self.rows[k].append(np.asarray(poses[k], np.float32))

    def row(self, i: int) -> dict[str, np.ndarray]:
        """Row i as saved: the state and the named pose columns (float32)."""
        return {"state": self.rows["state"][i], **_split(self.rows["pose"][i])}

    def arrays(self, records: list = (), hold_steps: list[int] = ()) -> dict[str, np.ndarray]:
        """All logged arrays (see the module docstring), with the scheduler's query records and held steps."""
        P = np.asarray(self.rows["pose"], np.float32).reshape(-1, 32)
        out: dict[str, np.ndarray] = {"state": np.asarray(self.rows["state"], np.float32).reshape(-1, 54),
                                      "action": np.asarray(self.rows["action"], np.float32).reshape(-1, 54),
                                      "poses": P, **_split(P)}
        for k in _EXTRA:
            out[k] = np.asarray(self.rows[k], np.float32).reshape(-1, 3)
        out["infer_ms"] = np.asarray(self.rows["infer_ms"], np.float32)
        out["query_step"] = np.asarray([r.obs_step for r in records], np.int32)
        out["query_ms"] = np.asarray([r.latency_ms for r in records], np.float32)
        out["query_apply_step"] = np.asarray([r.apply_step for r in records], np.int32)
        out["query_start_row"] = np.asarray([r.start_row for r in records], np.int16)
        out["hold_steps"] = np.asarray(hold_steps, np.int32)
        out.update(p_hang0=self.p_hang0, p_tube0=self.p_tube0)
        return out

    def save(self, path: Path, records: list, hold_steps: list[int]) -> None:
        np.savez_compressed(path, pose_cols=np.array(POSE_COLS), **self.arrays(records, hold_steps))
