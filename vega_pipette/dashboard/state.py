"""The JSON the dashboard page draws: the evaluation frame's state at the latest scored row, plus run information."""
from __future__ import annotations

import numpy as np

from ..progress import geometry as geo
from ..progress.rollout import PRESS_MM, milestones, status

TRAIL_ROWS = 120
"""The tip's trail in the top view: the last 4 s."""


def _f(v: float, nd: int = 2) -> float | None:
    v = float(v)
    return round(v, nd) if np.isfinite(v) else None


def snapshot(S: dict, outcome: bool) -> dict:
    """The page's state for frame_state S at its last row. outcome: the run has reached its outcome (the stop pill is
    shown only then, not the provisional "time limit")."""
    i = S["n"] - 1
    if i < 0:
        return {"rows": 0}
    held = S["s2row"] is not None and i >= S["s2row"] and S["tube_d"][i] < geo.HAND_DIST
    r, h, dp = float(S["r"][i]), float(S["h"][i]), float(S["depth"][i])
    active = held and r <= 45 and -30 <= h <= 120
    k0 = max(S["s2row"] if S["s2row"] is not None else i, i - TRAIL_ROWS)
    runs = [(a, min(b, i)) for a, b in S["press_runs"] if a <= i]
    return {
        "rows": S["n"],
        "stages": [{"code": m["code"], "name": m["name"], "t": None if m["row"] is None else round(m["row"] / 30.0, 1),
                    "bad": m["note"]} for m in milestones(S, i)],
        "pills": [[t, k] for t, k in status(S, i, show_stop=outcome)],
        "s1": S["s1row"] is not None and i >= S["s1row"], "held": bool(held), "active": bool(active),
        "r": _f(r), "h": _f(h), "depth": _f(dp), "pressing": bool(held and dp >= PRESS_MM),
        "in_window": bool(r <= geo.R_MAX and geo.H_LO <= h <= geo.H_HI),
        "tip": [_f(S["tx"][i]), _f(S["ty"][i])],
        "trail": [[_f(S["tx"][k]), _f(S["ty"][k])] for k in range(k0, i + 1)] if active else [],
        "press_paths": [[[_f(S["tx"][k]), _f(S["ty"][k])] for k in range(a, b + 1)] for a, b in runs],
        "press_h": [[_f(np.nanmin(S["h"][a:b + 1]), 1), _f(np.nanmax(S["h"][a:b + 1]), 1)] for a, b in runs][-4:],
        "cams": {name: ([_f(d[i][0], 3), _f(d[i][1], 3)] if np.isfinite(d[i][0]) else None) for name, d in S["cams"]},
    }
