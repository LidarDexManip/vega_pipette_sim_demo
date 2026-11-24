"""The JSON the dashboard page draws: the evaluation frame's state at the latest scored row, plus run information
(snapshot: the pipette task; pick_place_snapshot: the tube pick-and-place task)."""
from __future__ import annotations

import math

import numpy as np

from ..progress import geometry as geo
from ..progress import pick_place as pp
from ..progress import rack
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


VIEW_MM = 36.0
"""The pick-and-place top view shows +-36 mm around the target pocket (video/pick_place.py)."""
SHIFT_MAX_MM = 20.0
"""The end of the target-shift gauge (video/pick_place.py)."""


def pick_place_snapshot(S: dict, outcome: bool) -> dict:
    """The page's state for a pick-and-place frame_state S (progress/pick_place.py) at its last row; outcome: the run
    has reached its outcome."""
    i = S["n"] - 1
    out = {"task": "pick-place", "rows": S["n"],
           "holes": {"a": list(rack.column_row(S["hole_a"])), "b": list(rack.column_row(S["hole_b"]))},
           "limits": {"view_mm": VIEW_MM, "seat_lat_mm": round(1e3 * rack.SEAT_LAT, 3),
                      "capture_mm": round(1e3 * (rack.HOLE_R + rack.BEVEL[0] - rack.TUBE_R + rack.TUBE_BEVEL[0]), 3),
                      "tilt_deg": rack.SEAT_TILT_DEG, "lifted_mm": 1e3 * pp.PICK_CLEAR, "floor_mm": -1e3 * rack.TOP,
                      "seat_depth_mm": 1e3 * rack.SEAT_DEPTH, "bore_mm": 1e3 * rack.HOLE_R,
                      "outer_mm": 1e3 * (rack.HOLE_R + rack.BEVEL[0]), "shift_max_mm": SHIFT_MAX_MM,
                      "tube_mm": 1e3 * rack.TUBE_R, "disc_mm": 1e3 * (rack.TUBE_R - rack.TUBE_BEVEL[0])}}
    if i < 0:
        out.update(stages=[{"code": c, "name": n, "t": None, "bad": None} for c, n in pp.MILESTONES], pills=[])
        return out
    st = int(S["stage"][i])
    target = S["hole_a"] if st == pp.STAGES.index("pick") else S["hole_b"]
    c0 = rack.hole_xy(target)
    pockets = [[_f(1e3 * (rack.hole_xy(h) - c0)[0], 1), _f(1e3 * (rack.hole_xy(h) - c0)[1], 1), h == target]
               for h in rack.POCKETS]
    k0 = max(0, i - TRAIL_ROWS)
    while k0 < i and (S["stage"][k0] == pp.STAGES.index("pick")) != (st == pp.STAGES.index("pick")):
        k0 += 1
    out.update({
        "stages": [{"code": m["code"], "name": m["name"], "t": None if m["row"] is None else round(m["row"] / 30.0, 1),
                    "bad": None} for m in pp.milestones(S, i)],
        "pills": [[t_, k] for t_, k in pp.status(S, i, show_stop=outcome)],
        "row_stage": pp.STAGES[st], "target": "A" if st == pp.STAGES.index("pick") else "B",
        "read": S["read"][i], "hint": None if math.isnan(S["hint"][i][0]) else [_f(S["hint"][i][0]), _f(S["hint"][i][1])],
        "shift": None if math.isnan(S["shift"][i]) else _f(S["shift"][i], 1), "pockets": [p for p in pockets
                                                     if abs(p[0]) + out["limits"]["outer_mm"] <= VIEW_MM
                                                     and abs(p[1]) + out["limits"]["outer_mm"] <= VIEW_MM],
        "dx": _f(S["dx"][i]), "dy": _f(S["dy"][i]), "lat": _f(math.hypot(S["dx"][i], S["dy"][i])),
        "height": _f(S["height"][i], 1), "tilt": _f(S["tilt"][i], 1), "seated": bool(S["seat_b"][i]),
        "trail": [[_f(S["dx"][k]), _f(S["dy"][k])] for k in range(k0, i + 1)],
    })
    return out
