"""Task progress of one run from its per-step rows: the milestone scorer with the evaluation's stop rules, and the
state the evaluation frame and the live dashboard draw.

The evaluation stops a run on:
  success   S5 (release) after an on-target first-stop press
  failure   a dropped object (the scorer's fail flag), a first-stop press off target, no new milestone for 20 s,
            or no joint motion for 10 s
RolloutScore takes the rows one at a time (the live runner); score_rows runs it over a saved log. frame_state turns
either result into what is drawn: milestones, the tip's offset from the tube axis and height above the mouth, the
plunger depth and presses, the wrist cameras' directions, crooked grasps, failures and the stop. run_state(objlog)
does all of it for a saved run.
"""
from __future__ import annotations

import numpy as np

from . import geometry as geo
from . import scoring

STALL_ROWS, FROZEN_ROWS, FROZEN_MOVE = 600, 300, 0.05   # the evaluation's stall / frozen stop rules (rows, rad)
ROW_KEYS = ("pip_pos", "pip_quat", "tube_pos", "tube_quat", "lw_pos", "rw_pos", "tip_pos", "plunger")

# a milestone that fired with the pipette held crooked is not a grasp: pipette tilt in the right hand, from the scorer's
# hand-off check, and the pipette axis in the right wrist 30 rows after the regrasp against the demonstrations' axis
GRASP_TILT_MAX = 30.0
REGRASP_AXIS = np.array([0.9666, 0.0895, -0.2402])
REGRASP_DEV_MAX = 20.0

# the wrist cameras, in their wrist frames, for the top view's camera badges
LW_EYE, LW_TGT = np.array([0.0, 0.13, 0.02]), np.array([0.0, -0.30, -0.06])
RW_EYE, RW_TGT = np.array([0.0, -0.13, 0.02]), np.array([-0.1248, 0.0846, -0.0096])


STAGES = (("S1_unrack", "S1", "unrack"), ("R1_regrasp", "R1", "regrasp"), ("S2_tube", "S2", "tube"),
          ("S3_align", "S3", "in tube"), ("S4_press", "S4", "press in"), ("S5_release", "S5", "release"))
"""The milestones shown: (scorer key, code, name)."""
PRESS_MM = 1.0              # plunger depth that counts as pressing (a thumb resting on it reads 0.4-0.6 mm)
STOP_TEXT = {"success_S5": "success", "stalled": "stalled 20 s", "frozen": "frozen",
             "press_off_target": "press off target", "max_steps": "time limit"}


class RolloutScore:
    """The milestone scorer over a run's rows, one row at a time, with the evaluation's stop rules. Once the run would
    stop, further rows are ignored (n stays at the stop row + 1)."""

    def __init__(self, p_hang0: np.ndarray, p_tube0: np.ndarray) -> None:
        self._sc = scoring.Scorer(p_hang0, p_tube0)
        self.press: dict | None = None
        self.stop = "max_steps"
        self.n = 0
        self._stall = {"n_ms": -1, "since": 0, "q": None, "q_since": 0}

    @property
    def stopped(self) -> bool:
        return self.stop != "max_steps"

    def add(self, row: dict) -> None:
        """Score one row: ROW_KEYS plus "state", as logged (sim/objlog.py)."""
        if self.stopped:
            return
        i, st = self.n, self._stall
        r = {k: (float(row[k]) if k == "plunger" else np.asarray(row[k], np.float64)) for k in ROW_KEYS}
        state = np.asarray(row["state"], np.float32)
        fs = self._sc.update(i, r).first_step
        if self.press is None and fs["S1_unrack"] is not None:
            self.press = press_record(i, r, fs["S2_tube"])
        if self.press is not None and not self.press["on_target"]:
            self.stop = "press_off_target"
        elif self._sc.failed:
            self.stop = "fail"
        elif fs["S5_release"] is not None:
            self.stop = "success_S5"
        else:
            n_ms = sum(v is not None for v in fs.values())
            if n_ms > st["n_ms"]:
                st["n_ms"], st["since"] = n_ms, i
            if st["q"] is None or float(np.max(np.abs(state - st["q"]))) > FROZEN_MOVE:
                st["q"], st["q_since"] = state.copy(), i
            if i - st["since"] >= STALL_ROWS:
                self.stop = "stalled"
            elif i - st["q_since"] >= FROZEN_ROWS:
                self.stop = "frozen"
        self.n = i + 1

    def result(self) -> dict:
        """The score summary, the first-stop press, the stop reason, the rows scored and the success flag."""
        score = self._sc.summary()
        fired = all(score["first_step"].get(k) is not None for k in scoring.DEMO_STEPS)
        return {"score": score, "press": self.press, "stop_reason": self.stop, "n": self.n,
                "success": bool(fired and self.press is not None and self.press["on_target"])}


def score_rows(z: dict) -> dict:
    """RolloutScore over all rows of a log z (objlog arrays)."""
    rs = RolloutScore(z["p_hang0"], z["p_tube0"])
    for i in range(len(z["plunger"])):
        if rs.stopped:
            break
        rs.add({k: z[k][i] for k in (*ROW_KEYS, "state")})
    return rs.result()


def press_record(row: int, r: dict, s2_row: int | None) -> dict | None:
    """The first-stop press at this row (None unless the plunger is at the first stop): r, h and whether it is on target
    (tip in the press window and the tube held)."""
    if r["plunger"] > geo.FIRST_STOP:
        return None
    rr, hh = geo.tip_r_h(r["tip_pos"][None], r["tube_pos"][None], r["tube_quat"][None])
    d_tube = float(np.linalg.norm(r["tube_pos"] - r["lw_pos"]))
    held = s2_row is not None and s2_row <= row and d_tube < geo.HAND_DIST
    in_window = rr[0] <= geo.R_MAX and geo.H_LO <= hh[0] <= geo.H_HI
    return {"row": int(row), "r_mm": round(float(rr[0]), 3), "h_mm": round(float(hh[0]), 3), "tube_held": bool(held),
            "on_target": bool(in_window and held)}


def crooked_grasps(score: dict, z: dict) -> dict:
    """{"S1_unrack" | "R1_regrasp": (row, deg[, "regrasp"])} for milestones reached with a crooked pipette."""
    out = {}
    for key in ("S1_unrack", "R1_regrasp"):
        c = (score.get("contract_detail") or {}).get(key) or {}
        if c.get("tilt_deg") is not None and c["tilt_deg"] > GRASP_TILT_MAX:
            out[key] = (c["milestone_step"], round(c["tilt_deg"]))
    r1 = score["first_step"].get("R1_regrasp")
    if "R1_regrasp" not in out and r1 is not None:
        i = min(int(r1) + 30, len(z["plunger"]) - 1)
        ax = geo.quat_matrix(z["rw_quat"][i]).T @ geo.quat_matrix(z["pip_quat"][i])[:, 2]
        deg = float(np.degrees(np.arccos(np.clip(ax @ REGRASP_AXIS / np.linalg.norm(REGRASP_AXIS), -1.0, 1.0))))
        if deg > REGRASP_DEV_MAX:
            out["R1_regrasp"] = (i, round(deg), "regrasp")
    return out


def tip_stages(z: dict, s1: int | None, s2: int | None) -> dict:
    """S3 = the first row at or after S2 with the tip in the press window over the held tube; S4 = the first first-stop
    row after S1 if every row of that stroke (from the last row the plunger was released) is in the window."""
    r, h = geo.tip_r_h(z["tip_pos"], z["tube_pos"], z["tube_quat"])
    held = np.linalg.norm(z["tube_pos"] - z["lw_pos"], axis=1) < geo.HAND_DIST
    ok = held & (r <= geo.R_MAX) & (h >= geo.H_LO) & (h <= geo.H_HI)
    rows = np.arange(len(ok))
    out = {"S3": None, "S4": None, "stroke": None}
    if s2 is not None:
        hit = np.flatnonzero(ok & (rows >= s2))
        out["S3"] = int(hit[0]) if len(hit) else None
    if s1 is not None:
        pl = np.asarray(z["plunger"])
        fs = np.flatnonzero((pl <= geo.FIRST_STOP) & (rows >= s1))
        if len(fs):
            j = int(fs[0])
            up = np.flatnonzero(pl[:j] > geo.RELEASED)
            a = int(up[-1]) + 1 if len(up) else 0
            bad = np.flatnonzero(~ok[a:j + 1])
            out["stroke"] = {"from": a, "to": j, "left_at": int(a + bad[0]) if len(bad) else None}
            if not len(bad) and out["S3"] is not None:
                out["S4"] = j
    return out


def press_runs(depth: np.ndarray, press: np.ndarray, s1: int | None) -> list[tuple[int, int]]:
    """Row runs with the plunger down: gaps up to 10 rows joined, runs of at least 5 rows and 2.5 mm kept, each
    extended back to where the plunger left its rest reading."""
    edges = np.flatnonzero(np.diff(np.r_[0, press.astype(np.int8), 0]))
    runs: list[tuple[int, int]] = []
    for a, b in zip(edges[::2], edges[1::2] - 1):
        if runs and a - runs[-1][1] <= 10:
            runs[-1] = (runs[-1][0], b)
        else:
            runs.append((a, b))
    runs = [(int(a), int(b)) for a, b in runs if b - a + 1 >= 5 and depth[a:b + 1].max() >= 2.5]
    rest = float(np.median(depth[s1:s1 + 60])) if s1 is not None else 0.6
    for j, (a, b) in enumerate(runs):
        floor = runs[j - 1][1] + 1 if j else 0
        while a - 1 >= floor and depth[a - 1] > rest + 0.12:
            a -= 1
        runs[j] = (a, b)
    return runs


def camera_sight(z: dict, eye: np.ndarray, tgt: np.ndarray, quat_key: str, ex: np.ndarray, ey: np.ndarray) -> np.ndarray:
    """Per row: a wrist camera's line of sight in the top view's screen axes (unit), NaN when it looks along the tube."""
    v = (tgt - eye) / np.linalg.norm(tgt - eye)
    dw = np.einsum("nij,j->ni", geo.quat_matrix(z[quat_key]), v)
    sx, sy = np.einsum("ij,ij->i", dw, ex), -np.einsum("ij,ij->i", dw, ey)
    nn = np.hypot(sx, sy)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack([sx / nn, sy / nn], 1)
    out[nn < 0.3] = np.nan
    return out


def frame_state(z: dict, res: dict) -> dict:
    """Everything the frame draws, per row up to the scored rows (n = res["n"]), from the log z and a score result."""
    n = res["n"]
    z = {k: (v[:n] if np.ndim(v) and len(v) == len(z["plunger"]) else v) for k, v in z.items()}
    score, press = res["score"], res["press"]
    fs = dict(score["first_step"])
    crooked = crooked_grasps(score, z)
    r, h = geo.tip_r_h(z["tip_pos"], z["tube_pos"], z["tube_quat"])
    ax = geo.quat_z_axis(z["tube_quat"])
    off = np.asarray(z["tip_pos"], np.float64) - (np.asarray(z["tube_pos"], np.float64) + geo.TUBE_HALF * ax)
    off -= np.einsum("ij,ij->i", off, ax)[:, None] * ax
    ex = np.array([1.0, 0.0, 0.0]) - ax[:, 0:1] * ax
    ex /= np.linalg.norm(ex, axis=1, keepdims=True)
    ey = np.cross(ax, ex)
    tx, ty = 1e3 * np.einsum("ij,ij->i", off, ex), 1e3 * np.einsum("ij,ij->i", off, ey)
    s1, s2 = fs.get("S1_unrack"), fs.get("S2_tube")
    tube_d = np.linalg.norm(z["tube_pos"] - z["lw_pos"], axis=1)
    pip_d = np.linalg.norm(z["pip_pos"] - z["rw_pos"], axis=1)
    cams = (("L", camera_sight(z, LW_EYE, LW_TGT, "lw_quat", ex, ey)),
            ("R", camera_sight(z, RW_EYE, RW_TGT, "rw_quat", ex, ey)))
    ts = tip_stages(z, s1, s2)
    success = res["success"] and ts["S4"] is not None
    fs.update(S3_align=ts["S3"] if s2 is not None else None, S4_press=ts["S4"],
              S5_release=fs.get("S5_release") if success else None)

    fails = []                                   # (row, text), shown from the row they happened on
    if score.get("pipette_dropped") is not None:
        fails.append((int(score["pipette_dropped"]), "pipette dropped"))
    if score.get("tube_dropped") is not None:
        fails.append((int(score["tube_dropped"]), "tube dropped"))
    if press is not None and not press["on_target"]:
        fails.append((press["row"], "press off target"))
    st = ts["stroke"]
    if st and st["left_at"] is not None and press is not None and press["on_target"]:
        fails.append((st["left_at"], "press left the tube"))
    dropped = " + ".join(t for _, t in fails if "dropped" in t) or "fail flag"
    stop = dropped if res["stop_reason"] == "fail" else STOP_TEXT[res["stop_reason"]]
    if res["success"] and not success:
        stop = "released, press left the tube"

    depth = -1000.0 * z["plunger"]
    rows = np.arange(n)
    pressing = (depth >= 1.0) & (tube_d < geo.HAND_DIST) & (rows >= (s1 if s1 is not None else n)) \
        & (rows >= (s2 if s2 is not None else n))
    deep = np.flatnonzero(pressing & (depth >= 6.0))
    stop_row = fs["S5_release"] if success and fs.get("S5_release") is not None else n - 1
    return dict(n=n, fs=fs, crooked=crooked, r=r, h=h, tx=tx, ty=ty, s2row=s2, s1row=s1, tube_d=tube_d, pip_d=pip_d,
                depth=depth, fails=fails, stop=stop, success=success, cams=cams,
                press_runs=press_runs(depth, pressing, s1), first_stop=int(deep[0]) if len(deep) else None,
                stop_row=int(stop_row), stop_reason=res["stop_reason"])


def run_state(objlog: str) -> dict:
    """frame_state of a saved run (objlog.npz)."""
    with np.load(objlog) as raw:
        z = {k: raw[k] for k in raw.files}
    return frame_state(z, score_rows(z))


def status(S: dict, i: int, show_stop: bool = True) -> list[tuple[str, str]]:
    """Status at row i as (text, kind), kind ok / bad / warn: failures from their row, the outcome from the stop row
    (if show_stop), an object not in its hand."""
    out = [(t, "bad") for row, t in S["fails"] if i >= row]
    if show_stop and i >= S["stop_row"] and S["stop"] and not any(S["stop"] == t for t, _ in out):
        out.append((S["stop"], "ok" if S["success"] and not out else "bad"))
    if S["s2row"] is not None and i >= S["s2row"] and not S["tube_d"][i] < geo.HAND_DIST:
        out.append(("tube not in left hand", "warn"))
    if S["s1row"] is not None and i >= S["s1row"] and not S["pip_d"][i] < geo.HAND_DIST:
        out.append(("pipette not in right hand", "warn"))
    return out


def milestones(S: dict, i: int) -> list[dict]:
    """The milestones at row i: code, name, the row reached (None if not yet) and a crooked-grasp note."""
    out = []
    for key, code, name in STAGES:
        row = S["fs"].get(key)
        row = row if row is not None and i >= row else None
        c = S["crooked"].get(key) if row is not None else None
        note = None if c is None else (f"regrasp off {c[1]}°" if len(c) > 2 else f"tilt {c[1]}°, crooked")
        out.append({"key": key, "code": code, "name": name, "row": row, "note": note})
    return out
