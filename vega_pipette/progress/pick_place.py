"""Progress of a tube pick-and-place run: the instruction the policy is given, switched on the simulation's truth, the
milestones and the outcome, and the state the evaluation frame and the live dashboard draw.

Stages, one 30 Hz control row at a time (each row is judged on the state before its action):
  pick     "Pick up the tube from column c, row r of the blue rack."
           -> place once the tube's bottom is PICK_CLEAR above the rack's top; after PICK_ROWS the run ends.
  place    "Place the tube into column c, row r of the blue rack." (the runner appends the hand target, control/
           place_hint.py)
           -> release once the tube is seated (progress/rack.py) in any pocket on SEAT_ROWS rows in a row, or after
              PLACE_ROWS.
  release  "Release the tube and move the hand away."
           -> check once the hand is open (the left-hand joints sum to at most HAND_OPEN rad) and the wrist is HAND_AWAY
              from the tube, or after RELEASE_ROWS.
  check    CHECK_ROWS more rows with the release sentence, then the run ends.
A new stage asks the policy for a new chunk at once. Success: the release ended by its condition and the tube is
seated in hole B on every check row.

Milestones: S1 pick up (the switch to place), S2 in hole B (the first row the tube is seated in B), S3 release (the last
row of a successful check). Otherwise the outcome is the first failure that applies to the last row: pick failed, wrong
hole (seated in another pocket, hole A included), still held, not seated (in or on the rack), dropped, release not
done (its time limit), not seated through the check.
"""
from __future__ import annotations

import math

import numpy as np

from . import rack

STAGES = ("pick", "place", "release", "check", "done")
PICK_CLEAR = 0.100
SEAT_ROWS = 3
PICK_ROWS, PLACE_ROWS, RELEASE_ROWS, CHECK_ROWS = 960, 340, 510, 30
HAND_OPEN, HAND_AWAY = 3.0, 0.150
NOT_SEATED_R = 0.01152
"""A tube below the rack's top within this distance of a pocket's axis (the bevel's outer polygon corner) sits in it."""
LEFT_HAND = slice(7, 27)
RELEASE = "Release the tube and move the hand away."
MILESTONES = (("S1", "pick up"), ("S2", "in hole B"), ("S3", "release"))
ROW_KEYS = ("tube_pos", "tube_quat", "rack_pos", "rack_quat", "lw_pos")
OUTCOME_TEXT = {"success": "success", "pick_failed": "pick failed", "wrong_hole": "wrong hole", "held": "still held",
                "not_seated": "not seated", "dropped": "tube dropped", "release_not_done": "release not done",
                "check_failed": "not seated for 1 s"}


def sentences(hole_a: tuple[int, int], hole_b: tuple[int, int]) -> dict[str, str]:
    """The instruction of every stage."""
    (ca, ra), (cb, rb) = rack.column_row(hole_a), rack.column_row(hole_b)
    return {"pick": f"Pick up the tube from column {ca}, row {ra} of the blue rack.",
            "place": f"Place the tube into column {cb}, row {rb} of the blue rack.",
            "release": RELEASE, "check": RELEASE}


class PickPlaceProgress:
    """The stage switching and the record of one run, row by row (module docstring). Rows after the end are ignored;
    n is the number of rows that got an action."""

    def __init__(self, hole_a: tuple[int, int], hole_b: tuple[int, int]) -> None:
        self.hole_a, self.hole_b = tuple(int(x) for x in hole_a), tuple(int(x) for x in hole_b)
        self.sentences = sentences(self.hole_a, self.hole_b)
        self.stage, self.stage_row, self.n = "pick", 0, 0
        self.events: dict[str, int] = {"pick_start": 0}
        self.timeouts: list[str] = []
        self.place_seat: tuple[int, int] | None = None
        self._seat_run = 0
        self.rows: dict[str, list] = {k: [] for k in ("stage", "bottom", "axis", "seat_b", "seat_any", "near", "near_d",
                                                       "hand", "wrist")}
        self._last: dict | None = None

    @property
    def stopped(self) -> bool:
        return self.stage == "done"

    @property
    def sentence(self) -> str:
        return self.sentences[self.stage]

    def add(self, row: dict) -> bool:
        """Judge one row (ROW_KEYS plus "state", the 54 joint positions); True when the stage changed."""
        if self.stopped:
            return False
        i, before = self.n, self.stage
        state = np.asarray(row["state"], np.float32)
        tube = np.asarray(row["tube_pos"], np.float64)
        bottom, axis = rack.in_rack(tube, row["tube_quat"], row["rack_pos"], row["rack_quat"])
        near, near_d = rack.nearest_pocket(bottom[:2])
        seat_any, seat_b = rack.seated(bottom, axis, near), rack.seated(bottom, axis, self.hole_b)
        hand = float(state[LEFT_HAND].sum())
        wrist = float(np.linalg.norm(np.asarray(row["lw_pos"], np.float64) - tube))
        k = i - self.stage_row
        if self.stage == "pick":
            if bottom[2] - rack.TOP >= PICK_CLEAR:
                self._switch("place", i)
            elif k >= PICK_ROWS:
                self.timeouts.append("pick")
                self.events["pick_end"] = i
                self.stage = "done"
        elif self.stage == "place":
            self._seat_run = self._seat_run + 1 if seat_any else 0
            if self._seat_run >= SEAT_ROWS:
                self.place_seat = near
                self._switch("release", i)
            elif k >= PLACE_ROWS:
                self.timeouts.append("place")
                self._switch("release", i)
        elif self.stage == "release":
            if hand <= HAND_OPEN and wrist >= HAND_AWAY:
                self._switch("check", i)
            elif k >= RELEASE_ROWS:
                self.timeouts.append("release")
                self._switch("check", i)
        elif self.stage == "check" and k >= CHECK_ROWS:
            self.events["check_end"] = i
            self.stage = "done"
        if not self.stopped:
            for key, v in (("stage", STAGES.index(self.stage)), ("bottom", bottom), ("axis", axis),
                           ("seat_b", seat_b), ("seat_any", seat_any), ("near", near), ("near_d", near_d),
                           ("hand", hand), ("wrist", wrist)):
                self.rows[key].append(v)
            self._last = {"tube_pos": tube, "lw_pos": np.asarray(row["lw_pos"], np.float64)}
            self.n = i + 1
        return self.stage != before

    def _switch(self, to: str, row: int) -> None:
        self.events[f"{self.stage}_end"] = row
        self.events[f"{to}_start"] = row
        self.stage, self.stage_row = to, row

    def result(self) -> dict:
        """The outcome (OUTCOME_TEXT key), success, the milestone rows, the stage events and timeouts."""
        ev, n = self.events, self.n
        seat_b = np.asarray(self.rows["seat_b"], bool)
        release_done = "check_start" in ev and "release" not in self.timeouts
        check = seat_b[ev["check_start"]:] if "check_start" in ev else seat_b[:0]
        success = bool(release_done and len(check) >= CHECK_ROWS and check.all())
        hits = np.flatnonzero(seat_b)
        out = {"success": success, "n": n, "events": dict(ev), "timeouts": list(self.timeouts),
               "place_seat": self.place_seat, "hole_a": self.hole_a, "hole_b": self.hole_b,
               "milestones": {"S1": ev.get("place_start"), "S2": int(hits[0]) if len(hits) else None,
                              "S3": ev["check_start"] + CHECK_ROWS - 1 if success else None}}
        out["outcome"] = "success" if success else self._failure(release_done)
        return out

    def _failure(self, release_done: bool) -> str:
        if "place_start" not in self.events:
            return "pick_failed"
        bottom, axis = self.rows["bottom"][-1], self.rows["axis"][-1]
        near, near_d = self.rows["near"][-1], self.rows["near_d"][-1]
        tilt = math.degrees(math.acos(min(1.0, max(-1.0, float(axis[2])))))
        below_top = bottom[2] < rack.TOP - 0.002 and tilt < 30.0
        wrist = float(np.linalg.norm(self._last["lw_pos"] - self._last["tube_pos"]))
        if self.rows["seat_b"][-1]:
            return "check_failed" if release_done else "release_not_done"
        if (tuple(near) == self.hole_a and below_top) or self.rows["seat_any"][-1]:
            return "wrong_hole"
        if below_top and near_d <= NOT_SEATED_R:
            return "not_seated"
        if wrist <= 0.100 and bottom[2] > rack.TOP:
            return "held"
        px, py = rack.PLATE_HALF
        on_top = rack.TOP - 0.002 <= bottom[2] <= rack.TOP + 0.015
        if abs(bottom[0]) <= px + 0.01 and abs(bottom[1]) <= py + 0.01 and on_top:
            return "not_seated"
        return "dropped"


def progress_of(z: dict) -> PickPlaceProgress:
    """PickPlaceProgress over all rows of a saved log z (objlog arrays of a pick-and-place run)."""
    p = PickPlaceProgress(tuple(z["hole_a"]), tuple(z["hole_b"]))
    for i in range(len(z["state"])):
        if p.stopped:
            break
        p.add({"state": z["state"][i], **{k: z[k][i] for k in ROW_KEYS}})
    return p


def frame_state(p: PickPlaceProgress, hint: dict | None = None) -> dict:
    """Everything the frame and the dashboard draw, per row: the stage, the target pocket (hole A while picking, B
    after), the tube bottom's offset from it (rack frame, mm; x right, y away from the robot), the bottom's height above
    the rack's top (mm), the tube's tilt (deg), the seat in B; plus the milestones, the outcome and the stop row.
    With the run's place hint (hint_move, hint_shift: per row as logged; prompt_step, prompt_text: each query's
    instruction): the instruction the policy reads on each row ("read"), where the hint sends the tube bottom ("hint",
    offset like dx / dy, NaN when there is no hint) and the size of its in-hand correction ("shift", mm, NaN before
    the first one)."""
    res = p.result()
    n = res["n"]
    stage = np.asarray(p.rows["stage"], np.int8)
    bottom = np.asarray(p.rows["bottom"], np.float64).reshape(-1, 3)
    axis = np.asarray(p.rows["axis"], np.float64).reshape(-1, 3)
    on_b = stage >= STAGES.index("place")
    centre = np.where(on_b[:, None], rack.hole_xy(p.hole_b), rack.hole_xy(p.hole_a))
    ev = res["events"]
    fails = []
    for st in ("place", "release"):
        if st in res["timeouts"]:
            fails.append((ev[f"{st}_end"], f"{st} timed out", "warn"))
    if p.place_seat is not None and p.place_seat != p.hole_b:
        c, r = rack.column_row(p.place_seat)
        fails.append((ev["place_end"], f"seated in column {c}, row {r}", "bad"))
    dx, dy = 1e3 * (bottom[:, 0] - centre[:, 0]), 1e3 * (bottom[:, 1] - centre[:, 1])
    read = [p.sentences[STAGES[min(int(s), 2)]] for s in stage]
    marker, shift = np.full((n, 2), np.nan), np.full(n, np.nan)
    if hint is not None:
        steps, texts = np.asarray(hint["prompt_step"]), list(hint["prompt_text"])
        k = np.searchsorted(steps, np.arange(n), side="right") - 1
        read = [str(texts[j]) if j >= 0 else read[i] for i, j in enumerate(k)]
        marker = np.stack([dx, dy], 1) + np.asarray(hint["hint_move"], np.float64)[:n].reshape(n, 2)
        shift = np.asarray(hint["hint_shift"], np.float64)[:n]
    return dict(n=n, stage=stage, hole_a=p.hole_a, hole_b=p.hole_b, sentences=p.sentences, read=read,
                dx=dx, dy=dy, height=1e3 * (bottom[:, 2] - rack.TOP),
                tilt=np.degrees(np.arccos(np.clip(axis[:, 2], -1.0, 1.0))), hint=marker, shift=shift,
                seat_b=np.asarray(p.rows["seat_b"], bool), milestones=res["milestones"], events=ev, fails=fails,
                outcome=res["outcome"], success=res["success"], stop_row=n - 1)


HINT_KEYS = ("hint_move", "hint_shift", "prompt_step", "prompt_text")


def run_state(objlog: str) -> dict:
    """frame_state of a saved run (objlog.npz), with its place hint."""
    with np.load(objlog) as raw:
        z = {k: raw[k] for k in raw.files}
    return frame_state(progress_of(z), {k: z[k] for k in HINT_KEYS})


def milestones(S: dict, i: int) -> list[dict]:
    """The milestones at row i: code, name, the row reached (None if not yet) and no note."""
    return [{"key": code, "code": code, "name": name,
             "row": S["milestones"][code] if S["milestones"][code] is not None and i >= S["milestones"][code] else None,
             "note": None} for code, name in MILESTONES]


def status(S: dict, i: int, show_stop: bool = True) -> list[tuple[str, str]]:
    """Status at row i as (text, kind), kind ok / bad / warn: timeouts and a wrong seat from their row, the outcome from
    the stop row (if show_stop)."""
    out = [(t, kind) for row, t, kind in S["fails"] if i >= row]
    if show_stop and i >= S["stop_row"]:
        out.append((OUTCOME_TEXT[S["outcome"]], "ok" if S["success"] else "bad"))
    return out
