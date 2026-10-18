"""Task milestones of the Vega-U + Wuji pipette task, scored online from one record per control step.

Each milestone records the first control step at which it became true:
  S1 unrack   the pipette lifted >= UNRACK_LIFT above its hang height and then carried by the right wrist (its offset
              in the wrist frame stays within CARRY_OFFSET_TOL, no drop, while the wrist travels >= CARRY_MIN_TRAVEL)
  R1 regrasp  after S1: the pipette slid >= SLIDE_MIN along its own axis in the right hand (the press grip) while
              still at the wrist
  S2 tube     the tube lifted >= TUBE_LIFT above its rack height and carried by the left wrist the same way, upright
              (tilt <= S2_UPRIGHT) and never knocked over before (tilt <= S2_PREGRASP_TILT)
  S3 align    the tip within TUBE_INNER_R of the held tube's axis and -1..+5 cm from its opening
  S4 press    the plunger at the first stop (<= FIRST_STOP) with the tube held, the tip coarsely over the opening
              (PRESS_RADIAL, PRESS_Z_LO .. S4_Z_HI), after S3
  S5 release  after S4, the plunger back above RELEASED
Failure flags: pipette_dropped (after S1, far from both wrists and fallen), tube_dropped (after S2, out of the hand
and back down away from its pocket).

Hand-off checks (``contracts``, ``contract_detail``): a fixed number of steps after S1, R1, S2 and S3 the scorer also
measures the state the next phase starts from (where the object sits in the hand, its tilt, the best alignment)
against the demonstrations' bands. They never change a milestone.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial.transform import Rotation

# milestone thresholds (m unless marked)
UNRACK_LIFT = 0.030
HAND_DIST = 0.150            # an object within this distance of a wrist counts as in that hand
TUBE_LIFT = 0.050
TUBE_INNER_R = 0.014
ALIGN_Z_LO, ALIGN_Z_HI = -0.010, 0.050
FIRST_STOP = -0.0060         # plunger joint position of the first stop
RELEASED = -0.0010
S4_Z_HI = 0.035              # the tip at most this high above the opening at the press
PRESS_RADIAL = 0.030         # the tip coarsely over the opening at the press
PRESS_Z_LO, PRESS_Z_HI = -0.020, 0.080
POCKET_XY = 0.020            # the tube counts as back at its pocket within this distance (xy, z)
POCKET_Z = 0.020
DROP_FAR = 0.250
DROP_FALL = 0.150
TUBE_HEIGHT = 0.160
CARRY_OFFSET_TOL = 0.030     # a carried object's offset in the wrist frame may drift this much
CARRY_Z_DROP = 0.030         # a larger drop counts as a fall
CARRY_MIN_TRAVEL = 0.050     # the wrist must travel this far with the object attached
CARRY_FAIL_OFFSET = 0.050    # beyond this the object has separated from the hand
SLIDE_MIN = 0.055            # in-hand slide of the pipette that counts as the regrasp
S2_UPRIGHT = 20.0            # deg: the tube's tilt at the lift and through the carry
S2_PREGRASP_TILT = 25.0      # deg: the tube's worst tilt before it is lifted

# hand-off check bands: (centre, tolerance) of the demonstrations at that hand-off, or a one-sided limit
CONTRACT_DELAY = {"S1_unrack": 30, "R1_regrasp": 60, "S2_tube": 60, "S3_align": 150}   # steps after the milestone
C_S1_ALONG = (11.5, 1.0)     # cm: pipette origin along its axis from the right wrist
C_S1_RADIAL = (4.9, 1.0)     # cm: the perpendicular part of the same offset
C_S1_TILT = 16.0             # deg: pipette axis from vertical
C_S1_YAW = 12.0              # deg: rotation about vertical since the pipette hung on the stand
C_R1_ALONG = (2.5, 1.5)      # cm
C_R1_RADIAL = (4.5, 1.0)     # cm
C_R1_TILT = 20.0             # deg
C_S2_TILT = 15.0             # deg: tube axis from vertical
C_S2_DIST = (5.9, 1.5)       # cm: tube origin to the left wrist
C_S2_DZ = (-2.7, 1.5)        # cm: the z part of the same offset
C_S3_XY = 15.0               # mm: best tip-to-opening radial error in the window
C_S3_H = (20.0, 15.0)        # mm: tip height above the opening at that moment

STEPS = ("S1_unrack", "R1_regrasp", "S2_tube", "S3_align", "S4_press", "S5_release")
DEMO_STEPS = ("S1_unrack", "S2_tube", "S3_align", "S4_press", "S5_release")   # the milestones a success needs


def _vertical_tilt_deg(quat_xyzw: ArrayLike) -> float:
    """Angle between the body's +z axis and world vertical, sign-free."""
    z = Rotation.from_quat(quat_xyzw).apply(np.array([0.0, 0.0, 1.0]))[2]
    return float(np.degrees(np.arccos(min(1.0, abs(float(z))))))


def _in_hand_cm(obj_pos: ArrayLike, obj_quat: ArrayLike, wrist_pos: ArrayLike) -> tuple[float, float]:
    """The wrist-to-object offset split along the object's own axis and perpendicular to it, in cm."""
    d = np.asarray(obj_pos, dtype=float) - np.asarray(wrist_pos, dtype=float)
    axis = Rotation.from_quat(obj_quat).apply(np.array([0.0, 0.0, 1.0]))
    along = float(d @ axis)
    return along * 100.0, float(np.linalg.norm(d - along * axis)) * 100.0


def _yaw_about_vertical_deg(q0: ArrayLike, qt: ArrayLike) -> float:
    """Rotation of the pipette about vertical since q0 (on the stand): the z part of the rotation vector of
    R0^-1 * R_t, in deg."""
    rv = (Rotation.from_quat(q0).inv() * Rotation.from_quat(qt)).as_rotvec()
    return float(abs(np.degrees(rv[2])))


def _within(value: float, band: tuple[float, float]) -> bool:
    return abs(value - band[0]) <= band[1]


@dataclass
class ScoreState:
    first_step: dict[str, int | None] = field(default_factory=lambda: {k: None for k in STEPS})
    pipette_dropped: int | None = None
    tube_dropped: int | None = None
    max_plunger_depth: float = 0.0        # most negative plunger position seen (m)
    max_slide: float = 0.0                # largest in-hand slide of the pipette after S1 (m)
    min_tip_xy_err: float = 1e9           # best tip-to-opening radial error while the tube was held (m)
    press_radial_at_s4: float | None = None
    tube_tilt_max_pre: float = 0.0        # worst tube tilt before it was lifted (deg)
    tube_tilt_at_s2: float | None = None
    tube_tilt_max_carry: float = 0.0      # worst tube tilt while carried (deg)
    coarse_align_step: int | None = None   # first step the tip was coarsely over the opening
    contracts: dict[str, bool | None] = field(default_factory=lambda: {k: None for k in CONTRACT_DELAY})
    first_step_ok: dict[str, int | None] = field(default_factory=lambda: {k: None for k in CONTRACT_DELAY})
    contract_detail: dict[str, dict] = field(default_factory=dict)
    _last_align_step: int = -10**9
    _grip_axis_ref: float | None = None   # the right wrist's coordinate along the pipette axis at S1
    _pending: list = field(default_factory=list)   # scheduled hand-off checks

    @property
    def success(self) -> bool:
        return all(self.first_step[k] is not None for k in DEMO_STEPS)

    def summary(self) -> dict:
        d = {k: v for k, v in asdict(self).items() if not k.startswith("_")}
        d["success"] = self.success
        d["n_steps_reached"] = sum(v is not None for v in self.first_step.values())
        return d


class _CarryTracker:
    """An object is in a hand once, after a start condition, it keeps its offset in the wrist frame and does not fall
    while the wrist travels; separating from the wrist starts over."""

    def __init__(self) -> None:
        self.ref_offset = None
        self.ref_wrist = None
        self.ref_z = None

    def update(self, started: bool, obj: np.ndarray, wrist: np.ndarray, tilt_deg: float = 0.0,
               max_tilt: float = 180.0) -> bool:
        if not started or tilt_deg > max_tilt:
            return False
        if self.ref_offset is None:
            self.ref_offset, self.ref_wrist, self.ref_z = obj - wrist, wrist.copy(), float(obj[2])
            return False
        off_change = float(np.linalg.norm((obj - wrist) - self.ref_offset))
        z_drop = self.ref_z - float(obj[2])
        travel = float(np.linalg.norm(wrist - self.ref_wrist))
        if off_change > CARRY_FAIL_OFFSET or z_drop > CARRY_FAIL_OFFSET:
            self.ref_offset = None
            return False
        return off_change < CARRY_OFFSET_TOL and z_drop < CARRY_Z_DROP and travel >= CARRY_MIN_TRAVEL


class Scorer:
    """Incremental scorer: call update() once per control step with that step's record."""

    def __init__(self, p_hang0: np.ndarray, p_tube0: np.ndarray) -> None:
        self.p_hang0 = np.asarray(p_hang0, dtype=float)
        self.p_tube0 = np.asarray(p_tube0, dtype=float)
        self.s = ScoreState()
        self._carry_s1 = _CarryTracker()
        self._carry_s2 = _CarryTracker()
        self._q0 = None          # the pipette's orientation on the stand: the yaw reference
        self._last = None        # (step, record) of the last update, for checks still due at the end

    @staticmethod
    def _tube_opening(tube_pos: ArrayLike, tube_quat_xyzw: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        axis = Rotation.from_quat(tube_quat_xyzw).apply(np.array([0.0, 0.0, 1.0]))
        return np.asarray(tube_pos) + axis * (TUBE_HEIGHT / 2.0), axis

    def _schedule(self, key: str, step: int) -> None:
        """Queue the hand-off check of a milestone that just fired, CONTRACT_DELAY[key] steps later."""
        self.s._pending.append({"key": key, "milestone_step": step, "due": step + CONTRACT_DELAY[key],
                                "best_xy": float("inf"), "best_h": None, "best_step": None,
                                "at_xy": None, "at_h": None})

    def _contract_step(self, step: int, r: dict) -> None:
        """Track the S3 window's best alignment, then evaluate every check that is due."""
        s = self.s
        if not s._pending:
            return
        for e in s._pending:
            if e["key"] == "S3_align" and step >= e["milestone_step"]:
                opening, axis = self._tube_opening(r["tube_pos"], r["tube_quat"])
                v = np.asarray(r["tip_pos"], dtype=float) - opening
                along = float(v @ axis)
                radial = float(np.linalg.norm(v - along * axis)) * 1000.0
                if step == e["milestone_step"]:
                    e["at_xy"], e["at_h"] = radial, along * 1000.0
                if radial < e["best_xy"]:
                    e["best_xy"], e["best_h"], e["best_step"] = radial, along * 1000.0, step
        due = [e for e in s._pending if step >= e["due"]]
        for e in due:
            self._eval_contract(e, step, r)
            s._pending.remove(e)

    def _eval_contract(self, e: dict, step: int, r: dict) -> None:
        key = e["key"]
        if key in ("S1_unrack", "R1_regrasp"):
            along, radial = _in_hand_cm(r["pip_pos"], r["pip_quat"], r["rw_pos"])
            tilt = _vertical_tilt_deg(r["pip_quat"])
            d = {"along_cm": along, "radial_cm": radial, "tilt_deg": tilt}
            if key == "S1_unrack":
                d["yaw_deg"] = _yaw_about_vertical_deg(self._q0, r["pip_quat"])
                bad = ([] if _within(along, C_S1_ALONG) else ["along"]) \
                    + ([] if _within(radial, C_S1_RADIAL) else ["radial"]) \
                    + ([] if tilt <= C_S1_TILT else ["tilt"]) \
                    + ([] if d["yaw_deg"] <= C_S1_YAW else ["yaw"])
            else:
                bad = ([] if _within(along, C_R1_ALONG) else ["along"]) \
                    + ([] if _within(radial, C_R1_RADIAL) else ["radial"]) \
                    + ([] if tilt <= C_R1_TILT else ["tilt"])
        elif key == "S2_tube":
            v = np.asarray(r["tube_pos"], dtype=float) - np.asarray(r["lw_pos"], dtype=float)
            dist, dz = float(np.linalg.norm(v)) * 100.0, float(v[2]) * 100.0
            tilt = _vertical_tilt_deg(r["tube_quat"])
            d = {"tube_dist_cm": dist, "tube_dz_cm": dz, "tube_tilt_deg": tilt}
            bad = ([] if _within(dist, C_S2_DIST) else ["tube_dist"]) \
                + ([] if _within(dz, C_S2_DZ) else ["tube_dz"]) \
                + ([] if tilt <= C_S2_TILT else ["tube_tilt"])
        else:                    # S3_align: the best alignment reached in the window
            xy, h = e["best_xy"], e["best_h"]
            d = {"tip_xy_best_mm": xy, "tip_h_at_best_mm": h, "best_step": e["best_step"],
                 "tip_xy_at_mm": e["at_xy"], "tip_h_at_mm": e["at_h"]}
            bad = ([] if xy <= C_S3_XY else ["tip_xy_best"]) \
                + ([] if (h is not None and _within(h, C_S3_H)) else ["tip_height"])
        d["milestone_step"] = e["milestone_step"]
        d["checked_step"] = step
        d["violates"] = bad
        self.s.contract_detail[key] = d
        self.s.contracts[key] = not bad
        self.s.first_step_ok[key] = step if not bad else None

    def _flush_contracts(self) -> None:
        """Evaluate the checks still scheduled on the last step seen (a run that ends early)."""
        if self._last is None:
            return
        step, r = self._last
        for e in list(self.s._pending):
            self._eval_contract(e, step, r)
            self.s._pending.remove(e)

    def update(self, step: int, r: dict) -> ScoreState:
        """r: pip_pos, pip_quat, tube_pos, tube_quat, lw_pos, rw_pos, tip_pos (m, xyzw) and plunger (m) of this step."""
        s, fs = self.s, self.s.first_step
        if self._q0 is None:
            self._q0 = np.asarray(r["pip_quat"], dtype=float).copy()
        self._last = (step, r)
        fs_before = dict(fs)
        pip, tube = np.asarray(r["pip_pos"]), np.asarray(r["tube_pos"])
        lw, rw = np.asarray(r["lw_pos"]), np.asarray(r["rw_pos"])
        tip = np.asarray(r["tip_pos"])
        plunger = float(r["plunger"])
        d_pip_r = float(np.linalg.norm(pip - rw))
        d_pip_l = float(np.linalg.norm(pip - lw))
        d_tube_l = float(np.linalg.norm(tube - lw))
        s.max_plunger_depth = min(s.max_plunger_depth, plunger)

        pip_axis = Rotation.from_quat(r["pip_quat"]).apply(np.array([0.0, 0.0, 1.0]))
        rw_along = float((rw - pip) @ pip_axis)          # the right wrist along the pipette axis (for the slide)
        if fs["S1_unrack"] is None:
            lifted = pip[2] - self.p_hang0[2] >= UNRACK_LIFT and d_pip_r < HAND_DIST
            if self._carry_s1.update(lifted, pip, rw):
                fs["S1_unrack"] = step
                s._grip_axis_ref = rw_along
        if fs["S1_unrack"] is not None and d_pip_r < HAND_DIST and s._grip_axis_ref is not None:
            s.max_slide = max(s.max_slide, float(rw_along - s._grip_axis_ref))
        if fs["S1_unrack"] is not None and fs["R1_regrasp"] is None and s._grip_axis_ref is not None:
            if d_pip_r < HAND_DIST and rw_along - s._grip_axis_ref >= SLIDE_MIN:
                fs["R1_regrasp"] = step
        tube_tilt = _vertical_tilt_deg(r["tube_quat"])
        if fs["S2_tube"] is None:
            lifted = tube[2] - self.p_tube0[2] >= TUBE_LIFT and d_tube_l < HAND_DIST
            if not lifted:
                s.tube_tilt_max_pre = max(s.tube_tilt_max_pre, tube_tilt)
            upright_all_along = s.tube_tilt_max_pre <= S2_PREGRASP_TILT
            if self._carry_s2.update(lifted and upright_all_along, tube, lw, tube_tilt, S2_UPRIGHT):
                fs["S2_tube"] = step
                s.tube_tilt_at_s2 = tube_tilt
        elif d_tube_l < HAND_DIST:
            s.tube_tilt_max_carry = max(s.tube_tilt_max_carry, tube_tilt)

        tube_held = fs["S2_tube"] is not None and d_tube_l < HAND_DIST
        s4_over = False
        if tube_held:
            opening, axis = self._tube_opening(tube, r["tube_quat"])
            v = tip - opening
            along = float(v @ axis)
            radial = float(np.linalg.norm(v - along * axis))
            s.min_tip_xy_err = min(s.min_tip_xy_err, radial)
            if radial <= TUBE_INNER_R and ALIGN_Z_LO <= along <= ALIGN_Z_HI:
                s._last_align_step = step
                if fs["S3_align"] is None:
                    fs["S3_align"] = step
            coarse_over = radial <= PRESS_RADIAL and PRESS_Z_LO <= along <= PRESS_Z_HI
            if coarse_over and s.coarse_align_step is None:
                s.coarse_align_step = step
            s4_over = coarse_over and along <= S4_Z_HI

        if fs["S4_press"] is None and plunger <= FIRST_STOP and s4_over and s._last_align_step > -10**9:
            fs["S4_press"] = step
            s.press_radial_at_s4 = float(radial)          # the tip's offset at the moment of the press
        if fs["S4_press"] is not None and fs["S5_release"] is None and step > fs["S4_press"] and plunger >= RELEASED:
            fs["S5_release"] = step
        if fs["S1_unrack"] is not None and s.pipette_dropped is None:
            if min(d_pip_r, d_pip_l) > DROP_FAR and pip[2] < self.p_hang0[2] - DROP_FALL:
                s.pipette_dropped = step
        if fs["S2_tube"] is not None and s.tube_dropped is None:
            dxy = float(np.linalg.norm((tube - self.p_tube0)[:2]))
            if d_tube_l > HAND_DIST and tube[2] < self.p_tube0[2] + POCKET_Z and dxy > POCKET_XY:
                s.tube_dropped = step

        for k in CONTRACT_DELAY:                         # hand-off checks of milestones that just fired
            if fs_before.get(k) is None and fs.get(k) is not None:
                self._schedule(k, fs[k])
        self._contract_step(step, r)
        return s

    @property
    def failed(self) -> bool:
        return self.s.pipette_dropped is not None or self.s.tube_dropped is not None

    def summary(self) -> dict:
        """The final score, with the hand-off checks still due evaluated on the last step."""
        self._flush_contracts()
        return self.s.summary()
