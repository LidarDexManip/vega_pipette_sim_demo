"""The tube pick-and-place task's evaluation frame (video/frame.py's Panel PICK_PLACE), in the pipette frame's look.

Layout (1x units, 1544 x 798):
  left    the progress panel: title, subtitle, task time, status pills, the milestones S1 S2 S3
  right   the side view | the hole tile: a top view centred on the target pocket (hole A while picking, B after) with
          its neighbours, the tube's outline and bottom and their 4 s trail, and while placing where the place hint
          sends the tube bottom (a diamond); the bottom's height above the rack's top; its distance from the pocket's
          axis, the tube's tilt and the size of the hint's in-hand correction; under them the two policy cameras pixel
          for pixel as the policy sees them (head, palm), and the instruction tile: the instruction as the policy reads
          it (numbers included) and the three stages.
"""
from __future__ import annotations

import math

from ..progress import pick_place as pp
from ..progress import rack
from .frame import Panel
from .style import (
    AMBER,
    BG0,
    CARD,
    FAINT,
    FG,
    LINE,
    MUTED,
    OK,
    OK_DIM,
    PANEL,
    ROW_H,
    RW,
    SH,
    SW,
    VW,
    X0,
    F,
    G,
    M,
    Scaled,
    caps,
    gauge,
    mix,
    rtext,
)

EGO_W, PALM_W = 537, ROW_H                       # the head view at 16:9, the palm view square
INSTR_X, INSTR_W = X0 + EGO_W + PALM_W + 2 * G, RW - EGO_W - PALM_W - 2 * G
HOLE_X, ROW_Y = X0 + SW + G, M + SH + G
VIEW_MM = 36.0                                    # the top view shows +-36 mm around the target pocket
TRAIL_ROWS = 120
CAPTURE_MM = 1e3 * (rack.HOLE_R + rack.BEVEL[0]) - 1e3 * (rack.TUBE_R - rack.TUBE_BEVEL[0])
"""A tube lowered with its axis this close to a pocket's centre lands on the bevel, not on the rack's top."""
STAGE_NAMES = ("pick", "place", "release")
HINT = (94, 200, 235)
"""The place hint's colour (not a status colour)."""
SHIFT_MAX_MM = 20.0


def target_of(S: dict, i: int) -> tuple[int, int] | None:
    """The pocket the frame centres on at row i: hole A while picking, B after."""
    return S["hole_a"] if S["stage"][i] == pp.STAGES.index("pick") else S["hole_b"]


def wrap(d: Scaled, text: str, font: object, width: float) -> list[str]:
    """Words of `text` in lines no wider than `width`."""
    lines, cur = [], ""
    for word in text.split():
        cand = f"{cur} {word}".strip()
        if cur and d.textlength(cand, font=font) > width:
            lines.append(cur)
            cur = word
        else:
            cur = cand
    return lines + ([cur] if cur else [])


def diamond(d: Scaled, x: float, y: float, r: float, col: tuple[int, int, int], width: float = 2) -> None:
    d.line([(x, y - r), (x + r, y), (x, y + r), (x - r, y), (x, y - r)], fill=col, width=width)


def topview(d: Scaled, x0: float, y0: float, size: float, S: dict, i: int) -> None:
    """Top view down onto the target pocket (+-VIEW_MM): the pocket and its neighbours (bore and bevel), the tube's
    outline and flat bottom with its 4 s trail, and while placing where the hint sends the tube bottom (on top); the
    scale and the robot side under it."""
    cx, cy, mm = x0 + size / 2, y0 + size / 2, size / (2 * VIEW_MM)
    target = target_of(S, i)
    c0 = rack.hole_xy(target)
    outer, bore = 1e3 * (rack.HOLE_R + rack.BEVEL[0]), 1e3 * rack.HOLE_R
    for hole in rack.POCKETS:
        ox, oy = 1e3 * (rack.hole_xy(hole) - c0)
        if abs(ox) + outer > VIEW_MM or abs(oy) + outer > VIEW_MM:
            continue
        px, py = cx + ox * mm, cy - oy * mm
        d.ellipse([px - outer * mm, py - outer * mm, px + outer * mm, py + outer * mm],
                  fill=CARD if hole != target else LINE)
        d.ellipse([px - bore * mm, py - bore * mm, px + bore * mm, py + bore * mm], fill=PANEL)
    by = y0 + size + 14
    sb, bx0 = 10 * mm, x0 + 2
    d.line([(bx0, by), (bx0 + sb, by)], fill=MUTED, width=2)
    d.line([(bx0, by - 4), (bx0, by + 4)], fill=MUTED)
    d.line([(bx0 + sb, by - 4), (bx0 + sb, by + 4)], fill=MUTED)
    d.text((bx0 + sb + 6, by - 10), "10 mm", font=F["small"], fill=MUTED)
    rtext(d, x0 + size, by - 10, "robot side", F["small"], MUTED)
    lim = VIEW_MM - 1e3 * rack.TUBE_R

    def pt(k: int) -> tuple[float, float, bool]:
        ox, oy = S["dx"][k], S["dy"][k]
        m = max(math.hypot(ox, oy), 1e-9)
        f = min(1.0, lim / m)
        return cx + ox * f * mm, cy - oy * f * mm, m <= lim
    k0 = max(0, i - TRAIL_ROWS)
    while k0 < i and target_of(S, k0) != target:
        k0 += 1
    for k in range(k0, i):
        (ax, ay, ain), (bx, by_, bin_) = pt(k), pt(k + 1)
        if ain and bin_:
            t = (k - k0 + 1) / max(1, i - k0)
            d.line([(ax, ay), (bx, by_)], fill=mix(PANEL, (205, 214, 222), 0.25 + 0.75 * t), width=2 if t < 0.5 else 3)
    x, y, inside = pt(i)
    lat = math.hypot(S["dx"][i], S["dy"][i])
    col = OK if S["seat_b"][i] and target == S["hole_b"] else AMBER if lat <= CAPTURE_MM else FG
    if inside:
        tr, disc = 1e3 * rack.TUBE_R * mm, 1e3 * (rack.TUBE_R - rack.TUBE_BEVEL[0]) * mm
        d.ellipse([x - tr, y - tr, x + tr, y + tr], fill=mix(PANEL, col, 0.22), outline=col, width=2)
        d.ellipse([x - disc, y - disc, x + disc, y + disc], fill=col, outline=BG0, width=1)
    else:
        ux, uy = (x - cx) / max(math.hypot(x - cx, y - cy), 1e-9), (y - cy) / max(math.hypot(x - cx, y - cy), 1e-9)
        nx, ny = -uy, ux
        d.polygon([(x + 8 * ux, y + 8 * uy), (x - 6 * ux + 7 * nx, y - 6 * uy + 7 * ny),
                   (x - 6 * ux - 7 * nx, y - 6 * uy - 7 * ny)], fill=col, outline=BG0)
    hx, hy = S["hint"][i]
    if not math.isnan(hx):
        m = max(math.hypot(hx, hy), 1e-9)
        f = min(1.0, (VIEW_MM - 4) / m)
        diamond(d, cx + hx * f * mm, cy - hy * f * mm, 9, HINT)


def height_gauge(d: Scaled, x: float, y: float, hgt: float, S: dict, i: int, x_right: float) -> None:
    """The tube bottom's height above the rack's top as a vertical gauge, with the top face at 0, the seated depth and
    the height that ends the pick."""
    lo, hi = -60.0, 120.0
    val = float(S["height"][i])
    seated = bool(S["seat_b"][i])
    col = OK if seated else FG
    d.text((x, y), "h", font=F["stage"], fill=FG)
    uw = d.textlength(" mm", font=F["unit"])
    rtext(d, x_right - uw, y + 20, f"{val:.1f}", F["val"], col)
    d.text((x_right - uw + 4, y + 27), "mm", font=F["unit"], fill=MUTED)
    t0, t1, tx = y + 58, y + hgt, x + 40

    def ty(v: float) -> float:
        return t1 - (t1 - t0) * (min(max(v, lo), hi) - lo) / (hi - lo)
    d.rounded_rectangle([tx - 4, t0, tx + 4, t1], radius=4, fill=CARD)
    floor = -1e3 * rack.TOP
    d.rounded_rectangle([tx - 4, ty(floor + 1e3 * rack.SEAT_DEPTH) - 2, tx + 4, ty(floor) + 2], radius=3, fill=OK_DIM)
    for v, lab in ((1e3 * pp.PICK_CLEAR, "lifted"), (0.0, "top"), (floor, "seated")):
        d.line([(tx - 10, ty(v)), (tx + 10, ty(v))], fill=MUTED, width=2 if v == 0.0 else 1)
        d.text((tx - 14 - d.textlength(lab, font=F["small"]), ty(v) - 10), lab, font=F["small"], fill=MUTED)
    py = ty(val)
    if val > hi:
        d.polygon([(tx - 7, py + 7), (tx, py - 1), (tx + 7, py + 7)], fill=col)
    else:
        d.ellipse([tx - 7, py - 7, tx + 7, py + 7], fill=col, outline=BG0, width=2)


def hole_tile(d: Scaled, x0: float, y0: float, w: float, h: float, S: dict, i: int) -> None:
    """The hole tile: top view and height gauge; under them the distance from the pocket's axis, the tilt and the
    target shift of the hint's in-hand correction ("-" when there is none)."""
    x, wi = x0 + 16, w - 32
    target = target_of(S, i)
    caps(d, (x, y0 + 12), "tube vs hole")
    c, r = rack.column_row(target)
    rtext(d, x + wi, y0 + 10, f"{'A' if S['stage'][i] == 0 else 'B'}: column {c}, row {r}", F["tag"], FG)
    size = min(wi - 104, h - 34 - 14 - 152 - 62)
    topview(d, x, y0 + 38, size, S, i)
    if not math.isnan(S["hint"][i][0]):
        ly = y0 + 38 + size + 26
        diamond(d, x + 8, ly + 10, 6, HINT)
        d.text((x + 22, ly + 1), "hint sends tube here", font=F["small"], fill=MUTED)
    height_gauge(d, x + size + 22, y0 + 38, size, S, i, x_right=x + wi)
    lat = math.hypot(float(S["dx"][i]), float(S["dy"][i]))
    seated = bool(S["seat_b"][i])
    gy = y0 + 38 + size + 52
    gauge(d, x, gy, wi, "off the hole axis", lat, "mm", 0, 15, (0, 1e3 * rack.SEAT_LAT),
          OK if seated else AMBER if lat <= CAPTURE_MM else FG, (round(1e3 * rack.SEAT_LAT, 1), round(CAPTURE_MM, 1)),
          True)
    tilt = float(S["tilt"][i])
    gauge(d, x, gy + 54, wi, "tilt", tilt, "°", 0, 30, (0, rack.SEAT_TILT_DEG),
          OK if tilt <= rack.SEAT_TILT_DEG else FG, (rack.SEAT_TILT_DEG,), True)
    shift = float(S["shift"][i])
    on = not math.isnan(shift)
    gauge(d, x, gy + 108, wi, "target shift (in-hand correction)", shift if on else 0.0, "mm", 0, SHIFT_MAX_MM,
          (0, 0), HINT if on else MUTED, (), on)


COMPUTED_FROM = "Pocket:"
"""The first word of the hint, the part of the place instruction that the supervisor computes for the policy."""
COMPUTED_NOTE = "hint computed by the supervisor from the hole numbers and where the hand grasped the tube"


def instruction_tile(d: Scaled, x0: float, y0: float, w: float, h: float, S: dict, i: int) -> None:
    """The instruction as the policy reads it now, then the three stages. The supervisor's hint (from COMPUTED_FROM
    on) is drawn in the hint's colour, with a note saying where it comes from."""
    x, wi = x0 + 16, w - 32
    caps(d, (x, y0 + 12), "instruction the policy reads")
    st = int(S["stage"][i])
    y = y0 + 38
    computed = False
    for line in wrap(d, S["read"][i], F["val"], wi):
        wx = x
        for word in line.split():
            computed = computed or word == COMPUTED_FROM
            d.text((wx, y), word, font=F["val"], fill=HINT if computed else FG)
            wx += d.textlength(word + " ", font=F["val"])
        y += 27
    if computed:
        y += 4
        d.rectangle([x + 1, y + 6, x + 9, y + 14], fill=HINT)
        for line in wrap(d, COMPUTED_NOTE, F["small"], wi - 18):
            d.text((x + 18, y), line, font=F["small"], fill=MUTED)
            y += 19
    y = max(y + 12, y0 + 150)
    d.line([(x, y), (x + wi, y)], fill=LINE)
    y += 14
    for k, name in enumerate(STAGE_NAMES):
        here, done = min(st, 2) == k, min(st, 2) > k
        col = OK if done else FG if here else FAINT
        d.ellipse([x + 1, y + 5, x + 11, y + 15], fill=col if (done or here) else PANEL, outline=col, width=2)
        d.text((x + 20, y), name, font=F["stage"], fill=FG if here else MUTED)
        y += 30


def _draw(d: Scaled, S: dict, i: int) -> None:
    hole_tile(d, HOLE_X, M, VW, SH, S, i)
    instruction_tile(d, INSTR_X, ROW_Y, INSTR_W, ROW_H, S, i)


PICK_PLACE = Panel(
    cells=(("side_view", "side", X0, M, SW, SH), ("ego_view", "head", X0, ROW_Y, EGO_W, ROW_H),
           ("left_palm_view", "palm", X0 + EGO_W + G, ROW_Y, PALM_W, ROW_H)),
    tiles=((HOLE_X, M, VW, SH), (INSTR_X, ROW_Y, INSTR_W, ROW_H)),
    milestones=pp.milestones, status=pp.status, n_stages=len(pp.MILESTONES), draw=_draw, ego_crop=False,
    pixelated=("ego_view", "left_palm_view"))
"""The tube pick-and-place task's frame."""
