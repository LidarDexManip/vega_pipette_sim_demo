"""The evaluation video frame, drawn K times larger for high-definition output.

Layout (1x units, 1544 x 798):
  left    the progress panel: title, subtitle, task time, status pills, the six milestones (S1 R1 S2 S3 S4 S5)
  right   the side view | the alignment tile (top view down the tube mouth with the tip, its height above the mouth,
          its distance from the axis, the plunger depth), over the four policy cameras (ego, left wrist, right wrist,
          thumb); the wrist views are turned upright
One video frame per logged control row (30 Hz); the last frame is held for one second.
"""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ..progress import geometry as geo
from ..progress.rollout import PRESS_MM, STAGES, milestones, status
from .style import (
    AMBER,
    BAD,
    BG0,
    CAM_L,
    CAM_R,
    CARD,
    EGO_CROP,
    FAINT,
    FG,
    HEIGHT,
    LINE,
    MUTED,
    OK,
    OK_DIM,
    PANEL,
    PRESS,
    PW,
    RAD,
    RGB,
    ROW,
    ROW_H,
    RW,
    SH,
    SW,
    VW,
    WIDTH,
    X0,
    F,
    G,
    M,
    Scaled,
    caps,
    corner_mask,
    fit_title,
    gauge,
    mix,
    pills,
    round_corners,
    rtext,
    tag,
)

VIEWS = ("side_view", "ego_view", "left_wrist_view", "right_wrist_view", "right_thumb_view")
TURN = {"left_wrist_view": -1, "right_wrist_view": 1}     # np.rot90 k: -1 = 90 deg clockwise, 1 = counter-clockwise
CAM_COLOUR = {"L": CAM_L, "R": CAM_R}
KIND_COLOUR = {"ok": OK, "bad": BAD, "warn": AMBER}


# ---- the progress panel ------------------------------------------------------------------------------------------------
def stepper(d: Scaled, x: float, y: float, S: dict, i: int, rs: int) -> None:
    """The milestones as a vertical stepper: dots joined by a line (green up to the last reached), code and name,
    the time each was reached."""
    ms = milestones(S, i)
    for k in range(len(ms) - 1):
        d.line([(x + 8, y + rs * k + 21), (x + 8, y + rs * (k + 1) - 1)],
               fill=OK if ms[k + 1]["row"] is not None else LINE, width=2)
    for k, m in enumerate(ms):
        hit, bad = m["row"] is not None, m["note"] is not None
        col = BAD if bad else OK if hit else FAINT
        cy = y + rs * k + 10
        if hit:
            d.ellipse([x + 1, cy - 7, x + 15, cy + 7], fill=col)
        else:
            d.ellipse([x + 1, cy - 7, x + 15, cy + 7], fill=PANEL, outline=col, width=2)
        d.text((x + 28, cy - 12), m["code"], font=F["code"], fill=FG if hit else FAINT)
        d.text((x + 60, cy - 12), m["name"], font=F["step"], fill=FG if hit else MUTED)
        tt = m["note"] or (f"at {m['row'] / 30.0:.1f} s" if hit else "")
        if tt:
            d.text((x + 28, cy + 13), tt, font=F["small"], fill=BAD if bad else MUTED)


# ---- the alignment tile ------------------------------------------------------------------------------------------------
def topview(d: Scaled, x0: float, y0: float, size: int, S: dict, i: int) -> None:
    """Top view down the held tube (+-12 mm): the wall, the tip with a 4 s trail and its press path, each wrist camera
    as a lettered badge on a dotted orbit on the side it looks from; scale bar and legend along the bottom."""
    cx, cy = x0 + size // 2, y0 + size // 2
    mm_px = size / 24.0
    ro, ri = geo.TUBE_R_MM * mm_px, (geo.TUBE_R_MM - geo.WALL_MM) * mm_px
    r, hh, s2row, tube_d = S["r"], S["h"], S["s2row"], S["tube_d"]
    active = s2row is not None and i >= s2row and tube_d[i] < geo.HAND_DIST and r[i] <= 45 and -30 <= hh[i] <= 120
    r_orb = size // 2 - 16
    for k in range(0, 360, 5):
        a = np.radians(k)
        qx, qy = cx + r_orb * np.cos(a), cy + r_orb * np.sin(a)
        d.ellipse([qx - 1.2, qy - 1.2, qx + 1.2, qy + 1.2], fill=LINE)
    d.ellipse([cx - ro, cy - ro, cx + ro, cy + ro], fill=mix(PANEL, FG, 0.30) if active else LINE)
    d.ellipse([cx - ri, cy - ri, cx + ri, cy + ri], fill=PANEL)
    by0 = y0 + size - 4
    sb, bx0 = 5 * mm_px, x0 + 2
    d.line([(bx0, by0), (bx0 + sb, by0)], fill=MUTED, width=2)
    d.line([(bx0, by0 - 4), (bx0, by0 + 4)], fill=MUTED)
    d.line([(bx0 + sb, by0 - 4), (bx0 + sb, by0 + 4)], fill=MUTED)
    d.text((bx0 + sb + 6, by0 - 10), "5 mm", font=F["small"], fill=MUTED)
    lb = "robot side"
    d.text((cx - d.textlength(lb, font=F["small"]) / 2, by0 - 10), lb, font=F["small"], fill=MUTED)
    rtext(d, x0 + size, by0 - 10, "press path", F["small"], MUTED)
    lx = x0 + size - d.textlength("press path", font=F["small"]) - 8
    d.line([(lx - 18, by0), (lx, by0)], fill=PRESS, width=4)
    if not active:
        return
    if S.get("_tv_size") != size:      # screen offsets of every row from the mouth, clamped inside the orbit
        lim = r_orb - 18
        px, py = S["tx"] * mm_px, -S["ty"] * mm_px
        m = np.maximum(np.hypot(px, py), 1e-9)
        f = np.where(m > lim, lim / m, 1.0)
        S["_tv"], S["_tv_size"], S["_tv_in"] = np.stack([px * f, py * f], 1), size, m <= lim

    def pt(k: int) -> tuple[float, float]:
        return cx + S["_tv"][k, 0], cy + S["_tv"][k, 1]
    k0 = max(s2row, i - 120)
    inside = S["_tv_in"]
    for k in range(k0, i):              # the motion trail, the last 4 s, brighter and wider toward the tip
        if not (inside[k] and inside[k + 1]):
            continue
        t = (k - k0 + 1) / max(1, i - k0)
        d.line([pt(k), pt(k + 1)], fill=mix(PANEL, (205, 214, 222), 0.25 + 0.75 * t), width=2 if t < 0.5 else 3)
    for a, b in S["press_runs"]:        # the press path so far: one line that stays
        if a > i:
            break
        seg, segs = [], []
        for k in range(a, min(b, i) + 1):
            if inside[k]:
                seg.append(pt(k))
            elif seg:
                segs.append(seg)
                seg = []
        segs.append(seg)
        for sg in segs:
            if len(sg) > 1:
                d.line(sg, fill=BG0, width=8, joint="curve")
                d.line(sg, fill=PRESS, width=4, joint="curve")
    x, y = pt(i)
    in_window = r[i] <= geo.R_MAX and geo.H_LO <= hh[i] <= geo.H_HI
    pressing = S["depth"][i] >= PRESS_MM
    col = OK if in_window else BAD if pressing else AMBER if r[i] <= geo.TUBE_R_MM else FG
    if inside[i]:
        d.ellipse([x - 6, y - 6, x + 6, y + 6], fill=col, outline=BG0, width=2)
    else:                               # far off: an outward chevron on the rim
        ux, uy = S["_tv"][i] / max(np.hypot(*S["_tv"][i]), 1e-9)
        nx, ny = -uy, ux
        d.polygon([(x + 8 * ux, y + 8 * uy), (x - 6 * ux + 7 * nx, y - 6 * uy + 7 * ny),
                   (x - 6 * ux - 7 * nx, y - 6 * uy - 7 * ny)], fill=col, outline=BG0)
    for name, dirs in S["cams"]:
        vx, vy = dirs[i]
        if not np.isfinite(vx):
            continue
        ccol = CAM_COLOUR[name]
        bx_, by_, rr = cx - vx * r_orb, cy - vy * r_orb, 13
        ax_, ay_ = bx_ + vx * (rr + 10), by_ + vy * (rr + 10)
        d.polygon([(ax_, ay_), (bx_ + vx * rr * 0.6 - vy * 8, by_ + vy * rr * 0.6 + vx * 8),
                   (bx_ + vx * rr * 0.6 + vy * 8, by_ + vy * rr * 0.6 - vx * 8)], fill=ccol)
        d.ellipse([bx_ - rr, by_ - rr, bx_ + rr, by_ + rr], fill=ccol)
        t = d.textbbox((0, 0), name, font=F["badge"])
        d.text((bx_ - (t[2] + t[0]) / 2, by_ - (t[3] + t[1]) / 2), name, font=F["badge"], fill=BG0)


def height_gauge(d: Scaled, x: float, y: float, hgt: float, val: float, lo: float, hi: float, band: tuple[float, float],
                 col: RGB, active: bool, ranges: list[tuple[float, float]], x_right: float) -> None:
    """The tip's height above the mouth as a vertical gauge: the target band, the mouth line at 0, the value, and one
    bar per press beside the track (the height range the tip covered while the plunger was down)."""
    d.text((x, y), "h", font=F["stage"], fill=FG if active else FAINT)
    vs = f"{val:.1f}" if active and np.isfinite(val) else "–"
    uw = d.textlength(" mm", font=F["unit"])
    rtext(d, x_right - uw, y + 20, vs, F["val"], col if active else FAINT)
    d.text((x_right - uw + 4, y + 27), "mm", font=F["unit"], fill=MUTED if active else FAINT)
    t0, t1 = y + 58, y + hgt
    tx_ = x + 28

    def ty(v: float) -> float:
        return t1 - (t1 - t0) * (min(max(v, lo), hi) - lo) / (hi - lo)
    d.rounded_rectangle([tx_ - 4, t0, tx_ + 4, t1], radius=4, fill=CARD)
    d.rounded_rectangle([tx_ - 4, ty(band[1]), tx_ + 4, ty(band[0])], radius=4,
                        fill=OK_DIM if active else mix(CARD, OK_DIM, 0.4))
    for t in band:
        lab = f"{t:g}"
        d.text((tx_ - 12 - d.textlength(lab, font=F["small"]), ty(t) - 10), lab, font=F["small"], fill=MUTED)
    d.line([(tx_ - 10, ty(0)), (tx_ + 10, ty(0))], fill=MUTED, width=2)
    d.text((tx_ - 12 - d.textlength("mouth", font=F["small"]), ty(0) - 10), "mouth", font=F["small"], fill=MUTED)
    for k, (plo, phi) in enumerate(ranges[-4:]):
        bx_ = tx_ + 11 + 8 * k
        d.rounded_rectangle([bx_ - 1, ty(phi) - 4, bx_ + 7, ty(plo) + 4], radius=4, fill=BG0)
        d.rounded_rectangle([bx_, ty(phi) - 3, bx_ + 6, ty(plo) + 3], radius=3, fill=PRESS)
    if active and np.isfinite(val):
        py = ty(val)
        if val > hi:
            d.polygon([(tx_ - 7, py + 7), (tx_, py - 1), (tx_ + 7, py + 7)], fill=col)
        elif val < lo:
            d.polygon([(tx_ - 7, py - 7), (tx_, py + 1), (tx_ + 7, py - 7)], fill=col)
        else:
            d.ellipse([tx_ - 7, py - 7, tx_ + 7, py + 7], fill=col, outline=BG0, width=2)


def align_tile(d: Scaled, x0: float, y0: float, w: float, h: float, S: dict, i: int) -> None:
    """The alignment tile: the top view with the height gauge beside it; under them r, then the button press."""
    x, wi = x0 + 16, w - 32
    caps(d, (x, y0 + 12), "tip vs tube")
    size = min(wi - 92, h - 34 - 14 - 136)
    topview(d, x, y0 + 34, size, S, i)
    ri, hi_ = float(S["r"][i]), float(S["h"][i])
    held = S["s2row"] is not None and i >= S["s2row"] and S["tube_d"][i] < geo.HAND_DIST
    dp = float(S["depth"][i])
    pressing = held and dp >= PRESS_MM          # pressing out of the window is a failure: red
    rc = OK if ri <= 7.5 else BAD if pressing else FG
    hc = OK if geo.H_LO <= hi_ <= geo.H_HI else BAD if pressing else AMBER if ri <= geo.R_MAX else FG
    pc = OK if dp >= 6 else PRESS if dp >= PRESS_MM else FG
    ranges = [(float(np.nanmin(S["h"][a:min(b, i) + 1])), float(np.nanmax(S["h"][a:min(b, i) + 1])))
              for a, b in S["press_runs"] if a <= i]
    height_gauge(d, x + size + 22, y0 + 34, size, hi_, -10, 60, (geo.H_LO, geo.H_HI), hc, held, ranges, x_right=x + wi)
    gy = y0 + 34 + size + 14
    gauge(d, x, gy, wi, "r to axis", ri, "mm", 0, 20, (0, 7.5), rc, (7.5,), held)
    gauge(d, x, gy + 68, wi, "button press", dp, "mm", 0, 10, (6, 10), pc, (6,), i >= (S["s1row"] or 10 ** 9))


# ---- cameras -----------------------------------------------------------------------------------------------------------
def frames_of(path: Path, limit: int) -> Iterator[np.ndarray | None]:
    """The first `limit` decoded RGB frames of one camera mp4; None forever after the end or when the file is missing."""
    if path.exists():
        w, h = map(int, subprocess.check_output(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                                 "stream=width,height", "-of", "csv=p=0", str(path)]).decode()
                   .split(",")[:2])
        pr = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-frames:v", str(limit), "-f", "rawvideo",
                               "-pix_fmt", "rgb24", "-"],
                              stdout=subprocess.PIPE)
        n = w * h * 3
        try:
            while True:
                b = pr.stdout.read(n)
                if len(b) < n:
                    break
                yield np.frombuffer(b, np.uint8).reshape(h, w, 3)
        finally:
            pr.stdout.close()
            pr.wait()
    while True:
        yield None


def fit(fr: np.ndarray | None, view: str, w: int, h: int) -> np.ndarray | None:
    """A camera frame for its tile: the ego view's 4:3 centre, the wrist views turned upright, then a centre crop to
    the tile's aspect and a resize."""
    if fr is None:
        return None
    if view == "ego_view":
        sc = fr.shape[1] / 424.0
        x0, y0, x1, y1 = (int(round(c * sc)) for c in EGO_CROP)
        fr = fr[y0:y1, x0:x1]
    if view in TURN:
        fr = np.ascontiguousarray(np.rot90(fr, TURN[view]))
    fh, fw = fr.shape[:2]
    a = w / h
    if fw / fh > a:
        cw = int(round(fh * a))
        x0 = (fw - cw) // 2
        box = (x0, 0, x0 + cw, fh)
    else:
        ch = int(round(fw / a))
        y0 = (fh - ch) // 2
        box = (0, y0, fw, y0 + ch)
    return np.asarray(Image.fromarray(fr).resize((w, h), Image.LANCZOS, box=box))


# ---- assembly ----------------------------------------------------------------------------------------------------------
def encoder_args(encoder: str, crf: int) -> list[str]:
    """ffmpeg output options: libx264 (portable, default) or h264_nvenc (NVIDIA GPUs)."""
    if encoder == "nvenc":
        return ["-c:v", "h264_nvenc", "-preset", "p7", "-rc", "vbr", "-cq", str(crf), "-b:v", "0"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf)]


def compose(S: dict, camera_dir: Path, title: str, subtitle: str, dst: Path, k: int = 2, crf: int = 20,
            encoder: str = "libx264") -> tuple[int, int]:
    """Write the video of one run: S from run_state.run_state(), camera_dir holding <view>.mp4 for every view."""
    width, height = WIDTH * k, HEIGHT * k
    y_row = M + SH + G
    base = Image.new("RGB", (width, height), BG0)
    bd = Scaled(ImageDraw.Draw(base), k)
    bd.rounded_rectangle([M, M, M + PW, HEIGHT - M - 1], radius=RAD, fill=PANEL)
    bd.rounded_rectangle([X0 + SW + G, M, X0 + RW - 1, M + SH - 1], radius=RAD, fill=PANEL)
    cells, xx = [("side_view", X0, M, SW, SH, "side")], X0
    for view, name, tw in ROW:
        cells.append((view, xx, y_row, tw, ROW_H, name))
        xx += tw + G
    x, pwi = M + 16, PW - 32
    bd.text((x, M + 16), title, font=fit_title(bd, title, pwi), fill=FG)
    bd.text((x, M + 50), subtitle, font=F["sub"], fill=MUTED)
    y_time = M + 92
    y_stat = y_time + 44
    y_st = y_stat + 100
    rs = min(80, (HEIGHT - M - 12 - y_st) // len(STAGES))
    base = np.asarray(base).copy()
    tag_font = ImageFont.truetype(F["tag"].path, F["tag"].size * k)
    mask = corner_mask(RAD * k)
    views = {v: frames_of(camera_dir / f"{v}.mp4", S["n"]) for v in VIEWS}
    tmp = dst.with_name(dst.name + ".tmp.mp4")
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", "30",
           "-i", "-", *encoder_args(encoder, crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(tmp)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    im = None
    try:
        for i in range(S["n"]):
            cur = {v: next(g) for v, g in views.items()}
            cv = base.copy()
            for view, cx, cy, w, h, name in cells:
                t = fit(cur[view], view, w * k, h * k)
                if t is None:
                    continue
                cv[cy * k:(cy + h) * k, cx * k:(cx + w) * k] = t
                round_corners(cv, cx * k, cy * k, w * k, h * k, RAD * k, mask)
                tag(cv, cx * k, cy * k, name, k, tag_font)
            im = Image.fromarray(cv)
            d = Scaled(ImageDraw.Draw(im), k)
            ts = f"{i / 30.0:.1f}"
            d.text((x, y_time), ts, font=F["time"], fill=FG)
            d.text((x + d.textlength(ts, font=F["time"]) + 6, y_time + 12), "s", font=F["unit"], fill=MUTED)
            pills(d, x, y_stat, pwi, [(t, KIND_COLOUR[k]) for t, k in status(S, i)[:3]])
            stepper(d, x, y_st, S, i, rs)
            align_tile(d, X0 + SW + G, M, VW, SH, S, i)
            pr.stdin.write(im.tobytes())
        for _ in range(30):                     # hold the final state for one second
            pr.stdin.write(im.tobytes())
    finally:
        pr.stdin.close()
        pr.wait()
    if pr.returncode:
        raise RuntimeError(f"ffmpeg failed ({pr.returncode}) writing {dst}")
    tmp.replace(dst)
    return width, height
