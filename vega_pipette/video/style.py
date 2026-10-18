"""Look of the video frame: palette, fonts, layout constants and the small drawing helpers.

All coordinates here are in the frame's 1x units (a 1544 x 798 frame); frame.py draws them K times larger.
Font: IBM Plex Sans (fonts/, SIL Open Font License 1.1).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_DIR = Path(__file__).resolve().parent / "fonts"

# palette: one dark theme
BG0 = (11, 15, 19)          # frame ground, gutters
PANEL = (18, 24, 30)        # panel / top-view surface
CARD = (27, 36, 45)         # tracks, pills
LINE = (42, 54, 65)
FG = (233, 238, 242)
MUTED = (140, 153, 164)
FAINT = (88, 101, 112)
OK = (64, 206, 133)
OK_DIM = (31, 78, 57)
AMBER = (255, 146, 60)      # near: orange, so it never reads as the yellow press line
BAD = (255, 102, 90)
CAM_L, CAM_R = (40, 188, 168), (158, 130, 246)
PRESS = (255, 224, 0)       # the tip's path while the plunger is down

# layout (1x)
M, G, RAD = 6, 4, 8                     # outer margin, gutter, corner radius
TW = 316                                # a camera tile of the 1x frame (~4:3)
RW = 4 * TW + 3 * G                     # 1276: the camera row = the width of the side + alignment row
SW, SH = 853, 480                       # side view (16:9)
VW = RW - SW - G                        # 419: the alignment tile
PW = 250                                # the progress panel
X0 = M + PW + G + 2                     # the video area's left edge
ROW_H = 302                             # the camera row's height
# camera row: ego 4:3, the two wrist views turned upright (3:4), the thumb view fills the rest
ROW = (("ego_view", "ego", 404), ("left_wrist_view", "L wrist", 227), ("right_wrist_view", "R wrist", 227),
       ("right_thumb_view", "thumb", RW - 404 - 2 * 227 - 3 * G))
EGO_CROP = (52, 0, 372, 240)            # the 4:3 centre of the 424 x 240 ego view
WIDTH, HEIGHT = X0 + RW + M, M + SH + G + ROW_H + M


RGB = tuple[int, int, int]
Font = ImageFont.FreeTypeFont
XY = Sequence[float] | Sequence[Sequence[float]]
"""A point (x, y), a box (x0, y0, x1, y1) or a list of points, in 1x units."""


class Scaled:
    """An ImageDraw proxy: callers draw in 1x coordinates, this draws them k times larger on the big canvas."""

    def __init__(self, d: ImageDraw.ImageDraw, k: int) -> None:
        self.d, self.k, self._fonts = d, k, {}

    def f(self, font: Font | None) -> Font | None:
        if font is None:
            return None
        if id(font) not in self._fonts:
            self._fonts[id(font)] = ImageFont.truetype(font.path, int(round(font.size * self.k)))
        return self._fonts[id(font)]

    def s(self, v: XY | float) -> XY | float:
        if isinstance(v, (list, tuple)):
            return type(v)(self.s(x) for x in v)
        return v * self.k

    def w(self, width: float | None) -> int:
        return max(1, int(round((width or 1) * self.k)))

    def text(self, xy: XY, text: str, font: Font | None = None, fill: RGB | None = None, **kw: object) -> None:
        self.d.text(self.s(xy), text, font=self.f(font), fill=fill, **kw)

    def textlength(self, text: str, font: Font | None = None) -> float:
        return self.d.textlength(text, font=self.f(font)) / self.k

    def textbbox(self, xy: XY, text: str, font: Font | None = None, **kw: object) -> tuple[float, ...]:
        return tuple(v / self.k for v in self.d.textbbox(self.s(xy), text, font=self.f(font), **kw))

    def line(self, xy: XY, fill: RGB | None = None, width: float = 0, joint: str | None = None) -> None:
        self.d.line(self.s(xy), fill=fill, width=self.w(width), joint=joint)

    def ellipse(self, xy: XY, fill: RGB | None = None, outline: RGB | None = None, width: float = 1) -> None:
        self.d.ellipse(self.s(xy), fill=fill, outline=outline, width=self.w(width))

    def polygon(self, xy: XY, fill: RGB | None = None, outline: RGB | None = None, width: float = 1) -> None:
        self.d.polygon(self.s(xy), fill=fill, outline=outline, width=self.w(width))

    def rectangle(self, xy: XY, fill: RGB | None = None, outline: RGB | None = None, width: float = 1) -> None:
        self.d.rectangle(self.s(xy), fill=fill, outline=outline, width=self.w(width))

    def rounded_rectangle(self, xy: XY, radius: float = 0, fill: RGB | None = None, outline: RGB | None = None,
                          width: float = 1) -> None:
        self.d.rounded_rectangle(self.s(xy), radius=radius * self.k, fill=fill, outline=outline, width=self.w(width))


def _font(face: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_DIR / f"IBMPlexSans-{face}.ttf"), size)


F = dict(title=_font("SemiBold", 24), sub=_font("Regular", 15), cap=_font("SemiBold", 13), time=_font("SemiBold", 28),
         stage=_font("Medium", 15), code=_font("SemiBold", 17), step=_font("Medium", 17), small=_font("Regular", 14),
         val=_font("SemiBold", 20), unit=_font("Regular", 14), pill=_font("SemiBold", 14), tag=_font("Medium", 14),
         badge=_font("Bold", 14))


def mix(a: RGB, b: RGB, t: float) -> RGB:
    """Linear blend of two RGB colours."""
    return tuple(int(round(x * (1 - t) + y * t)) for x, y in zip(a, b))


def caps(d: Scaled, xy: tuple[float, float], text: str, fill: RGB = MUTED, font: Font | None = None,
         track: float = 1.2) -> float:
    """Letter-spaced small-caps label; returns the x after it."""
    font = font or F["cap"]
    x, y = xy
    for ch in text.upper():
        d.text((x, y), ch, font=font, fill=fill)
        x += d.textlength(ch, font=font) + track
    return x


def rtext(d: Scaled, x_right: float, y: float, text: str, font: Font, fill: RGB) -> None:
    """Right-aligned text."""
    d.text((x_right - d.textlength(text, font=font), y), text, font=font, fill=fill)


def fit_title(d: Scaled, text: str, width: float) -> Font:
    """The largest title font the text fits in."""
    for size in ("title", "val", "step"):
        if d.textlength(text, font=F[size]) <= width:
            return F[size]
    return F["step"]


def gauge(d: Scaled, x: float, y: float, w: float, name: str, val: float, unit: str, lo: float, hi: float,
          band: tuple[float, float], col: RGB, ticks: tuple[float, ...], active: bool = True) -> float:
    """A labelled horizontal gauge: name and value, a track lo..hi with the target band, tick labels and a marker."""
    d.text((x, y + 4), name, font=F["stage"], fill=FG if active else FAINT)
    vs = f"{val:.1f}" if active and np.isfinite(val) else "–"      # no number before it means anything
    uw = d.textlength(" " + unit, font=F["unit"])
    rtext(d, x + w - uw, y, vs, F["val"], col if active else FAINT)
    d.text((x + w - uw + 4, y + 7), unit, font=F["unit"], fill=MUTED if active else FAINT)
    ty = y + 32
    d.rounded_rectangle([x, ty, x + w, ty + 8], radius=4, fill=CARD)

    def bx(v: float) -> float:
        return x + w * (min(max(v, lo), hi) - lo) / (hi - lo)
    d.rounded_rectangle([bx(band[0]), ty, bx(band[1]), ty + 8], radius=4, fill=OK_DIM if active else mix(CARD, OK_DIM, 0.4))
    for t in ticks:
        lab = f"{t:g}"
        d.text((bx(t) - d.textlength(lab, font=F["small"]) / 2, ty + 11), lab, font=F["small"], fill=MUTED)
    if active and np.isfinite(val):
        px = bx(val)
        if val > hi:                    # past the end: a chevron instead of the knob
            d.polygon([(px - 7, ty - 3), (px + 1, ty + 4), (px - 7, ty + 11)], fill=col)
        else:
            d.ellipse([px - 7, ty - 3, px + 7, ty + 11], fill=col, outline=BG0, width=2)
    return ty + 28


def pills(d: Scaled, x: float, y: float, w: float, items: list[tuple[str, RGB]], font: Font | None = None) -> float:
    """Status pills [(text, colour)], wrapped into rows inside width w; returns the bottom y."""
    font = font or F["pill"]
    cx, cy, hh = x, y, 26
    for text, col in items:
        tw = d.textlength(text, font=font) + 22
        if cx > x and cx + tw > x + w:
            cx, cy = x, cy + hh + 6
        d.rounded_rectangle([cx, cy, cx + tw, cy + hh], radius=hh // 2, fill=mix(PANEL, col, 0.22))
        d.ellipse([cx + 9, cy + 10, cx + 15, cy + 16], fill=col)
        d.text((cx + 19, cy + 4), text, font=font, fill=mix(col, FG, 0.35))
        cx += tw + 6
    return cy + hh


def corner_mask(r: int) -> np.ndarray:
    """Anti-aliased quarter-circle alpha mask (r x r x 1) for rounding a tile's corner."""
    yy, xx = np.mgrid[0:r, 0:r] + 0.5
    return np.clip(r - np.hypot(r - xx, r - yy) + 0.5, 0, 1)[..., None]


def round_corners(cv: np.ndarray, x: int, y: int, w: int, h: int, r: int, mask: np.ndarray) -> None:
    """Blend the four corners of the rectangle (x, y, w, h) of the canvas into the frame ground."""
    b = np.array(BG0, np.float32)
    for sy, sx, m in ((slice(y, y + r), slice(x, x + r), mask), (slice(y, y + r), slice(x + w - r, x + w), mask[:, ::-1]),
                      (slice(y + h - r, y + h), slice(x, x + r), mask[::-1]),
                      (slice(y + h - r, y + h), slice(x + w - r, x + w), mask[::-1, ::-1])):
        cv[sy, sx] = (cv[sy, sx] * m + b * (1 - m)).astype(np.uint8)


def tag(cv: np.ndarray, x: int, y: int, text: str, k: int, font: Font) -> None:
    """A camera name on a translucent pill at the top-left of a tile (canvas pixels, scale k)."""
    tw, th = int(font.getlength(text)) + 18 * k, 24 * k
    sl = (slice(y + 8 * k, y + 8 * k + th), slice(x + 8 * k, x + 8 * k + tw))
    reg = cv[sl].astype(np.float32)
    im = Image.new("L", (tw, th), 0)
    ImageDraw.Draw(im).rounded_rectangle([0, 0, tw - 1, th - 1], radius=th // 2, fill=255)
    a = np.asarray(im, np.float32)[..., None] / 255 * 0.62
    reg = reg * (1 - a) + np.array(BG0, np.float32) * a
    pim = Image.fromarray(reg.astype(np.uint8))
    ImageDraw.Draw(pim).text((9 * k, 3 * k), text, font=font, fill=FG)
    cv[sl] = np.asarray(pim)
