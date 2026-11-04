"""Teaching figures drawn by code (never by an image model) for shots that have no picture of their own.

The old pipeline explained an abstract sentence ("the tangent is perpendicular to the radius", "light bends towards the normal") with
text cards. A teacher draws the figure instead, so this module draws it: circle and tangents, right triangles, angles of elevation,
sequences, dice, refraction, lens ray diagrams, Ohm's-law circuits and graphs, motors ... (chemistry and biology: figures_sci.py).

Rules shared by every figure
- chosen from the WORDS of the shot's narration (choose()), never from a topic name; the part is only drawn when the words name it;
- every label is drawn with the video's own fonts at a known position, so a figure can never carry wrong or garbled lettering;
- the parts the voice is naming right now are lit (the "focus" is the shot's narration) and the figure builds up part by part;
- drawn at twice the size and scaled down, so lines and curves are crisp at 1080p;
- no skin, no body outline: structure, apparatus and symbols only (school-student rule).
"""
from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw

from .typography import font

DW, DH = 1600, 820          # design space of a figure; every drawer works in these units
SS = 2                      # supersampling factor


class Pal:
    """Colours for a light notebook page (mathematics) or the dark teacher's board (other subjects)."""

    def __init__(self, light: bool):
        self.light = light
        if light:
            self.ink, self.muted = (26, 36, 44), (122, 136, 144)
            self.a1, self.a2, self.hot, self.blue = (46, 125, 99), (209, 139, 50), (206, 58, 54), (44, 98, 182)
            self.fill, self.fill2, self.grid = (214, 236, 226), (250, 233, 200), (208, 218, 214)
            self.glass = (190, 222, 245)
        else:
            self.ink, self.muted = (234, 240, 248), (150, 166, 186)
            self.a1, self.a2, self.hot, self.blue = (64, 210, 230), (245, 214, 90), (255, 122, 96), (122, 170, 255)
            self.fill, self.fill2, self.grid = (28, 62, 84), (84, 66, 30), (52, 70, 96)
            self.glass = (40, 84, 128)


class Cv:
    """A supersampled drawing surface in design units. `focus` is the sentence being spoken, `t` the seconds since the shot began."""

    def __init__(self, size: tuple[int, int], pal: Pal, focus: str = "", t: float = 99.0):
        self.size, self.pal, self.focus, self.t = size, pal, (focus or "").lower(), t
        self.k = min(size[0] / DW, size[1] / DH)
        self.s = self.k * SS
        self.img = Image.new("RGBA", (round(DW * self.s), round(DH * self.s)), (0, 0, 0, 0))
        self.d = ImageDraw.Draw(self.img, "RGBA")

    # ---- state ----
    def lit(self, *patterns: str) -> bool:
        return any(re.search(p, self.focus) for p in patterns)

    def shown(self, index: int, step: float = 0.28) -> bool:
        return self.t >= index * step

    def col(self, active: bool, base=None):
        return self.pal.hot if active else (base or self.pal.ink)

    # ---- primitives (design units in, pixels out) ----
    def P(self, p):
        return (p[0] * self.s, p[1] * self.s)

    def line(self, a, b, color=None, w=4, dash: int = 0):
        color = color or self.pal.ink
        if not dash:
            self.d.line((*self.P(a), *self.P(b)), fill=color, width=max(1, round(w * self.s)))
            return
        length = math.dist(a, b) or 1
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        pos = 0.0
        while pos < length:
            end = min(length, pos + dash)
            self.d.line((*self.P((a[0] + ux * pos, a[1] + uy * pos)), *self.P((a[0] + ux * end, a[1] + uy * end))),
                        fill=color, width=max(1, round(w * self.s)))
            pos += dash * 1.9

    def path(self, pts, color=None, w=4):
        color = color or self.pal.ink
        self.d.line([self.P(p) for p in pts], fill=color, width=max(1, round(w * self.s)), joint="curve")

    def arrow(self, a, b, color=None, w=5, head=26, dash: int = 0):
        color = color or self.pal.ink
        length = math.dist(a, b) or 1
        ux, uy = (b[0] - a[0]) / length, (b[1] - a[1]) / length
        base = (b[0] - ux * head * 0.8, b[1] - uy * head * 0.8)
        self.line(a, base, color, w, dash)
        self.d.polygon([self.P(b), self.P((b[0] - ux * head - uy * head * 0.42, b[1] - uy * head + ux * head * 0.42)),
                        self.P((b[0] - ux * head + uy * head * 0.42, b[1] - uy * head - ux * head * 0.42))], fill=color)

    def _soft(self, kind, xy, fill, radius=0) -> bool:
        """A translucent fill is composited onto the figure (drawing it straight onto transparent pixels darkens it). True when handled."""
        if fill is None or len(fill) < 4 or fill[3] >= 255:
            return False
        pts = xy if kind == "poly" else [xy[:2], xy[2:]]
        xs, ys = [q[0] for q in pts], [q[1] for q in pts]
        box = (max(0, int(min(xs)) - 2), max(0, int(min(ys)) - 2), min(self.img.width, int(max(xs)) + 3), min(self.img.height, int(max(ys)) + 3))
        if box[2] <= box[0] or box[3] <= box[1]:
            return True
        layer = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        shift = [(q[0] - box[0], q[1] - box[1]) for q in pts]
        if kind == "ellipse":
            ld.ellipse((*shift[0], *shift[1]), fill=fill)
        elif kind == "rect":
            ld.rectangle((*shift[0], *shift[1]), fill=fill)
        elif kind == "rrect":
            ld.rounded_rectangle((*shift[0], *shift[1]), radius=radius, fill=fill)
        else:
            ld.polygon(shift, fill=fill)
        region = Image.alpha_composite(self.img.crop(box), layer)
        self.img.paste(region, box[:2])
        self.d = ImageDraw.Draw(self.img, "RGBA")
        return True

    def circle(self, c, r, fill=None, outline=None, w=4):
        x, y = self.P(c)
        r *= self.s
        box = (x - r, y - r, x + r, y + r)
        if self._soft("ellipse", box, fill):
            fill = None
        self.d.ellipse(box, fill=fill, outline=outline, width=max(1, round(w * self.s)))

    def ellipse(self, box, fill=None, outline=None, w=4):
        b = (*self.P(box[:2]), *self.P(box[2:]))
        if self._soft("ellipse", b, fill):
            fill = None
        self.d.ellipse(b, fill=fill, outline=outline, width=max(1, round(w * self.s)))

    def rect(self, box, fill=None, outline=None, w=4, radius=0):
        b = (*self.P(box[:2]), *self.P(box[2:]))
        if self._soft("rrect" if radius else "rect", b, fill, radius * self.s):
            fill = None
        if radius:
            self.d.rounded_rectangle(b, radius=radius * self.s, fill=fill, outline=outline, width=max(1, round(w * self.s)))
        else:
            self.d.rectangle(b, fill=fill, outline=outline, width=max(1, round(w * self.s)))

    def poly(self, pts, fill=None, outline=None, w=3):
        pp = [self.P(p) for p in pts]
        if self._soft("poly", pp, fill):
            fill = None
        if fill is not None:
            self.d.polygon(pp, fill=fill)
        if outline:
            self.d.line(pp + [pp[0]], fill=outline, width=max(1, round(w * self.s)), joint="curve")

    def arc(self, c, r, a0, a1, color=None, w=4):
        """Angles in degrees, counter-clockwise as seen on screen (like mathematics), from a0 to a1."""
        x, y = self.P(c)
        r *= self.s
        self.d.arc((x - r, y - r, x + r, y + r), -a1, -a0, fill=color or self.pal.ink, width=max(1, round(w * self.s)))

    def curve(self, p0, p1, bulge, color=None, w=4, arrow=False):
        """A quadratic curve from p0 to p1 bulging by `bulge` units to the left of the direction of travel."""
        mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
        dx, dy = p1[0] - p0[0], p1[1] - p0[1]
        n = math.hypot(dx, dy) or 1
        ctrl = (mx + dy / n * bulge, my - dx / n * bulge)
        pts = [((1 - u) ** 2 * p0[0] + 2 * (1 - u) * u * ctrl[0] + u * u * p1[0],
                (1 - u) ** 2 * p0[1] + 2 * (1 - u) * u * ctrl[1] + u * u * p1[1]) for u in [i / 28 for i in range(29)]]
        if arrow:
            self.path(pts[:-2], color, w)
            self.arrow(pts[-3], pts[-1], color, w, head=22)
        else:
            self.path(pts, color, w)

    def right_angle(self, vertex, to_a, to_b, size=30, color=None, w=4):
        """The square mark of a right angle at `vertex` between the directions towards points to_a and to_b."""
        ua = ((to_a[0] - vertex[0]), (to_a[1] - vertex[1]))
        ub = ((to_b[0] - vertex[0]), (to_b[1] - vertex[1]))
        na, nb = math.hypot(*ua) or 1, math.hypot(*ub) or 1
        ua, ub = (ua[0] / na * size, ua[1] / na * size), (ub[0] / nb * size, ub[1] / nb * size)
        corner = (vertex[0] + ua[0] + ub[0], vertex[1] + ua[1] + ub[1])
        self.path([(vertex[0] + ua[0], vertex[1] + ua[1]), corner, (vertex[0] + ub[0], vertex[1] + ub[1])], color or self.pal.ink, w)

    def text(self, p, s, size=34, color=None, anchor="mm", bold=True):
        self.d.text(self.P(p), s, font=font(max(8, round(size * self.s)), bold), fill=color or self.pal.ink, anchor=anchor)

    def dot(self, p, r=9, color=None):
        self.circle(p, r, fill=color or self.pal.ink)

    def result(self) -> Image.Image:
        out = self.img.resize((round(DW * self.k), round(DH * self.k)), Image.LANCZOS)
        canvas = Image.new("RGBA", self.size, (0, 0, 0, 0))
        canvas.paste(out, ((self.size[0] - out.width) // 2, (self.size[1] - out.height) // 2), out)
        return canvas


# ============================================================================== mathematics

def _tangent_circle(c: Cv):
    p = c.pal
    O, r = (470, 420), 250
    P_ = (1180, 420)
    phi = math.acos(r / math.dist(O, P_))
    A = (O[0] + r * math.cos(phi), O[1] - r * math.sin(phi))
    B = (O[0] + r * math.cos(phi), O[1] + r * math.sin(phi))
    two = c.lit(r"two tangents|both tangents|pair of tangents|external point|outside (the|a) circle|\bpb\b|from a point|"
                r"equal|same length|lengths? of (the )?tangent")
    hot_radius, hot_tan = c.lit(r"radius|radii|centre|center|perpendicular|right angle|90"), c.lit(r"tangent")
    hot_right = c.lit(r"right angle|perpendicular|90|at right|angle between")
    hot_equal = c.lit(r"equal|same length|\bpa\b|\bpb\b|congruent")
    hot_tri = c.lit(r"triangle|pythag|hypotenuse|\bop\b|square")
    hot_touch = c.lit(r"touch|point of contact|single point|exactly one|only one|meets? the circle")
    if c.shown(0):
        c.circle(O, r, fill=(*p.fill, 120), outline=p.a1, w=6)
    if hot_tri and two:
        c.poly([O, A, P_], fill=(*p.a2, 70))
    if c.shown(1):
        c.dot(O, 8)
        c.text((O[0] - 30, O[1] + 22), "O", 38)
    if c.shown(2):
        c.line(O, A, p.hot if hot_radius else p.ink, 7 if hot_radius else 5)
        c.text(((O[0] + A[0]) / 2 - 24, (O[1] + A[1]) / 2 - 6), "r", 34, p.hot if hot_radius else p.ink)
        if two:
            c.line(O, B, p.ink, 5)
            c.text(((O[0] + B[0]) / 2 - 24, (O[1] + B[1]) / 2 + 12), "r", 34)
    if c.shown(3):
        col = p.hot if hot_tan else p.a2
        if two:
            c.line(P_, A, col, 7 if hot_tan else 6)
            c.line(P_, B, col, 7 if hot_tan else 6)
        else:
            ux, uy = (A[0] - P_[0]) / math.dist(A, P_), (A[1] - P_[1]) / math.dist(A, P_)
            c.line((A[0] - ux * 260, A[1] - uy * 260), (A[0] + ux * 400, A[1] + uy * 400), col, 7 if hot_tan else 6)
            c.text((A[0] + ux * 300 + 30, A[1] + uy * 300 - 40), "tangent", 32, col)
    if c.shown(4):
        c.dot(A, 12 if hot_touch else 9, p.hot if hot_touch else p.ink)
        c.text((A[0] - 22, A[1] - 34), "A", 38)
        c.right_angle(A, O, P_, 34, p.hot if hot_right else p.ink, 5 if hot_right else 4)
        if two:
            c.dot(B, 9)
            c.text((B[0] - 22, B[1] + 38), "B", 38)
            c.right_angle(B, O, P_, 34, p.hot if hot_right else p.ink, 5 if hot_right else 4)
    if two and c.shown(5):
        c.dot(P_, 10)
        c.text((P_[0] + 34, P_[1]), "P", 38)
        if hot_tri:
            c.line(O, P_, p.hot, 5, dash=14)
            c.text((820, 470), "OP", 32, p.hot)
        for end in (A, B):                                    # equal-length ticks on the two tangent segments
            mx, my = (P_[0] + end[0]) / 2, (P_[1] + end[1]) / 2
            ux, uy = (end[0] - P_[0]), (end[1] - P_[1])
            n = math.hypot(ux, uy)
            nx, ny = -uy / n, ux / n
            for off in (-8, 8):
                c.line((mx + off * ux / n + nx * 16, my + off * uy / n + ny * 16), (mx + off * ux / n - nx * 16, my + off * uy / n - ny * 16),
                       p.hot if hot_equal else p.ink, 5)
        if hot_equal:
            c.text((1010, 640), "PA = PB", 54, p.hot)


def _right_triangle(c: Cv):
    p = c.pal
    A, C, B = (150, 690), (860, 690), (860, 190)
    ang = math.degrees(math.atan2(A[1] - B[1], B[0] - A[0]))
    hot_h, hot_o, hot_a = c.lit(r"hypotenuse"), c.lit(r"opposite|perpendicular"), c.lit(r"adjacent|base")
    if c.shown(0):
        c.poly([A, C, B], fill=(*p.fill, 150))
        c.line(A, C, p.hot if hot_a else p.a1, 8 if hot_a else 6)
        c.line(C, B, p.hot if hot_o else p.a1, 8 if hot_o else 6)
        c.line(A, B, p.hot if hot_h else p.a1, 8 if hot_h else 6)
    if c.shown(1):
        c.right_angle(C, A, B, 40, p.ink, 4)
        c.arc(A, 130, 0, ang, p.a2, 6)
        c.text((A[0] + 175, A[1] - 28), "θ", 52, p.a2)
    if c.shown(2):
        c.text(((A[0] + B[0]) / 2 - 90, (A[1] + B[1]) / 2 - 40), "hypotenuse", 34, p.hot if hot_h else p.ink)
        c.text((B[0] + 20, (B[1] + C[1]) / 2), "opposite", 34, p.hot if hot_o else p.ink, anchor="lm")
        c.text(((A[0] + C[0]) / 2, A[1] + 48), "adjacent", 34, p.hot if hot_a else p.ink)
    ratios = [("sin θ", "opposite", "hypotenuse", r"\bsin\b|sine"), ("cos θ", "adjacent", "hypotenuse", r"\bcos\b|cosine"),
              ("tan θ", "opposite", "adjacent", r"\btan\b|tangent")]
    named = [rt for rt in ratios if c.lit(rt[3])]
    if not named and c.lit(r"trigonometric ratio|ratios"):
        named = ratios
    for i, (name, num, den, _pat) in enumerate(named):
        if c.shown(3 + i):
            y = 300 + i * 150
            c.text((1010, y), name, 46, p.hot, anchor="lm")
            c.text((1160, y), "=", 46, p.ink, anchor="lm")
            c.text((1370, y - 26), num, 34, p.ink)
            c.line((1250, y), (1490, y), p.ink, 4)
            c.text((1370, y + 28), den, 34, p.ink)


def _elevation(c: Cv):
    p = c.pal
    ground = 700
    down = c.lit(r"depression")
    c.rect((60, ground, 1540, ground + 12), fill=p.muted)
    tower_x, top = (1180, 290) if not down else (250, 290)
    c.rect((tower_x - 45, top, tower_x + 45, ground), fill=(*p.fill, 255), outline=p.a1, w=5)
    for i in range(1, 5):                                              # windows: it is a building or a tower
        c.rect((tower_x - 22, top + i * 70 - 20, tower_x + 22, top + i * 70 + 12), fill=(*p.a2, 120))
    if not down:
        eye = (260, 590)
        target = (tower_x, top)
        horiz_end = (tower_x, eye[1])
    else:
        eye = (tower_x, top - 100)
        target = (1180, ground - 8)
        horiz_end = (1180, eye[1])
    # the observer: a stick figure
    base = (eye[0], eye[1] + 110) if not down else (eye[0], top)
    c.circle((eye[0], eye[1] - 8), 22, outline=p.ink, w=5)
    c.line((eye[0], eye[1] + 14), (eye[0], base[1] - 40), p.ink, 6)
    c.line((eye[0], base[1] - 40), (eye[0] - 24, base[1]), p.ink, 6)
    c.line((eye[0], base[1] - 40), (eye[0] + 24, base[1]), p.ink, 6)
    c.line((eye[0], eye[1] + 40), (eye[0] - 30, eye[1] + 76), p.ink, 6)
    c.line((eye[0], eye[1] + 40), (eye[0] + 30, eye[1] + 76), p.ink, 6)
    if down:
        c.circle(target, 18, fill=p.a2)
    sight = c.lit(r"line of sight|sight|looking|view")
    ang = math.degrees(math.atan2(abs(target[1] - eye[1]), abs(target[0] - eye[0])))
    if c.shown(1):
        c.line(eye, horiz_end, p.muted, 4, dash=16)
    if c.shown(2):
        c.line(eye, target, p.hot if sight else p.a1, 8 if sight else 6)
    hot_angle = c.lit(r"angle|elevation|depression")
    if c.shown(3):
        c.arc(eye, 150, -ang if down else 0, 0 if down else ang, p.hot if hot_angle else p.a2, 7 if hot_angle else 6)
        c.text((eye[0] + 215, eye[1] + (34 if down else -30)), "α", 54, p.hot if hot_angle else p.a2)
        c.text((eye[0] + 300, eye[1] + (96 if down else -80)), "angle of depression" if down else "angle of elevation", 30,
               p.hot if hot_angle else p.ink, anchor="lm")
    if c.shown(4):
        hot_h = c.lit(r"height|tall|top of")
        hx = tower_x + (80 if not down else -80)
        c.arrow((hx, ground - 6), (hx, top + 4), p.hot if hot_h else p.ink, 4, head=20)
        c.arrow((hx, top + 4), (hx, ground - 6), p.hot if hot_h else p.ink, 4, head=20)
        c.text((hx + (24 if not down else -24), (top + ground) / 2), "h", 46, p.hot if hot_h else p.ink, anchor="lm" if not down else "rm")
    if c.shown(5):
        hot_d = c.lit(r"distance|away|foot|base")
        y = ground + 56
        c.arrow((eye[0] + 20, y), (tower_x - 60 if not down else 1180, y), p.hot if hot_d else p.ink, 4, head=20)
        c.arrow(((tower_x - 60) if not down else 1180, y), (eye[0] + 20, y), p.hot if hot_d else p.ink, 4, head=20)
        c.text(((eye[0] + (tower_x if not down else 1180)) / 2, y + 36), "distance d", 34, p.hot if hot_d else p.ink)


_NUM_SEQ = re.compile(r"(-?\d+(?:\.\d+)?)(?:\s*,\s*(-?\d+(?:\.\d+)?)){2,}")


def _sequence_terms(text: str) -> list[float] | None:
    m = re.search(r"(-?\d+(?:\.\d+)?(?:\s*,\s*-?\d+(?:\.\d+)?){2,})", text or "")
    if not m:
        return None
    return [float(x) for x in re.split(r"\s*,\s*", m.group(1))][:6]


def _ap_sequence(c: Cv):
    p = c.pal
    numbers = _sequence_terms(c.focus)
    fmt = lambda v: str(int(v)) if float(v).is_integer() else f"{v:g}"
    diffs = {round(b - a, 6) for a, b in zip(numbers or [], (numbers or [])[1:])}
    if numbers and len(diffs) == 1:
        d = diffs.pop()
        boxes = [fmt(v) for v in numbers] + ["…"]
        step = ("+" if d >= 0 else "−") + fmt(abs(d))
    else:
        boxes = ["a", "a + d", "a + 2d", "a + 3d", "…", "a + (n − 1)d"]
        step = "+ d"
    n = len(boxes)
    w, gap = 190, 56
    x0 = (DW - (n * w + (n - 1) * gap)) / 2
    y0, h = 400, 130
    hot_d = c.lit(r"common difference|difference|increase|decrease|add|subtract|constant")
    for i, label in enumerate(boxes):
        if not c.shown(i, 0.22):
            break
        bx = x0 + i * (w + gap)
        dots = label == "…"
        if not dots:
            c.rect((bx, y0, bx + w, y0 + h), fill=(*p.fill, 255), outline=p.a1, w=5, radius=22)
        c.text((bx + w / 2, y0 + h / 2), label, 46 if len(label) < 7 else 36, p.ink)
        if i < n - 1 and not dots and boxes[i + 1] != "…":
            c.curve((bx + w * 0.7, y0 - 6), (bx + w + gap + w * 0.3, y0 - 6), 70, p.hot if hot_d else p.a2, 6, arrow=True)
            c.text((bx + w + gap / 2, y0 - 108), step, 40, p.hot if hot_d else p.a2)
    if c.shown(n):
        first = c.lit(r"first term|starting|begins|initial|\ba\b")
        c.text((x0 + w / 2, y0 + h + 52), "first term", 32, p.hot if first else p.muted)
        if c.lit(r"nth term|last term|general term") and boxes[-1] != "…":
            c.text((x0 + (n - 1) * (w + gap) + w / 2, y0 + h + 52), "nth term", 32, p.hot)


_DICE = {1: [(0, 0)], 2: [(-1, -1), (1, 1)], 3: [(-1, -1), (0, 0), (1, 1)], 4: [(-1, -1), (1, -1), (-1, 1), (1, 1)],
         5: [(-1, -1), (1, -1), (0, 0), (-1, 1), (1, 1)], 6: [(-1, -1), (1, -1), (-1, 0), (1, 0), (-1, 1), (1, 1)]}


def _dice_event(focus: str) -> set[int]:
    fav: set[int] = set()
    if re.search(r"\beven\b", focus):
        fav |= {2, 4, 6}
    if re.search(r"\bodd\b", focus):
        fav |= {1, 3, 5}
    if re.search(r"\bprime\b", focus):
        fav |= {2, 3, 5}
    m = re.search(r"multiples? of (\d)", focus)
    if m:
        fav |= {n for n in range(1, 7) if n % int(m.group(1)) == 0}
    m = re.search(r"(?:greater|more|larger) than (\d)", focus)
    if m:
        fav |= {n for n in range(1, 7) if n > int(m.group(1))}
    m = re.search(r"(?:less|smaller|fewer) than (\d)", focus)
    if m:
        fav |= {n for n in range(1, 7) if n < int(m.group(1))}
    if not fav:
        m = re.search(r"(?:getting|get|rolling|roll|shows?|showing|number|an?|appear\w*)\s+(?:an?\s+)?([1-6])\b", focus)
        if m:
            fav.add(int(m.group(1)))
    return fav


def _probability(c: Cv):
    p = c.pal
    focus = c.focus
    if re.search(r"\bdic?e\b|\bdie\b", focus):
        fav = _dice_event(focus)
        size, gap = 190, 40
        x0 = (DW - (6 * size + 5 * gap)) / 2
        for i in range(6):
            if not c.shown(i, 0.2):
                break
            bx = x0 + i * (size + gap)
            on = (i + 1) in fav
            c.rect((bx, 300, bx + size, 300 + size), fill=(*p.fill, 255) if not on else (*p.a1, 255) if p.light else (*p.a1, 90),
                   outline=p.hot if on else p.a1, w=7 if on else 4, radius=28)
            for dx, dy in _DICE[i + 1]:
                c.circle((bx + size / 2 + dx * 46, 300 + size / 2 + dy * 46), 15, fill=(255, 255, 255) if on and p.light else p.ink)
            c.text((bx + size / 2, 300 + size + 44), str(i + 1), 34, p.hot if on else p.muted)
        if c.shown(7):
            if re.search(r"sample space|all possible|total number of outcomes|outcomes", focus):
                c.path([(x0, 590), (x0, 610), (x0 + 6 * size + 5 * gap, 610), (x0 + 6 * size + 5 * gap, 590)], p.muted, 4)
                c.text((DW / 2, 656), "all possible outcomes (sample space)", 34, p.ink)
            if fav and re.search(r"favou?rable|event|chance|probability", focus):
                c.text((DW / 2, 150), "favourable outcomes are lit", 38, p.hot)
        return
    if re.search(r"\bcoins?\b|\bheads?\b|\btails?\b", focus):
        for i, (name, col) in enumerate((("H", p.a2), ("T", p.a1))):
            if c.shown(i, 0.4):
                cx = 520 + i * 560
                c.circle((cx, 400), 170, fill=(*col, 70), outline=col, w=9)
                c.circle((cx, 400), 138, outline=col, w=3)
                c.text((cx, 392), name, 120, col)
                c.text((cx, 640), "heads" if name == "H" else "tails", 38, p.ink)
        if c.shown(3) and re.search(r"equally likely|fair|equal chance|two possible", focus):
            c.text((DW / 2, 150), "two equally likely outcomes", 40, p.hot)
        return
    # the probability scale from impossible (0) to certain (1)
    y, x0, x1 = 420, 160, 1440
    if c.shown(0):
        c.arrow((x0 - 20, y), (x1 + 30, y), p.ink, 7, head=30)
    marks = [(0.0, "0", "impossible event", r"impossible|\b0\b|zero"), (0.5, "½", "equally likely", r"equally likely|half|even chance|\b1/2\b"),
             (1.0, "1", "certain event", r"certain|sure|\b1\b|always")]
    for i, (v, num, name, pat) in enumerate(marks):
        if c.shown(1 + i, 0.4):
            hot = c.lit(pat)
            x = x0 + v * (x1 - x0)
            c.line((x, y - 24), (x, y + 24), p.hot if hot else p.ink, 6)
            c.text((x, y + 70), num, 52, p.hot if hot else p.ink)
            if hot or c.lit(r"between 0 and 1|scale|range"):
                c.text((x, y - 76), name, 34, p.hot if hot else p.muted)
    if c.shown(4):
        c.text((DW / 2, 200), "P(E)", 70, p.a2)
        if c.lit(r"between 0 and 1|never (less|more)|cannot (be|exceed)|lies between"):
            c.text((DW / 2, 290), "0 ≤ P(E) ≤ 1", 58, p.hot)


# ============================================================================== physics

def _refraction(c: Cv):
    p = c.pal
    away = c.lit(r"away from (the )?normal|denser to rarer|glass to air|water to air|rarer medium|speeds? up")
    i_deg, r_deg = (28, 50) if away else (50, 30)
    O, L = (800, 420), 330
    if c.shown(0):
        c.rect((60, 420, 1540, 780), fill=(*p.glass, 150))
        c.line((60, 420), (1540, 420), p.muted, 5)
        c.text((140, 760), "denser medium" if not away else "denser medium", 30, p.muted, anchor="lm")
        c.text((140, 80), "rarer medium", 30, p.muted, anchor="lm")
    if c.shown(1):
        c.line((800, 90), (800, 750), p.muted, 4, dash=16)
        c.text((830, 120), "normal", 30, p.ink, anchor="lm")
    si, ci = math.sin(math.radians(i_deg)), math.cos(math.radians(i_deg))
    sr, cr = math.sin(math.radians(r_deg)), math.cos(math.radians(r_deg))
    hot_i, hot_r = c.lit(r"incident|incidence|angle i\b"), c.lit(r"refracted|refraction|angle r\b|bend")
    if not away:
        start, end = (O[0] - L * si, O[1] - L * ci), (O[0] + L * sr, O[1] + L * cr)
    else:
        start, end = (O[0] - L * si, O[1] + L * ci), (O[0] + L * sr, O[1] - L * cr)
    if c.shown(2):
        c.arrow(start, O, p.hot if hot_i else p.a2, 8, head=34)
        c.text((start[0] - 20, start[1] - (30 if not away else -32)), "incident ray", 30, p.hot if hot_i else p.ink, anchor="rm")
    if c.shown(3):
        c.arrow(O, end, p.hot if hot_r else p.a1, 8, head=34)
        c.text((end[0] + 24, end[1] + (22 if not away else -26)), "refracted ray", 30, p.hot if hot_r else p.ink, anchor="lm")
    if c.shown(4):
        up_i = 90 if not away else -90
        c.arc(O, 120, 90 + 0 if not away else 270 - i_deg, 90 + i_deg if not away else 270, p.hot if hot_i else p.ink, 5)
        c.text((O[0] - 150 * math.sin(math.radians(i_deg / 2)), O[1] + (-150 if not away else 150) * math.cos(math.radians(i_deg / 2))),
               "i", 40, p.hot if hot_i else p.ink)
        c.arc(O, 160, 270 + 0 if not away else 90 - r_deg, 270 + r_deg if not away else 90, p.hot if hot_r else p.ink, 5)
        c.text((O[0] + 190 * math.sin(math.radians(r_deg / 2)), O[1] + (190 if not away else -190) * math.cos(math.radians(r_deg / 2))),
               "r", 40, p.hot if hot_r else p.ink)
    if c.shown(5) and c.lit(r"snell|refractive index|constant|sin"):
        c.text((1150, 600), "sin i / sin r = constant", 44, p.hot)


def _thin_lens(c: Cv, f: float, u: float, h: float = 130):
    """Principal rays of a thin lens at (800, 420): object at distance u (positive, to the left) of height h, focal length f (signed)."""
    p = c.pal
    x0, y0 = 800, 420
    top = (x0 - u, y0 - h)
    q = (x0, y0 - h)
    f2 = (x0 + f, y0)
    # the ray that arrives parallel to the axis leaves the lens through (or away from) the far focus; the ray through the centre goes straight
    d1 = (q[0] - f2[0], q[1] - f2[1]) if f < 0 else (f2[0] - q[0], f2[1] - q[1])
    d2 = (x0 - top[0], y0 - top[1])
    # intersection of line q + a d1 with line top + b d2
    det = d1[0] * -d2[1] - d1[1] * -d2[0]
    if abs(det) < 1e-9:
        return None
    a = ((top[0] - q[0]) * -d2[1] - (top[1] - q[1]) * -d2[0]) / det
    image = (q[0] + a * d1[0], q[1] + a * d1[1])
    real = image[0] > x0 + 1
    return top, q, d1, d2, image, real


def _lens_rays(c: Cv):
    p = c.pal
    concave = c.lit(r"concave|diverging")
    virtual = not concave and c.lit(r"virtual|erect|upright|magnif|between (the )?(optical centre|lens|focus)|closer than")
    f, u = (-250.0, 500.0) if concave else ((250.0, 150.0) if virtual else (250.0, 520.0))
    x0, y0 = 800, 420
    c.line((60, y0), (1540, y0), p.muted, 4)                       # principal axis
    if c.shown(0):                                                  # the lens
        if concave:
            c.path([(770, 190), (800, 215), (806, 420), (800, 625), (770, 650)], p.a1, 7)
            c.path([(830, 190), (800, 215)], p.a1, 7)
            c.path([(830, 650), (800, 625)], p.a1, 7)
            c.poly([(770, 190), (830, 190), (806, 420), (830, 650), (770, 650), (794, 420)], fill=(*p.glass, 140))
        else:
            c.ellipse((762, 180, 838, 660), fill=(*p.glass, 150), outline=p.a1, w=6)
    for i, (x, name) in enumerate(((x0 - abs(f), "F"), (x0 + abs(f), "F"), (x0 - 2 * abs(f), "2F"), (x0 + 2 * abs(f), "2F"))):
        if c.shown(1):
            c.line((x, y0 - 14), (x, y0 + 14), p.ink, 4)
            c.text((x, y0 + 46), name, 30, p.hot if c.lit(r"focus|focal|\bf\b") and name == "F" else p.ink)
    c.text((x0, y0 + 46), "O", 30, p.ink)
    geo = _thin_lens(c, f, u)
    if not geo:
        return
    top, q, d1, d2, image, real = geo
    h = 130
    hot_obj, hot_img, hot_ray = c.lit(r"object"), c.lit(r"image"), c.lit(r"ray|rays|refract|bend|converge|diverge|parallel")
    if c.shown(2):
        c.arrow((top[0], y0), top, p.a2, 7, head=26)
        c.text((top[0], y0 + 96), "object", 28, p.hot if hot_obj else p.ink)
    n1 = math.hypot(*d1)
    n2 = math.hypot(*d2)
    far = 700
    if c.shown(3):
        c.line(top, q, p.hot if hot_ray else p.a1, 5)
        c.line(top, (x0 + d2[0] / n2 * far, y0 + d2[1] / n2 * far), p.hot if hot_ray else p.a1, 5)
        sign = 1 if (f > 0 and real) or f < 0 else 1
        end = (q[0] + d1[0] / n1 * far * sign, q[1] + d1[1] / n1 * far * sign) if f > 0 else (q[0] + d1[0] / n1 * far, q[1] + d1[1] / n1 * far)
        c.line(q, end, p.hot if hot_ray else p.a1, 5)
        if not real:                                                # a virtual image: the rays only seem to come from it
            back = (q[0] - d1[0] / n1 * 900, q[1] - d1[1] / n1 * 900) if f > 0 else (f and (x0 + f, y0))
            c.line(image, q if f > 0 else (x0 + f, y0), p.muted, 3, dash=12)
            c.line(image, (x0, y0), p.muted, 3, dash=12)
    if c.shown(4):
        if image[1] > y0 - 2 and real:
            c.arrow((image[0], y0), image, p.hot if hot_img else p.a2, 7, head=26)
        elif real:
            c.arrow((image[0], y0), image, p.hot if hot_img else p.a2, 7, head=26)
        else:
            c.arrow((image[0], y0), image, p.hot if hot_img else p.a2, 6, head=24, dash=10)
        c.text((image[0], image[1] + (46 if image[1] > y0 else -34)), "image", 28, p.hot if hot_img else p.ink)


def _ohm(c: Cv):
    p = c.pal
    if c.lit(r"graph|slope|straight line|plot|proportional|directly|linear"):
        o = (300, 700)
        c.arrow((o[0] - 30, o[1]), (1400, o[1]), p.ink, 6, head=28)
        c.arrow((o[0], o[1] + 30), (o[0], 110), p.ink, 6, head=28)
        c.text((1430, o[1] + 6), "I", 46, p.ink, anchor="lm")
        c.text((o[0], 80), "V", 46, p.ink)
        if c.shown(1):
            hot = c.lit(r"straight|proportional|directly|linear|slope|resistance")
            c.arrow(o, (1260, 190), p.hot if hot else p.a1, 9, head=34)
            c.dot(o, 9)
            c.text((o[0] - 26, o[1] + 34), "0", 32, p.muted)
            c.text((1180, 520), "slope = R" if c.lit(r"slope|resistance") else "V = IR", 48, p.hot if hot else p.ink)
        return
    TL, TR, BR, BL = (300, 200), (1250, 200), (1250, 640), (300, 640)
    hot_cell, hot_r = c.lit(r"cell|battery|source|emf"), c.lit(r"resistor|resistance|\br\b|ohm")
    hot_a, hot_v = c.lit(r"ammeter|current"), c.lit(r"voltmeter|potential difference|voltage")
    wire = p.ink
    if c.shown(0):
        c.path([(300, 400), TL, (590, 200)], wire, 6)
        c.path([(700, 200), (940, 200)], wire, 6)
        c.path([(1060, 200), TR, (1250, 340)], wire, 6)
        c.path([(1250, 520), BR, BL, (300, 470)], wire, 6)
        c.line((270, 400), (330, 400), p.hot if hot_cell else wire, 9)              # the long plate (+)
        c.line((285, 470), (315, 470), p.hot if hot_cell else wire, 9)              # the short plate (-)
        c.text((236, 398), "+", 32, wire)
        c.text((236, 472), "−", 32, wire)
        c.text((196, 436), "cell", 30, p.hot if hot_cell else p.muted, anchor="rm")
    if c.shown(1):
        c.circle((645, 200), 52, fill=(*p.fill, 255), outline=p.hot if hot_a else p.a1, w=6)
        c.text((645, 198), "A", 52, p.hot if hot_a else p.ink)
        c.text((645, 140), "ammeter", 28, p.hot if hot_a else p.muted)
        c.line((940, 200), (1030, 178), wire, 6)                                   # the switch
        c.dot((940, 200), 8)
        c.dot((1060, 200), 8)
        c.text((1000, 150), "switch", 28, p.muted)
    if c.shown(2):
        c.rect((1210, 340, 1290, 520), fill=(*p.fill2, 255), outline=p.hot if hot_r else p.a2, w=6)
        c.text((1250, 430), "R", 52, p.hot if hot_r else p.ink)
        c.text((1330, 430), "resistor", 28, p.hot if hot_r else p.muted, anchor="lm")
    if c.shown(3):
        c.path([(1210, 340), (1120, 340), (1120, 400)], wire, 5)
        c.path([(1120, 470), (1120, 520), (1210, 520)], wire, 5)
        c.circle((1120, 435), 40, fill=(*p.fill, 255), outline=p.hot if hot_v else p.blue, w=6)
        c.text((1120, 433), "V", 46, p.hot if hot_v else p.ink)
        c.text((1120, 580), "voltmeter", 28, p.hot if hot_v else p.muted)
    if c.shown(4) and c.lit(r"current|flow"):
        c.arrow((820, 640), (560, 640), p.a2, 6, head=26)
        c.text((690, 686), "current I", 32, p.a2)


def _motor(c: Cv):
    p = c.pal
    gen = c.lit(r"generator|induced|induction|dynamo|emf|slip ring")
    hot_field, hot_coil = c.lit(r"field|magnet|pole"), c.lit(r"coil|armature|conductor|wire|rotat|loop")
    if c.shown(0):
        c.rect((60, 190, 280, 650), fill=(*p.hot, 200), outline=p.hot, w=4, radius=14)
        c.text((170, 420), "N", 110, (255, 255, 255))
        c.rect((1320, 190, 1540, 650), fill=(*p.blue, 220), outline=p.blue, w=4, radius=14)
        c.text((1430, 420), "S", 110, (255, 255, 255))
    if c.shown(1):
        for y in (250, 335, 420, 505, 590):
            c.arrow((300, y), (1300, y), p.hot if hot_field else p.muted, 4, head=22)
        c.text((800, 150), "magnetic field", 32, p.hot if hot_field else p.muted)
    if c.shown(2):
        c.rect((700, 290, 900, 550), outline=p.hot if hot_coil else p.a1, w=8, radius=10)
        c.circle((700, 420), 30, outline=p.ink, w=5)
        c.dot((700, 420), 8)                                         # current out of the page
        c.circle((900, 420), 30, outline=p.ink, w=5)
        c.line((886, 406), (914, 434), p.ink, 5)                      # current into the page
        c.line((886, 434), (914, 406), p.ink, 5)
        c.text((800, 610), "coil", 34, p.hot if hot_coil else p.ink)
    if c.shown(3):
        if gen:
            c.curve((660, 330), (940, 330), -110, p.a2, 7, arrow=True)
            c.text((800, 705), "the coil is turned: a current is induced", 32, p.a2)
        else:
            c.arrow((700, 380), (700, 250), p.a2, 8, head=30)
            c.arrow((900, 460), (900, 590), p.a2, 8, head=30)
            c.text((640, 300), "F", 40, p.a2)
            c.text((960, 540), "F", 40, p.a2)
            c.text((800, 705), "forces on the two sides turn the coil", 32, p.a2)
            c.curve((730, 235), (870, 235), -70, p.hot, 6, arrow=True)


# ============================================================================== registry

# name -> (strong words, weak words, drawer, subjects it applies to or None for any)
FIGURES: dict = {
    "tangent_circle": (r"\btangents?\b|point of contact", r"\bcircles?\b|\bradius\b|\bradii\b", _tangent_circle, {"maths"}),
    "right_triangle": (r"\bhypotenuse\b|\badjacent\b|\bopposite (?:side|to)\b|\bsine\b|\bcosine\b|\bsin\b|\bcos\b|trigonometric ratios?",
                       r"\btriangle\b|\bangle\b|\bratio", _right_triangle, {"maths"}),
    "elevation": (r"angle of (?:elevation|depression)|line of sight|\btower\b|\bflag ?pole\b|\bkite\b|\bminaret\b|\bladder\b|\blighthouse\b",
                  r"\bheight\b|\bdistance\b|\bangle\b", _elevation, {"maths"}),
    "ap_sequence": (r"arithmetic (?:progression|sequence)|common difference|\bfirst term\b|\bnth term\b|\bAP\b|\d\s*,\s*\d+\s*,\s*\d+",
                    r"\bterms?\b|\bsequence\b|\bsum\b", _ap_sequence, {"maths"}),
    "probability": (r"\bprobabilit\w*|\bdic?e\b|\bcoins?\b|sample space|favou?rable|equally likely|\boutcomes?\b",
                    r"\bevent\b|\bchance\b|\brandom", _probability, {"maths"}),
    "refraction": (r"refract\w*|snell|angle of incidence|angle of refraction|refractive index|\bnormal\b",
                   r"\bray\b|\bmedium\b|\bbends?\b|\bspeed of light", _refraction, {"physics"}),
    "lens_rays": (r"\blens(?:es)?\b|focal length|principal axis|optical cent(?:re|er)|real image|virtual image",
                  r"\bimage\b|\bfocus\b|\bfocal\b|\brays?\b|magnif\w*", _lens_rays, {"physics"}),
    "ohm": (r"ohm'?s law|\bresistors?\b|\bammeter\b|\bvoltmeter\b|potential difference|\bresistance\b",
            r"\bcurrent\b|\bvoltage\b|\bcircuit\b|\bcell\b|\bbattery\b", _ohm, {"physics"}),
    "motor": (r"electric motor|\bmotors?\b|\bgenerators?\b|armature|commutator|left[- ]hand rule|electromagnetic induction|slip rings?|\bdynamo\b",
              r"\bcoil\b|magnetic field|\bmagnets?\b|induced", _motor, {"physics"}),
}


def concept_for(narration: str, context: str, subject: str) -> str | None:
    """The figure that explains this sentence, or None. A figure needs one strong word (or two weak ones) in the narration itself."""
    from . import figures_sci
    text = narration or ""
    low = f"{text} {context or ''}".lower()
    best, score = None, 0
    for name, (strong, weak, _drawer, subjects) in {**FIGURES, **figures_sci.FIGURES}.items():
        subj = (subject or "").strip().lower()
        if subjects and subj:
            subjs_low = {s.lower() for s in subjects}
            if subj not in subjs_low and subj != "general":
                if not any(s in subj or subj in s for s in subjs_low):
                    continue
        s, w = len(re.findall(strong, text, re.I)), len(re.findall(weak, text, re.I))
        if name == "tangent_circle" and "circle" not in low:
            continue
        if name == "right_triangle" and re.search(r"\bcircle|\btangents? (?:to|of|from)", low):
            continue
        if name == "ohm" and not re.search(r"ohm|resist|circuit|ammeter|voltmeter|potential difference", low):
            continue
        value = s * 3 + w if (s or w >= 2) else 0
        if value > score:
            best, score = name, value
    return best


def render(concept: str, size: tuple[int, int], t: float, focus: str, light: bool) -> Image.Image | None:
    entry = FIGURES.get(concept)
    if entry is None:
        from . import figures_sci
        entry = figures_sci.FIGURES.get(concept)
    if entry is None:
        return None
    canvas = Cv(size, Pal(light), focus, t)
    entry[2](canvas)
    return canvas.result()


REVEAL_SECONDS = 4.0          # every figure has finished building by then (Cv.shown steps are 0.28 s)
FIGURE_SECONDS = 5.5          # shortest time a figure stays up so the viewer can read it


def frame(size: tuple[int, int], config, concept: str, heading: str, t: float, focus: str, light: bool) -> Image.Image:
    """A complete 1080p page: notebook paper (maths) or the dark teacher board, a heading, and the figure below it above the subtitle band."""
    from .typography import draw_lines, fit_text, hex_rgb
    width, height = size
    s = height / 1080
    if light:
        image = Image.new("RGB", size, hex_rgb(config.render.background))
        ink, accent = hex_rgb(config.render.ink), hex_rgb(config.render.accent)
    else:
        from . import board as teacher_board
        image = teacher_board._background(size, teacher_board.theme_for(heading)).copy()
        ink, accent = (236, 242, 250), (90, 200, 230)
    draw = ImageDraw.Draw(image)
    top = round(40 * s)
    if heading:
        fnt, lines, line_h = fit_text(draw, heading, (round(96 * s), top, width - round(96 * s), top + round(110 * s)), round(56 * s), round(34 * s), bold=True, max_lines=1)
        draw_lines(draw, lines, fnt, line_h, round(96 * s), top, ink)
        top += line_h + round(8 * s)
        draw.rounded_rectangle((round(96 * s), top, round(96 * s) + round(180 * s), top + round(6 * s)), radius=round(3 * s), fill=accent)
        top += round(14 * s)
    area = (width, max(100, height - top - round(190 * s)))        # the bottom 190 px belong to the subtitles
    art = render(concept, area, t, focus, light)
    if art is not None:
        image.paste(art, (0, top), art)
    return image
