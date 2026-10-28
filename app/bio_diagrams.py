"""Biology concept diagrams drawn by code (never by an image model): the main visual beside the text on a board when a biology part has no
picture of its own. Like diagrams.py (physics, chemistry) each is a reusable concept chosen from the WORDS of the narration, never from a
topic name. Every label is drawn with the board's own fonts at a known position, so a diagram can never carry wrong lettering, a dropped
label or a body: internal organs, cells, structures and timelines only (school-student rule: no skin, no body outline, no external genitals).

- female_system / male_system: schematic internal organs of the reproductive systems (the male one without external parts);
- hormone_loop: hypothalamus -> pituitary -> endocrine gland -> hormone in the blood -> target organ, with feedback;
- menstrual_cycle: a 28-day bar with menstruation, follicular phase, ovulation and luteal phase;
- flower: a flower in section with its parts; fertilisation: sperm meets egg -> fertilisation -> zygote; seed_germination: four stages.
"""
from __future__ import annotations

import math
import re

from PIL import Image, ImageDraw

from .typography import font

_FEMALE = re.compile(r"(?i)\b(uterus|uteri|ovary|ovaries|ovarian|fallopian|oviducts?|cervix|cervical|vagina\w*|womb|female reproductive|endometri\w*)\b")
_MALE = re.compile(r"(?i)\b(testes|testis|testicles?|epididym\w*|vas deferens|seminal vesicles?|prostate|urethra|male reproductive|sperm ducts?|semen)\b")
_HORMONE = re.compile(r"(?i)\b(hormones?|pituitary|hypothalam\w*|endocrine|glands?|feedback|secret\w*|target (?:organ|cell|tissue)s?|bloodstream)\b")
_CYCLE = re.compile(r"(?i)\b(menstru\w*|ovulat\w*|follicular|luteal|menstrual cycle|menarche|menopause|28 days?)\b")
_FLOWER = re.compile(r"(?i)\b(petals?|sepals?|stamens?|anthers?|stigma|pistil|filament|carpels?|style|receptacle|pollen)\b")
_FERTILISE = re.compile(r"(?i)\b(fertili[sz]\w*|zygote|sperm|ovum|egg cell|embryo|implant\w*)\b")
_SEED = re.compile(r"(?i)\b(germinat\w*|radicle|plumule|cotyledons?|seedling|seed coat|sprout\w*)\b")
BIO_CONCEPTS = {"female_system": _FEMALE, "male_system": _MALE, "hormone_loop": _HORMONE, "menstrual_cycle": _CYCLE,
                "flower": _FLOWER, "fertilisation": _FERTILISE, "seed_germination": _SEED}

INK = (232, 238, 246, 255)
CYAN = (64, 210, 230, 255)
YELLOW = (245, 214, 90, 255)
PINK = (244, 150, 128, 255)
PINK_DARK = (214, 104, 92, 255)
RED = (226, 92, 92, 255)
GREEN = (110, 200, 120, 255)
VIOLET = (170, 130, 230, 255)
BROWN = (176, 128, 84, 255)
OUTLINE = (255, 255, 255, 70)


def concept_for(text: str) -> str | None:
    """The biology concept the part is about (the concept named most, at least twice), or None."""
    scores = {name: len(pattern.findall(text or "")) for name, pattern in BIO_CONCEPTS.items()}
    best = max(scores, key=scores.get)                   # ties keep the order above
    return best if scores[best] >= 2 else None


def render(concept: str, size: tuple[int, int], t: float) -> Image.Image | None:
    drawer = {"female_system": _female, "male_system": _male, "hormone_loop": _hormone, "menstrual_cycle": _cycle, "flower": _flower,
              "fertilisation": _fertilisation, "seed_germination": _seed}.get(concept)
    return drawer(size, t) if drawer else None


# ---- helpers ----------------------------------------------------------------------------------------------------------------------

def _canvas(size):
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    return image, ImageDraw.Draw(image, "RGBA")


def _shown(index: int, t: float, step: float = 0.45) -> bool:
    """Parts appear one after the other as the clip plays."""
    return t >= index * step


def _label(draw, size, text: str, target: tuple, at: tuple, anchor: str = "lm", color=INK):
    """A name with a thin leader line to the part it names (all positions are fractions of the canvas)."""
    w, h = size
    fnt = font(max(15, round(h * 0.043)), True)
    tx, ty = target[0] * w, target[1] * h
    ax, ay = at[0] * w, at[1] * h
    draw.line((tx, ty, ax, ay), fill=(220, 228, 238, 150), width=2)
    draw.ellipse((tx - 5, ty - 5, tx + 5, ty + 5), fill=CYAN)
    draw.text((ax + (8 if anchor.startswith("l") else -8), ay), text, font=fnt, fill=color, anchor=anchor)


def _blob(draw, box, fill, outline=OUTLINE, width=3):
    draw.ellipse(box, fill=fill, outline=outline, width=width)


def _tube(draw, points, color, width):
    draw.line(points, fill=color, width=width, joint="curve")
    for x, y in (points[0], points[-1]):
        draw.ellipse((x - width / 2, y - width / 2, x + width / 2, y + width / 2), fill=color)


def _arrow(draw, start, end, color=INK, width=4, head=14):
    draw.line((*start, *end), fill=color, width=width)
    dx, dy = end[0] - start[0], end[1] - start[1]
    n = math.hypot(dx, dy) or 1
    dx, dy = dx / n, dy / n
    x, y = end
    draw.polygon([(x, y), (x - dx * head - dy * head * 0.55, y - dy * head + dx * head * 0.55),
                  (x - dx * head + dy * head * 0.55, y - dy * head - dx * head * 0.55)], fill=color)


# ---- diagrams ---------------------------------------------------------------------------------------------------------------------

def _female(size, t):
    w, h = size
    image, d = _canvas(size)
    cx = w * 0.50
    # vagina and cervix (a canal below the uterus), uterus (a pear), tubes and ovaries
    d.rounded_rectangle((cx - w * 0.045, h * 0.62, cx + w * 0.045, h * 0.92), radius=int(w * 0.03), fill=PINK_DARK, outline=OUTLINE, width=3)
    d.rounded_rectangle((cx - w * 0.06, h * 0.55, cx + w * 0.06, h * 0.64), radius=int(w * 0.03), fill=(196, 84, 84, 255), outline=OUTLINE, width=3)
    d.polygon([(cx - w * 0.17, h * 0.30), (cx + w * 0.17, h * 0.30), (cx + w * 0.07, h * 0.56), (cx - w * 0.07, h * 0.56)], fill=PINK, outline=OUTLINE)
    d.ellipse((cx - w * 0.17, h * 0.25, cx + w * 0.17, h * 0.36), fill=PINK, outline=OUTLINE, width=3)
    d.polygon([(cx - w * 0.07, h * 0.36), (cx + w * 0.07, h * 0.36), (cx + w * 0.03, h * 0.52), (cx - w * 0.03, h * 0.52)], fill=(250, 196, 150, 255))
    for side in (-1, 1):
        _tube(d, [(cx + side * w * 0.15, h * 0.29), (cx + side * w * 0.26, h * 0.22), (cx + side * w * 0.34, h * 0.26), (cx + side * w * 0.36, h * 0.33)], PINK, 12)
        _blob(d, (cx + side * w * 0.36 - w * 0.07, h * 0.34, cx + side * w * 0.36 + w * 0.07, h * 0.46), (250, 212, 190, 255))
    if _shown(0, t): _label(d, size, "Ovary", (0.14, 0.40), (0.02, 0.30), "lm")
    if _shown(1, t): _label(d, size, "Fallopian tube", (0.27, 0.215), (0.02, 0.10), "lm")
    if _shown(2, t): _label(d, size, "Uterus", (0.58, 0.42), (0.78, 0.30), "lm")
    if _shown(3, t): _label(d, size, "Cervix", (0.52, 0.60), (0.70, 0.60), "lm")
    if _shown(4, t): _label(d, size, "Vagina", (0.51, 0.78), (0.68, 0.80), "lm")
    return image


def _male(size, t):
    w, h = size
    image, d = _canvas(size)
    cx = w * 0.50
    _blob(d, (cx - w * 0.13, h * 0.07, cx + w * 0.13, h * 0.27), (140, 190, 230, 255))                     # urinary bladder
    d.rounded_rectangle((cx - w * 0.015, h * 0.26, cx + w * 0.015, h * 0.93), radius=6, fill=YELLOW)         # urethra
    _blob(d, (cx - w * 0.07, h * 0.30, cx + w * 0.07, h * 0.40), PINK)                                       # prostate gland
    for side in (-1, 1):
        _blob(d, (cx + side * w * 0.13 - w * 0.045, h * 0.24, cx + side * w * 0.13 + w * 0.045, h * 0.34), (250, 196, 150, 255))     # seminal vesicle
        _tube(d, [(cx + side * w * 0.05, h * 0.34), (cx + side * w * 0.20, h * 0.30), (cx + side * w * 0.30, h * 0.50), (cx + side * w * 0.31, h * 0.66)], GREEN, 7)   # vas deferens
        _blob(d, (cx + side * w * 0.30 - w * 0.075, h * 0.64, cx + side * w * 0.30 + w * 0.075, h * 0.86), PINK_DARK)                # testis
        d.arc((cx + side * w * 0.30 - w * 0.055, h * 0.62, cx + side * w * 0.30 + w * 0.095, h * 0.84), 250, 110, fill=(255, 235, 200, 255), width=8)   # epididymis
    if _shown(0, t): _label(d, size, "Urinary bladder", (0.40, 0.14), (0.02, 0.07), "lm")
    if _shown(1, t): _label(d, size, "Seminal vesicle", (0.63, 0.29), (0.70, 0.12), "lm")
    if _shown(2, t): _label(d, size, "Prostate gland", (0.57, 0.35), (0.66, 0.50), "lm")
    if _shown(3, t): _label(d, size, "Urethra", (0.50, 0.72), (0.52, 0.94), "lm")
    if _shown(4, t): _label(d, size, "Vas deferens", (0.35, 0.36), (0.02, 0.40), "lm")
    if _shown(5, t): _label(d, size, "Testis", (0.80, 0.76), (0.88, 0.62), "lm")
    if _shown(6, t): _label(d, size, "Epididymis", (0.84, 0.68), (0.99, 0.90), "rm")
    return image


def _hormone(size, t):
    w, h = size
    image, d = _canvas(size)
    fnt = font(max(16, round(h * 0.046)), True)
    small = font(max(13, round(h * 0.036)))
    boxes = [("Hypothalamus", 0.10, VIOLET), ("Pituitary gland", 0.36, CYAN), ("Endocrine gland", 0.62, YELLOW)]
    x0, x1 = w * 0.04, w * 0.52
    for i, (name, y, color) in enumerate(boxes):
        if _shown(i, t, 0.6):
            d.rounded_rectangle((x0, h * y, x1, h * y + h * 0.15), radius=int(h * 0.04), fill=color[:3] + (60,), outline=color, width=4)
            d.text(((x0 + x1) / 2, h * y + h * 0.075), name, font=fnt, fill=INK, anchor="mm")
            if i < len(boxes) - 1:
                _arrow(d, ((x0 + x1) / 2, h * y + h * 0.16), ((x0 + x1) / 2, h * (boxes[i + 1][1]) - 6), color)
    if _shown(3, t, 0.6):
        _arrow(d, (x1 + 8, h * 0.695), (w * 0.745, h * 0.695), RED, 5)
        d.text(((x1 + w * 0.745) / 2, h * 0.645), "hormone", font=small, fill=RED, anchor="mm")
        d.text(((x1 + w * 0.745) / 2, h * 0.755), "in the blood", font=small, fill=RED, anchor="mm")
        d.rounded_rectangle((w * 0.75, h * 0.62, w * 0.98, h * 0.77), radius=int(h * 0.04), fill=GREEN[:3] + (60,), outline=GREEN, width=4)
        d.text((w * 0.865, h * 0.695), "Target organ", font=small, fill=INK, anchor="mm")
    if _shown(4, t, 0.6):
        d.line((w * 0.865, h * 0.62, w * 0.865, h * 0.50, w * 0.62, h * 0.50), fill=INK, width=3)                 # feedback back to the pituitary
        _arrow(d, (w * 0.62, h * 0.50), (x1 + 6, h * 0.43), INK, 3)
        d.text((w * 0.80, h * 0.46), "feedback", font=small, fill=INK, anchor="mm")
    return image


def _cycle(size, t):
    w, h = size
    image, d = _canvas(size)
    fnt = font(max(15, round(h * 0.042)), True)
    small = font(max(13, round(h * 0.036)))
    x0, x1, y0, y1 = w * 0.04, w * 0.96, h * 0.42, h * 0.60
    phases = [(1, 5, "Menstruation", RED), (6, 13, "Follicular", CYAN), (14, 14, "", YELLOW), (15, 28, "Luteal phase", VIOLET)]
    day = lambda n: x0 + (x1 - x0) * (n - 1) / 28
    for i, (a, b, name, color) in enumerate(phases):
        if not _shown(i, t, 0.55):
            continue
        left, right = day(a), day(b + 1)
        d.rectangle((left, y0, right, y1), fill=color[:3] + (150,), outline=color, width=3)
        if name:
            tiny = font(max(12, round(h * 0.031)), True)
            d.text(((left + right) / 2, (y0 + y1) / 2), name, font=tiny, fill=INK, anchor="mm")
    if _shown(2, t, 0.55):
        d.polygon([(day(14.5), y0 - 4), (day(14.5) - 14, y0 - 28), (day(14.5) + 14, y0 - 28)], fill=YELLOW)
        d.text((day(14.5), y0 - h * 0.09), "Ovulation (day 14)", font=fnt, fill=YELLOW, anchor="mm")
    for n in (1, 5, 14, 28):
        x = min(day(n) if n != 28 else day(28.5), w * 0.94)
        d.text((x, y1 + h * 0.07), f"day {n}", font=small, fill=INK, anchor="mm")
    return image


def _flower(size, t):
    w, h = size
    image, d = _canvas(size)
    cx = w * 0.46
    d.rounded_rectangle((cx - 6, h * 0.58, cx + 6, h * 0.95), radius=5, fill=GREEN)                           # stem
    _blob(d, (cx - w * 0.07, h * 0.56, cx + w * 0.07, h * 0.62), (150, 190, 100, 255))                        # receptacle
    for side in (-1, 1):
        d.ellipse((cx + side * w * 0.12 - w * 0.07, h * 0.58, cx + side * w * 0.12 + w * 0.07, h * 0.64), fill=GREEN)               # sepals
        d.ellipse((cx + side * w * 0.17 - w * 0.07, h * 0.30, cx + side * w * 0.17 + w * 0.07, h * 0.58), fill=(214, 140, 210, 255), outline=OUTLINE)   # petals
    d.rounded_rectangle((cx - 4, h * 0.30, cx + 4, h * 0.58), radius=4, fill=(150, 220, 150, 255))              # style
    _blob(d, (cx - w * 0.025, h * 0.26, cx + w * 0.025, h * 0.32), YELLOW)                                    # stigma
    _blob(d, (cx - w * 0.04, h * 0.50, cx + w * 0.04, h * 0.58), (250, 196, 150, 255))                         # ovary
    d.line((cx + w * 0.07, h * 0.56, cx + w * 0.11, h * 0.33), fill=(150, 220, 150, 255), width=5)             # filament
    _blob(d, (cx + w * 0.085, h * 0.28, cx + w * 0.14, h * 0.34), (240, 170, 60, 255))                         # anther
    if _shown(0, t): _label(d, size, "Stigma", (0.46, 0.29), (0.60, 0.10), "lm")
    if _shown(1, t): _label(d, size, "Anther", (0.58, 0.31), (0.80, 0.18), "lm")
    if _shown(2, t): _label(d, size, "Filament", (0.56, 0.45), (0.78, 0.42), "lm")
    if _shown(3, t): _label(d, size, "Petal", (0.29, 0.44), (0.02, 0.30), "lm")
    if _shown(4, t): _label(d, size, "Style", (0.46, 0.40), (0.02, 0.52), "lm")
    if _shown(5, t): _label(d, size, "Ovary", (0.46, 0.54), (0.64, 0.64), "lm")
    if _shown(6, t): _label(d, size, "Sepal", (0.34, 0.61), (0.02, 0.70), "lm")
    return image


def _tadpole(d, x, y, r, angle, color=CYAN):
    d.ellipse((x - r, y - r * 0.7, x + r, y + r * 0.7), fill=color)
    tx, ty = x - math.cos(angle) * r * 3.2, y - math.sin(angle) * r * 3.2
    d.line((x, y, tx, ty), fill=color, width=max(2, int(r * 0.35)))


def _fertilisation(size, t):
    w, h = size
    image, d = _canvas(size)
    fnt = font(max(15, round(h * 0.046)), True)
    cy, r = h * 0.40, min(w, h) * 0.12
    centres = [w * 0.17, w * 0.50, w * 0.83]
    names = ["Sperm meets egg", "Fertilisation", "Zygote"]
    for i, (cx, name) in enumerate(zip(centres, names)):
        if not _shown(i, t, 0.7):
            continue
        _blob(d, (cx - r, cy - r, cx + r, cy + r), (250, 210, 170, 255) if i < 2 else (250, 225, 150, 255))
        if i == 0:
            for k, a in enumerate((-0.9, 0.0, 0.9)):
                _tadpole(d, cx - r * 1.9 * math.cos(a), cy + r * 1.9 * math.sin(a) + 0.0, r * 0.22, -a + math.pi)
        elif i == 1:
            _tadpole(d, cx - r * 0.65, cy, r * 0.22, math.pi)
            d.ellipse((cx - r * 0.2, cy - r * 0.2, cx + r * 0.3, cy + r * 0.3), fill=(214, 120, 90, 255))
        else:
            d.line((cx, cy - r, cx, cy + r), fill=(214, 120, 90, 255), width=4)
        d.text((cx, cy + r * 1.7), name, font=fnt, fill=INK, anchor="mm")
        if i < 2 and _shown(i + 1, t, 0.7):
            _arrow(d, (cx + r * 1.25, cy), (centres[i + 1] - r * 1.25, cy), INK, 4)
    return image


def _seed(size, t):
    w, h = size
    image, d = _canvas(size)
    fnt = font(max(14, round(h * 0.044)), True)
    names = ["Seed", "Radicle", "Plumule", "Seedling"]
    xs = [w * (0.12 + 0.25 * i) for i in range(4)]
    ground = h * 0.62
    d.line((w * 0.02, ground, w * 0.98, ground), fill=BROWN, width=5)
    for i, (x, name) in enumerate(zip(xs, names)):
        if not _shown(i, t, 0.6):
            continue
        sx, sy = x, ground + h * 0.08
        d.ellipse((sx - w * 0.04, sy - h * 0.045, sx + w * 0.04, sy + h * 0.045), fill=BROWN)
        if i >= 1:
            d.line((sx - w * 0.03, sy + h * 0.02, sx - w * 0.04, sy + h * (0.10 + 0.04 * i)), fill=(240, 226, 200, 255), width=6)   # radicle -> root
        if i >= 2:
            d.line((sx, sy - h * 0.04, sx, ground - h * (0.05 + 0.10 * (i - 2))), fill=GREEN, width=7)                            # plumule -> shoot
        if i == 3:
            d.ellipse((sx - w * 0.07, ground - h * 0.20, sx, ground - h * 0.13), fill=GREEN)
            d.ellipse((sx, ground - h * 0.22, sx + w * 0.07, ground - h * 0.15), fill=GREEN)
        d.text((x, h * 0.93), name, font=fnt, fill=INK, anchor="mm")
    return image
