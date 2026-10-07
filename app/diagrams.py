"""Physics concept diagrams drawn by code (never by an image model), used as the main visual of a board page when a physics part has
no picture of its own. Each diagram is a reusable concept, chosen from the words of the part's narration - never from a topic name:

- wave: a travelling sine wave on axes; when the narration speaks of both electric and magnetic fields, the two fields at right
  angles (E vertical, B horizontal), like the classic electromagnetic-wave figure;
- conductor: a wire of length L and cross-section A with current flowing through it (resistance, resistivity, current, wire);
- circuit: a cell, a resistor and a glowing bulb in one loop with the current moving round it;
- lens: a convex lens with parallel rays meeting at the focus F;
- magnet: a bar magnet (N, S) with field lines; charge: + and - charges with the electric field lines between them;
- force: a block on a surface with applied force, friction, weight and normal force; pendulum: a simple pendulum swinging.

All labels are drawn with the board's own fonts, so a picture can never contain wrong lettering.
"""
from __future__ import annotations

import math
import re

import numpy as np
from PIL import Image, ImageDraw

from .typography import font

_WAVE = re.compile(r"(?i)\b(wave|waves|oscillat\w*|frequency|wavelength|amplitude|sinusoid\w*)\b")
_CONDUCTOR = re.compile(r"(?i)\b(wire|conductor|resistance|resistivity|resistor|cross[- ]section\w*|current flow\w*|ohm'?s law)\b")
_EM = re.compile(r"(?i)\belectric\b.*\bmagnetic\b|\bmagnetic\b.*\belectric\b|\belectromagnetic\b")
_CIRCUIT = re.compile(r"(?i)\b(circuit|battery|cell|bulb|switch|ammeter|voltmeter|potential difference|emf|in series|in parallel)\b")
_LENS = re.compile(r"(?i)\b(lens|lenses|refraction|refract\w*|focal|focus|converg\w*|diverg\w*|ray|rays|optical centre|optical center)\b")
_MAGNET = re.compile(r"(?i)\b(magnet|magnets|magnetic field|field lines|north pole|south pole|compass|bar magnet)\b")
_CHARGE = re.compile(r"(?i)\b(charge|charges|coulomb|electric field|electrostatic|attract\w*|repel\w*|positive charge|negative charge)\b")
_FORCE = re.compile(r"(?i)\b(force|forces|friction|newton'?s|acceleration|net force|push|pull|inertia|momentum)\b")
_PENDULUM = re.compile(r"(?i)\b(pendulum|bob|time period|simple harmonic|shm|swing\w*)\b")
CONCEPTS = {"wave": _WAVE, "conductor": _CONDUCTOR, "circuit": _CIRCUIT, "lens": _LENS, "magnet": _MAGNET, "charge": _CHARGE,
            "force": _FORCE, "pendulum": _PENDULUM}


_CONDUCTIVITY = re.compile(r"(?i)\b(molar conductivit\w*|conductance|equivalent conductivit\w*|limiting molar|dilution|square root of concentration|"
                           r"strong electrolytes?|weak electrolytes?)\b")
_GRAPH = re.compile(r"(?i)\b(graph|plot|curve|versus|vs\.?|concentration|extrapolat\w*|asymptot\w*)\b")
_ELECTROLYSIS = re.compile(r"(?i)\b(electrolysis|electrolytic cell|electroplating|electrorefining|electrodes?|cathode|anode|electroly[sz]e\w*)\b")
CHEM_CONCEPTS = {"conductivity_graph": _CONDUCTIVITY, "electrolysis": _ELECTROLYSIS}


def concept_for(text: str, chemistry: bool = False) -> str | None:
    """Which diagram fits the narration (the concept mentioned most, at least twice), or None. Chemistry has its own figures."""
    if chemistry:
        scores = {name: len(pattern.findall(text or "")) for name, pattern in CHEM_CONCEPTS.items()}
        if not _GRAPH.search(text or ""):
            scores["conductivity_graph"] = min(scores["conductivity_graph"], 1)     # conductivity talk without a graph: no figure
        best = max(scores, key=scores.get)
        return best if scores[best] >= 2 else None
    scores = {name: len(pattern.findall(text or "")) for name, pattern in CONCEPTS.items()}
    if scores["magnet"] and scores["wave"] and _EM.search(text or ""):
        scores["magnet"] = 0                                    # an electromagnetic wave is a wave, not a bar magnet
    best = max(scores, key=scores.get)                          # ties keep the order above (the older, more specific concepts first)
    return best if scores[best] >= 2 else None


def render(concept: str, size: tuple[int, int], t: float, text: str = "") -> Image.Image | None:
    if concept == "wave":
        return _wave(size, t, bool(_EM.search(text or "")))
    drawers = {"conductivity_graph": _conductivity_graph, "electrolysis": _electrolysis, "conductor": _conductor, "circuit": _circuit, "lens": _lens, "magnet": _magnet, "charge": _charges,
               "force": _force, "pendulum": _pendulum}
    if concept in drawers:
        return drawers[concept](size, t)
    return None


def _arrowhead(draw, tip, direction, color, size=14):
    dx, dy = direction
    n = math.hypot(dx, dy) or 1
    dx, dy = dx / n, dy / n
    x, y = tip
    draw.polygon([(x, y), (x - dx * size - dy * size * 0.55, y - dy * size + dx * size * 0.55),
                  (x - dx * size + dy * size * 0.55, y - dy * size - dx * size * 0.55)], fill=color)


def _conductivity_graph(size, t):
    """Molar conductivity against the square root of concentration: a strong electrolyte falls along a straight line (extrapolated
    to the axis, dashed), a weak electrolyte rises steeply only at very low concentration. The curves draw themselves."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(14, round(height * 0.045)), True)
    small = font(max(12, round(height * 0.038)))
    ox, oy = width * 0.16, height * 0.82
    w, h = width * 0.76, height * 0.70
    axis = (210, 216, 226, 255)
    draw.line((ox, oy, ox + w, oy), fill=axis, width=3)
    draw.line((ox, oy, ox, oy - h), fill=axis, width=3)
    _arrowhead(draw, (ox + w + 14, oy), (1, 0), axis, 14)
    _arrowhead(draw, (ox, oy - h - 14), (0, -1), axis, 14)
    draw.text((ox + w - 40, oy + 12), "√c", font=lab, fill=axis)
    draw.text((ox - 56, oy - h - 6), "Λₘ", font=lab, fill=axis)
    reveal = min(1.0, 0.2 + t * 0.3)
    n = max(2, round(80 * reveal))
    strong = [(ox + w * 0.06 + w * 0.88 * i / 79, oy - h * (0.78 - 0.30 * i / 79)) for i in range(80)][:n]
    weak = [(ox + w * 0.03 + w * 0.91 * i / 79, oy - h * (0.10 + 0.80 * math.exp(-i / 6.0))) for i in range(80)][:n]
    draw.line(strong, fill=(64, 210, 230, 255), width=5)
    draw.line(weak, fill=(245, 214, 90, 255), width=5)
    if reveal >= 1:
        y0 = oy - h * 0.78 - (h * 0.30 / 0.88) * 0.06                    # the strong line extended to c = 0
        for k in range(6):                                                 # dashed extrapolation to the axis
            x = ox + w * 0.06 * k / 6
            draw.line((x, y0 + (oy - h * 0.78 - y0) * k / 6, x + w * 0.006, y0 + (oy - h * 0.78 - y0) * (k + 0.5) / 6),
                      fill=(64, 210, 230, 255), width=3)
        draw.text((ox + w * 0.08, y0 - 52), "Λ°ₘ", font=lab, fill=(64, 210, 230, 255))
    draw.text((ox + w * 0.52, oy - h * 0.52), "strong electrolyte", font=small, fill=(64, 210, 230, 255))
    draw.text((ox + w * 0.30, oy - h * 0.22), "weak electrolyte", font=small, fill=(245, 214, 90, 255))
    return image


def _electrolysis(size, t):
    """An electrolytic cell: a battery drives current through two electrodes in a solution; cations move to the cathode (-),
    anions to the anode (+)."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(14, round(height * 0.045)), True)
    small = font(max(12, round(height * 0.04)), True)
    bx0, bx1, by0, by1 = width * 0.12, width * 0.88, height * 0.36, height * 0.86
    draw.rectangle((bx0, by0 + (by1 - by0) * 0.15, bx1, by1), fill=(80, 160, 230, 70))             # the solution
    draw.line((bx0, by0, bx0, by1, bx1, by1, bx1, by0), fill=(200, 220, 240, 255), width=4)       # the beaker
    ax, cx = width * 0.30, width * 0.70                                                           # anode left (+), cathode right (-)
    for x, color in ((ax, (230, 110, 90, 255)), (cx, (150, 160, 180, 255))):
        draw.rectangle((x - 14, by0 - height * 0.08, x + 14, by1 - height * 0.08), fill=color)
    top = height * 0.12
    draw.line((ax, by0 - height * 0.08, ax, top, cx, top, cx, by0 - height * 0.08), fill=(200, 210, 224, 255), width=3)
    mid = width / 2
    draw.rectangle((mid - 50, top - 22, mid + 50, top + 22), fill=(20, 30, 50, 255), outline=(245, 214, 90, 255), width=3)
    draw.line((mid - 14, top - 16, mid - 14, top + 16), fill=(245, 214, 90, 255), width=4)
    draw.line((mid + 14, top - 9, mid + 14, top + 9), fill=(245, 214, 90, 255), width=7)
    draw.text((ax - 60, by0 - height * 0.15), "+", font=lab, fill=(230, 110, 90, 255))
    draw.text((cx + 34, by0 - height * 0.15), "−", font=lab, fill=(190, 200, 220, 255))
    draw.text((ax - 40, by1 + 8), "anode", font=small, fill=(230, 110, 90, 255))
    draw.text((cx - 50, by1 + 8), "cathode", font=small, fill=(190, 200, 220, 255))
    span = cx - ax - 60
    for k in range(5):                                                                            # ions drifting to their electrodes
        u = ((k / 5) + t * 0.15) % 1.0
        y = by0 + (by1 - by0) * (0.30 + 0.12 * k)
        xp = ax + 30 + span * u                                                                   # cation -> cathode
        xn = cx - 30 - span * u                                                                   # anion -> anode
        draw.ellipse((xp - 15, y - 15, xp + 15, y + 15), fill=(230, 80, 80, 255))
        draw.text((xp - 7, y - 15), "+", font=small, fill=(255, 255, 255, 255))
        yn = y + (by1 - by0) * 0.06
        draw.ellipse((xn - 15, yn - 15, xn + 15, yn + 15), fill=(70, 120, 235, 255))
        draw.text((xn - 6, yn - 17), "−", font=small, fill=(255, 255, 255, 255))
    return image


def _circuit(size, t):
    """A cell, a resistor and a bulb in one loop; the current (conventional, + to -) moves round it."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ink, wire, gold = (236, 240, 246, 255), (190, 200, 214, 255), (245, 214, 90, 255)
    lab = font(max(14, round(height * 0.05)), True)
    x0, x1, y0, y1 = width * 0.14, width * 0.86, height * 0.20, height * 0.78
    cx = (x0 + x1) / 2
    # wires with gaps for the parts
    draw.line((x0, y0, cx - 70, y0), fill=wire, width=4)
    draw.line((cx + 70, y0, x1, y0), fill=wire, width=4)
    draw.line((x1, y0, x1, y1), fill=wire, width=4)
    draw.line((x1, y1, cx + 34, y1), fill=wire, width=4)
    draw.line((cx - 34, y1, x0, y1), fill=wire, width=4)
    draw.line((x0, y1, x0, y0), fill=wire, width=4)
    # resistor (zig-zag) on top
    zig = [(cx - 70, y0)] + [(cx - 70 + 14 * k + 7, y0 + (-16 if k % 2 == 0 else 16)) for k in range(10)] + [(cx + 70, y0)]
    draw.line(zig, fill=(240, 150, 90, 255), width=4)
    draw.text((cx - 10, y0 - 70), "R", font=lab, fill=(240, 150, 90, 255))
    # cell at the bottom: long (+) and short (-) plates
    draw.line((cx - 34, y1 - 40, cx - 34, y1 + 40), fill=ink, width=5)
    draw.line((cx + 34, y1 - 22, cx + 34, y1 + 22), fill=ink, width=9)
    draw.text((cx - 64, y1 - 92), "+", font=lab, fill=ink)
    draw.text((cx + 46, y1 - 92), "−", font=lab, fill=ink)
    # bulb on the right side
    by = (y0 + y1) / 2
    glow = 0.6 + 0.4 * math.sin(t * 3)
    draw.ellipse((x1 - 36, by - 36, x1 + 36, by + 36), fill=(255, 230, 120, round(120 + 100 * glow)), outline=gold, width=3)
    draw.line((x1 - 18, by - 18, x1 + 18, by + 18), fill=(120, 90, 30, 255), width=3)
    draw.line((x1 - 18, by + 18, x1 + 18, by - 18), fill=(120, 90, 30, 255), width=3)
    # current: dots travelling from + round the loop (counter-clockwise from the long plate, through the left side)
    loop = [(cx - 34, y1), (x0, y1), (x0, y0), (x1, y0), (x1, y1), (cx + 34, y1)]
    lengths = [math.dist(loop[i], loop[i + 1]) for i in range(len(loop) - 1)]
    total = sum(lengths)
    for k in range(10):
        d = ((k / 10) + t * 0.12) % 1.0 * total
        for i, seg_len in enumerate(lengths):
            if d <= seg_len:
                (ax, ay), (bx, by2) = loop[i], loop[i + 1]
                u = d / seg_len
                x, y = ax + (bx - ax) * u, ay + (by2 - ay) * u
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=gold)
                break
            d -= seg_len
    draw.text((x0 - 40, by - 20), "I", font=lab, fill=gold)
    _arrowhead(draw, (x0, by + 30), (0, -1), gold, 16)
    return image


def _lens(size, t):
    """A convex lens: parallel rays refract and meet at the focus F."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(14, round(height * 0.05)), True)
    cx, cy = width * 0.45, height * 0.48
    axis = (170, 180, 196, 255)
    draw.line((width * 0.04, cy, width * 0.96, cy), fill=axis, width=2)
    lens_h, lens_w = height * 0.62, width * 0.07
    draw.ellipse((cx - lens_w / 2, cy - lens_h / 2, cx + lens_w / 2, cy + lens_h / 2), fill=(120, 200, 255, 70),
                 outline=(150, 220, 255, 255), width=3)
    f = width * 0.30
    fx = cx + f
    draw.ellipse((fx - 6, cy - 6, fx + 6, cy + 6), fill=(245, 214, 90, 255))
    draw.text((fx - 8, cy + 12), "F", font=lab, fill=(245, 214, 90, 255))
    draw.ellipse((cx - f - 6, cy - 6, cx - f + 6, cy + 6), fill=(245, 214, 90, 255))
    draw.text((cx - f - 8, cy + 12), "F", font=lab, fill=(245, 214, 90, 255))
    reveal = min(1.0, 0.25 + t * 0.35)                           # the rays are drawn out from the left
    for k, offset in enumerate((-0.22, -0.11, 0.11, 0.22)):
        y = cy + offset * height
        color = (250, 110, 110, 255) if k % 2 == 0 else (250, 170, 90, 255)
        start, hit = (width * 0.05, y), (cx, y)
        end = (fx + (fx - cx) * 0.55, cy + (cy - y) * 0.55)
        total = math.dist(start, hit) + math.dist(hit, end)
        d = reveal * total
        first = min(d, math.dist(start, hit))
        draw.line((start[0], y, start[0] + first, y), fill=color, width=3)
        _arrowhead(draw, (start[0] + first * 0.5, y), (1, 0), color, 12)
        if d > first:
            u = min(1.0, (d - first) / math.dist(hit, end))
            draw.line((hit[0], hit[1], hit[0] + (end[0] - hit[0]) * u, hit[1] + (end[1] - hit[1]) * u), fill=color, width=3)
    return image


def _magnet(size, t):
    """A bar magnet with its field lines from N to S."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(16, round(height * 0.07)), True)
    cx, cy = width / 2, height * 0.48
    mw, mh = width * 0.34, height * 0.13
    line_color = (120, 200, 255, 200)
    for k in range(1, 6):                                       # field loops round the magnet
        ry = mh * 0.5 + k * height * 0.07
        rx = mw * 0.5 + k * width * 0.045
        draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), outline=line_color, width=2)
        phi = math.pi * ((t * 0.25 + k * 0.17) % 1.0)          # arrows travel outside the magnet from N (right) to S (left)
        for sign in (-1, 1):
            ax, ay = cx + rx * math.cos(phi), cy + sign * ry * math.sin(phi)
            _arrowhead(draw, (ax, ay), (-rx * math.sin(phi), sign * ry * math.cos(phi)), (150, 220, 255, 255), 12)
    draw.rectangle((cx - mw / 2, cy - mh / 2, cx, cy + mh / 2), fill=(70, 120, 230, 255))
    draw.rectangle((cx, cy - mh / 2, cx + mw / 2, cy + mh / 2), fill=(225, 70, 70, 255))
    draw.text((cx - mw / 4 - 10, cy - mh / 2 + 4), "S", font=lab, fill=(255, 255, 255, 255))
    draw.text((cx + mw / 4 - 10, cy - mh / 2 + 4), "N", font=lab, fill=(255, 255, 255, 255))
    return image


def _charges(size, t):
    """A positive and a negative charge with the electric field lines from + to -."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(18, round(height * 0.08)), True)
    cy = height * 0.48
    left, right = (width * 0.25, cy), (width * 0.75, cy)
    gap = right[0] - left[0]
    for k in range(-3, 4):                                      # curved lines bulging away from the axis
        bulge = k * height * 0.09
        points = []
        for i in range(41):
            u = i / 40
            points.append((left[0] + gap * u, cy - bulge * math.sin(math.pi * u) * (1.6 if k else 1)))
        draw.line(points, fill=(245, 214, 90, 170), width=2)
        u = (t * 0.3 + abs(k) * 0.11) % 1.0
        i = max(1, min(39, round(u * 40)))
        _arrowhead(draw, points[i], (points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]), (245, 214, 90, 255), 12)
    r = height * 0.09
    for (x, y), color, sign in ((left, (230, 70, 70, 255), "+"), (right, (70, 120, 235, 255), "−")):
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color, outline=(255, 255, 255, 220), width=2)
        box = draw.textbbox((0, 0), sign, font=lab)
        draw.text((x - (box[2] - box[0]) / 2, y - (box[3] - box[1]) / 2 - box[1]), sign, font=lab, fill=(255, 255, 255, 255))
    return image


def _force(size, t):
    """A block on a surface: applied force to the right, friction to the left, weight and normal force; it slides slowly."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(14, round(height * 0.05)), True)
    ground = height * 0.66
    draw.line((width * 0.05, ground, width * 0.95, ground), fill=(170, 180, 196, 255), width=3)
    for k in range(18):
        x = width * 0.05 + k * width * 0.05
        draw.line((x, ground, x - 14, ground + 14), fill=(120, 130, 146, 255), width=2)
    bw, bh = width * 0.22, height * 0.20
    bx = width * 0.30 + (t * 12) % (width * 0.12)
    draw.rectangle((bx, ground - bh, bx + bw, ground), fill=(90, 150, 210, 255), outline=(200, 230, 255, 255), width=3)
    draw.text((bx + bw / 2 - 12, ground - bh / 2 - 18), "m", font=lab, fill=(255, 255, 255, 255))
    mid_x, mid_y = bx + bw / 2, ground - bh / 2
    arrows = [((bx + bw, mid_y), (bx + bw + width * 0.22, mid_y), (250, 110, 110, 255), "F"),
              ((bx, ground - 12), (bx - width * 0.13, ground - 12), (245, 214, 90, 255), "f"),
              ((mid_x, ground), (mid_x, ground + height * 0.20), (150, 230, 110, 255), "mg"),
              ((mid_x, ground - bh), (mid_x, ground - bh - height * 0.20), (64, 210, 230, 255), "N")]
    for (sx, sy), (ex, ey), color, name in arrows:
        draw.line((sx, sy, ex, ey), fill=color, width=5)
        _arrowhead(draw, (ex, ey), (ex - sx, ey - sy), color, 16)
        draw.text((ex + (8 if ex >= sx else -40), ey - (40 if ey <= sy else -4)), name, font=lab, fill=color)
    return image


def _pendulum(size, t):
    """A simple pendulum swinging, with its length L and the angle from the vertical."""
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    lab = font(max(14, round(height * 0.05)), True)
    px, py = width / 2, height * 0.12
    length = height * 0.62
    draw.line((px - width * 0.18, py, px + width * 0.18, py), fill=(190, 200, 214, 255), width=6)
    draw.line((px, py, px, py + length + 30), fill=(120, 130, 146, 180), width=2)      # the vertical (rest position)
    theta0 = 0.42
    theta = theta0 * math.cos(t * 2.2)
    bx, by = px + length * math.sin(theta), py + length * math.cos(theta)
    for sign in (-1, 1):                                                              # the extreme positions, faint
        ex, ey = px + length * math.sin(sign * theta0), py + length * math.cos(sign * theta0)
        draw.ellipse((ex - 22, ey - 22, ex + 22, ey + 22), outline=(120, 130, 146, 160), width=2)
    draw.arc((px - length - 22, py - length - 22, px + length + 22, py + length + 22), 90 - math.degrees(theta0), 90 + math.degrees(theta0),
             fill=(120, 130, 146, 160), width=2)
    draw.line((px, py, bx, by), fill=(236, 240, 246, 255), width=3)
    draw.ellipse((bx - 24, by - 24, bx + 24, by + 24), fill=(245, 214, 90, 255), outline=(255, 240, 180, 255), width=2)
    draw.text((px + length * math.sin(theta) / 2 + 14, py + length * 0.45), "L", font=lab, fill=(236, 240, 246, 255))
    if abs(theta) > 0.08:
        draw.arc((px - 60, py - 60, px + 60, py + 60), 90 - math.degrees(max(theta, 0)), 90 - math.degrees(min(theta, 0)),
                 fill=(64, 210, 230, 255), width=3)
        draw.text((px + (10 if theta < 0 else -34) , py + 66), "θ", font=lab, fill=(64, 210, 230, 255))
    return image


def _wave(size, t, electromagnetic):
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    ox, oy = width * 0.10, height * 0.55
    length = width * 0.82
    axis = (210, 216, 226, 255)
    draw.line((ox, oy, ox + length, oy), fill=axis, width=3)                         # direction of travel
    draw.polygon([(ox + length, oy - 9), (ox + length + 18, oy), (ox + length, oy + 9)], fill=axis)
    draw.line((ox, oy, ox, oy - height * 0.40), fill=(230, 80, 80, 255), width=3)  # E axis (up)
    lab = font(max(14, round(height * 0.05)), True)
    draw.text((ox + length - 8, oy + 14), "x", font=lab, fill=axis)
    phase = t * 2.4
    amp = height * 0.28
    points_e, points_b = [], []
    for i in range(0, 241):
        u = i / 240
        x = ox + u * length
        value = math.sin(u * 4 * math.pi - phase)
        points_e.append((x, oy - amp * value))
        # B drawn in perspective: towards the viewer and down-left
        points_b.append((x - amp * 0.55 * value * 0.6, oy + amp * 0.55 * value * 0.45))
    for i in range(0, 241, 8):                                                       # field arrows like the textbook figure
        x, ye = points_e[i]
        draw.line((x, oy, x, ye), fill=(240, 90, 90, 150), width=2)
        if electromagnetic:
            xb, yb = points_b[i]
            draw.line((ox + (i / 240) * length, oy, xb, yb), fill=(90, 140, 250, 150), width=2)
    draw.line(points_e, fill=(250, 90, 90, 255), width=5)
    draw.text((ox + 10, oy - height * 0.44), "E", font=lab, fill=(250, 110, 110, 255))
    if electromagnetic:
        draw.line(points_b, fill=(90, 150, 255, 255), width=5)
        draw.line((ox, oy, ox - height * 0.22, oy + height * 0.18), fill=(90, 150, 255, 255), width=3)
        draw.text((ox - height * 0.26, oy + height * 0.19), "B", font=lab, fill=(110, 160, 255, 255))
    return image


def _conductor(size, t):
    width, height = size
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    x0, x1 = width * 0.12, width * 0.84
    cy, r = height * 0.45, height * 0.11
    ell = r * 0.45                                                                   # ellipse half-width of the end faces
    body = (176, 112, 70, 255)
    shade = Image.new("RGBA", size, (0, 0, 0, 0))
    sd = ImageDraw.Draw(shade)
    for k in range(int(2 * r)):                                                      # metallic shading across the wire
        v = 1 - abs(k - r) / r
        c = tuple(int(ch * (0.55 + 0.6 * v)) for ch in body[:3])
        sd.line((x0, cy - r + k, x1, cy - r + k), fill=c + (255,))
    image.alpha_composite(shade)
    draw.ellipse((x1 - ell, cy - r, x1 + ell, cy + r), fill=(214, 150, 100, 255), outline=(250, 210, 170, 255), width=2)
    draw.ellipse((x0 - ell, cy - r, x0 + ell, cy + r), fill=(150, 92, 58, 255), outline=(220, 170, 130, 255), width=2)
    lab = font(max(14, round(height * 0.06)), True)
    ink = (236, 240, 246, 255)
    yl = cy + r + height * 0.10                                                      # length L
    draw.line((x0, yl, x1, yl), fill=ink, width=2)
    for x in (x0, x1):
        draw.line((x, yl - 10, x, yl + 10), fill=ink, width=2)
    draw.text(((x0 + x1) / 2 - 8, yl + 8), "L", font=lab, fill=ink)
    ax = min(x1 + ell + 40, width - lab.size - 12)                                   # area A, kept inside the picture
    draw.line((x1 + ell * 0.5, cy - r * 0.5, ax, cy - r - 30), fill=(64, 210, 230, 255), width=2)
    draw.text((ax - lab.size * 0.2, cy - r - 30 - lab.size * 1.3), "A", font=lab, fill=(64, 210, 230, 255))
    for k in range(6):                                                               # current: dots moving along the wire
        u = ((k / 6) + t * 0.18) % 1.0
        x = x0 + ell + u * (x1 - x0 - 2 * ell)
        draw.ellipse((x - 7, cy - 7, x + 7, cy + 7), fill=(245, 214, 90, 255))
    ya = cy - r - height * 0.12
    draw.line((x0, ya, x0 + width * 0.22, ya), fill=(245, 214, 90, 255), width=3)
    draw.polygon([(x0 + width * 0.22, ya - 8), (x0 + width * 0.22 + 16, ya), (x0 + width * 0.22, ya + 8)], fill=(245, 214, 90, 255))
    draw.text((x0 + width * 0.24, ya - 22), "I", font=lab, fill=(245, 214, 90, 255))
    return image
