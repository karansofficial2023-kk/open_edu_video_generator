"""Chemistry and biology teaching figures drawn by code (see figures.py for the shared rules and the drawing surface).

Each entry in FIGURES:
  name -> (strong_pattern, weak_pattern, drawer_function, subjects)

The drawer receives a Cv canvas (see figures.py).  Labels are always drawn at
known positions using the video's own fonts — the image model never produces any
text.  No skin or body outlines (school-student rule).
"""
from __future__ import annotations

import math

from .figures import Cv, DH, DW


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _bubble(cv: Cv, cx: float, cy: float, r: float = 9, col=None) -> None:
    """A small gas bubble circle."""
    c = col or cv.pal.glass
    cv.circle((cx, cy), r, fill=(*c, 180), outline=c, w=2)


def _flask(cv: Cv, x: float, y: float, w: float, h: float, fill_frac: float, fill_col, active: bool = False) -> None:
    """A conical Erlenmeyer flask outline with liquid at fill_frac of h."""
    neck_w = w * 0.28
    # Body trapezoid points (bottom wide, top narrow)
    bx0, bx1 = x, x + w
    tx0, tx1 = x + (w - neck_w) / 2, x + (w + neck_w) / 2
    body = [(tx0, y), (tx1, y), (bx1, y + h), (bx0, y + h)]
    # Fill polygon (liquid)
    fy = y + h * (1 - fill_frac)
    # interpolate x at fy
    frac_down = 1 - fill_frac
    fl0 = tx0 + (bx0 - tx0) * frac_down
    fl1 = tx1 + (bx1 - tx1) * frac_down
    liq = [(fl0, fy), (fl1, fy), (bx1, y + h), (bx0, y + h)]
    liq_col = (*fill_col, 140)
    cv.d.polygon([cv.P(p) for p in liq], fill=liq_col)
    # Outline
    col = cv.pal.hot if active else cv.pal.ink
    cv.d.polygon([cv.P(p) for p in body], outline=col, width=max(1, round(3 * cv.s)))
    # Neck rectangle
    nh = h * 0.18
    cv.rect((tx0, y - nh, tx1, y), outline=col, w=3)
    # Stopper nub
    cv.rect((tx0 - 6, y - nh - 10, tx1 + 6, y - nh + 2), fill=cv.pal.muted, outline=col, w=2)


def _test_tube(cv: Cv, cx: float, top: float, height: float, tilt_deg: float = 0,
               fill_frac: float = 0.5, fill_col=None, active: bool = False) -> None:
    """A test tube, optionally tilted."""
    fill_col = fill_col or cv.pal.glass
    w2 = height * 0.18
    # Untilted outline: rectangle + semicircle at bottom
    r = w2
    # We draw untilted then rotate each point
    raw_pts = [
        (-w2, 0), (w2, 0),          # opening
        (w2, height - r),           # right side
    ]
    # Arc bottom via polygon approximation
    for a in range(0, 181, 15):
        rad = math.radians(a)
        raw_pts.append((r * math.cos(math.radians(180 - a)), height - r + r * math.sin(math.radians(180 - a))))
    raw_pts.append((-w2, height - r))   # left side close

    tr = math.radians(tilt_deg)
    cos_t, sin_t = math.cos(tr), math.sin(tr)

    def rotate(px, py):
        return cx + px * cos_t - py * sin_t, top + px * sin_t + py * cos_t

    pts = [rotate(*p) for p in raw_pts]
    col = cv.pal.hot if active else cv.pal.ink

    # Fill (liquid level)
    fill_y = height * (1 - fill_frac)
    fill_pts = []
    for (px, py) in raw_pts:
        if py >= fill_y or py >= height - r:
            fill_pts.append(rotate(px, max(py, fill_y)))
    if fill_pts:
        liq_col = (*fill_col, 150)
        cv.d.polygon([cv.P(p) for p in fill_pts], fill=liq_col)

    cv.d.polygon([cv.P(p) for p in pts], outline=col, width=max(1, round(3 * cv.s)))


# ---------------------------------------------------------------------------
# Electrolysis cell
# ---------------------------------------------------------------------------

def _electrolysis_cell(cv: Cv) -> None:
    """Electrolysis cell: battery, two electrodes, electrolyte, gas bubbles."""
    lit_anode = cv.lit(r"anode|positive electrode|oxidation|anion|chlorine|oxygen gas")
    lit_cathode = cv.lit(r"cathode|negative electrode|reduction|cation|hydrogen|metal deposit")
    lit_battery = cv.lit(r"batter|power supply|current|voltage|electri")
    lit_electrolyte = cv.lit(r"electrolyte|solution|ions?|dissol")

    p = cv.pal
    # Beaker / container
    bx0, bx1, by0, by1 = 340, 1260, 220, 650
    # Fill electrolyte
    if cv.shown(0):
        ey0 = by0 + (by1 - by0) * 0.25
        liq_col = (*p.glass, 100) if not lit_electrolyte else (*p.blue, 120)
        cv.d.rectangle([cv.P((bx0 + 4, ey0)), cv.P((bx1 - 4, by1 - 4))], fill=liq_col)
    # Container outline
    cv.rect((bx0, by0, bx1, by1), outline=p.ink, w=5)

    electrode_top = by0 - 80
    electrode_bottom = by0 + (by1 - by0) * 0.75

    # Anode (right, +)
    if cv.shown(1):
        anode_x = bx1 - 200
        a_col = p.hot if lit_anode else p.a2
        cv.rect((anode_x - 16, electrode_top, anode_x + 16, electrode_bottom), fill=a_col, outline=p.ink, w=3)
        cv.text((anode_x, electrode_top - 28), "Anode (+)", size=30, color=p.hot if lit_anode else p.muted)
        # Gas bubbles rising from anode
        if cv.shown(3):
            for i, (bcy, bcx_off) in enumerate([(by0 + 60, -8), (by0 + 120, 10), (by0 + 30, 5)]):
                if cv.shown(3 + i):
                    _bubble(cv, anode_x + bcx_off, bcy, r=11)

    # Cathode (left, -)
    if cv.shown(2):
        cathode_x = bx0 + 200
        c_col = p.blue if lit_cathode else p.muted
        cv.rect((cathode_x - 16, electrode_top, cathode_x + 16, electrode_bottom), fill=c_col, outline=p.ink, w=3)
        cv.text((cathode_x, electrode_top - 28), "Cathode (−)", size=30, color=p.blue if lit_cathode else p.muted)
        # Bubbles / deposits
        if cv.shown(4):
            for i, (bcy, bcx_off) in enumerate([(by0 + 80, 6), (by0 + 140, -9)]):
                if cv.shown(4 + i):
                    _bubble(cv, cathode_x + bcx_off, bcy, r=10, col=p.a1)

    # Battery on top
    if cv.shown(0):
        mid_x = (bx0 + bx1) // 2
        bat_y = electrode_top - 90
        bat_col = p.hot if lit_battery else p.ink
        # Battery symbol
        cv.rect((mid_x - 70, bat_y - 24, mid_x + 70, bat_y + 24), fill=(*p.fill, 200), outline=bat_col, w=4)
        cv.line((mid_x - 20, bat_y - 30), (mid_x - 20, bat_y + 30), bat_col, w=6)
        cv.line((mid_x + 20, bat_y - 20), (mid_x + 20, bat_y + 20), bat_col, w=3)
        cv.text((mid_x - 45, bat_y), "+", size=26, color=p.hot)
        cv.text((mid_x + 35, bat_y), "−", size=26, color=p.blue)
        cv.text((mid_x, bat_y - 50), "Battery", size=28, color=bat_col)

        # Wires: battery → electrodes
        anode_x = bx1 - 200
        cathode_x = bx0 + 200
        wire_y = bat_y
        cv.line((mid_x + 70, wire_y), (anode_x, wire_y), bat_col, w=4)
        cv.line((anode_x, wire_y), (anode_x, electrode_top), bat_col, w=4)
        cv.line((mid_x - 70, wire_y), (cathode_x, wire_y), bat_col, w=4)
        cv.line((cathode_x, wire_y), (cathode_x, electrode_top), bat_col, w=4)

    # Electrolyte label
    if cv.shown(1) and lit_electrolyte:
        cv.text(((bx0 + bx1) // 2, by0 + (by1 - by0) * 0.6), "Electrolyte", size=30, color=p.blue)


# ---------------------------------------------------------------------------
# Test-tube reaction
# ---------------------------------------------------------------------------

def _test_tube_reaction(cv: Cv) -> None:
    """Test tube with reactants, heating, and gas collection."""
    lit_heat = cv.lit(r"heat|flame|bunsen|warm|boil|thermal")
    lit_gas = cv.lit(r"\bgas\b|bubble|vapour|vapor|collect")
    lit_liquid = cv.lit(r"liquid|solution|dissolv|reactant")

    p = cv.pal
    cx, top, h = DW // 2, 140, 420
    fill_col = p.blue if lit_liquid else p.glass
    active = cv.lit(r"test.?tube|reaction|reactant|product|precipitate")

    if cv.shown(0):
        _test_tube(cv, cx, top, h, tilt_deg=-15, fill_frac=0.55, fill_col=fill_col, active=active)

    # Flame below tube
    if cv.shown(1) and lit_heat:
        flame_cx = cx - h * math.sin(math.radians(15)) + 0
        flame_base_y = top + h * math.cos(math.radians(15)) + 20
        # Simple flame shape as polygon
        flame_pts = [
            (flame_cx, flame_base_y),
            (flame_cx - 28, flame_base_y - 50),
            (flame_cx - 12, flame_base_y - 90),
            (flame_cx, flame_base_y - 130),
            (flame_cx + 12, flame_base_y - 90),
            (flame_cx + 28, flame_base_y - 50),
        ]
        cv.d.polygon([cv.P(fp) for fp in flame_pts], fill=(*p.hot, 200))
        cv.text((flame_cx + 50, flame_base_y - 70), "Heat", size=28, color=p.hot)

    # Gas arrow / bubbles rising from open end
    if cv.shown(2) and lit_gas:
        tube_open_x = cx + h * math.sin(math.radians(15)) * 0
        tube_open_y = top - 10
        for i in range(3):
            if cv.shown(2 + i):
                _bubble(cv, cx - 20 + i * 18, tube_open_y - 30 - i * 30, r=10)
        cv.arrow((cx, tube_open_y - 30), (cx, tube_open_y - 120), p.a1, w=4)
        cv.text((cx + 60, tube_open_y - 90), "Gas", size=28, color=p.a1)

    # Labels
    if cv.shown(0):
        cv.text((cx + 130, top + h * 0.3), "Reactants", size=28, color=p.blue if lit_liquid else p.muted)
    if cv.shown(1):
        cv.text((cx + 130, top + h * 0.7), "Test tube", size=26, color=p.ink)


# ---------------------------------------------------------------------------
# Distillation setup
# ---------------------------------------------------------------------------

def _distillation(cv: Cv) -> None:
    """Distillation: round-bottom flask, condenser, collection flask."""
    lit_flask = cv.lit(r"flask|mixture|solution|heat|boil")
    lit_condenser = cv.lit(r"condenser|cool|water.jacket|coil")
    lit_collect = cv.lit(r"collect|distillate|pure|product|beaker|receivin")

    p = cv.pal

    # Round-bottom flask (left)
    if cv.shown(0):
        fc = p.blue if lit_flask else p.glass
        cv.circle((340, 480), 130, fill=(*fc, 120), outline=p.ink, w=4)
        cv.rect((315, 330, 365, 360), outline=p.ink, w=4)   # neck
        cv.text((340, 610), "Mixture" if lit_flask else "Flask", size=28, color=p.muted)

        # Flame / heat
        if cv.shown(1):
            cv.d.polygon([cv.P(fp) for fp in [
                (310, 640), (330, 590), (340, 560), (350, 590), (370, 640)
            ]], fill=(*p.hot, 190))
            cv.text((340, 670), "Heat", size=24, color=p.hot if lit_flask else p.muted)

    # Connecting arm / condenser (diagonal)
    if cv.shown(1):
        cond_col = p.a1 if lit_condenser else p.muted
        # Outer condenser tube
        cv.line((365, 350), (900, 420), cond_col, w=20, dash=0)
        cv.line((365, 350), (900, 420), p.ink, w=4)
        cv.line((385, 360), (920, 430), p.ink, w=4)
        # Cooling water arrows
        if lit_condenser and cv.shown(2):
            cv.arrow((700, 380), (600, 365), p.blue, w=4)
            cv.text((650, 350), "Water in", size=24, color=p.blue)
            cv.arrow((500, 395), (600, 410), p.blue, w=4)
            cv.text((450, 420), "Water out", size=24, color=p.blue)
        cv.text((640, 300), "Condenser", size=28, color=cond_col)

    # Collection flask (right)
    if cv.shown(2):
        dist_col = p.a2 if lit_collect else p.glass
        _flask(cv, 870, 350, 200, 250, 0.3, dist_col, active=lit_collect)
        cv.text((970, 640), "Distillate" if lit_collect else "Collection", size=28, color=p.a2 if lit_collect else p.muted)

    # Vapour arrow
    if cv.shown(1):
        cv.arrow((365, 360), (440, 375), p.hot, w=3)


# ---------------------------------------------------------------------------
# Titration
# ---------------------------------------------------------------------------

def _titration(cv: Cv) -> None:
    """Burette, conical flask, indicator colour change at equivalence point."""
    lit_burette = cv.lit(r"burette|buret|titre|titrant|standard solution")
    lit_flask = cv.lit(r"flask|analyte|indicator|colour change|end ?point|equivale")
    lit_equiv = cv.lit(r"equivalence|end.?point|neutral|colour.change|pink|colour")

    p = cv.pal

    cx = DW // 2

    # Burette (tall, narrow, graduated)
    if cv.shown(0):
        bur_x, bur_top, bur_h, bur_w = cx - 20, 60, 440, 38
        bur_col = p.blue if lit_burette else p.glass
        # Liquid in burette
        cv.rect((bur_x - bur_w // 2 + 3, bur_top + 3, bur_x + bur_w // 2 - 3, bur_top + bur_h - 3),
                fill=(*bur_col, 140))
        # Burette outline
        cv.rect((bur_x - bur_w // 2, bur_top, bur_x + bur_w // 2, bur_top + bur_h), outline=p.ink, w=4)
        # Graduation marks
        for i in range(1, 6):
            gy = bur_top + bur_h * i // 6
            cv.line((bur_x - bur_w // 2, gy), (bur_x, gy), p.muted, w=2)
            cv.text((bur_x + 30, gy), f"{i * 5}", size=18, color=p.muted)
        # Stopcock
        cv.circle((bur_x, bur_top + bur_h + 10), 12, fill=p.muted, outline=p.ink, w=3)
        # Drip
        if cv.shown(2):
            cv.line((bur_x, bur_top + bur_h + 22), (bur_x, bur_top + bur_h + 55), p.blue, w=3)
            cv.circle((bur_x, bur_top + bur_h + 60), 8, fill=(*p.blue, 200))
        cv.text((bur_x, bur_top - 30), "Burette", size=28, color=p.blue if lit_burette else p.muted)

    # Conical flask (below burette)
    if cv.shown(1):
        flask_bot = bur_top + bur_h + 200
        flask_col = p.hot if lit_equiv else (p.glass if not lit_flask else p.a1)
        _flask(cv, cx - 130, flask_bot - 200, 260, 200, 0.55, flask_col, active=lit_flask)
        label = "Equivalence!" if lit_equiv else ("Analyte + Indicator" if lit_flask else "Flask")
        cv.text((cx, flask_bot + 60), label, size=26, color=p.hot if lit_equiv else p.muted)

    # Indicator note
    if cv.shown(2) and lit_equiv:
        cv.text((cx + 220, bur_top + bur_h + 150), "Colour\nchange", size=28, color=p.hot)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FIGURES: dict = {
    "electrolysis_cell": (
        r"electrolysis|electrolytic|electroplat|electrode|anode|cathode|electrolyte",
        r"\bcell\b|decompos|copper sulphate|brine|salt solution|ions? migrat",
        _electrolysis_cell,
        {"chemistry", "physics"},
    ),
    "test_tube_reaction": (
        r"test.?tube|boiling.?tube|reagent tube|precipitate|effervesc",
        r"\breaction\b|\bgas\b|\bheat\b|\bliquid\b|chemical change",
        _test_tube_reaction,
        {"chemistry"},
    ),
    "distillation": (
        r"\bdistillat\w*\b|fractional distillation|condenser|distil",
        r"\bpure\b|\bseparate\b|\bboiling point\b|\bmixture\b|volatile",
        _distillation,
        {"chemistry"},
    ),
    "titration": (
        r"\btitrat\w*\b|\bburette\b|\bstandardis\w*\b|equivalence point",
        r"\bindicator\b|colour change|end.?point|analyte|neutralis",
        _titration,
        {"chemistry"},
    ),
}
