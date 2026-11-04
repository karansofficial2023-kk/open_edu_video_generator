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

# Electrolysis & Electroplating cell

# ---------------------------------------------------------------------------



def _electrolysis_cell(cv: Cv) -> None:

    """Enhanced electrolysis / electroplating cell:

    Explicit red wire (+) to positive terminal, blue wire (-) to negative terminal with switch,

    prominent electrolytic solution annotation, dynamic anode/cathode callouts,

    and flowing electron (e⁻) and silver cation (Ag⁺) particle animation.

    """

    lit_anode = cv.lit(r"anode|positive terminal|positive side|pure silver rod|oxidation")

    lit_cathode = cv.lit(r"cathode|negative terminal|negative side|keychain|reduction")

    lit_battery = cv.lit(r"batter|power supply|current|voltage|electri|switch|wire")

    lit_electrolyte = cv.lit(r"electrolyte|solution|silver nitrate|agno|ions?|dissol")

    lit_electrons = cv.lit(r"electron|flow|current|travel|reduc|oxida|mov")



    p = cv.pal

    is_electroplating = True



    # Beaker bounds

    bx0, bx1 = 220, 1380

    by0, by1 = 260, 740

    ey0 = by0 + 70  # liquid level: 330



    # 1. Beaker & Electrolytic Solution (Point 3 fix)

    if cv.shown(0):

        liq_alpha = 150 if lit_electrolyte else 115

        cv.d.rectangle([cv.P((bx0 + 6, ey0)), cv.P((bx1 - 6, by1 - 6))], fill=(*p.blue, liq_alpha))

        cv.rect((bx0, by0, bx1, by1), outline=p.ink, w=5, radius=14)

        cv.line((bx0 + 6, ey0), (bx1 - 6, ey0), color=(*p.blue, 230), w=3)



        # Prominent Electrolytic Solution Label Banner

        sol_box_y = by1 - 38

        cv.rect((DW // 2 - 310, sol_box_y - 20, DW // 2 + 310, sol_box_y + 20),

                fill=(*p.fill, 245), outline=p.blue if lit_electrolyte else p.muted, w=3, radius=8)

        sol_text = "Electrolytic Solution: Silver Nitrate (AgNO₃)" if is_electroplating else "Electrolytic Solution (Electrolyte)"

        cv.text((DW // 2, sol_box_y), sol_text, size=24, color=p.ink, bold=True)



    # 2. Coordinates

    electrode_top = 220

    electrode_bot = by1 - 85

    cathode_x = 420   # Left, Negative Cathode (Keychain)

    anode_x = 1180    # Right, Positive Anode (Pure Silver Rod)



    # 3. DC Battery / Power Supply (Point 1 fix)

    mid_x = DW // 2

    bat_y = 52

    bat_w, bat_h = 240, 56

    bat_pos_x = mid_x + bat_w // 2   # 920

    bat_neg_x = mid_x - bat_w // 2   # 680



    if cv.shown(0):

        cv.rect((mid_x - bat_w // 2, bat_y - bat_h // 2, mid_x + bat_w // 2, bat_y + bat_h // 2),

                fill=(*p.fill, 245), outline=p.hot if lit_battery else p.ink, w=4, radius=10)

        cv.text((mid_x, bat_y - 42), "DC Power Supply / Battery", size=22, color=p.hot if lit_battery else p.ink, bold=True)



        # Polarity plates inside battery box

        cv.line((mid_x - 30, bat_y - 16), (mid_x - 30, bat_y + 16), color=p.blue, w=8)

        cv.line((mid_x + 30, bat_y - 22), (mid_x + 30, bat_y + 22), color=p.hot, w=4)



        # Terminal Pill Badges

        cv.rect((mid_x - 110, bat_y - 16, mid_x - 45, bat_y + 16), fill=(*p.blue, 220), outline=p.ink, w=2, radius=6)

        cv.text((mid_x - 77, bat_y), "− Term", size=16, color=(255, 255, 255), bold=True)



        cv.rect((mid_x + 45, bat_y - 16, mid_x + 110, bat_y + 16), fill=(*p.hot, 220), outline=p.ink, w=2, radius=6)

        cv.text((mid_x + 77, bat_y), "+ Term", size=16, color=(255, 255, 255), bold=True)



    # 4. Wiring & Switch (Point 1 fix)

    red_wire_col = (245, 60, 60)

    blue_wire_col = (45, 140, 255)

    sw_cx = 560



    if cv.shown(0):

        # Red Wire (+ terminal -> anode_x -> electrode_top)

        cv.line((bat_pos_x, bat_y), (anode_x, bat_y), color=red_wire_col, w=6)

        cv.line((anode_x, bat_y), (anode_x, electrode_top), color=red_wire_col, w=6)



        # Red Wire Annotation Badge

        red_mid_x = (bat_pos_x + anode_x) // 2

        cv.rect((red_mid_x - 80, bat_y - 28, red_mid_x + 80, bat_y - 4), fill=(*p.hot, 225), outline=p.ink, w=2, radius=6)

        cv.text((red_mid_x, bat_y - 16), "Red Wire (+)", size=16, color=(255, 255, 255), bold=True)



        # Blue Wire (- terminal -> switch -> cathode_x -> electrode_top)

        cv.line((bat_neg_x, bat_y), (sw_cx + 25, bat_y), color=blue_wire_col, w=6)



        # Switch Symbol

        sw_closed = True

        cv.circle((sw_cx + 25, bat_y), 6, fill=blue_wire_col, outline=p.ink, w=2)

        cv.circle((sw_cx - 25, bat_y), 6, fill=blue_wire_col, outline=p.ink, w=2)

        if sw_closed:

            cv.line((sw_cx + 25, bat_y), (sw_cx - 25, bat_y), color=blue_wire_col, w=6)

        else:

            cv.line((sw_cx + 25, bat_y), (sw_cx - 20, bat_y - 18), color=p.muted, w=5)



        cv.text((sw_cx, bat_y - 25), "Switch [Closed]", size=16, color=p.blue, bold=True)



        # Blue wire from switch to cathode_x

        cv.line((sw_cx - 25, bat_y), (cathode_x, bat_y), color=blue_wire_col, w=6)

        cv.line((cathode_x, bat_y), (cathode_x, electrode_top), color=blue_wire_col, w=6)



        # Blue Wire Annotation Badge

        cv.rect((cathode_x - 70, bat_y - 28, cathode_x + 70, bat_y - 4), fill=(*p.blue, 225), outline=p.ink, w=2, radius=6)

        cv.text((cathode_x, bat_y - 16), "Blue Wire (−)", size=16, color=(255, 255, 255), bold=True)



    # 5. Cathode: Copper Keychain (Point 2 highlight fix)

    if cv.shown(1):

        callout_y = 150

        box_w, box_h = 240, 56

        bg_col = (20, 50, 80) if lit_cathode else (*p.fill, 245)

        border_col = p.blue if lit_cathode else p.muted

        cv.rect((cathode_x - box_w // 2, callout_y - box_h // 2, cathode_x + box_w // 2, callout_y + box_h // 2),

                fill=bg_col, outline=border_col, w=4 if lit_cathode else 2, radius=8)

        cv.text((cathode_x, callout_y - 12), "CATHODE (−)", size=22, color=p.blue if lit_cathode else p.ink, bold=True)

        cv.text((cathode_x, callout_y + 14), "Copper Keychain (Object)", size=16, color=p.ink)



        # Hanging Wire Clip

        cv.line((cathode_x, electrode_top), (cathode_x, electrode_top + 45), color=p.blue, w=4)



        # Keychain Silhouette

        ring_cy = electrode_top + 80

        cv.circle((cathode_x, ring_cy), 34, fill=(*p.fill2, 220), outline=p.a2, w=5)

        cv.circle((cathode_x, ring_cy), 16, fill=(*p.blue, 150), outline=p.a2, w=3)

        fob_y0, fob_y1 = ring_cy + 34, electrode_bot

        copper_col = (195, 115, 65)

        cv.rect((cathode_x - 30, fob_y0, cathode_x + 30, fob_y1), fill=copper_col, outline=p.ink, w=4, radius=10)

        cv.line((cathode_x - 18, fob_y0 + 35), (cathode_x + 18, fob_y0 + 35), color=p.ink, w=3)

        cv.line((cathode_x - 18, fob_y0 + 75), (cathode_x + 18, fob_y0 + 75), color=p.ink, w=3)



        # Electroplated Silver Layer (grows with cv.t)

        silver_t = min(1.0, max(0.0, (cv.t - 0.4) / 3.0))

        if silver_t > 0.05:

            plat_w = max(2, round(6 * silver_t))

            cv.rect((cathode_x - 30 - plat_w, fob_y0 - plat_w, cathode_x + 30 + plat_w, fob_y1 + plat_w),

                    outline=(235, 242, 252), w=plat_w, radius=10)

            if silver_t > 0.4:

                cv.text((cathode_x, fob_y0 + 55), "Ag Layer", size=16, color=(255, 255, 255), bold=True)



    # 6. Anode: Pure Silver Rod (Point 2 highlight fix)

    if cv.shown(1):

        callout_y = 150

        box_w, box_h = 240, 56

        bg_col = (80, 30, 30) if lit_anode else (*p.fill, 245)

        border_col = p.hot if lit_anode else p.muted

        cv.rect((anode_x - box_w // 2, callout_y - box_h // 2, anode_x + box_w // 2, callout_y + box_h // 2),

                fill=bg_col, outline=border_col, w=4 if lit_anode else 2, radius=8)

        cv.text((anode_x, callout_y - 12), "ANODE (+)", size=22, color=p.hot if lit_anode else p.ink, bold=True)

        cv.text((anode_x, callout_y + 14), "Pure Silver Rod (Ag)", size=16, color=p.ink)



        # Silver Rod

        silver_col = (220, 228, 238)

        cv.rect((anode_x - 20, electrode_top, anode_x + 20, electrode_bot), fill=silver_col, outline=p.ink, w=4, radius=4)

        cv.line((anode_x - 8, electrode_top + 10), (anode_x - 8, electrode_bot - 10), color=(255, 255, 255), w=3)



    # 7. Animated Electron & Ion Motion (Point 5 & Point 8 fixes)

    if cv.shown(2):

        # Electrons along Blue Wire

        for i in range(4):

            frac = ((cv.t * 0.45) + i * 0.25) % 1.0

            horiz_len = bat_neg_x - cathode_x

            vert_len = electrode_top - bat_y

            tot_len = horiz_len + vert_len

            d = frac * tot_len

            if d < horiz_len:

                ex = bat_neg_x - d

                ey = bat_y

            else:

                ex = cathode_x

                ey = bat_y + (d - horiz_len)

            cv.circle((ex, ey), 8, fill=(255, 240, 70), outline=p.ink, w=2)

            cv.text((ex, ey - 14), "e⁻", size=15, color=(255, 240, 70), bold=True)



        cv.arrow((sw_cx + 70, bat_y + 20), (sw_cx - 70, bat_y + 20), color=(255, 240, 70), w=3, head=14)

        cv.text((sw_cx, bat_y + 35), "e⁻ flow to Cathode", size=16, color=(255, 240, 70), bold=True)



        # Silver Cations (Ag⁺) migrating across the solution

        span_x = anode_x - cathode_x - 140

        for k in range(5):

            u = ((cv.t * 0.2) + k * 0.2) % 1.0

            ix = anode_x - 70 - span_x * u

            iy = ey0 + 45 + (by1 - ey0 - 130) * (0.2 + 0.16 * k)

            cv.circle((ix, iy), 18, fill=(225, 240, 255), outline=p.a1, w=3)

            cv.text((ix, iy), "Ag⁺", size=18, color=(20, 40, 70), bold=True)



        # Nitrate Anions (NO₃⁻)

        for k in range(3):

            nx = cathode_x + 110 + (anode_x - cathode_x - 220) * ((k * 0.35 + 0.2) % 1.0)

            ny = ey0 + 75 + k * 65

            cv.circle((nx, ny), 16, fill=(*p.fill, 200), outline=p.blue, w=2)

            cv.text((nx, ny), "NO₃⁻", size=15, color=p.blue, bold=True)



        # Migration arrow

        cv.arrow((anode_x - 140, by1 - 90), (cathode_x + 140, by1 - 90), color=p.a1, w=4, head=18)

        cv.text((DW // 2, by1 - 85), "Ag⁺ cations migrate toward negative Cathode keychain", size=20, color=p.a1, bold=True)



        # Reaction Badges inside beaker

        cv.rect((cathode_x - 110, by1 - 145, cathode_x + 110, by1 - 105), fill=(*p.fill, 235), outline=p.blue, w=2, radius=6)

        cv.text((cathode_x, by1 - 125), "Ag⁺ + e⁻ → Ag (Reduction)", size=16, color=p.blue, bold=True)



        cv.rect((anode_x - 110, by1 - 145, anode_x + 110, by1 - 105), fill=(*p.fill, 235), outline=p.hot, w=2, radius=6)

        cv.text((anode_x, by1 - 125), "Ag → Ag⁺ + e⁻ (Oxidation)", size=16, color=p.hot, bold=True)





# ---------------------------------------------------------------------------

# Silver Ion Outermost Shell (Point 7 fix)

# ---------------------------------------------------------------------------



def _silver_ion_shell(cv: Cv) -> None:

    """Bohr electron configuration of Silver (Ag⁺ cation):

    Nucleus +47, filled inner shells (n=1..4), outermost shell (n=5) with empty electron vacancy,

    and an incoming electron (e⁻) from cathode filling the vacancy to form neutral Ag.

    """

    p = cv.pal

    cx = DW // 2

    cy = 425



    # Shell radii

    radii = [55, 95, 140, 185, 235]  # n=1 (K), n=2 (L), n=3 (M), n=4 (N), n=5 (O valence)



    # Nucleus

    if cv.shown(0):

        cv.circle((cx, cy), 32, fill=(225, 65, 65), outline=p.ink, w=3)

        cv.text((cx, cy - 8), "Ag⁺", size=20, color=(255, 255, 255), bold=True)

        cv.text((cx, cy + 12), "+47p", size=14, color=(255, 255, 255), bold=True)



    # Shells 1-4 (Filled Inner Orbitals)

    electrons_per_shell = [2, 8, 18, 18]

    for s_idx, (r, count) in enumerate(zip(radii[:4], electrons_per_shell)):

        if cv.shown(s_idx):

            cv.circle((cx, cy), r, outline=p.muted, w=2)

            cv.text((cx + r - 12, cy - 8), f"n={s_idx+1}", size=14, color=p.muted)

            for e_idx in range(count):

                angle = 2 * math.pi * e_idx / count

                ex = cx + r * math.cos(angle)

                ey = cy + r * math.sin(angle)

                cv.circle((ex, ey), 5, fill=(*p.blue, 230), outline=p.ink, w=1)



    # Shell 5: Outermost Valence Shell (5s orbital) with Empty Spot

    if cv.shown(3):

        r5 = radii[4]

        cv.circle((cx, cy), r5, outline=p.a2, w=3)

        cv.text((cx + r5 - 20, cy - 12), "n=5 (5s)", size=16, color=p.a2, bold=True)



        # The Empty Spot (Electron Vacancy) at top of shell 5

        spot_x = cx

        spot_y = cy - r5

        cv.circle((spot_x, spot_y), 16, fill=(*p.fill, 180), outline=p.hot, w=3)

        cv.text((spot_x, spot_y), "0", size=14, color=p.hot, bold=True)



        # Vacancy Callout Box

        call_x = cx + 240

        call_y = spot_y - 45

        cv.rect((call_x - 170, call_y - 24, call_x + 170, call_y + 24), fill=(*p.fill, 235), outline=p.hot, w=2, radius=8)

        cv.text((call_x, call_y), "Empty Spot (5s orbital vacancy)", size=18, color=p.hot, bold=True)

        cv.arrow((call_x - 170, call_y + 10), (spot_x + 20, spot_y - 6), color=p.hot, w=3, head=14)



        # Incoming electron (e⁻) from cathode keychain

        e_anim = min(1.0, max(0.0, (cv.t - 1.0) * 0.5))

        start_x, start_y = cx - 180, spot_y - 75

        curr_x = start_x + (spot_x - 24 - start_x) * e_anim

        curr_y = start_y + (spot_y - start_y) * e_anim



        cv.circle((curr_x, curr_y), 12, fill=(255, 235, 60), outline=p.ink, w=2)

        cv.text((curr_x, curr_y), "e⁻", size=16, color=(20, 20, 20), bold=True)

        cv.arrow((start_x, start_y), (spot_x - 26, spot_y - 4), color=(255, 235, 60), w=3, head=14, dash=6)

        cv.text((start_x - 20, start_y - 20), "Incoming e⁻ from Cathode", size=16, color=p.a2, bold=True)



    # Bottom Explanation Cards

    if cv.shown(1):

        by0, by1 = DH - 110, DH - 20

        cv.rect((cx - 520, by0, cx + 520, by1), fill=(*p.fill, 240), outline=p.a1, w=3, radius=10)

        cv.text((cx, by0 + 26), "Ag⁺ Cation: [Kr] 4d¹⁰ 5s⁰   +   1 e⁻   ->   Neutral Ag Atom: [Kr] 4d¹⁰ 5s¹", size=24, color=p.a2, bold=True)

        cv.text((cx, by0 + 62), "Attracting 1 electron fills the vacant 5s orbital, transforming the dissolved ion into a neutral silver atom.", size=19, color=p.ink)





# ---------------------------------------------------------------------------

# Dual Redox Reaction Balance Board (Point 6 fix)

# ---------------------------------------------------------------------------



def _redox_reaction_board(cv: Cv) -> None:

    """Dual Redox Balance: Cathode Reduction vs Anode Oxidation and Net Reaction."""

    p = cv.pal

    cx = DW // 2



    # Split Two Columns: Left Cathode Reduction, Right Anode Oxidation

    col_w = 640

    col_h = 470

    top_y = 50

    lx0 = cx - col_w - 25

    rx0 = cx + 25



    # Left Column: CATHODE (Reduction)

    if cv.shown(0):

        cv.rect((lx0, top_y, lx0 + col_w, top_y + col_h), fill=(*p.fill, 235), outline=p.blue, w=4, radius=12)

        # Header strip

        cv.rect((lx0, top_y, lx0 + col_w, top_y + 55), fill=(*p.blue, 210), outline=p.blue, w=2, radius=12)

        cv.text((lx0 + col_w // 2, top_y + 28), "CATHODE (−) : REDUCTION (Gain of e⁻)", size=22, color=(255, 255, 255), bold=True)

        cv.text((lx0 + col_w // 2, top_y + 80), "[Copper Keychain to be Plated]", size=19, color=p.blue, bold=True)



        # Equation Card

        cv.rect((lx0 + 35, top_y + 110, lx0 + col_w - 35, top_y + 185), fill=(*p.glass, 140), outline=p.blue, w=2, radius=8)

        cv.text((lx0 + col_w // 2, top_y + 148), "Ag⁺(aq)  +  e⁻   ->   Ag(solid)", size=32, color=p.ink, bold=True)



        # Mechanism Bullet Points

        points = [

            "• Positive Ag⁺ ions migrate to the negative keychain",

            "• Keychain supplies electrons (e⁻) from battery",

            "• Each Ag⁺ ion is REDUCED to neutral metallic silver",

            "• Forms a shiny, uniform electroplated silver layer",

        ]

        for idx, pt in enumerate(points):

            cv.text((lx0 + 40, top_y + 230 + idx * 48), pt, size=19, color=p.ink, anchor="lm")



    # Right Column: ANODE (Oxidation)

    if cv.shown(0):

        cv.rect((rx0, top_y, rx0 + col_w, top_y + col_h), fill=(*p.fill, 235), outline=p.hot, w=4, radius=12)

        # Header strip

        cv.rect((rx0, top_y, rx0 + col_w, top_y + 55), fill=(*p.hot, 210), outline=p.hot, w=2, radius=12)

        cv.text((rx0 + col_w // 2, top_y + 28), "ANODE (+) : OXIDATION (Loss of e⁻)", size=22, color=(255, 255, 255), bold=True)

        cv.text((rx0 + col_w // 2, top_y + 80), "[Pure Silver Rod (Ag)]", size=19, color=p.hot, bold=True)



        # Equation Card

        cv.rect((rx0 + 35, top_y + 110, rx0 + col_w - 35, top_y + 185), fill=(*p.fill2, 140), outline=p.hot, w=2, radius=8)

        cv.text((rx0 + col_w // 2, top_y + 148), "Ag(solid)   ->   Ag⁺(aq)  +  e⁻", size=32, color=p.ink, bold=True)



        # Mechanism Bullet Points

        points = [

            "• Pure silver rod connects to positive battery terminal",

            "• Silver atoms LOSE electrons (e⁻) → Oxidation",

            "• Enters electrolyte as Ag⁺ ions to replenish solution",

            "• Released electrons travel through wire to battery",

        ]

        for idx, pt in enumerate(points):

            cv.text((rx0 + 40, top_y + 230 + idx * 48), pt, size=19, color=p.ink, anchor="lm")



    # Bottom Full-Width Banner: Net Redox Reaction

    if cv.shown(1):

        ny0 = top_y + col_h + 20

        ny1 = DH - 20

        cv.rect((lx0, ny0, rx0 + col_w, ny1), fill=(*p.fill, 245), outline=p.a2, w=3, radius=10)

        cv.text((cx, ny0 + 35), "NET OVERALL REACTION:   Ag(anode rod)   ->   Ag(keychain coating)", size=26, color=p.a2, bold=True)

        cv.text((cx, ny0 + 78), "Silver mass lost by Anode = Silver mass deposited on Keychain. Ag⁺ concentration in solution stays CONSTANT.", size=20, color=p.ink)

        cv.text((cx, ny0 + 112), "Electric current acts as the driving pump, transferring silver atoms onto Vivant's gift.", size=18, color=p.muted)





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

# ---------------------------------------------------------------------------

# Combustion & Thermodynamics figures

# ---------------------------------------------------------------------------



def _draw_flame(cv: Cv, cx: float, cy: float, w: float, h: float, flame_type: str, t: float = 0.0) -> None:

    flicker = 1.0 + 0.08 * math.sin(t * 14.0)

    w_f = w * flicker

    h_f = h * flicker

    

    if flame_type == "lean":

        cv.circle((cx, cy - h_f * 0.2), w_f * 0.9, fill=(0, 180, 240, 35))

        cv.ellipse((cx - w_f * 0.45, cy - h_f, cx + w_f * 0.45, cy + h_f * 0.1), fill=(0, 210, 255, 120))

        cv.ellipse((cx - w_f * 0.25, cy - h_f * 0.8, cx + w_f * 0.25, cy), fill=(180, 240, 255, 180))

        cv.circle((cx, cy - h_f * 0.4), w_f * 0.15, fill=(255, 255, 255, 220))

    elif flame_type == "stoich":

        cv.circle((cx, cy - h_f * 0.25), w_f * 1.1, fill=(80, 120, 255, 45))

        cv.ellipse((cx - w_f * 0.5, cy - h_f * 1.1, cx + w_f * 0.5, cy + h_f * 0.1), fill=(60, 150, 255, 160))

        cv.ellipse((cx - w_f * 0.32, cy - h_f * 0.95, cx + w_f * 0.32, cy), fill=(130, 210, 255, 220))

        cv.ellipse((cx - w_f * 0.18, cy - h_f * 0.75, cx + w_f * 0.18, cy - h_f * 0.1), fill=(240, 250, 255, 250))

    else:  # rich

        cv.circle((cx, cy - h_f * 0.3), w_f * 1.2, fill=(255, 120, 0, 40))

        cv.ellipse((cx - w_f * 0.55, cy - h_f * 1.15, cx + w_f * 0.55, cy + h_f * 0.1), fill=(255, 90, 0, 150))

        cv.ellipse((cx - w_f * 0.4, cy - h_f * 0.95, cx + w_f * 0.4, cy), fill=(255, 180, 0, 210))

        cv.ellipse((cx - w_f * 0.22, cy - h_f * 0.7, cx + w_f * 0.22, cy - h_f * 0.05), fill=(255, 240, 120, 240))

        soot_drift = (t * 28.0) % 25.0

        cv.circle((cx - 6, cy - h_f * 1.18 - soot_drift), 4.5, fill=(60, 60, 60, 160))

        cv.circle((cx + 7, cy - h_f * 1.25 - soot_drift * 0.8), 3.5, fill=(40, 40, 40, 140))





def _equivalence_ratio_dial(cv: Cv) -> None:

    """Equivalence ratio (φ) spectrum analyzer: Lean, Stoichiometric, and Rich regimes."""

    cv.rect((60, 30, 1540, 790), radius=16, fill=(12, 19, 32, 245), outline=(55, 80, 120, 220), w=3)

    

    # Title Header

    cv.text((800, 65), "EQUIVALENCE RATIO (φ) SPECTRUM ANALYZER", color=(255, 255, 255), size=28, bold=True, anchor="mm")

    cv.text((800, 98), "φ = (Actual Fuel-to-Air Ratio) / (Stoichiometric Fuel-to-Air Ratio)", color=(180, 215, 245), size=18, anchor="mm")

    

    is_lean = cv.lit(r"lean|excess air|below one|less than one|phi.*0|cooler flame|fuel economy")

    is_rich = cv.lit(r"rich|excess fuel|exceeds one|greater than one|more than one|soot|unburned")

    is_stoich = cv.lit(r"stoichiometric|equals one|equal to one|perfect balance|complete combustion") or (not is_lean and not is_rich and cv.lit(r"phi|ratio|mixture"))

    

    lean_active = is_lean or (not is_rich and not is_stoich and (cv.t % 9.0 < 3.0))

    stoich_active = is_stoich or (not is_lean and not is_rich and (3.0 <= cv.t % 9.0 < 6.0))

    rich_active = is_rich or (not is_lean and not is_stoich and (cv.t % 9.0 >= 6.0))

    

    card_y = 130

    card_h = 515

    card_w = 440

    

    # 1. LEAN CARD

    lx0 = 90

    l_bg = (16, 48, 76, 245) if lean_active else (14, 26, 40, 190)

    l_border = (0, 215, 255, 255) if lean_active else (40, 72, 105, 160)

    cv.rect((lx0, card_y, lx0 + card_w, card_y + card_h), radius=12, fill=l_bg, outline=l_border, w=4 if lean_active else 2)

    

    # Header badge

    cv.rect((lx0 + 50, card_y + 16, lx0 + card_w - 50, card_y + 54), radius=8, fill=(0, 140, 205) if lean_active else (25, 55, 88), outline=(0, 215, 255) if lean_active else None, w=2)

    cv.text((lx0 + card_w // 2, card_y + 35), "LEAN REGIME", color=(255, 255, 255), size=19, bold=True, anchor="mm")

    

    cv.text((lx0 + card_w // 2, card_y + 82), "φ < 1.0", color=(0, 240, 255) if lean_active else (145, 190, 220), size=32, bold=True, anchor="mm")

    cv.text((lx0 + card_w // 2, card_y + 114), "Excess Air (Oxygen Rich)", color=(210, 238, 255), size=17, bold=True, anchor="mm")

    

    _draw_flame(cv, lx0 + card_w // 2, card_y + 270, 75, 125, "lean", cv.t)

    

    pts_lean = [

        "• More air than chemically required",

        "• Pale blue, lower peak temperature",

        "• Peak thermal / fuel efficiency",

        "• Low carbon monoxide (CO) emission",

        "• Lean misfire limit if φ too low"

    ]

    for idx, pt in enumerate(pts_lean):

        cv.text((lx0 + 26, card_y + 360 + idx * 30), pt, color=(220, 238, 252), size=17, anchor="lm")

        

    # 2. STOICHIOMETRIC CARD

    sx0 = 580

    s_bg = (18, 62, 44, 245) if stoich_active else (14, 35, 28, 190)

    s_border = (46, 225, 125, 255) if stoich_active else (40, 95, 68, 160)

    cv.rect((sx0, card_y, sx0 + card_w, card_y + card_h), radius=12, fill=s_bg, outline=s_border, w=4 if stoich_active else 2)

    

    cv.rect((sx0 + 50, card_y + 16, sx0 + card_w - 50, card_y + 54), radius=8, fill=(35, 180, 98) if stoich_active else (25, 78, 48), outline=(50, 235, 135) if stoich_active else None, w=2)

    cv.text((sx0 + card_w // 2, card_y + 35), "STOICHIOMETRIC", color=(255, 255, 255), size=19, bold=True, anchor="mm")

    

    cv.text((sx0 + card_w // 2, card_y + 82), "φ = 1.0", color=(80, 255, 165) if stoich_active else (145, 215, 175), size=32, bold=True, anchor="mm")

    cv.text((sx0 + card_w // 2, card_y + 114), "Perfect Chemical Balance", color=(215, 255, 230), size=17, bold=True, anchor="mm")

    

    _draw_flame(cv, sx0 + card_w // 2, card_y + 270, 85, 135, "stoich", cv.t)

    

    pts_stoich = [

        "• 100% complete combustion",

        "• Intense high-velocity blue flame",

        "• Theoretical peak heat release",

        "• Clean exhaust: CO₂ + H₂O only",

        "• Baseline reference standard"

    ]

    for idx, pt in enumerate(pts_stoich):

        cv.text((sx0 + 26, card_y + 360 + idx * 30), pt, color=(225, 248, 232), size=17, anchor="lm")

        

    # 3. RICH CARD

    rx0 = 1070

    r_bg = (68, 26, 22, 245) if rich_active else (40, 18, 16, 190)

    r_border = (255, 105, 65, 255) if rich_active else (120, 48, 38, 160)

    cv.rect((rx0, card_y, rx0 + card_w, card_y + card_h), radius=12, fill=r_bg, outline=r_border, w=4 if rich_active else 2)

    

    cv.rect((rx0 + 50, card_y + 16, rx0 + card_w - 50, card_y + 54), radius=8, fill=(220, 58, 36) if rich_active else (100, 34, 26), outline=(255, 115, 75) if rich_active else None, w=2)

    cv.text((rx0 + card_w // 2, card_y + 35), "RICH REGIME", color=(255, 255, 255), size=19, bold=True, anchor="mm")

    

    cv.text((rx0 + card_w // 2, card_y + 82), "φ > 1.0", color=(255, 135, 95) if rich_active else (220, 150, 130), size=32, bold=True, anchor="mm")

    cv.text((rx0 + card_w // 2, card_y + 114), "Excess Fuel (Air Deficit)", color=(255, 228, 218), size=17, bold=True, anchor="mm")

    

    _draw_flame(cv, rx0 + card_w // 2, card_y + 270, 88, 142, "rich", cv.t)

    

    pts_rich = [

        "• More fuel than oxygen can burn",

        "• Luminous orange flame with soot",

        "• Peak mechanical power & torque",

        "• Produces unburned HC and toxic CO",

        "• Cooler combustion wall protection"

    ]

    for idx, pt in enumerate(pts_rich):

        cv.text((rx0 + 26, card_y + 360 + idx * 30), pt, color=(252, 228, 222), size=17, anchor="lm")

        

    # Bottom Spectrum Gauge Bar

    bar_y = 695

    bar_x1 = 160

    bar_x2 = 1440

    bar_w = bar_x2 - bar_x1

    cv.rect((bar_x1, bar_y - 6, bar_x2, bar_y + 6), radius=5, fill=(28, 42, 60, 220))

    # Lean track

    cv.rect((bar_x1, bar_y - 6, bar_x1 + bar_w * 0.44, bar_y + 6), radius=4, fill=(0, 195, 245, 220))

    # Stoich notch

    stoich_px = bar_x1 + bar_w * 0.5

    cv.circle((stoich_px, bar_y), 11, fill=(46, 225, 125, 255), outline=(255, 255, 255), w=3)

    # Rich track

    cv.rect((bar_x1 + bar_w * 0.56, bar_y - 6, bar_x2, bar_y + 6), radius=4, fill=(255, 80, 50, 220))

    

    if lean_active:

        target_pos = bar_x1 + bar_w * 0.22

        val_text = "CURRENT OPERATING POINT: φ = 0.82 (LEAN COMBUSTION)"

        needle_col = (0, 230, 255)

    elif rich_active:

        target_pos = bar_x1 + bar_w * 0.78

        val_text = "CURRENT OPERATING POINT: φ = 1.30 (RICH COMBUSTION)"

        needle_col = (255, 95, 65)

    else:

        target_pos = stoich_px

        val_text = "CURRENT OPERATING POINT: φ = 1.00 (STOICHIOMETRIC COMBUSTION)"

        needle_col = (80, 255, 160)

        

    cv.arrow((target_pos, bar_y + 42), (target_pos, bar_y + 12), color=needle_col, w=5, head=18)

    cv.rect((target_pos - 260, bar_y + 44, target_pos + 260, bar_y + 80), radius=6, fill=(16, 26, 42, 240), outline=needle_col, w=2)

    cv.text((target_pos, bar_y + 62), val_text, color=(255, 255, 255), size=17, bold=True, anchor="mm")





def _combustion_chamber(cv: Cv) -> None:

    """Internal combustion engine cylinder with reciprocating piston, spark discharge, and turbulent flame propagation."""

    cv.rect((60, 30, 1540, 790), radius=16, fill=(12, 18, 30, 245), outline=(55, 78, 110, 215), w=3)

    cv.text((800, 65), "INTERNAL COMBUSTION CHAMBER DYNAMICS", color=(255, 255, 255), size=28, bold=True, anchor="mm")

    cv.text((800, 98), "Air-Fuel Mixing, Spark Ignition, and High-Pressure Flame Propagation", color=(175, 210, 245), size=18, anchor="mm")

    

    cyl_x1 = 530

    cyl_x2 = 1070

    cyl_top = 145

    cyl_bottom = 690

    

    wall_w = 55

    cv.rect((cyl_x1 - wall_w, cyl_top, cyl_x1, cyl_bottom), fill=(36, 46, 62, 255), outline=(75, 95, 125), w=3)

    for hy in range(cyl_top + 12, cyl_bottom, 22):

        cv.line((cyl_x1 - wall_w + 6, hy), (cyl_x1 - 6, hy + 16), color=(60, 78, 102, 210), w=3)

        

    cv.rect((cyl_x2, cyl_top, cyl_x2 + wall_w, cyl_bottom), fill=(36, 46, 62, 255), outline=(75, 95, 125), w=3)

    for hy in range(cyl_top + 12, cyl_bottom, 22):

        cv.line((cyl_x2 + 6, hy), (cyl_x2 + wall_w - 6, hy + 16), color=(60, 78, 102, 210), w=3)

        

    cv.rect((cyl_x1, cyl_top, cyl_x2, cyl_bottom), fill=(18, 25, 38, 255))

    

    cycle_speed = 0.85

    stroke_amplitude = 65.0

    piston_top_base = 470.0

    piston_displacement = stroke_amplitude * math.cos(2.0 * math.pi * cycle_speed * cv.t)

    piston_y = piston_top_base + piston_displacement

    

    flame_cx = (cyl_x1 + cyl_x2) / 2

    flame_cy = cyl_top + 35

    

    is_rich = cv.lit("rich", "soot", "excess fuel")

    is_lean = cv.lit("lean", "excess air")

    flame_type = "rich" if is_rich else ("lean" if is_lean else "stoich")

    

    kernel_r = min(175.0, 45.0 + (cv.t * 48.0) % 130.0)

    if flame_type == "stoich":

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 1.3, fill=(60, 125, 255, 35))

        cv.circle((flame_cx, flame_cy + 35), kernel_r, fill=(80, 185, 255, 85))

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 0.6, fill=(205, 245, 255, 190))

    elif flame_type == "lean":

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 1.15, fill=(0, 165, 235, 30))

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 0.8, fill=(0, 205, 255, 75))

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 0.45, fill=(185, 235, 255, 170))

    else:

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 1.35, fill=(255, 105, 25, 40))

        cv.circle((flame_cx, flame_cy + 35), kernel_r, fill=(255, 145, 0, 95))

        cv.circle((flame_cx, flame_cy + 35), kernel_r * 0.55, fill=(255, 235, 110, 200))

        

    # Moving reactant molecules in cylinder

    for i in range(16):

        px = cyl_x1 + 45 + (i * 39 + int(cv.t * 55)) % (cyl_x2 - cyl_x1 - 90)

        py = cyl_top + 35 + (i * 26 + int(cv.t * 35)) % int(max(15, piston_y - cyl_top - 55))

        if i % 2 == 0:

            cv.circle((px, py), 6.0, fill=(0, 225, 255, 220))

            cv.circle((px + 7, py), 6.0, fill=(0, 225, 255, 220))

        else:

            cv.circle((px, py), 7.5, fill=(255, 185, 35, 235))

            

    # Spark Plug Assembly at top

    sp_w = 38

    cv.rect((flame_cx - sp_w // 2, cyl_top - 28, flame_cx + sp_w // 2, cyl_top + 14), fill=(215, 220, 232), outline=(135, 145, 160), w=3)

    cv.line((flame_cx, cyl_top + 14), (flame_cx, cyl_top + 30), color=(245, 245, 245), w=4)

    cv.line((flame_cx - 9, cyl_top + 30), (flame_cx + 3, cyl_top + 30), color=(205, 205, 205), w=4)

    

    spark_active = math.sin(cv.t * 16.0) > 0.35

    if spark_active:

        cv.circle((flame_cx, cyl_top + 30), 16, fill=(255, 255, 255, 240))

        cv.circle((flame_cx, cyl_top + 30), 30, fill=(255, 245, 110, 110))

        

    # Piston Head & Compression Rings

    piston_h = 125

    cv.rect((cyl_x1 + 3, piston_y, cyl_x2 - 3, piston_y + piston_h), radius=8, fill=(82, 98, 120), outline=(155, 175, 202), w=4)

    for ry in range(int(piston_y + 18), int(piston_y + 54), 12):

        cv.line((cyl_x1 + 4, ry), (cyl_x2 - 4, ry), color=(30, 42, 54), w=4)

        

    # Wrist Pin & Connecting Rod

    pin_y = piston_y + 65

    cv.circle((flame_cx, pin_y), 22, fill=(48, 58, 72), outline=(185, 200, 220), w=3)

    cv.line((flame_cx - 15, pin_y + 18), (flame_cx - 20, cyl_bottom + 10), color=(98, 115, 136), w=10)

    cv.line((flame_cx + 15, pin_y + 18), (flame_cx + 20, cyl_bottom + 10), color=(98, 115, 136), w=10)

    cv.rect((flame_cx - 18, pin_y + 18, flame_cx + 18, cyl_bottom + 10), fill=(78, 92, 110))

    

    # Left Callout Card

    cv.rect((90, 175, 440, 600), radius=10, fill=(16, 32, 50, 240), outline=(0, 205, 255), w=2)

    cv.rect((110, 190, 420, 230), radius=6, fill=(0, 140, 195), outline=(0, 220, 255), w=2)

    cv.text((265, 210), "INCOMING CHARGE", color=(255, 255, 255), size=18, bold=True, anchor="mm")

    cv.text((115, 260), "• O₂ Oxidizer Molecules", color=(0, 230, 255), size=17, anchor="lm")

    cv.text((115, 298), "• Hydrocarbon Fuel Spray", color=(255, 195, 60), size=17, anchor="lm")

    cv.text((115, 336), "• Tumble & Swirl Flow", color=(210, 230, 250), size=17, anchor="lm")

    cv.text((115, 374), "• Equivalence Ratio (φ)", color=(210, 230, 250), size=17, anchor="lm")

    cv.text((115, 412), "• Governs Flame Velocity", color=(210, 230, 250), size=17, anchor="lm")

    cv.arrow((265, 450), (cyl_x1 - 15, cyl_top + 60), color=(0, 210, 255), w=4, head=16)

    

    # Right Callout Card

    cv.rect((1160, 175, 1510, 600), radius=10, fill=(48, 24, 20, 240), outline=(255, 120, 60), w=2)

    cv.rect((1180, 190, 1490, 230), radius=6, fill=(215, 75, 40), outline=(255, 130, 70), w=2)

    cv.text((1335, 210), "FLAME PROPAGATION", color=(255, 255, 255), size=18, bold=True, anchor="mm")

    cv.text((1185, 260), "• High-Pressure Wavefront", color=(255, 225, 210), size=17, anchor="lm")

    cv.text((1185, 298), "• Turbulent Reaction Zone", color=(255, 225, 210), size=17, anchor="lm")

    cv.text((1185, 336), "• Rapid Enthalpy Release", color=(255, 225, 210), size=17, anchor="lm")

    cv.text((1185, 374), "• Drives Piston Face Down", color=(255, 225, 210), size=17, anchor="lm")

    cv.text((1185, 412), "• Converts Heat to Work", color=(255, 225, 210), size=17, anchor="lm")

    cv.arrow((1335, 450), (cyl_x2 + 15, piston_y - 25), color=(255, 125, 60), w=4, head=16)

    

    # Bottom HUD summary banner

    cv.rect((150, 715, 1450, 765), radius=8, fill=(18, 30, 46, 230), outline=(55, 82, 120), w=2)

    cv.text((800, 740), "Combustion Power Stroke: Chemical Bond Energy -> Thermal Expansion -> Mechanical Shaft Work", color=(240, 248, 255), size=19, bold=True, anchor="mm")





# ---------------------------------------------------------------------------

# Automotive & Internal Combustion Engine Dynamics (4-Stroke & AFR Tuning)

# ---------------------------------------------------------------------------



def _draw_flame(cv: Cv, cx: float, cy: float, w: float, h: float, flame_type: str, t: float = 0.0) -> None:

    flicker = 1.0 + 0.08 * math.sin(t * 14.0)

    w_f = w * flicker

    h_f = h * flicker

    

    if flame_type == "lean":

        cv.circle((cx, cy - h_f * 0.2), w_f * 0.9, fill=(0, 180, 240, 35))

        cv.ellipse((cx - w_f * 0.45, cy - h_f, cx + w_f * 0.45, cy + h_f * 0.1), fill=(0, 210, 255, 120))

        cv.ellipse((cx - w_f * 0.25, cy - h_f * 0.8, cx + w_f * 0.25, cy), fill=(180, 240, 255, 180))

        cv.circle((cx, cy - h_f * 0.4), w_f * 0.15, fill=(255, 255, 255, 220))

    elif flame_type == "stoich":

        cv.circle((cx, cy - h_f * 0.25), w_f * 1.1, fill=(80, 120, 255, 45))

        cv.ellipse((cx - w_f * 0.5, cy - h_f * 1.1, cx + w_f * 0.5, cy + h_f * 0.1), fill=(60, 150, 255, 160))

        cv.ellipse((cx - w_f * 0.32, cy - h_f * 0.95, cx + w_f * 0.32, cy), fill=(130, 210, 255, 220))

        cv.ellipse((cx - w_f * 0.18, cy - h_f * 0.75, cx + w_f * 0.18, cy - h_f * 0.1), fill=(240, 250, 255, 250))

    else:  # rich

        cv.circle((cx, cy - h_f * 0.3), w_f * 1.2, fill=(255, 120, 0, 40))

        cv.ellipse((cx - w_f * 0.55, cy - h_f * 1.15, cx + w_f * 0.55, cy + h_f * 0.1), fill=(255, 90, 0, 150))

        cv.ellipse((cx - w_f * 0.4, cy - h_f * 0.95, cx + w_f * 0.4, cy), fill=(255, 180, 0, 210))

        cv.ellipse((cx - w_f * 0.22, cy - h_f * 0.7, cx + w_f * 0.22, cy - h_f * 0.05), fill=(255, 240, 120, 240))

        soot_drift = (t * 28.0) % 25.0

        cv.circle((cx - 6, cy - h_f * 1.18 - soot_drift), 4.5, fill=(60, 60, 60, 160))

        cv.circle((cx + 7, cy - h_f * 1.25 - soot_drift * 0.8), 3.5, fill=(40, 40, 40, 140))





def _draw_spark_plug_tip(cv: Cv, cx: float, cy: float, condition: str) -> None:

    """Renders a realistic spark plug tip with insulator condition:

    'lean' -> overheated white/blistered

    'stoich' -> ideal healthy grey-tan

    'rich' -> dark black carbon fouled

    """

    # Outer threaded shell cutaway

    cv.rect((cx - 42, cy - 50, cx - 18, cy + 15), fill=(140, 150, 165), outline=(70, 80, 95), w=2)

    cv.rect((cx + 18, cy - 50, cx + 42, cy + 15), fill=(140, 150, 165), outline=(70, 80, 95), w=2)

    for ty in range(int(cy - 40), int(cy + 10), 10):

        cv.line((cx - 42, ty), (cx - 18, ty + 3), color=(60, 70, 80), w=2)

        cv.line((cx + 18, ty), (cx + 42, ty + 3), color=(60, 70, 80), w=2)

        

    # Ceramic insulator nose

    if condition == "lean":

        ins_col = (250, 250, 255) # stark white / overheated

        tip_detail = "Overheated (White / Blistered)"

        detail_col = (255, 120, 120)

    elif condition == "stoich":

        ins_col = (215, 195, 160) # healthy grey-tan

        tip_detail = "Healthy (Light Tan / Grey)"

        detail_col = (80, 255, 160)

    else: # rich

        ins_col = (45, 45, 48)    # carbon fouled black

        tip_detail = "Carbon Fouled (Soot Black)"

        detail_col = (255, 140, 60)

        

    cv.rect((cx - 17, cy - 45, cx + 17, cy + 22), radius=4, fill=ins_col, outline=(90, 95, 105), w=2)

    

    # Center electrode

    cv.rect((cx - 4, cy - 20, cx + 4, cy + 32), fill=(180, 185, 195), outline=(100, 105, 115), w=2)

    # Ground electrode hook

    cv.line((cx - 30, cy + 10), (cx - 30, cy + 39), color=(150, 155, 165), w=6)

    cv.line((cx - 30, cy + 39), (cx + 2, cy + 39), color=(150, 155, 165), w=6)

    

    # Spark gap annotation

    cv.text((cx, cy + 62), tip_detail, size=15, color=detail_col, bold=True, anchor="mm")





def _four_stroke_cycle(cv: Cv) -> None:

    """Realistic 4-stroke internal combustion engine animation:

    1. Intake: Intake valve open, mixture enters, piston moves down (TDC -> BDC).

    2. Compression: Both valves closed, charge compressed, piston moves up (BDC -> TDC).

    3. Combustion / Power: Spark plug discharges, intense expanding flame front, piston driven down.

    4. Exhaust: Exhaust valve open, burned gases swept out, piston moves up.

    Includes rotating crankshaft counterweight and articulated connecting rod!

    """

    cv.rect((60, 30, 1540, 790), radius=16, fill=(12, 18, 30, 245), outline=(55, 80, 120, 220), w=3)

    cv.text((800, 62), "FOUR-STROKE INTERNAL COMBUSTION ENGINE DYNAMICS", color=(255, 255, 255), size=27, bold=True, anchor="mm")

    cv.text((800, 94), "1. Intake Stroke  •  2. Compression Stroke  •  3. Power (Combustion) Stroke  •  4. Exhaust Stroke", color=(180, 215, 245), size=17, anchor="mm")

    

    # Stroke timing calculation: 6.0s full cycle (1.5s per stroke)

    cycle_period = 6.0

    curr_t = cv.t % cycle_period

    stroke_idx = int(curr_t / 1.5) # 0: Intake, 1: Compression, 2: Power, 3: Exhaust

    

    # Check if narration specifically focuses on a stroke

    if cv.lit(r"\bintake\b|suction|fuel[- ]air enters?|air enters?|intake valve|downward stroke"):

        stroke_idx = 0

    elif cv.lit(r"compress\w*|squeez\w*|both valves closed|upward stroke.*valves closed|compression ratio"):

        stroke_idx = 1

    elif cv.lit(r"power\b|combust\w*|spark plug fires?|ignit\w*|flame front|flame expands?|drives the piston"):

        stroke_idx = 2

    elif cv.lit(r"exhaust\b|burned gas\w*|exhaust valve|expel\w*|scaveng\w*|emissions?"):

        stroke_idx = 3

        

    stroke_names = ["1. INTAKE STROKE", "2. COMPRESSION STROKE", "3. COMBUSTION / POWER", "4. EXHAUST STROKE"]

    stroke_colors = [(0, 215, 255), (255, 210, 60), (255, 90, 50), (180, 190, 205)]

    

    # Engine cylinder geometry

    cyl_cx = 710

    cyl_w = 420

    cyl_x1 = cyl_cx - cyl_w // 2 # 500

    cyl_x2 = cyl_cx + cyl_w // 2 # 920

    cyl_top = 180

    cyl_bot = 530

    tdc_y = cyl_top + 80

    bdc_y = cyl_top + 230

    

    # Piston position interpolation based on stroke

    stroke_progress = (curr_t % 1.5) / 1.5

    # Smooth sinusoidal travel

    if stroke_idx == 0:   # Intake: TDC -> BDC

        piston_y = tdc_y + (bdc_y - tdc_y) * (0.5 - 0.5 * math.cos(math.pi * stroke_progress))

    elif stroke_idx == 1: # Compression: BDC -> TDC

        piston_y = bdc_y - (bdc_y - tdc_y) * (0.5 - 0.5 * math.cos(math.pi * stroke_progress))

    elif stroke_idx == 2: # Power: TDC -> BDC

        piston_y = tdc_y + (bdc_y - tdc_y) * (0.5 - 0.5 * math.cos(math.pi * stroke_progress))

    else:                 # Exhaust: BDC -> TDC

        piston_y = bdc_y - (bdc_y - tdc_y) * (0.5 - 0.5 * math.cos(math.pi * stroke_progress))

        

    # Crankshaft center and rotation

    crank_cy = 680

    crank_r = 55.0

    crank_angle = 2.0 * math.pi * (curr_t / 1.5)

    crank_px = cyl_cx + crank_r * math.sin(crank_angle)

    crank_py = crank_cy + crank_r * math.cos(crank_angle)

    

    # 1. Cast Cylinder Walls & Water Jackets

    wall_w = 55

    # Left wall

    cv.rect((cyl_x1 - wall_w, cyl_top, cyl_x1, cyl_bot), fill=(36, 46, 60, 255), outline=(75, 92, 118), w=3)

    cv.rect((cyl_x1 - wall_w + 10, cyl_top + 25, cyl_x1 - 10, cyl_bot - 25), fill=(20, 65, 105, 180), outline=(30, 95, 150), w=2)

    cv.text((cyl_x1 - wall_w // 2, (cyl_top + cyl_bot) // 2), "COOLANT", size=13, color=(140, 210, 255), bold=True, anchor="mm")

    # Right wall

    cv.rect((cyl_x2, cyl_top, cyl_x2 + wall_w, cyl_bot), fill=(36, 46, 60, 255), outline=(75, 92, 118), w=3)

    cv.rect((cyl_x2 + 10, cyl_top + 25, cyl_x2 + wall_w - 10, cyl_bot - 25), fill=(20, 65, 105, 180), outline=(30, 95, 150), w=2)

    cv.text((cyl_x2 + wall_w // 2, (cyl_top + cyl_bot) // 2), "COOLANT", size=13, color=(140, 210, 255), bold=True, anchor="mm")

    

    # Cylinder bore chamber interior

    cv.rect((cyl_x1, cyl_top, cyl_x2, cyl_bot), fill=(16, 22, 34, 255))

    

    # 2. Intake & Exhaust Ports at Cylinder Head

    intake_valve_open = (stroke_idx == 0)

    exhaust_valve_open = (stroke_idx == 3)

    

    # Left intake port

    cv.rect((cyl_x1 - 20, cyl_top - 45, cyl_cx - 60, cyl_top), fill=(24, 45, 68, 220), outline=(60, 85, 115), w=2)

    cv.text((cyl_x1 + 35, cyl_top - 25), "INTAKE PORT", size=13, color=(0, 215, 255), bold=True, anchor="mm")

    # Intake valve (moves down 18px when open)

    iv_drop = 18 if intake_valve_open else 0

    cv.line((cyl_cx - 100, cyl_top - 55), (cyl_cx - 100, cyl_top + iv_drop), color=(180, 190, 205), w=6)

    cv.rect((cyl_cx - 125, cyl_top + iv_drop, cyl_cx - 75, cyl_top + iv_drop + 8), radius=3, fill=(150, 160, 175), outline=(220, 230, 245), w=2)

    

    # Right exhaust port

    cv.rect((cyl_cx + 60, cyl_top - 45, cyl_x2 + 20, cyl_top), fill=(55, 32, 28, 220), outline=(100, 65, 55), w=2)

    cv.text((cyl_x2 - 35, cyl_top - 25), "EXHAUST PORT", size=13, color=(255, 130, 90), bold=True, anchor="mm")

    # Exhaust valve (moves down 18px when open)

    ev_drop = 18 if exhaust_valve_open else 0

    cv.line((cyl_cx + 100, cyl_top - 55), (cyl_cx + 100, cyl_top + ev_drop), color=(180, 190, 205), w=6)

    cv.rect((cyl_cx + 75, cyl_top + ev_drop, cyl_cx + 125, cyl_top + ev_drop + 8), radius=3, fill=(150, 160, 175), outline=(220, 230, 245), w=2)

    

    # Spark Plug Assembly at top center

    sp_w = 34

    cv.rect((cyl_cx - sp_w // 2, cyl_top - 50, cyl_cx + sp_w // 2, cyl_top + 10), fill=(225, 230, 240), outline=(130, 140, 155), w=2)

    cv.line((cyl_cx, cyl_top + 10), (cyl_cx, cyl_top + 22), color=(245, 245, 245), w=4)

    cv.line((cyl_cx - 8, cyl_top + 22), (cyl_cx + 2, cyl_top + 22), color=(200, 200, 200), w=4)

    

    # 3. Dynamic In-Cylinder Phenomena based on active stroke

    if stroke_idx == 0: # Intake: fresh fuel-air mist flowing in

        for i in range(14):

            flow_p = (curr_t * 60 + i * 25) % 180

            fx = cyl_cx - 100 + (i % 5 - 2) * 22

            fy = cyl_top + 15 + flow_p * 0.7

            if fy < piston_y - 15:

                cv.circle((fx, fy), 5.0, fill=(0, 215, 255, 200))

                cv.circle((fx + 5, fy + 2), 4.0, fill=(255, 200, 50, 190))

    elif stroke_idx == 1: # Compression: squeezing mixture

        density_alpha = int(120 + 100 * stroke_progress)

        cv.rect((cyl_x1 + 4, cyl_top + 10, cyl_x2 - 4, piston_y), fill=(255, 220, 80, int(density_alpha * 0.25)))

        for i in range(12):

            px = cyl_x1 + 35 + (i * 31) % (cyl_w - 70)

            py = cyl_top + 15 + (i * 19) % int(max(10, piston_y - cyl_top - 25))

            cv.circle((px, py), 5.5, fill=(255, 210, 60, density_alpha))

    elif stroke_idx == 2: # Power / Combustion: spark and expanding flame front

        spark_fire = (stroke_progress < 0.25)

        if spark_fire:

            cv.circle((cyl_cx, cyl_top + 22), 16, fill=(255, 255, 255, 245))

            cv.circle((cyl_cx, cyl_top + 22), 32, fill=(255, 240, 100, 140))

        # Expanding flame front filling chamber

        flame_r = min(190.0, 35.0 + stroke_progress * 160.0)

        cv.circle((cyl_cx, cyl_top + 35), flame_r * 1.3, fill=(255, 80, 20, 45))

        cv.circle((cyl_cx, cyl_top + 35), flame_r, fill=(255, 140, 0, 110))

        cv.circle((cyl_cx, cyl_top + 35), flame_r * 0.55, fill=(255, 235, 120, 220))

        # High pressure arrows pushing piston down

        for arr_x in (cyl_cx - 120, cyl_cx, cyl_cx + 120):

            cv.arrow((arr_x, piston_y - 45), (arr_x, piston_y - 8), color=(255, 100, 40), w=4, head=14)

    else: # Exhaust: burned gas flowing out

        cv.rect((cyl_x1 + 4, cyl_top + 10, cyl_x2 - 4, piston_y), fill=(80, 70, 70, 70))

        for i in range(12):

            flow_p = (curr_t * 65 + i * 22) % 160

            ex = cyl_cx + 100 + (i % 4 - 2) * 20

            ey = piston_y - flow_p * 0.8

            if ey > cyl_top - 40:

                cv.circle((ex, ey), 6.5, fill=(120, 115, 120, 160))

                

    # 4. Piston Body & Compression Rings

    piston_h = 105

    cv.rect((cyl_x1 + 3, piston_y, cyl_x2 - 3, piston_y + piston_h), radius=6, fill=(78, 92, 112), outline=(150, 170, 195), w=3)

    # Compression and oil control ring grooves

    for ry in range(int(piston_y + 14), int(piston_y + 45), 10):

        cv.line((cyl_x1 + 4, ry), (cyl_x2 - 4, ry), color=(28, 38, 48), w=3)

        

    # Wrist Pin

    pin_y = piston_y + 54

    cv.circle((cyl_cx, pin_y), 18, fill=(45, 55, 68), outline=(180, 195, 215), w=3)

    

    # 5. Articulated Connecting Rod to Rotating Crankshaft

    cv.line((cyl_cx, pin_y), (crank_px, crank_py), color=(95, 112, 134), w=14)

    cv.circle((crank_px, crank_py), 16, fill=(60, 72, 88), outline=(200, 215, 235), w=3)

    

    # Crankcase & Rotating Counterweight

    cv.rect((cyl_x1 - 40, cyl_bot, cyl_x2 + 40, crank_cy + 85), fill=(24, 32, 44), outline=(60, 75, 95), w=3)

    cv.circle((cyl_cx, crank_cy), 22, fill=(40, 50, 64), outline=(160, 175, 195), w=3)

    cv.arc((cyl_cx, crank_cy), 68, int(math.degrees(crank_angle) + 90), int(math.degrees(crank_angle) + 270), color=(80, 95, 118), w=12)

    cv.text((cyl_cx, crank_cy + 58), "CRANKSHAFT ROTATION", size=13, color=(160, 185, 210), bold=True, anchor="mm")

    

    # 6. Right Side Panel: 4-Stroke Interactive Process Cards (1000 to 1510)

    card_x = 1010

    card_w = 500

    card_h = 135

    for i in range(4):

        cy0 = 145 + i * 148

        is_cur = (stroke_idx == i)

        c_fill = (*stroke_colors[i], 45) if is_cur else (16, 26, 38, 220)

        c_border = stroke_colors[i] if is_cur else (50, 70, 95, 180)

        cv.rect((card_x, cy0, card_x + card_w, cy0 + card_h), radius=10, fill=c_fill, outline=c_border, w=4 if is_cur else 2)

        

        # Pill badge

        cv.rect((card_x + 16, cy0 + 12, card_x + 220, cy0 + 40), radius=5, fill=stroke_colors[i] if is_cur else (30, 48, 68), outline=None)

        cv.text((card_x + 118, cy0 + 26), stroke_names[i], color=(0, 0, 0) if is_cur else (220, 235, 250), size=14, bold=True, anchor="mm")

        

        if i == 0:

            details = [

                "• Intake valve opens; exhaust valve stays sealed",

                "• Piston descends from TDC to BDC creating vacuum",

                "• Fresh atomized fuel-air charge sucked into cylinder",

                "• Mixture prepares for high-density compression"

            ]

        elif i == 1:

            details = [

                "• Both intake and exhaust valves firmly closed",

                "• Piston ascends from BDC to TDC compressing mixture",

                "• Pressure rises ~10-15 bar; temperature rises ~400°C",

                "• High compression ratio optimizes thermodynamic burn"

            ]

        elif i == 2:

            details = [

                "• Spark plug fires high-voltage arc near TDC",

                "• Rapid flame propagation; cylinder pressure spikes ~60 bar",

                "• Superheated expanding gas forces piston down to BDC",

                "• Produces mechanical torque & usable engine horsepower"

            ]

        else:

            details = [

                "• Exhaust valve opens; intake valve remains closed",

                "• Piston moves from BDC to TDC sweeping chamber",

                "• Burned gases (CO₂, H₂O, N₂) exit to exhaust manifold",

                "• Cycle completes; engine immediately resets to Intake"

            ]

        for d_idx, d_txt in enumerate(details):

            cv.text((card_x + 20, cy0 + 54 + d_idx * 21), d_txt, color=(240, 245, 255) if is_cur else (180, 195, 215), size=14, anchor="lm")

            

    # 7. Left Side Panel: Cycle Telemetry & Technical Specs (90 to 440)

    spec_x = 90

    spec_w = 350

    cv.rect((spec_x, 145, spec_x + spec_w, 735), radius=10, fill=(16, 26, 40, 235), outline=(50, 75, 110), w=2)

    # Header

    cv.rect((spec_x + 20, 160, spec_x + spec_w - 20, 192), radius=6, fill=(28, 55, 85), outline=(0, 215, 255), w=1)

    cv.text((spec_x + spec_w // 2, 176), "ENGINE CYCLE METRICS", color=(255, 255, 255), size=15, bold=True, anchor="mm")

    

    # Active Stroke Highlight Box

    cv.rect((spec_x + 18, 208, spec_x + spec_w - 18, 280), radius=8, fill=(*stroke_colors[stroke_idx], 35), outline=stroke_colors[stroke_idx], w=2)

    cv.text((spec_x + spec_w // 2, 226), "CURRENT CYCLE STATE", color=stroke_colors[stroke_idx], size=13, bold=True, anchor="mm")

    cv.text((spec_x + spec_w // 2, 254), stroke_names[stroke_idx], color=(255, 255, 255), size=17, bold=True, anchor="mm")

    

    spec_items = [

        ("Crankshaft Speed:", f"{int(60.0 / cycle_period * 2)} RPM (Simulation)"),

        ("Thermodynamic Cycle:", "Otto Cycle (Spark Ignition)"),

        ("Stroke-to-Bore Ratio:", "Square Engine (~1.0)"),

        ("Intake Valve State:", "OPEN" if intake_valve_open else "CLOSED"),

        ("Exhaust Valve State:", "OPEN" if exhaust_valve_open else "CLOSED"),

        ("Spark Plug Discharge:", "ACTIVE ARC" if (stroke_idx == 2 and stroke_progress < 0.25) else "STANDBY"),

        ("Peak Cylinder Press:", "50 - 75 bar (Power Stroke)"),

        ("Combustion Velocity:", "15 - 30 m/s Flame Front"),

        ("Chemical Energy:", "Hydrocarbon Fuel Enthalpy"),

        ("Output Delivery:", "Reciprocating Torque to Crank"),

    ]

    for idx, (label, val) in enumerate(spec_items):

        item_y = 300 + idx * 40

        cv.line((spec_x + 15, item_y + 30), (spec_x + spec_w - 15, item_y + 30), color=(35, 52, 75), w=1)

        cv.text((spec_x + 18, item_y + 12), label, color=(160, 185, 210), size=13, anchor="lm")

        val_col = (0, 235, 255) if "OPEN" in val or "ACTIVE" in val else (240, 248, 255)

        cv.text((spec_x + spec_w - 18, item_y + 12), val, color=val_col, size=13, bold=True, anchor="rm")

        

    # Bottom HUD summary banner

    cv.rect((spec_x, 748, 1510, 788), radius=6, fill=(16, 26, 40, 235), outline=(50, 75, 110), w=1)

    cv.text((800, 768), "Four-Stroke Reciprocating Cycle: Chemical Energy In -> High-Pressure Expansion -> Useful Crankshaft Mechanical Work", color=(240, 248, 255), size=16, bold=True, anchor="mm")





def _air_fuel_ratio_gauge(cv: Cv) -> None:

    """Detailed Air-Fuel Ratio (AFR) & Lambda / Phi Meter with Real-World Spark Plug Diagnostics:

    Inspired by automotive SME videos (Panda Bewok, engineering breakdowns).

    - Gasoline Stoichiometric: 14.7 : 1 (Ideal Chemical Balance)

    - Lean Mixture: > 14.7 AFR (phi < 1.0, lambda > 1.0) with blistered white overheated spark plug

    - Rich Mixture: < 14.7 AFR (phi > 1.0, lambda < 1.0) with black carbon fouled spark plug

    - Visual flame color and combustion efficiency spectrum!

    """

    cv.rect((60, 30, 1540, 790), radius=16, fill=(12, 19, 32, 245), outline=(55, 80, 120, 220), w=3)

    cv.text((800, 62), "AIR-FUEL RATIO (AFR) & COMBUSTION TUNING METER", color=(255, 255, 255), size=27, bold=True, anchor="mm")

    cv.text((800, 94), "Gasoline Stoichiometric Ratio: 14.7:1 (by mass)  •  Lambda (λ) = AFR_actual / 14.7  •  Phi (φ) = 1 / λ", color=(180, 215, 245), size=17, anchor="mm")

    

    is_lean = cv.lit(r"lean|excess air|greater than 14|above 14|hotter flame|fuel economy|pinging|detonat|blister")

    is_rich = cv.lit(r"rich|excess fuel|less than 14|below 14|carbon|soot|fouled|black plug|torque|acceleration")

    is_stoich = cv.lit(r"stoichiometric|14\.7|perfect balance|ideal ratio|clean emissions|catalytic") or (not is_lean and not is_rich and cv.lit(r"ratio|mixture|afr|lambda|phi"))

    

    lean_active = is_lean or (not is_rich and not is_stoich and (cv.t % 9.0 < 3.0))

    stoich_active = is_stoich or (not is_lean and not is_rich and (3.0 <= cv.t % 9.0 < 6.0))

    rich_active = is_rich or (not is_lean and not is_stoich and (cv.t % 9.0 >= 6.0))

    

    card_y = 125

    card_h = 510

    card_w = 440

    

    # 1. LEAN CARD (AFR > 14.7)

    lx0 = 90

    l_bg = (16, 48, 76, 245) if lean_active else (14, 26, 40, 190)

    l_border = (0, 215, 255, 255) if lean_active else (40, 72, 105, 160)

    cv.rect((lx0, card_y, lx0 + card_w, card_y + card_h), radius=12, fill=l_bg, outline=l_border, w=4 if lean_active else 2)

    

    cv.rect((lx0 + 40, card_y + 14, lx0 + card_w - 40, card_y + 48), radius=6, fill=(0, 140, 205) if lean_active else (25, 55, 88), outline=None)

    cv.text((lx0 + card_w // 2, card_y + 31), "LEAN MIXTURE (EXCESS AIR)", color=(255, 255, 255), size=17, bold=True, anchor="mm")

    

    cv.text((lx0 + card_w // 2, card_y + 75), "AFR > 14.7 : 1", color=(0, 240, 255) if lean_active else (145, 190, 220), size=28, bold=True, anchor="mm")

    cv.text((lx0 + card_w // 2, card_y + 102), "λ > 1.0  |  φ < 1.0  (e.g. 16.5:1)", color=(200, 230, 250), size=16, anchor="mm")

    

    # Spark plug tip condition

    _draw_spark_plug_tip(cv, lx0 + 110, card_y + 195, "lean")

    # Flame preview

    _draw_flame(cv, lx0 + card_w - 110, card_y + 205, 65, 110, "lean", cv.t)

    

    pts_lean = [

        "• Excess air relative to fuel mass",

        "• Pale blue, transparent flame front",

        "• Higher internal combustion temp",

        "• Spark plug: Light grey / blistered white",

        "• Max fuel economy; risk of detonation"

    ]

    for idx, pt in enumerate(pts_lean):

        cv.text((lx0 + 22, card_y + 300 + idx * 35), pt, color=(220, 238, 252), size=16, anchor="lm")

        

    # 2. STOICHIOMETRIC CARD (AFR = 14.7)

    sx0 = 580

    s_bg = (18, 62, 44, 245) if stoich_active else (14, 35, 28, 190)

    s_border = (46, 225, 125, 255) if stoich_active else (40, 95, 68, 160)

    cv.rect((sx0, card_y, sx0 + card_w, card_y + card_h), radius=12, fill=s_bg, outline=s_border, w=4 if stoich_active else 2)

    

    cv.rect((sx0 + 40, card_y + 14, sx0 + card_w - 40, card_y + 48), radius=6, fill=(35, 180, 98) if stoich_active else (25, 78, 48), outline=None)

    cv.text((sx0 + card_w // 2, card_y + 31), "STOICHIOMETRIC (IDEAL)", color=(255, 255, 255), size=17, bold=True, anchor="mm")

    

    cv.text((sx0 + card_w // 2, card_y + 75), "AFR = 14.7 : 1", color=(80, 255, 165) if stoich_active else (145, 215, 175), size=28, bold=True, anchor="mm")

    cv.text((sx0 + card_w // 2, card_y + 102), "λ = 1.0  |  φ = 1.0  (Perfect Balance)", color=(210, 255, 230), size=16, anchor="mm")

    

    _draw_spark_plug_tip(cv, sx0 + 110, card_y + 195, "stoich")

    _draw_flame(cv, sx0 + card_w - 110, card_y + 205, 75, 120, "stoich", cv.t)

    

    pts_stoich = [

        "• Chemically complete fuel-air reaction",

        "• Intense electric blue flame wavefront",

        "• Optimal catalytic converter conversion",

        "• Spark plug: Healthy light tan / grey deposit",

        "• Cleanest emissions: CO₂ + H₂O"

    ]

    for idx, pt in enumerate(pts_stoich):

        cv.text((sx0 + 22, card_y + 300 + idx * 35), pt, color=(225, 248, 232), size=16, anchor="lm")

        

    # 3. RICH CARD (AFR < 14.7)

    rx0 = 1070

    r_bg = (68, 26, 22, 245) if rich_active else (40, 18, 16, 190)

    r_border = (255, 105, 65, 255) if rich_active else (120, 48, 38, 160)

    cv.rect((rx0, card_y, rx0 + card_w, card_y + card_h), radius=12, fill=r_bg, outline=r_border, w=4 if rich_active else 2)

    

    cv.rect((rx0 + 40, card_y + 14, rx0 + card_w - 40, card_y + 48), radius=6, fill=(220, 58, 36) if rich_active else (100, 34, 26), outline=None)

    cv.text((rx0 + card_w // 2, card_y + 31), "RICH MIXTURE (EXCESS FUEL)", color=(255, 255, 255), size=17, bold=True, anchor="mm")

    

    cv.text((rx0 + card_w // 2, card_y + 75), "AFR < 14.7 : 1", color=(255, 135, 95) if rich_active else (220, 150, 130), size=28, bold=True, anchor="mm")

    cv.text((rx0 + card_w // 2, card_y + 102), "λ < 1.0  |  φ > 1.0  (e.g. 12.5:1)", color=(255, 225, 215), size=16, anchor="mm")

    

    _draw_spark_plug_tip(cv, rx0 + 110, card_y + 195, "rich")

    _draw_flame(cv, rx0 + card_w - 110, card_y + 205, 78, 128, "rich", cv.t)

    

    pts_rich = [

        "• Excess fuel cannot find enough O₂ to burn",

        "• Luminous orange flame with heavy soot",

        "• Combustion chamber runs cooler",

        "• Spark plug: Black dry/wet carbon fouled",

        "• Peak torque / acceleration; high CO & HC"

    ]

    for idx, pt in enumerate(pts_rich):

        cv.text((rx0 + 22, card_y + 300 + idx * 35), pt, color=(252, 228, 222), size=16, anchor="lm")

        

    # Bottom Spectrum Gauge Bar with AFR ticks

    bar_y = 690

    bar_x1 = 160

    bar_x2 = 1440

    bar_w = bar_x2 - bar_x1

    cv.rect((bar_x1, bar_y - 8, bar_x2, bar_y + 8), radius=6, fill=(28, 42, 60, 220))

    # Rich segment (left on AFR scale: 10:1 to 14.7:1)

    cv.rect((bar_x1, bar_y - 8, bar_x1 + bar_w * 0.44, bar_y + 8), radius=5, fill=(255, 80, 50, 220))

    # Stoich notch (14.7:1)

    stoich_px = bar_x1 + bar_w * 0.5

    cv.circle((stoich_px, bar_y), 13, fill=(46, 225, 125, 255), outline=(255, 255, 255), w=3)

    # Lean segment (right on AFR scale: 14.7:1 to 19:1)

    cv.rect((bar_x1 + bar_w * 0.56, bar_y - 8, bar_x2, bar_y + 8), radius=5, fill=(0, 195, 245, 220))

    

    # Scale tick marks

    ticks = [

        (bar_x1 + bar_w * 0.12, "11.0:1 (Rich)"),

        (bar_x1 + bar_w * 0.28, "13.0:1 (Power)"),

        (stoich_px, "14.7:1 (Stoichiometric)"),

        (bar_x1 + bar_w * 0.72, "16.5:1 (Economy)"),

        (bar_x1 + bar_w * 0.88, "18.0:1 (Lean)"),

    ]

    for tx, tlabel in ticks:

        cv.line((tx, bar_y - 12), (tx, bar_y + 12), color=(255, 255, 255), w=2)

        cv.text((tx, bar_y - 20), tlabel, color=(200, 220, 240), size=12, bold=True, anchor="mm")

        

    if rich_active:

        target_pos = bar_x1 + bar_w * 0.25

        val_text = "CURRENT AIR-FUEL RATIO: 12.8 : 1 (RICH POWER TUNE — λ = 0.87)"

        needle_col = (255, 95, 65)

    elif lean_active:

        target_pos = bar_x1 + bar_w * 0.75

        val_text = "CURRENT AIR-FUEL RATIO: 16.8 : 1 (LEAN ECONOMY TUNE — λ = 1.14)"

        needle_col = (0, 230, 255)

    else:

        target_pos = stoich_px

        val_text = "CURRENT AIR-FUEL RATIO: 14.7 : 1 (IDEAL STOICHIOMETRIC — λ = 1.00)"

        needle_col = (80, 255, 160)

        

    cv.arrow((target_pos, bar_y + 44), (target_pos, bar_y + 14), color=needle_col, w=5, head=18)

    cv.rect((target_pos - 280, bar_y + 46, target_pos + 280, bar_y + 82), radius=6, fill=(16, 26, 42, 240), outline=needle_col, w=2)

    cv.text((target_pos, bar_y + 64), val_text, color=(255, 255, 255), size=16, bold=True, anchor="mm")







# ---------------------------------------------------------------------------

# Registry

# ---------------------------------------------------------------------------



FIGURES: dict = {

    "four_stroke_cycle": (

        r"four[- ]stroke|4[- ]stroke|intake stroke|compression stroke|power stroke|exhaust stroke|engine cycle|internal combustion engine|reciprocating engine|four strokes|combustion applications?|engine combustion",

        r"stroke|piston|crankshaft|intake valve|exhaust valve|spark plug|cylinder",

        _four_stroke_cycle,

        {"engineering", "thermodynamics", "physics", "general"},

    ),

    "air_fuel_ratio_gauge": (

        r"air[- ]fuel ratio|\bafr\b|lambda sensor|stoichiometric 14\.7|lean mixture.*spark plug|rich mixture.*spark plug|spark plug condition|fuel[- ]air tuning|air fuel ratio|strength of a fuel[- ]air mixture",

        r"14\.7|air-fuel|mixture|lambda|lean|rich|spark plug|carburetor|fuel injection",

        _air_fuel_ratio_gauge,

        {"engineering", "thermodynamics", "chemistry", "physics", "general"},

    ),

    "equivalence_ratio_dial": (

        r"equivalence ratio|\bphi\b|stoichiometric ratio|fuel[- ]to[- ]air|air[- ]to[- ]fuel|lean mixture|rich mixture|excess air|excess fuel",

        r"stoichiometric|combustion ratio|fuel ratio|mixture ratio|\bphi\b < 1|\bphi\b > 1|\bphi\b = 1|lean regime|rich regime|combustion",

        _equivalence_ratio_dial,

        {"engineering", "thermodynamics", "chemistry", "physics", "general"},

    ),

    "combustion_chamber": (

        r"combustion chamber|engine cylinder|internal combustion|cylinder cross[- ]section|four[- ]stroke|piston.*spark|spark plug.*cylinder|flame front|flame propagation",

        r"piston|cylinder|combustion|spark plug|ignition|flame kernel|combustion process|power stroke|reactants",

        _combustion_chamber,

        {"engineering", "thermodynamics", "chemistry", "physics", "general"},

    ),



    "electrolysis_cell": (

        r"electrolysis|electrolytic|electroplat|electrode|anode|cathode|electrolyte|wire.*terminal|battery.*terminal|positive terminal|negative terminal|electrons? flow|silver rod.*keychain|power supply|rod into the wire|back into the battery|current flow in the circuit",

        r"\bcell\b|decompos|copper sulphate|brine|salt solution|ions? migrat|silver rod|keychain|current flow|power supply",

        _electrolysis_cell,

        {"chemistry", "physics"},

    ),

    "silver_ion_shell": (

        r"outermost shell|empty spot|valence shell|electron configuration|shell with empty|vacan|attracts an electron into this position",

        r"silver ion|incoming electron|attracts an electron|orbital|settles onto the.*keychain",

        _silver_ion_shell,

        {"chemistry"},

    ),

    "redox_reaction_board": (

        r"redox reaction|reduction occurs at the cathode|oxidation takes place at the anode|dual reaction|net reaction",

        r"cathode and oxidation|redox summary|half[- ]reactions?",

        _redox_reaction_board,

        {"chemistry"},

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

