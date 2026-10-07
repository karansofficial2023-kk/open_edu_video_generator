"""Deterministic charts and panel layouts. Values come only from the storyboard contract (never invented)."""
from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageOps

from .config import AppConfig
from .typography import draw_lines, fit_text, font, hex_rgb, is_complex_script, put_text, text_width

_NUMBER = re.compile(r"^\s*(?P<label>.+?)\s*[:=|]\s*(?P<value>-?\d[\d,]*(?:\.\d+)?)\s*(?P<unit>%|[^\s\d:=|]{0,12})\s*$")      # any short unit: Ω, m/s², °C, कि.मी.


def parse_rows(columns: list[str]) -> list[tuple[str, float, str]]:
    rows = []
    for column in columns:
        match = _NUMBER.match(column)
        if match:
            rows.append((match.group("label").strip(), float(match.group("value").replace(",", "")), match.group("unit")))
    return rows


def nice_max(value: float, percent: bool) -> float:
    if percent and value <= 100:
        return 100.0
    if value <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(value))
    for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if value <= step * magnitude:
            return step * magnitude
    return 10 * magnitude


def graph_frame(size: tuple[int, int], config: AppConfig, heading: str, rows: list[tuple[str, float, str]], unit_note: str,
                t: float) -> Image.Image:
    width, height = size
    s = height / 1080
    image = Image.new("RGB", size, config.render.background)
    draw = ImageDraw.Draw(image)
    ink, accent, second = hex_rgb(config.render.ink), hex_rgb(config.render.accent), hex_rgb(config.render.second_accent)
    fnt, lines, line_h = fit_text(draw, heading, (round(96 * s), round(40 * s), width - round(96 * s), round(160 * s)),
                                  round(58 * s), round(32 * s), bold=True, max_lines=2)
    draw_lines(draw, lines, fnt, line_h, round(96 * s), round(40 * s), ink)
    percent = any(unit == "%" for _l, _v, unit in rows) or "%" in unit_note
    axis_max = nice_max(max(v for _l, v, _u in rows), percent)
    label_w = min(round(width * 0.28), max(round(text_width(draw, l, font(round(34 * s), False, is_complex_script(l)))) for l, _v, _u in rows) + round(30 * s))
    left, right = round(96 * s) + label_w, width - round(230 * s)
    top, bottom = round(215 * s), round(740 * s)
    row_h = (bottom - top) / len(rows)
    bar_h = min(round(78 * s), row_h * 0.62)
    palette = [accent, second, (61, 110, 170), (150, 82, 130), (192, 84, 64)]
    label_font, value_font = font(round(34 * s)), font(round(34 * s), True)
    tick_font = font(round(26 * s))
    for k in range(0, 5):
        x = left + (right - left) * k / 4
        draw.line([(x, top - round(8 * s)), (x, bottom)], fill=(205, 212, 206), width=max(1, round(2 * s)))
        label = f"{axis_max * k / 4:g}"
        put_text(draw, (x - text_width(draw, label, tick_font) / 2, bottom + round(10 * s)), label, tick_font, (90, 105, 96))
    for i, (label, value, unit) in enumerate(rows):
        begin = 0.7 + i * 0.6
        progress = min(1.0, max(0.0, (t - begin) / 0.7))
        progress = 1 - (1 - progress) ** 3
        cy = top + row_h * i + row_h / 2
        lf = font(round(34 * s), False, is_complex_script(label))
        put_text(draw, (round(96 * s), cy - lf.size * 0.6), label, lf, ink)
        draw.rounded_rectangle((left, cy - bar_h / 2, right, cy + bar_h / 2), radius=round(8 * s), fill=(232, 236, 232))
        end = left + (right - left) * (value / axis_max) * progress
        if end > left + 2:
            draw.rounded_rectangle((left, cy - bar_h / 2, end, cy + bar_h / 2), radius=round(8 * s), fill=palette[i % len(palette)])
        if progress > 0.95:
            shown = f"{value:g}{unit if unit else ''}"
            put_text(draw, (right + round(22 * s), cy - value_font.size * 0.6), shown, value_font, ink)
    if unit_note:
        nf, nl, nh = fit_text(draw, unit_note, (left, bottom + round(52 * s), right, bottom + round(100 * s)), round(28 * s), round(20 * s))
        draw_lines(draw, nl, nf, nh, left, bottom + round(52 * s), (90, 105, 96), "center", right - left)
    return image


def panels_frame(size: tuple[int, int], config: AppConfig, heading: str, panels: list[Image.Image], captions: list[str],
                 t: float, contain: bool = False) -> Image.Image:
    width, height = size
    s = height / 1080
    image = Image.new("RGB", size, config.render.background)
    draw = ImageDraw.Draw(image)
    ink = hex_rgb(config.render.ink)
    accent = hex_rgb(config.render.accent)
    top = round(40 * s)
    if heading:
        fnt, lines, line_h = fit_text(draw, heading, (round(96 * s), top, width - round(96 * s), top + round(110 * s)),
                                      round(60 * s), round(34 * s), bold=True, max_lines=2)
        draw_lines(draw, lines, fnt, line_h, round(96 * s), top, ink, "center", width - round(192 * s))
        top += len(lines) * line_h + round(24 * s)
    count = len(panels)
    gap = round(36 * s)
    margin = round(70 * s)
    panel_w = (width - 2 * margin - gap * (count - 1)) // count
    caption_h = round(120 * s)
    panel_h = min(height - top - caption_h - round(200 * s), round(panel_w * 1.1))
    for i, (panel, caption) in enumerate(zip(panels, captions)):
        alpha = min(1.0, max(0.0, (t - 0.3 - i * 0.5) / 0.4))
        x0 = margin + i * (panel_w + gap)
        if contain:      # diagrams must never be cropped
            fitted = ImageOps.pad(panel.convert("RGB"), (panel_w, panel_h), Image.Resampling.LANCZOS, color=(255, 255, 255))
        else:
            fitted = ImageOps.fit(panel.convert("RGB"), (panel_w, panel_h), Image.Resampling.LANCZOS)
        mask = Image.new("L", (panel_w, panel_h), 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, panel_w - 1, panel_h - 1), radius=round(20 * s), fill=round(255 * alpha))
        frame_box = (x0 - round(6 * s), top - round(6 * s), x0 + panel_w + round(6 * s), top + panel_h + round(6 * s))
        if alpha > 0.02:
            draw.rounded_rectangle(frame_box, radius=round(24 * s), fill=(255, 255, 255), outline=(215, 222, 216), width=max(2, round(3 * s)))
            image.paste(fitted, (x0, top), mask)
        if alpha > 0.5 and caption:
            cf, cl, ch = fit_text(draw, caption, (x0, top + panel_h + round(22 * s), x0 + panel_w, top + panel_h + caption_h),
                                  round(38 * s), round(26 * s), bold=True, max_lines=2)
            draw_lines(draw, cl, cf, ch, x0, top + panel_h + round(22 * s), ink, "center", panel_w)
    return image


_SEQUENCE = re.compile(r"(?i)\b(then|first|second|third|next|after|afterwards|finally|followed|leads? to|results? in|before|later|once|step)\b")


def is_sequential(narration: str) -> bool:
    return bool(_SEQUENCE.search(narration or ""))


@lru_cache(maxsize=256)
def card_shadow(box: tuple[int, int, int, int], s: float) -> tuple[Image.Image, tuple[int, int]]:
    """Soft drop shadow of one card as a small RGBA patch plus its top-left position.

    It used to be drawn and blurred on a full-frame layer (twice) for every card in every frame, which was 60% of the time of a card
    clip. The patch is blurred on a canvas just large enough that the blur reaches zero before the edge, so the pixels are identical;
    it is cached per card position (a card only moves during its 0.4 s lift)."""
    blur, radius = round(12 * s), round(28 * s)
    margin = blur * 6 + 12
    width, height = box[2] - box[0], box[3] - box[1]
    patch = Image.new("RGBA", (width + 2 * margin + 6, height + 2 * margin + 10), (0, 0, 0, 0))
    ImageDraw.Draw(patch).rounded_rectangle((margin + 6, margin + 10, margin + 6 + width, margin + 10 + height), radius=radius, fill=(0, 0, 0, 40))
    return patch.filter(ImageFilter.GaussianBlur(blur)), (box[0] - margin, box[1] - margin)


def cards_frame(size: tuple[int, int], config: AppConfig, heading: str, steps: list[str], stage: int, age: float,
                sequential: bool, backdrop: Image.Image | None = None) -> Image.Image:
    """Key-idea / sequence cards for abstract concepts that have no single photographable subject.
    backdrop: an optional soft picture of the lesson behind the cards (app.backdrops.soft); without it a plain tinted page."""
    width, height = size
    s = height / 1080
    if backdrop is not None:
        image = backdrop.convert("RGB").resize(size) if backdrop.size != size else backdrop.convert("RGB").copy()
    else:
        image = Image.new("RGB", size, config.render.background)
        shade = Image.linear_gradient("L").resize(size).point(lambda v: round(v * 0.08))
        image.paste(Image.new("RGB", size, (20, 60, 45)), (0, 0), shade)
    draw = ImageDraw.Draw(image)
    ink, accent, second = hex_rgb(config.render.ink), hex_rgb(config.render.accent), hex_rgb(config.render.second_accent)
    top = round(56 * s)
    if heading:
        fnt, lines, line_h = fit_text(draw, heading, (round(96 * s), top, width - round(96 * s), top + round(150 * s)),
                                      round(64 * s), round(36 * s), bold=True, max_lines=2)
        draw_lines(draw, lines, fnt, line_h, round(96 * s), top, ink)
        top += len(lines) * line_h + round(10 * s)
        draw.rounded_rectangle((round(96 * s), top, round(96 * s) + round(200 * s), top + round(7 * s)), radius=round(3 * s), fill=accent)
        top += round(50 * s)
    count = len(steps)
    two_rows = count == 4 and max(len(t) for t in steps) > 28
    cols = 2 if two_rows else count
    rows = 2 if two_rows else 1
    gap = round(44 * s)
    margin = round(96 * s)
    area_bottom = height - round(200 * s)
    tile_w = (width - 2 * margin - gap * (cols - 1)) // cols
    tile_h = min(round(460 * s), (area_bottom - top - gap * (rows - 1)) // rows)
    top += max(0, (area_bottom - top - (tile_h * rows + gap * (rows - 1))) // 2)   # centre the block in the free area
    number_font = font(round(40 * s), True)
    for index, step in enumerate(steps):
        if index > stage:
            continue
        row, col = divmod(index, cols)
        x0 = margin + col * (tile_w + gap)
        y0 = top + row * (tile_h + gap)
        active = index == stage
        appear = min(1.0, age / 0.4) if active else 1.0
        lift = round((1 - appear) * 24 * s)
        box = (x0, y0 + lift, x0 + tile_w, y0 + tile_h + lift)
        patch, origin = card_shadow(box, s)
        image.paste(patch, origin, patch)
        draw = ImageDraw.Draw(image)
        color = accent if index % 2 == 0 else second
        draw.rounded_rectangle(box, radius=round(28 * s), fill=(255, 255, 255), outline=color if active else (214, 222, 216),
                               width=max(3, round((7 if active else 3) * s)))
        badge = round(64 * s)
        bx, by = box[0] + round(30 * s), box[1] + round(28 * s)
        draw.ellipse((bx, by, bx + badge, by + badge), fill=color)
        label = str(index + 1)
        put_text(draw, (bx + (badge - text_width(draw, label, number_font)) / 2, by + badge * 0.12), label, number_font, (255, 255, 255))
        text_box = (box[0] + round(34 * s), by + badge + round(24 * s), box[2] - round(34 * s), box[3] - round(30 * s))
        term, sep, body = step.partition(": ")
        # "Term: explanation" takeaway card - only for a genuinely short term; a sentence that merely contains a colon
        # ("... which asks: Which of the following ...") is one block of text, not a term with a squeezed body
        if sep and term and body and len(term.split()) <= 4 and len(term) <= 40:
            term_box = (text_box[0], text_box[1], text_box[2], text_box[1] + round(120 * s))
            tf, tl, th = fit_text(draw, term, term_box, round(54 * s), round(32 * s), bold=True, max_lines=2)
            draw_lines(draw, tl, tf, th, text_box[0], text_box[1], color)
            body_top = text_box[1] + len(tl) * th + round(14 * s)
            bf, bl, bh = fit_text(draw, body, (text_box[0], body_top, text_box[2], text_box[3]), round(38 * s), round(26 * s), max_lines=5)
            draw_lines(draw, bl, bf, bh, text_box[0], body_top, ink if active else (70, 84, 76))
        else:
            tf, tl, th = fit_text(draw, step, text_box, round(50 * s), round(28 * s), bold=active, max_lines=5)
            draw_lines(draw, tl, tf, th, text_box[0], text_box[1], ink if active else (70, 84, 76))
        if sequential and col < cols - 1 and index < count - 1 and index < stage:
            cx, cy = box[2] + gap // 2, (box[1] + box[3]) // 2
            arrow = round(16 * s)
            draw.polygon([(cx - arrow, cy - arrow), (cx + arrow, cy), (cx - arrow, cy + arrow)], fill=color)
    return image


# ------------------------------------------------------------------ function plots (mathematics / physics)
_SAFE_NAMES = {"sin": "sin", "cos": "cos", "tan": "tan", "sqrt": "sqrt", "ln": "log", "log": "log10", "exp": "exp", "abs": "fabs", "pi": "pi", "e": "e"}


def parse_function(columns: list[str]):
    """('y = x^2 - 5x + 6', x_range, y_range|None, python_callable) from contract columns, or None if not a function plot."""
    import math
    equation = next((c for c in columns if "=" in c and re.search(r"\bx\b|\bx[\^*/+\-]|[\d)]x\b|\bx\)", c)), None)
    if equation is None:
        return None
    rhs = equation.split("=", 1)[1].strip()
    from .mathparse import math_to_latex  # noqa: F401  (validates syntax early; raises MathParseError)
    expression = _to_python(rhs)

    def f(x: float) -> float:
        import warnings
        with warnings.catch_warnings():          # an unusable expression is rejected by the caller; the compiler's SyntaxWarning is noise
            warnings.simplefilter("ignore", SyntaxWarning)
            return eval(expression, {"__builtins__": {}}, {**{k: getattr(math, v) for k, v in _SAFE_NAMES.items()}, "x": x})

    f(1.0)      # reject unusable expressions now
    ranges = {}
    for column in columns:
        match = re.match(r"(?i)^\s*([xy])(?:\s*range)?\s*[:=]\s*(-?\d+(?:\.\d+)?)\s*(?:to|\.\.|,|-)\s*(-?\d+(?:\.\d+)?)\s*$", column)
        if match:
            ranges[match.group(1).lower()] = (float(match.group(2)), float(match.group(3)))
    return equation, ranges.get("x", (-5.0, 5.0)), ranges.get("y"), f


def _to_python(rhs: str) -> str:
    text = rhs.replace("^", "**").replace("\u00d7", "*").replace("\u00b7", "*").replace("\u2212", "-")
    text = re.sub(r"(\d)\s*(x|\()", r"\1*\2", text)                    # 5x -> 5*x
    text = re.sub(r"(\))\s*(x|\(|\d)", r"\1*\2", text)                 # )x -> )*x
    text = re.sub(r"\bx\s*(\()", r"x*\1", text)
    if re.search(r"[^0-9a-z.+\-*/()\s,]", text.replace("**", "*")):
        raise ValueError(f"unsupported characters in {rhs!r}")
    for word in set(re.findall(r"[a-z]+", text)):
        if word != "x" and word not in _SAFE_NAMES:
            raise ValueError(f"unsupported name {word!r}")
    # bound the work: a chain of powers (9**9**9**9) or a huge exponent would hang the render while the curve is tested
    if len(text) > 120 or re.search(r"\*\*\s*[\w.]+\s*\*\*", text) or any(int(m) > 12 for m in re.findall(r"\*\*\s*\(?\s*(\d+)", text))             or re.search(r"\d{7,}", text):
        raise ValueError(f"expression too large to plot safely: {rhs!r}")
    return text


def plot_frame(size: tuple[int, int], config: AppConfig, heading: str, equation: str, x_range, y_range, func, t: float) -> Image.Image:
    width, height = size
    s = height / 1080
    image = Image.new("RGB", size, config.render.background)
    draw = ImageDraw.Draw(image)
    ink, accent = hex_rgb(config.render.ink), hex_rgb(config.render.accent)
    fnt, lines, line_h = fit_text(draw, heading, (round(96 * s), round(40 * s), width - round(96 * s), round(150 * s)),
                                  round(58 * s), round(32 * s), bold=True, max_lines=2)
    draw_lines(draw, lines, fnt, line_h, round(96 * s), round(40 * s), ink)
    left, right, top, bottom = round(180 * s), width - round(180 * s), round(190 * s), round(740 * s)
    x0, x1 = x_range
    samples = [x0 + (x1 - x0) * i / 400 for i in range(401)]
    values = []
    for x in samples:
        try:
            v = func(x)
            values.append(v if abs(v) < 1e6 else float("nan"))
        except (ValueError, ZeroDivisionError, OverflowError):
            values.append(float("nan"))
    finite = [v for v in values if v == v]
    y0, y1 = y_range if y_range else ((min(finite), max(finite)) if finite else (-1, 1))
    pad = (y1 - y0) * 0.12 or 1.0
    y0, y1 = (y0, y1) if y_range else (y0 - pad, y1 + pad)

    def px(x, y):
        return left + (x - x0) / (x1 - x0) * (right - left), bottom - (y - y0) / (y1 - y0) * (bottom - top)

    grid_font = font(round(24 * s))
    for xv in _nice_ticks(x0, x1, 8):
        gx, _ = px(xv, 0)
        draw.line([(gx, top), (gx, bottom)], fill=(222, 228, 223), width=max(1, round(2 * s)))
        label = f"{xv:g}"
        put_text(draw, (gx - text_width(draw, label, grid_font) / 2, bottom + round(8 * s)), label, grid_font, (90, 105, 96))
    for yv in _nice_ticks(y0, y1, 6):
        _, gy = px(0, yv)
        draw.line([(left, gy), (right, gy)], fill=(222, 228, 223), width=max(1, round(2 * s)))
        label = f"{yv:g}"
        put_text(draw, (left - text_width(draw, label, grid_font) - round(12 * s), gy - round(14 * s)), label, grid_font, (90, 105, 96))
    if x0 <= 0 <= x1:
        ax, _ = px(0, 0)
        draw.line([(ax, top), (ax, bottom)], fill=ink, width=max(2, round(3 * s)))
    if y0 <= 0 <= y1:
        _, ay = px(0, 0)
        draw.line([(left, ay), (right, ay)], fill=ink, width=max(2, round(3 * s)))
    progress = min(1.0, max(0.0, (t - 0.6) / 2.0))
    shown = int(len(samples) * progress)
    run = []
    for x, v in zip(samples[:shown], values[:shown]):
        if v == v and y0 - 5 * (y1 - y0) < v < y1 + 5 * (y1 - y0):
            run.append(px(x, v))
        elif len(run) > 1:
            draw.line(run, fill=accent, width=max(4, round(6 * s)), joint="curve")
            run = []
    if len(run) > 1:
        draw.line(run, fill=accent, width=max(4, round(6 * s)), joint="curve")
    if progress >= 1.0 and t > 3.0:                    # mark the real zero crossings once the curve is complete
        second = hex_rgb(config.render.second_accent)
        mark_font = font(round(28 * s), True)
        for index in range(1, len(samples)):
            a, b = values[index - 1], values[index]
            if a == a and b == b and a * b < 0:
                lo, hi = samples[index - 1], samples[index]
                for _ in range(40):
                    mid = (lo + hi) / 2
                    if func(lo) * func(mid) <= 0:
                        hi = mid
                    else:
                        lo = mid
                root = (lo + hi) / 2
                cx, cy = px(root, 0)
                r = round(11 * s)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=second, outline=(255, 255, 255), width=max(2, round(3 * s)))
                tag = f"x = {root:.2f}".rstrip("0").rstrip(".")
                put_text(draw, (cx - text_width(draw, tag, mark_font) / 2, cy + round(22 * s)), tag, mark_font, second)
    eq_font = font(round(34 * s), True)
    put_text(draw, (right - text_width(draw, equation, eq_font), top - round(44 * s)), equation, eq_font, accent)
    return image


def _nice_ticks(lo: float, hi: float, target: int) -> list[float]:
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / max(1, target)
    magnitude = 10 ** math.floor(math.log10(raw))
    step = min((m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw), default=10 * magnitude)
    first = math.ceil(lo / step) * step
    ticks, value = [], first
    while value <= hi + 1e-9:
        ticks.append(round(value, 10))
        value += step
    return ticks
