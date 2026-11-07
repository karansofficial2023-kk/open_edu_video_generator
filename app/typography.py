"""Font selection, script-aware measurement/wrapping, and the subtitle band (open-source Pillow)."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

_LATIN_BOLD = ["C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf", "DejaVuSans-Bold.ttf",
               "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf"]
_LATIN = ["C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf",
          "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]
# Verified glyph coverage for Indic scripts / Arabic-script (Nirmala UI ships with Windows; Noto elsewhere).
_COMPLEX = ["C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/NirmalaB.ttc",
            "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf",
            "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"]

_CONFIG_FONT = {"path": ""}


def set_font_override(path: str) -> None:
    _CONFIG_FONT["path"] = path or ""
    font.cache_clear()


def is_complex_script(text: str) -> bool:
    for ch in text:
        if ord(ch) >= 0x0590 and unicodedata.category(ch)[0] == "L":
            name = unicodedata.name(ch, "")
            if not name.startswith(("LATIN", "GREEK", "CYRILLIC")):
                return True
    return False


@lru_cache(maxsize=256)
def font(size: int, bold: bool = False, complex_script: bool = False):
    """Latin text uses Pillow; complex scripts use the Qt/HarfBuzz shaped font (Pillow here has no Raqm), else Pillow as fallback."""
    if complex_script:
        from .shaping import shaped_font
        shaped = shaped_font(size, bold)
        if shaped is not None:
            return shaped
    candidates = list(_COMPLEX if complex_script else (_LATIN_BOLD if bold else _LATIN))
    if _CONFIG_FONT["path"]:
        candidates.insert(0, _CONFIG_FONT["path"])
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _is_shaped(fnt) -> bool:
    return getattr(fnt, "path", "") == "qt-shaped"


def text_width(draw: ImageDraw.ImageDraw, text: str, fnt) -> float:
    return fnt.length(text) if _is_shaped(fnt) else draw.textlength(text, font=fnt)


def put_text(draw, xy, text: str, fnt, fill, stroke: int = 0, stroke_fill=None) -> None:
    """Draw one line: shaped (Qt) for complex scripts, Pillow otherwise. Colours may carry alpha."""
    if _is_shaped(fnt):
        rendered = fnt.render(text, fill)
        target = draw._image
        x, y = int(round(xy[0])) - 2, int(round(xy[1])) - 2
        if target.mode == "RGBA":
            region = target.crop((x, y, x + rendered.width, y + rendered.height))
            target.paste(Image.alpha_composite(region, rendered), (x, y))
        else:
            target.paste(rendered, (x, y), rendered)
        return
    draw.text(xy, text, font=fnt, fill=fill, stroke_width=stroke, stroke_fill=stroke_fill)


def wrap_text(draw, text: str, fnt, max_width: float) -> list[str]:
    """Greedy wrap by measured width. Words that cannot fit are split only between grapheme clusters."""
    if _is_shaped(fnt):
        return fnt.wrap(text, max_width)
    words = text.split()
    if is_complex_script(text) and len(words) <= 1:
        words = _graphemes(text)
        joiner = ""
    else:
        joiner = " "
    lines, line = [], ""
    for word in words:
        candidate = f"{line}{joiner}{word}" if line else word
        if text_width(draw, candidate, fnt) <= max_width:
            line = candidate
            continue
        if line:
            lines.append(line)
        if text_width(draw, word, fnt) <= max_width:
            line = word
            continue
        line = ""
        for cluster in _graphemes(word):
            if line and text_width(draw, line + cluster, fnt) > max_width:
                lines.append(line)
                line = ""
            line += cluster
    if line:
        lines.append(line)
    return lines


def _graphemes(text: str) -> list[str]:
    clusters: list[str] = []
    for ch in text:
        if clusters and (unicodedata.combining(ch) or unicodedata.category(ch) in {"Mn", "Mc", "Me"} or ch in "\u200d\u200c"):
            clusters[-1] += ch
        else:
            clusters.append(ch)
    return clusters


def fit_text(draw, text: str, box: tuple[int, int, int, int], max_px: int, min_px: int = 14,
             bold: bool = False, max_lines: int | None = None):
    """Largest font that fits `text` in `box`. Returns (font, lines, line_height). Raises when nothing fits."""
    left, top, right, bottom = box
    complex_script = is_complex_script(text)
    for px in range(max_px, min_px - 1, -1):
        fnt = font(px, bold, complex_script)
        lines = wrap_text(draw, text, fnt, right - left)
        ascent, descent = fnt.getmetrics()
        line_h = ascent + descent + max(2, px // 8)
        if len(lines) * line_h <= bottom - top and (max_lines is None or len(lines) <= max_lines):
            return fnt, lines, line_h
    raise ValueError(f"Text does not fit its box; shorten or split it: {text[:80]!r}")


def draw_lines(draw, lines, fnt, line_h, x, y, fill, align="left", width=0, stroke=0, stroke_fill=None):
    for i, line in enumerate(lines):
        lx = x
        if align == "center":
            lx = x + (width - text_width(draw, line, fnt)) / 2
        put_text(draw, (lx, y + i * line_h), line, fnt, fill, stroke, stroke_fill)


def hex_rgb(value: str, default=(255, 255, 255)) -> tuple[int, int, int]:
    value = (value or "").strip().lstrip("#")
    if len(value) == 6:
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    return default


def parse_style(text: str) -> dict[str, str]:
    """Parse 'key=value; key2: value2; flag' contract style strings."""
    result: dict[str, str] = {}
    for part in (text or "").split(";"):
        part = part.strip()
        if not part:
            continue
        match = re.match(r"^([A-Za-z_][\w -]*?)\s*[=:]\s*(.+)$", part)
        if match:
            result[match.group(1).strip().lower().replace(" ", "_")] = match.group(2).strip()
        else:
            result[part.lower()] = "true"
    return result


def subtitle_band(size: tuple[int, int], text: str, style, opacity: float | None = None) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """RGBA overlay with a charcoal band sized to the real line count. Returns (overlay, band_box)."""
    if not text.strip():
        return None
    width, height = size
    scale = height / 1080
    px = round(style.font_px * scale)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin_x = round(width * style.margin_x)
    pad = round(style.padding_px * scale)
    max_w = width - 2 * margin_x - 2 * pad
    complex_script = is_complex_script(text)
    fnt = None
    lines: list[str] = []
    for size_px in range(px, max(round(24 * scale), 12) - 1, -1):
        fnt = font(size_px, False, complex_script)
        lines = wrap_text(draw, text, fnt, max_w)
        if len(lines) <= style.max_lines:
            break
    if len(lines) > style.max_lines:
        raise ValueError(f"Subtitle needs more than {style.max_lines} lines at the minimum size: {text[:80]!r}")
    ascent, descent = fnt.getmetrics()
    line_h = ascent + descent + max(2, fnt.size // 8)
    band_h = len(lines) * line_h + 2 * pad
    bottom = height - round(height * style.margin_bottom)
    band = (margin_x - pad // 2, bottom - band_h, width - margin_x + pad // 2, bottom)
    r, g, b = hex_rgb(style.band_color, (17, 20, 23))
    draw.rounded_rectangle(band, radius=round(14 * scale), fill=(r, g, b, round(255 * (style.band_opacity if opacity is None else opacity))))
    fr, fg, fb = hex_rgb(style.text_color)
    draw_lines(draw, lines, fnt, line_h, margin_x, band[1] + pad, (fr, fg, fb, 255), "center", width - 2 * margin_x)
    return overlay, band


def primary_karaoke_band(
    size: tuple[int, int],
    cue: object,
    t: float,
    style=None,
    opacity: float | None = None,
    topic_style: str = "general",
) -> tuple[Image.Image, tuple[int, int, int, int]] | None:
    """Card overlay with word-by-word karaoke highlight for primary school (<= 5th standard).
    Renders warm ivory rounded card with drop shadow and yellow pill highlight behind active word."""
    text = getattr(cue, "text", str(cue or "")).strip()
    if not text:
        return None
    width, height = size
    scale = height / 1080
    from .primary_style import get_topic_style
    theme = get_topic_style(topic_style)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    margin_x = round(28 * scale)
    card_w = width - 2 * margin_x
    card_bottom = round(1000 * scale)
    card_top = round(815 * scale)
    card_h = card_bottom - card_top
    card_box = (margin_x, card_top, margin_x + card_w, card_bottom)

    # Soft drop shadow
    shadow_offset = round(4 * scale)
    shadow_box = (margin_x, card_top + shadow_offset, margin_x + card_w, card_bottom + shadow_offset)
    draw.rounded_rectangle(shadow_box, radius=round(24 * scale), fill=(0, 0, 0, 45))

    # Warm cream / ivory card body with warm chocolate border
    draw.rounded_rectangle(
        card_box,
        radius=round(24 * scale),
        fill=(255, 252, 240, 255),
        outline=hex_rgb(theme.accent) + (255,),
        width=round(3 * scale)
    )
    accent_h = max(3, round(7 * scale))
    draw.rounded_rectangle((margin_x + round(30 * scale), card_top + round(8 * scale),
                            margin_x + card_w - round(30 * scale), card_top + round(8 * scale) + accent_h),
                           radius=accent_h // 2, fill=hex_rgb(theme.accent) + (225,))

    # Typography
    font_size = round(42 * scale)
    complex_script = is_complex_script(text)
    fnt = font(font_size, bold=False, complex_script=complex_script)
    fnt_bold = font(font_size, bold=True, complex_script=complex_script)

    ascent, descent = fnt.getmetrics()
    line_h = ascent + descent + round(8 * scale)
    max_text_w = card_w - round(60 * scale)

    # Words and timings
    cue_words = getattr(cue, "words", []) or []
    raw_words = text.split()
    space_w = text_width(draw, " ", fnt)

    lines: list[list[dict]] = []
    current_line: list[dict] = []
    current_line_w = 0.0

    for i, w_text in enumerate(raw_words):
        w_len = text_width(draw, w_text, fnt)
        if current_line and (current_line_w + space_w + w_len > max_text_w):
            lines.append(current_line)
            current_line = []
            current_line_w = 0.0

        w_caption = cue_words[i] if i < len(cue_words) else None
        current_line.append({
            "text": w_text,
            "width": w_len,
            "caption": w_caption,
            "index": i,
        })
        current_line_w += (space_w if current_line_w > 0 else 0) + w_len

    if current_line:
        lines.append(current_line)

    total_text_h = len(lines) * line_h
    start_y = card_top + (card_h - total_text_h) / 2

    active_idx = -1
    for i, c in enumerate(cue_words):
        if c.start <= t < c.end:
            active_idx = i
            break

    for line_idx, line in enumerate(lines):
        line_w = sum(item["width"] for item in line) + max(0, len(line) - 1) * space_w
        line_x = margin_x + (card_w - line_w) / 2
        line_y = start_y + line_idx * line_h

        cursor_x = line_x
        for item in line:
            is_active = (item["index"] == active_idx)
            w_w = item["width"]

            if is_active:
                pill_pad_x = round(6 * scale)
                pill_pad_y = round(3 * scale)
                pill_box = (
                    cursor_x - pill_pad_x,
                    line_y - pill_pad_y,
                    cursor_x + w_w + pill_pad_x,
                    line_y + line_h - round(5 * scale) + pill_pad_y,
                )
                draw.rounded_rectangle(
                    pill_box,
                    radius=round(6 * scale),
                    fill=hex_rgb(theme.highlight) + (255,),
                    outline=hex_rgb(theme.accent) + (255,),
                    width=round(1.5 * scale)
                )
                put_text(draw, (cursor_x, line_y), item["text"], fnt_bold, (15, 15, 15, 255))
            else:
                put_text(draw, (cursor_x, line_y), item["text"], fnt, (35, 38, 42, 255))

            cursor_x += w_w + space_w

    return overlay, card_box


def primary_scene_banner(size: tuple[int, int], text: str, topic_style: str = "general") -> Image.Image | None:
    """Topic-colored cream pill banner; long scene titles wrap and stay inside the frame."""
    clean = text.strip()
    if not clean:
        return None
    width, height = size
    scale = height / 1080
    from .primary_style import get_topic_style
    theme = get_topic_style(topic_style)
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    max_banner_w = width - round(96 * scale)
    pad_x = round(38 * scale)
    pad_y = round(12 * scale)
    chosen = None
    for font_size in range(round(52 * scale), round(30 * scale) - 1, -1):
        fnt = font(font_size, bold=True, complex_script=is_complex_script(clean))
        lines = wrap_text(draw, clean, fnt, max_banner_w - 2 * pad_x)
        ascent, descent = fnt.getmetrics()
        line_h = ascent + descent + max(2, round(3 * scale))
        if len(lines) <= 2:
            chosen = (fnt, lines, line_h)
            break
    if chosen is None:
        fnt = font(round(30 * scale), bold=True, complex_script=is_complex_script(clean))
        lines = wrap_text(draw, clean, fnt, max_banner_w - 2 * pad_x)
        ascent, descent = fnt.getmetrics()
        chosen = (fnt, lines[:2], ascent + descent + max(2, round(3 * scale)))
    fnt, lines, line_h = chosen
    text_w = max(text_width(draw, line, fnt) for line in lines)
    banner_w = min(max_banner_w, text_w + 2 * pad_x)
    banner_h = max(round(82 * scale), len(lines) * line_h + 2 * pad_y)

    banner_x = (width - banner_w) // 2
    banner_y = round(52 * scale)
    banner_box = (banner_x, banner_y, banner_x + banner_w, banner_y + banner_h)

    shadow_offset = round(3 * scale)
    draw.rounded_rectangle(
        (banner_x, banner_y + shadow_offset, banner_x + banner_w, banner_y + banner_h + shadow_offset),
        radius=round(14 * scale),
        fill=(0, 0, 0, 35)
    )

    draw.rounded_rectangle(
        banner_box,
        radius=round(14 * scale),
        fill=(255, 253, 245, 240),
        outline=hex_rgb(theme.accent) + (245,),
        width=round(3 * scale)
    )
    badge_r = round(8 * scale)
    badge_x = banner_x + round(14 * scale)
    badge_y = banner_y + banner_h // 2
    draw.ellipse((badge_x, badge_y - badge_r, badge_x + badge_r * 2, badge_y + badge_r),
                 fill=hex_rgb(theme.accent) + (255,))
    content_left = banner_x + pad_x + round(10 * scale)
    content_width = banner_w - 2 * pad_x - round(10 * scale)
    total_text_h = len(lines) * line_h
    ty = banner_y + (banner_h - total_text_h) / 2
    draw_lines(draw, lines, fnt, line_h, content_left, ty, (28, 36, 48, 255), "center", content_width)
    return overlay


def primary_topic_cards(size: tuple[int, int], items: list[str], topic_style: str = "general") -> Image.Image | None:
    """Exact, readable topic panels over primary illustrations; the image model never writes their text."""
    items = [" ".join((item or "").split()) for item in items if (item or "").strip()][:4]
    if len(items) < 2:
        return None
    from .primary_style import get_topic_style
    theme = get_topic_style(topic_style)
    width, height = size
    scale = height / 1080
    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    margin_x = round(76 * scale)
    gap_x = round(24 * scale)
    gap_y = round(18 * scale)
    if len(items) <= 3:
        columns, rows = len(items), 1
        top, card_h = round(515 * scale), round(235 * scale)
    else:
        columns, rows = 2, 2
        top, card_h = round(380 * scale), round(185 * scale)
    card_w = (width - 2 * margin_x - gap_x * (columns - 1)) // columns
    accent = hex_rgb(theme.accent)
    for index, text in enumerate(items):
        row, col = divmod(index, columns)
        left = margin_x + col * (card_w + gap_x)
        y = top + row * (card_h + gap_y)
        box = (left, y, left + card_w, y + card_h)
        radius = round(26 * scale)
        draw.rounded_rectangle((left + round(4 * scale), y + round(8 * scale), box[2] + round(4 * scale), box[3] + round(8 * scale)),
                               radius=radius, fill=(21, 33, 42, 48))
        draw.rounded_rectangle(box, radius=radius, fill=(255, 251, 235, 238),
                               outline=accent + (245,), width=max(2, round(3 * scale)))
        badge = round(19 * scale)
        badge_x, badge_y = left + round(22 * scale), y + round(22 * scale)
        draw.ellipse((badge_x, badge_y, badge_x + 2 * badge, badge_y + 2 * badge), fill=accent + (255,))
        number_font = font(round(22 * scale), True)
        number = str(index + 1)
        nw = text_width(draw, number, number_font)
        na, nd = number_font.getmetrics()
        put_text(draw, (badge_x + badge - nw / 2, badge_y + (2 * badge - na - nd) / 2), number, number_font, (255, 255, 255, 255))

        pad_l = round(28 * scale)
        text_box = (left + pad_l, y + round(18 * scale), box[2] - round(24 * scale), box[3] - round(18 * scale))
        max_px, min_px = round(34 * scale), max(14, round(20 * scale))
        chosen = None
        for px in range(max_px, min_px - 1, -1):
            fnt = font(px, True, is_complex_script(text))
            lines = wrap_text(draw, text, fnt, text_box[2] - text_box[0])
            ascent, descent = fnt.getmetrics()
            line_h = ascent + descent + max(2, round(4 * scale))
            if len(lines) * line_h <= text_box[3] - text_box[1]:
                chosen = (fnt, lines, line_h)
                break
        if chosen is None:
            fnt = font(min_px, True, is_complex_script(text))
            lines = wrap_text(draw, text, fnt, text_box[2] - text_box[0])
            ascent, descent = fnt.getmetrics()
            line_h = ascent + descent + max(2, round(4 * scale))
            chosen = (fnt, lines, line_h)
        fnt, lines, line_h = chosen
        total_h = len(lines) * line_h
        ty = y + (card_h - total_h) / 2
        draw_lines(draw, lines, fnt, line_h, text_box[0], ty, (38, 48, 51, 255), "center", text_box[2] - text_box[0])
    return overlay

