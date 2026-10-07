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
