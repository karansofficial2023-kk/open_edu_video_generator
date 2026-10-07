"""Teacher's board for physics and chemistry lessons (the style of good classroom videos, built generically).

One board per part of the lesson: the section title stays on top while every new sentence ADDS to the board instead of replacing the
screen - a definition in a framed box, points as bullets with an arrow, a comparison as a table filled row by row, a derivation line
by line, a "uses" list as chips, a summary at the end. Text types in with the voice; earlier items stay, slightly dimmed; when the board
is full the next shot starts a fresh page. Dark theme (navy / deep purple / dark teal, chosen per lesson) with cyan, green and yellow
accents.

Which shots use it is decided in production.py from the lesson's subject (subject.board_style); nothing here depends on a topic.
The layout of each shot comes from the storyboard (`layout`, `table_rows`, `gallery_items`, written by Transcribe's LayoutPlanner) or,
for older storyboards, from the same simple rules applied to the narration.
"""
from __future__ import annotations

import hashlib
import math
import re
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from .typography import font, hex_rgb, text_width, wrap_text

THEMES = [  # (top, bottom) background colours
    ((12, 22, 48), (5, 9, 22)),        # navy
    ((30, 16, 40), (12, 6, 18)),       # deep purple
    ((10, 34, 38), (4, 14, 17)),       # dark teal
]
INK, DIM = (238, 242, 246), (150, 160, 172)
CYAN, GREEN, YELLOW = (64, 210, 230), (150, 230, 110), (245, 214, 90)

X0, X1, TOP, BOTTOM = 120, 1800, 176, 838          # content area at 1920x1080 (subtitles live below BOTTOM)
FULL_X0, SPLIT_X0 = 120, 820                       # with a main visual on the left the board uses the right part of the screen
ANCHOR_BOX = (60, 176, 770, 838)


@contextmanager
def columns(split: bool):
    """Board geometry for one page: full width, or the right part when the part's main visual stays on the left."""
    global X0
    previous = X0
    X0 = SPLIT_X0 if split else FULL_X0
    try:
        yield
    finally:
        X0 = previous
BODY_PX, LINE_H = 42, 56
TYPE_RATE = 60          # characters per second: the line is on the board quickly, the voice then explains it

_DEFINITION = re.compile(r"(?i)\b(is|are) (called|known as|defined as|termed|referred to as)\b|\bis the (property|ability|process|measure|ratio)\b|\brefers to\b")
_CONTRAST = re.compile(r"(?i)\b(whereas|unlike|in contrast|on the other hand|compared (to|with)|versus|vs\.?)\b")
_SUMMARY = re.compile(r"(?i)\b(to summari[sz]e|in summary|summary|to conclude|in conclusion|recap)\b")
_RESULT = re.compile(r"(?i)\b(therefore|thus|hence|finally|so the|which gives|we get|is equal to|equals)\b")

_probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def theme_for(title: str) -> tuple:
    return THEMES[int(hashlib.sha1((title or "").encode()).hexdigest(), 16) % len(THEMES)]


def _sentences(text: str) -> list[str]:
    from .assets import is_filler_narration
    parts = [s.strip() for s in re.split(r"(?<=[.!?])\s+", (text or "").strip()) if s.strip()]
    kept = [s for s in parts if not is_filler_narration(s)]             # remarks are spoken, never written on the board
    return kept or parts


_LEAD_IN = re.compile(r"(?i)^(so|now|well|okay|ok|then|also|and|but|basically|actually|in fact|you see|as you can see|as we know|"
                      r"as we (?:have )?(?:saw|seen|discussed|learned|learnt)|we know that|we can (?:see|say) that|we see that|"
                      r"it means that|this means that|that means|remember that|note that|notice that|let(?:'s| us) (?:see|consider|look at|take)|here)\b[,:]?\s+")
_CLAUSE = re.compile(r";\s|,\s|\s[-–—]\s")
POINT_WORDS = 16


def key_point(sentence: str) -> str:
    """What a teacher writes on the board for a spoken sentence: no lead-in words, a long sentence cut at its first clause, no full
    stop. The words are the narration's own (nothing is rephrased, so nothing new can be claimed)."""
    text = (sentence or "").strip()
    while True:
        shorter = _LEAD_IN.sub("", text, count=1)
        if shorter == text or len(shorter.split()) < 3:
            break
        text = shorter
    if len(text.split()) > POINT_WORDS:
        for match in _CLAUSE.finditer(text):
            head = text[:match.start()].strip()
            if 6 <= len(head.split()) <= POINT_WORDS and not re.search(r"(?i)\b(the|a|an|of|to|and|or|but|is|are|not)$", head):
                text = head
                break
    text = text.rstrip(" .;:,")
    return text[:1].upper() + text[1:] if text else (sentence or "").strip()


def layout_of(segment) -> str:
    layout = (getattr(segment, "layout", "") or "").strip().lower()
    if segment.formula_lines:
        return "derivation"
    if layout in {"definition", "bullets", "table", "gallery", "summary"}:
        if layout == "table" and len([r for r in segment.table_rows if "|" in r]) < 2:
            return "bullets"
        if layout == "gallery" and len(segment.gallery_items) < 2:
            return "bullets"
        if layout == "definition" and not re.search(r"(?i)(is|are|means|refers)", segment.narration or ""):
            return "bullets"                    # "Now, let's consider resistivity" introduces a term, it does not define it
        if layout == "summary" and not _SUMMARY.search(segment.narration or ""):
            return "bullets"                    # only a real recap gets the "Summary" heading
        return layout
    text = segment.narration or ""
    if _SUMMARY.search(text):
        return "summary"
    if _DEFINITION.search(text) and len(_sentences(text)) == 1:
        return "definition"
    return "bullets"


def items_for(segment) -> list[dict]:
    """What this shot adds to the board."""
    layout = layout_of(segment)
    if layout == "derivation":
        # like a teacher: the statement first ("The velocity of the wave in free space is given by"), then the equations line by line
        first = _sentences(segment.narration)[:1]
        lead = [{"kind": "lead", "text": first[0]}] if first and len(first[0].split()) <= 45 else []
        notes = list(segment.explain_steps or [])
        lines = [line for line in segment.formula_lines if line.strip()]
        if len(lines) >= 3 and all(len(line) <= 18 for line in lines):
            return lead + [{"kind": "eqrow", "items": lines}]            # short steps (R ∝ l -> R ∝ l/a -> R = ρl/a) read as one row
        items = [{"kind": "eq", "text": line, "note": notes[i] if i < len(notes) else ""} for i, line in enumerate(lines)]
        if items and _RESULT.search(segment.narration or ""):
            items[-1]["key"] = True                                       # the result of the derivation keeps its box
        return lead + items
    if layout == "table":
        rows = [r for r in segment.table_rows if "|" in r]
        return [{"kind": "row", "cells": [c.strip() for c in row.split("|", 1)], "header": i == 0} for i, row in enumerate(rows)]
    if layout == "definition":
        return [{"kind": "box", "text": " ".join(_sentences(segment.narration))}]
    if layout == "gallery":
        sentences = _sentences(segment.narration)
        return ([{"kind": "bullet", "text": key_point(sentences[0])}] if sentences else []) + \
               [{"kind": "chips", "items": [str(i)[:40] for i in segment.gallery_items[:6]]}]
    if layout == "summary":
        return [{"kind": "heading", "text": "Summary"}] + [{"kind": "dot", "text": key_point(s)} for s in _sentences(segment.narration)]
    return [{"kind": "bullet", "text": key_point(s)} for s in _sentences(segment.narration)]


def _note_lines(note: str) -> list[str]:
    """The words under an equation, at most two lines (a third is cut with an ellipsis)."""
    if not note:
        return []
    lines = wrap_text(_probe, note, font(30), X1 - X0 - 70)
    if len(lines) > 2:
        lines = lines[:2]
        lines[1] = lines[1].rstrip(" ,.;")[:-2].rstrip() + "…"
    return lines


def _chip_rows(items: list[str]) -> list[list[str]]:
    fnt, rows, row, width = font(30, True), [], [], 0
    for item in items:
        w = text_width(_probe, item, fnt) + 60
        if row and width + w > X1 - X0:
            rows.append(row)
            row, width = [], 0
        row.append(item)
        width += w + 18
    return rows + ([row] if row else [])


def height_of(item: dict) -> int:
    kind = item["kind"]
    body = font(BODY_PX)
    if kind in {"bullet", "dot"}:
        return len(wrap_text(_probe, item["text"], body, X1 - X0 - 70)) * LINE_H + 22
    if kind == "box":
        return len(wrap_text(_probe, item["text"], body, X1 - X0 - 120)) * LINE_H + 64
    if kind == "row":
        half = (X1 - X0) // 2 - 40
        lines = max(len(wrap_text(_probe, c, font(32, item["header"]), half)) for c in item["cells"])
        return lines * 42 + 30
    if kind == "eq":
        return 132 + 40 * len(_note_lines(item.get("note", "")))
    if kind == "lead":
        return len(wrap_text(_probe, item["text"], font(38), X1 - X0 - 40)) * 50 + 16
    if kind == "heading":
        return 70
    if kind == "eqrow":
        return 130
    if kind == "chips":
        return len(_chip_rows(item["items"])) * 74 + 10
    if kind == "thumbs":
        return round(_thumb_width(len(item["items"])) * 0.75) + 74
    return 60


def _thumb_width(count: int) -> int:
    return min(320, (X1 - X0 - 24 * (count - 1)) // max(1, count))


MIN_EQ_HEIGHT = 62      # an equation squeezed below this height (at 1080p) cannot be read from the back of a class


@lru_cache(maxsize=256)
def _eq_size(text: str) -> tuple[int, int]:
    from .formulas import render_line
    try:
        eq = render_line(text, "#FFFFFF", 84)
        return eq.width, eq.height
    except Exception:
        return 0, 0


def wide(items: list[dict]) -> bool:
    """True when an equation of this shot would be too small in the right-hand column (it then gets the whole board)."""
    room = X1 - SPLIT_X0 - 120                         # the equation's width in the right-hand column
    for item in items:
        if item["kind"] == "eq":
            w, h = _eq_size(item["text"])
            if w and h * min(room / w, 112 / h, 2.2) < MIN_EQ_HEIGHT:
                return True
    return False


def fits(items: list[dict], split: bool = False) -> bool:
    with columns(split):
        return sum(height_of(i) for i in items) <= BOTTOM - TOP


def _chars(item: dict) -> int:
    kind = item["kind"]
    if kind == "row":
        return sum(len(c) + 4 for c in item["cells"])           # the drawing spends one extra per wrapped line: never cut a cell short
    if kind in {"chips", "thumbs"}:
        return 12 * len(item["items"])
    if kind == "eq":
        return 30
    if kind == "eqrow":
        return 20 * len(item["items"])
    return len(item.get("text", ""))


@lru_cache(maxsize=8)
def _background(size: tuple[int, int], theme: tuple) -> Image.Image:
    width, height = size
    top, bottom = theme
    ramp = Image.linear_gradient("L").resize(size)
    image = Image.composite(Image.new("RGB", size, bottom), Image.new("RGB", size, top), ramp)
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((width * 0.45, -height * 0.35, width * 1.25, height * 0.55), fill=(*CYAN, 22))
    glow = glow.filter(ImageFilter.GaussianBlur(height // 5))
    return Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")


def _arrow(draw, x, y, color):
    draw.polygon([(x, y), (x + 14, y + 9), (x, y + 18)], fill=color)


def _draw_item(draw: ImageDraw.ImageDraw, image: Image.Image, item: dict, y: int, shown: int, current: bool, dim: bool, scale: float) -> None:
    """Draw one item at y, with only `shown` characters of its text revealed (typewriter)."""
    px = lambda v: round(v * scale)
    ink = DIM if dim else INK
    kind = item["kind"]
    body = font(px(BODY_PX))
    if kind in {"bullet", "dot"}:
        lines = wrap_text(_probe, item["text"], font(BODY_PX), X1 - X0 - 70)
        marker = (CYAN if current else (90, 130, 150)) if kind == "bullet" else (YELLOW if current else DIM)
        if kind == "bullet":
            _arrow(draw, px(X0), px(y + 12), marker)
        else:
            draw.ellipse((px(X0), px(y + 14), px(X0 + 14), px(y + 28)), fill=marker)
        left = shown
        for i, line in enumerate(lines):
            if left <= 0:
                break
            draw.text((px(X0 + 40), px(y + i * LINE_H)), line[:left], font=body, fill=ink)
            left -= len(line) + 1
    elif kind == "box":
        lines = wrap_text(_probe, item["text"], font(BODY_PX), X1 - X0 - 120)
        h = len(lines) * LINE_H + 40
        border = CYAN if current else (70, 110, 130)
        draw.rounded_rectangle((px(X0), px(y), px(X1), px(y + h)), radius=px(16), fill=(255, 255, 255, 12),
                               outline=border, width=max(1, px(3 if current else 2)))
        left = shown
        for i, line in enumerate(lines):
            if left <= 0:
                break
            draw.text((px(X0 + 60), px(y + 20 + i * LINE_H)), line[:left], font=body, fill=ink)
            left -= len(line) + 1
    elif kind == "row":
        half = (X1 - X0) // 2
        header = item["header"]
        fnt = font(px(32), header)
        lines_per = [wrap_text(_probe, c, font(32, header), half - 40) for c in item["cells"]]
        h = max(len(l) for l in lines_per) * 42 + 30
        line_color = (90, 120, 140)
        draw.rectangle((px(X0), px(y), px(X1), px(y + h)), outline=line_color, width=max(1, px(2)))
        draw.line((px(X0 + half), px(y), px(X0 + half), px(y + h)), fill=line_color, width=max(1, px(2)))
        left = shown
        for col, lines in enumerate(lines_per):
            color = (GREEN if header else ink)
            for i, line in enumerate(lines):
                if left <= 0:
                    break
                segment_text = line[:left]
                x = px(X0 + col * half + 22)
                yy = px(y + 15 + i * 42)
                if not header and segment_text.lower().startswith("example"):
                    head, _, rest = segment_text.partition(":")
                    draw.text((x, yy), head + ":", font=fnt, fill=CYAN)
                    draw.text((x + text_width(draw, head + ": ", fnt), yy), rest.lstrip(), font=fnt, fill=ink)
                else:
                    draw.text((x, yy), segment_text, font=fnt, fill=color)
                left -= len(line) + 1
    elif kind == "eq":
        if shown <= 0:
            return
        from .formulas import render_line
        color = "#F5D65A" if current else ("#9AA6B2" if dim else "#EEF2F6")
        try:
            eq = render_line(item["text"], color, px(84))
        except Exception:
            draw.text((px(X0 + 40), px(y + 30)), item["text"][:shown], font=body, fill=ink)
            return
        max_w, max_h = px(X1 - X0 - 120), px(112)
        ratio = min(max_w / eq.width, max_h / eq.height, 2.2)
        eq = eq.resize((max(1, round(eq.width * ratio)), max(1, round(eq.height * ratio))), Image.Resampling.LANCZOS)
        fade = min(1.0, shown / 30)
        if fade < 1:
            eq.putalpha(eq.getchannel("A").point(lambda v: round(v * fade)))
        x, yy = px(X0 + 60), px(y + 10) + (max_h - eq.height) // 2
        image.paste(eq, (x, yy), eq)
        if (current or item.get("key")) and fade >= 1:
            draw.rounded_rectangle((x - px(18), yy - px(10), x + eq.width + px(18), yy + eq.height + px(10)), radius=px(12),
                                   outline=YELLOW if current else GREEN, width=max(1, px(3 if item.get("key") else 2)))
        if item.get("note") and fade >= 1:          # what the line says, in words, under it (as a teacher would add)
            for i, line in enumerate(_note_lines(item["note"])):
                draw.text((px(X0 + 60), px(y + 128 + i * 40)), line, font=font(px(30)), fill=(DIM if dim else (170, 205, 220)))
    elif kind == "lead":
        lines = wrap_text(_probe, item["text"], font(38), X1 - X0 - 40)
        left = shown
        for i, line in enumerate(lines):
            if left <= 0:
                break
            draw.text((px(X0), px(y + i * 50)), line[:left], font=font(px(38)), fill=(DIM if dim else YELLOW))
            left -= len(line) + 1
    elif kind == "eqrow":
        from .formulas import render_line
        visible = max(0, shown // 20 + (1 if shown % 20 else 0))
        x = px(X0)
        for i, text in enumerate(item["items"][:visible]):
            if i:
                draw.text((x + px(6), px(y + 38)), "→", font=font(px(40), True), fill=CYAN)
                x += px(62)
            try:
                eq = render_line(text, "#F5D65A" if current and i == visible - 1 else ("#9AA6B2" if dim else "#EEF2F6"), px(56))
            except Exception:
                continue
            ratio = min(px(80) / eq.height, 1.4)
            eq = eq.resize((max(1, round(eq.width * ratio)), max(1, round(eq.height * ratio))), Image.Resampling.LANCZOS)
            if x + eq.width + px(40) > px(X1):
                break
            draw.rounded_rectangle((x, px(y + 10), x + eq.width + px(36), px(y + 110)), radius=px(14), fill=(255, 255, 255, 16),
                                   outline=(70, 110, 130), width=max(1, px(2)))
            image.paste(eq, (x + px(18), px(y + 60) - eq.height // 2), eq)
            x += eq.width + px(36)
    elif kind == "heading":
        draw.text((px(X0), px(y + 8)), item["text"][:max(0, shown)], font=font(px(44), True), fill=YELLOW)
    elif kind == "thumbs":                     # a row of picture cards with the item's name under each (pictures from gallery.py)
        count = len(item["items"])
        w = _thumb_width(count)
        h = round(w * 0.75)
        visible = max(0, shown // 12 + (1 if shown % 12 else 0))
        fnt = font(px(28), True)
        for i, (label, path) in enumerate(zip(item["items"], item.get("images") or [None] * count)):
            if i >= visible:
                return
            x = X0 + i * (w + 24)
            box = (px(x), px(y), px(x + w), px(y + h))
            if path and Path(path).is_file():
                picture = _anchor_picture(path, (box[2] - box[0], box[3] - box[1]))
                if dim:
                    picture = Image.blend(picture, Image.new("RGB", picture.size, (20, 26, 34)), 0.35)
                mask = Image.new("L", picture.size, 0)
                ImageDraw.Draw(mask).rounded_rectangle((0, 0, picture.width - 1, picture.height - 1), radius=px(14), fill=255)
                image.paste(picture, box[:2], mask)
            else:
                draw.rounded_rectangle(box, radius=px(14), fill=(30, 70, 84) if not dim else (34, 44, 52))
            draw.rounded_rectangle(box, radius=px(14), outline=CYAN if current else (70, 110, 130), width=max(1, px(2)))
            text = label
            while text and text_width(draw, text, fnt) > px(w):
                text = text[:-2].rstrip() + "…" if len(text) > 2 else ""
            draw.text((px(x + w / 2) - text_width(draw, text, fnt) / 2, px(y + h + 14)), text, font=fnt, fill=ink)
    elif kind == "chips":
        fnt = font(px(30), True)
        visible = max(0, shown // 12 + (1 if shown % 12 else 0))
        count = 0
        for r, row in enumerate(_chip_rows(item["items"])):
            x = X0
            for label in row:
                if count >= visible:
                    return
                w = text_width(_probe, label, font(30, True)) + 60
                draw.rounded_rectangle((px(x), px(y + r * 74), px(x + w), px(y + r * 74 + 56)), radius=px(28),
                                       fill=(30, 70, 84) if not dim else (34, 44, 52), outline=CYAN if current else (70, 110, 130),
                                       width=max(1, px(2)))
                draw.text((px(x + 30), px(y + r * 74 + 11)), label, font=fnt, fill=ink)
                x += w + 18
                count += 1


def _draw_anchor(image: Image.Image, anchor: dict, t: float, scale: float) -> None:
    """The part's main visual on the left: its photograph (framed) or, in chemistry, the named molecule turning slowly."""
    px = lambda v: round(v * scale)
    x0, y0, x1, y1 = (px(v) for v in ANCHOR_BOX)
    draw = ImageDraw.Draw(image, "RGBA")
    if anchor.get("kind") == "image" and anchor.get("path") and Path(anchor["path"]).is_file():
        picture = _anchor_picture(anchor["path"], (x1 - x0, y1 - y0))
        mask = Image.new("L", picture.size, 0)
        ImageDraw.Draw(mask).rounded_rectangle((0, 0, picture.width - 1, picture.height - 1), radius=px(18), fill=255)
        image.paste(picture, (x0, y0), mask)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=px(18), outline=(90, 130, 150), width=max(1, px(2)))
    elif anchor.get("kind") == "diagram":
        from . import diagrams
        picture = diagrams.render(anchor["concept"], (x1 - x0, y1 - y0), t, anchor.get("text", ""))
        if picture is not None:
            image.paste(picture, (x0, y0), picture)
    elif anchor.get("kind") == "molecule":
        from . import molecules
        model = molecules.render(anchor["smiles"], (x1 - x0, y1 - y0 - px(80)), 0.6 + 0.35 * t)
        if model is not None:
            image.paste(model, (x0, y0), model)
        name = anchor.get("name", "")
        if name:
            fnt = font(px(34), True)
            w = round(text_width(draw, name, fnt)) + px(56)
            cx = (x0 + x1) // 2
            draw.rounded_rectangle((cx - w // 2, y1 - px(70), cx + w // 2, y1 - px(14)), radius=px(28), fill=(40, 30, 90, 230),
                                   outline=CYAN, width=max(1, px(2)))
            draw.text((cx - w // 2 + px(28), y1 - px(62)), name, font=fnt, fill=INK)


@lru_cache(maxsize=16)
def _anchor_picture(path: str, size: tuple) -> Image.Image:
    with Image.open(path) as source:
        picture = source.convert("RGB")
    ratio = max(size[0] / picture.width, size[1] / picture.height)
    picture = picture.resize((round(picture.width * ratio), round(picture.height * ratio)), Image.Resampling.LANCZOS)
    left, top = (picture.width - size[0]) // 2, (picture.height - size[1]) // 2
    return picture.crop((left, top, left + size[0], top + size[1]))


def frame(size: tuple[int, int], theme: tuple, title: str, previous: list[dict], current: list[dict], t: float, reveal_end: float,
          anchor: dict | None = None) -> Image.Image:
    """The board at time t: earlier items dimmed, the shot's own items typed in between 0.25 s and reveal_end."""
    with columns(anchor is not None):
        image = _frame(size, theme, title, previous, current, t, reveal_end)
        if anchor is not None:
            _draw_anchor(image, anchor, t, size[1] / 1080)
        return image


def _frame(size: tuple[int, int], theme: tuple, title: str, previous: list[dict], current: list[dict], t: float, reveal_end: float) -> Image.Image:
    scale = size[1] / 1080
    px = lambda v: round(v * scale)
    image = _background(size, theme).copy()
    draw = ImageDraw.Draw(image, "RGBA")
    heading = (title or "").upper()
    hfont = font(px(44), True)
    while text_width(draw, heading, hfont) > px(X1 - FULL_X0) and hfont.size > px(26):
        hfont = font(hfont.size - 2, True)
    draw.text((px(FULL_X0 - 24), px(58)), heading, font=hfont, fill=INK)          # the section title always spans the screen
    draw.rounded_rectangle((px(FULL_X0 - 24), px(124), px(FULL_X0 + 156), px(130)), radius=px(3), fill=CYAN)
    y = TOP
    for item in previous:
        _draw_item(draw, image, item, y, 10 ** 6, False, True, scale)
        y += height_of(item)
    total = max(1, sum(_chars(i) for i in current))
    start = 0.25
    end = typing_end(current, reveal_end)
    progress = 1.0 if end <= start else min(1.0, max(0.0, (t - start) / (end - start)))
    budget = round(progress * total)
    reveals, left = [], budget
    for item in current:
        reveals.append(max(0, min(_chars(item), left)))
        left -= _chars(item)
    # the item being written now (or the last one once everything is written) is highlighted
    active = next((i for i, (item, shown) in enumerate(zip(current, reveals)) if shown < _chars(item)), len(current) - 1)
    for index, item in enumerate(current):
        _draw_item(draw, image, item, y, reveals[index], index == active, False, scale)
        y += height_of(item)
    return image


def typing_end(current: list[dict], reveal_end: float) -> float:
    """Typing ends at TYPE_RATE characters a second, and never later than about half of the narration (reveal_end)."""
    total = sum(_chars(i) for i in current)
    return 0.25 + min(max(0.4, total / TYPE_RATE), max(0.4, reveal_end * 0.5))


def done_at(current: list[dict], reveal_end: float) -> float:
    return typing_end(current, reveal_end) + 0.05
