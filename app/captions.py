"""Subtitle cues that always fit the subtitle band (<= max_lines, safe margins), timed to speech."""
from __future__ import annotations

import re

from PIL import Image, ImageDraw

from .schema import Caption, Segment
from .typography import font, is_complex_script, wrap_text

_probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))


def _fits(text: str, style, size: tuple[int, int]) -> bool:
    width, height = size
    scale = height / 1080
    max_w = width - 2 * round(width * style.margin_x) - 2 * round(style.padding_px * scale)
    fnt = font(round(style.font_px * scale), False, is_complex_script(text))
    return len(wrap_text(_probe, text, fnt, max_w)) <= style.max_lines


def _display_text(segment: Segment) -> str:
    """Use the punctuated narration when the subtitle only differs by punctuation/case (e.g. stripped commas)."""
    norm = lambda t: re.sub(r"[^\w]+", " ", (t or "").lower()).split()
    if not segment.subtitle or norm(segment.subtitle) == norm(segment.narration):
        return segment.narration
    return segment.subtitle


def _with_punctuation(timed: list[Caption], words: list[str]) -> list[Caption]:
    """The speech engine's word timings carry no punctuation, so cues never ended at a sentence. When the timed words line up one to one
    with the narration, show the narration's own words (with their commas and full stops) at the voice's timings."""
    if len(timed) != len(words):
        return timed
    norm = lambda t: re.sub(r"[^\w]+", "", (t or "").lower())
    same = sum(1 for caption, word in zip(timed, words) if norm(caption.text) == norm(word))
    if same < 0.85 * len(words):                       # numbers, symbols or abbreviations were split differently: keep the engine's words
        return timed
    return [Caption(text=word, start=caption.start, end=caption.end) for caption, word in zip(timed, words)]


def segment_words(segment: Segment) -> list[Caption]:
    """Word timings for the subtitle text; falls back to proportional timing (preview or missing audio)."""
    text = _display_text(segment).split()
    duration = segment.speech_duration or max(0.1, segment.end - segment.start)
    narration_words = segment.narration.split()
    if segment.captions and text == narration_words:
        timed = _with_punctuation(list(segment.captions), narration_words)
        if [c.text for c in timed] == narration_words:
            return timed
        # the engine split the words differently ("1:1.21" -> "1", "1", ".", "21"): never show its pieces; show the narration's own
        # words, timed across the span the voice actually spoke, in proportion to their length
        start, end = segment.captions[0].start, segment.captions[-1].end
        weights = [len(w) + 1 for w in narration_words]
        total, cursor, words = float(sum(weights)), start, []
        for word, weight in zip(narration_words, weights):
            step = (end - start) * weight / total
            words.append(Caption(text=word, start=cursor, end=cursor + step))
            cursor += step
        return words
    if not text:
        return []
    step = duration / len(text)
    return [Caption(text=w, start=i * step, end=(i + 1) * step) for i, w in enumerate(text)]


def build_cues(segment: Segment, style, size: tuple[int, int]) -> list[Caption]:
    words = segment_words(segment)
    cues: list[Caption] = []
    group: list[Caption] = []

    def flush():
        nonlocal group
        if group:
            text = re.sub(r"\s+([.,;:!?])", r"\1", " ".join(w.text for w in group))        # the speech engine can report a full stop as its own word
            cues.append(Caption(text=text, start=group[0].start, end=group[-1].end))
            group = []

    for word in words:
        candidate = " ".join(w.text for w in group + [word])
        if group and not _fits(candidate, style, size):
            flush()
        group.append(word)
        spoken = group[-1].end - group[0].start
        if word.text.endswith((".", "?", "!", ";", ":", "।", "۔", "؟", "…")) and len(group) >= 3:   # incl. danda, Urdu stop, question mark
            flush()
        elif word.text.endswith((",", "،")) and len(group) >= 6 and spoken >= 4.0:
            flush()            # a long, slowly spoken sentence: change the subtitle at its commas, not after 12 s of the same two lines
        elif spoken >= 7.0 and len(group) >= 6:
            flush()
    flush()
    return _extend_to_next(cues, segment)


def _extend_to_next(cues: list[Caption], segment: Segment) -> list[Caption]:
    """Hold each cue until the next begins so the band does not flicker between phrases. The last cue leaves 1.5 s after the voice
    stops: a silent hold at the end of a shot (an equation being read, labels appearing) shows the picture, not an old sentence."""
    total = max(segment.end - segment.start, cues[-1].end if cues else 0)
    result = []
    for i, cue in enumerate(cues):
        end = cues[i + 1].start if i + 1 < len(cues) else min(total, cue.end + 1.5)
        result.append(Caption(text=cue.text, start=cue.start, end=max(end, cue.start + 0.05)))
    return result
