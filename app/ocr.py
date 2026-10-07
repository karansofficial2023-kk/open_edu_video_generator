"""Stray-lettering detector for generated stills (RapidOCR, Apache-2.0, runs on the CPU in about a second).

Image models draw fake captions, numerals and logos that a vision-language reviewer often overlooks. OCR is deterministic and
independent of that reviewer, so a still with readable lettering is rejected no matter what the reviewer said.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

_engine = None
_unavailable = False


def _get_engine():
    global _engine, _unavailable
    if _engine is None and not _unavailable:
        try:
            from rapidocr_onnxruntime import RapidOCR
            _engine = RapidOCR()
        except Exception:      # package missing or runtime error: the check is optional
            _unavailable = True
    return _engine


SMALL_TEXT_LINES = 5            # this many small text lines in one picture = a page/label of writing


def stray_text(path: str | Path, min_conf: float = 0.85, min_height: float = 0.035) -> list[str] | None:
    """Readable words in the picture, or None when OCR is unavailable.

    Only words of two or more characters, recognised confidently and tall enough for a viewer to notice, count; single letters are
    usually object edges and tiny marks are ignored.
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        import cv2
        # cv2.imread cannot open non-ASCII paths on Windows (Tamil/Telugu/Hindi folder names), so decode the bytes ourselves
        # and give the engine the pixels instead of the path
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        height = image.shape[0]
        result, _ = engine(image)
    except Exception:
        return None
    words, small = [], []
    for box, text, confidence in result or []:
        text = str(text).strip()
        if len([c for c in text if c.isalnum()]) < 2:
            continue
        if float(confidence) >= min_conf and abs(box[2][1] - box[0][1]) / height >= min_height:
            words.append(text)
        elif float(confidence) >= 0.5:
            small.append(text)
    if len(small) >= SMALL_TEXT_LINES:
        # one tiny mark is ignored, but a page, chart or label covered in small writing is still a picture full of text
        words.extend(small[:6])
    return words


def has_stray_text(words: list[str] | None) -> bool:
    """One longer word (3+ characters) or two or more words of any length are noticeable lettering."""
    return bool(words) and (len(words) >= 2 or len(words[0]) >= 3)
