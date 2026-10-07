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


MAX_ERASE_BOXES = 4             # a label or two can be painted out; a page of writing cannot
MAX_ERASE_AREA = 0.08           # share of the picture all erased boxes may cover
MAX_ERASE_HEIGHT = 0.12         # one box taller than this is a sign or a poster, not a small label


def erase_text(path: str | Path) -> list[str] | None:
    """Paint out a few small printed labels (a medicine box, a bottle label) so an otherwise good picture can be used.

    Returns the erased words when the picture was changed in place and a second OCR pass finds no lettering, else None (the picture
    is left untouched and stays rejected). The original is kept next to it as `<name>.with_text.png` for the teacher.
    """
    engine = _get_engine()
    if engine is None:
        return None
    try:
        import cv2
        image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        result, _ = engine(image)
    except Exception:
        return None
    boxes = [(np.array(box, dtype=np.int32), str(text)) for box, text, conf in result or []
             if float(conf) >= 0.5 and len([c for c in str(text) if c.isalnum()]) >= 2]
    if not boxes or len(boxes) > MAX_ERASE_BOXES:
        return None
    mask = np.zeros((height, width), np.uint8)
    for box, _text in boxes:
        x0, y0 = box.min(axis=0)
        x1, y1 = box.max(axis=0)
        if (y1 - y0) / height > MAX_ERASE_HEIGHT:
            return None
        pad = max(4, round((y1 - y0) * 0.35))
        cv2.rectangle(mask, (max(0, x0 - pad), max(0, y0 - pad)), (min(width - 1, x1 + pad), min(height - 1, y1 + pad)), 255, -1)
    if mask.mean() / 255 > MAX_ERASE_AREA:
        return None
    cleaned = cv2.inpaint(image, mask, 9, cv2.INPAINT_TELEA)
    cleaned = cv2.GaussianBlur(cleaned, (0, 0), 1.2) * (mask[..., None] / 255) + cleaned * (1 - mask[..., None] / 255)
    cleaned = cleaned.astype(np.uint8)
    path = Path(path)
    backup = path.with_name(path.stem + ".with_text.png")
    try:
        ok, encoded = cv2.imencode(".png", image)
        backup.write_bytes(encoded.tobytes()) if ok else None
        ok, encoded = cv2.imencode(path.suffix or ".png", cleaned)
        if not ok:
            return None
        tmp = path.with_name(path.stem + ".tmp" + path.suffix)
        tmp.write_bytes(encoded.tobytes())
        tmp.replace(path)
    except OSError:
        return None
    left = stray_text(path, min_conf=0.6, min_height=0.02)
    if left is None or left:                          # writing still readable (or cannot be verified): restore the original, keep it rejected
        backup.replace(path)
        return None
    return [text for _box, text in boxes]
