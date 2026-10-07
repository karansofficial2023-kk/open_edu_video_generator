"""Complex-script text shaping (Devanagari, Tamil, Telugu, Bengali, Urdu, ...) via Qt's HarfBuzz layout (open source, offscreen).

Pillow on Windows ships without Raqm, so Indic conjuncts and vowel signs would be drawn in the wrong order. Qt shapes
correctly and also provides script-aware measurement and line breaking. Qt is imported lazily: Latin-only lessons never load it.
"""
from __future__ import annotations

import os
import sys
from functools import lru_cache

from PIL import Image

_state: dict = {"app": None, "failed": False, "fonts": []}
FAMILIES = ["Nirmala UI", "Nirmala Text", "Segoe UI", "Noto Sans", "Arial"]


def available() -> bool:
    return _app() is not None


def _app():
    if _state["app"] is not None or _state["failed"]:
        return _state["app"]
    try:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.*=false;qt.text.*=false")
        from PySide6.QtGui import QGuiApplication, QFontDatabase
        app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])
        for path in ("C:/Windows/Fonts/Nirmala.ttc", "C:/Windows/Fonts/NirmalaB.ttc", "C:/Windows/Fonts/segoeui.ttf",
                     "/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf"):
            if os.path.exists(path):
                QFontDatabase.addApplicationFont(path)
        _state["app"] = app
    except Exception:         # PySide6 missing or no platform plugin: callers fall back to Pillow
        _state["failed"] = True
    return _state["app"]


def add_font_file(path: str) -> None:
    if path and _app() is not None and os.path.exists(path):
        from PySide6.QtGui import QFontDatabase
        QFontDatabase.addApplicationFont(path)


class ShapedFont:
    """Duck-types the parts of PIL's FreeTypeFont that typography.py relies on (size, getmetrics)."""

    def __init__(self, px: int, bold: bool = False):
        self.size = px
        self.bold = bold
        self.path = "qt-shaped"

    @property
    def _qfont(self):
        from PySide6.QtGui import QFont
        font = QFont()
        font.setFamilies(FAMILIES)
        font.setPixelSize(self.size)
        font.setBold(self.bold)
        return font

    def getmetrics(self) -> tuple[int, int]:
        from PySide6.QtGui import QFontMetricsF
        metrics = QFontMetricsF(self._qfont)
        return round(metrics.ascent()), round(metrics.descent())

    def length(self, text: str) -> float:
        from PySide6.QtGui import QFontMetricsF
        return float(QFontMetricsF(self._qfont).horizontalAdvance(text))

    def wrap(self, text: str, width: float) -> list[str]:
        """Script-aware wrapping using Qt's line breaker (never splits inside a conjunct or before a combining mark)."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QTextLayout, QTextOption
        layout = QTextLayout(text, self._qfont)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        layout.setTextOption(option)
        lines: list[str] = []
        layout.beginLayout()
        y = 0.0
        while True:
            line = layout.createLine()
            if not line.isValid():
                break
            line.setLineWidth(max(1.0, width))
            line.setPosition(QPointF(0, y))
            y += line.height()
            lines.append(text[line.textStart():line.textStart() + line.textLength()].strip())
        layout.endLayout()
        return [line for line in lines if line] or [text]

    def render(self, text: str, fill) -> Image.Image:
        """Tight RGBA image of one shaped line in the given colour."""
        from PySide6.QtCore import QPointF
        from PySide6.QtGui import QColor, QImage, QPainter, QTextLayout
        ascent, descent = self.getmetrics()
        width = max(2, int(self.length(text)) + 8)
        height = ascent + descent + 8
        image = QImage(width, height, QImage.Format.Format_ARGB32_Premultiplied)
        image.fill(0)
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r, g, b = fill[0], fill[1], fill[2]
        painter.setPen(QColor(r, g, b))
        layout = QTextLayout(text, self._qfont)
        layout.beginLayout()
        line = layout.createLine()
        line.setLineWidth(float(width + 1000))
        line.setPosition(QPointF(0, 0))
        layout.endLayout()
        layout.draw(painter, QPointF(2, 2))
        painter.end()
        pil = Image.frombuffer("RGBA", (width, height), bytes(image.constBits()), "raw", "BGRA", 0, 1)
        alpha = pil.getchannel("A")
        if len(fill) > 3 and fill[3] < 255:
            alpha = alpha.point(lambda v: round(v * fill[3] / 255))
        colored = Image.new("RGBA", pil.size, (r, g, b, 255))
        colored.putalpha(alpha)
        return colored


@lru_cache(maxsize=128)
def shaped_font(px: int, bold: bool = False) -> ShapedFont | None:
    return ShapedFont(px, bold) if available() else None
