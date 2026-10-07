"""Deterministic educational templates. No generated Python is executed."""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from .schema import Segment
from .subtitles import phrase_captions
from .visuals import _font, _pixel_wrap


def stage_times(segment: Segment) -> list[float]:
    shot = segment.shot
    count = len(shot.steps) if shot.template in {"process", "comparison"} else 2
    duration = segment.speech_duration or segment.end - segment.start
    if shot.cues:
        normalize = lambda s: re.findall(r"\w+", s.lower())
        tokens, starts = [], []
        for word in segment.captions:
            parts = normalize(word.text)
            tokens.extend(parts)
            starts.extend([word.start] * len(parts))
        # Preview or older audio: use explicitly approximate word timing.
        if not tokens:
            tokens = normalize(segment.narration)
            starts = [i * duration / max(1, len(tokens)) for i in range(len(tokens))]
        result, cursor = [], 0
        for cue in shot.cues:
            target = normalize(cue)
            match = next((i for i in range(cursor, len(tokens) - len(target) + 1)
                          if target and tokens[i:i + len(target)] == target), None)
            if match is None:
                raise ValueError(f"Animation cue not found in narration: {cue!r}")
            result.append(starts[match])
            cursor = match + len(target)
        result[0] = 0.0
        return result
    fractions = shot.stage_fractions or [i / count for i in range(count)]
    return [x * duration for x in fractions]


class Canvas:
    def __init__(self, config):
        self.width = config.output_resolution.width
        self.height = config.output_resolution.height
        self.scale = self.width / 1280
        self.image = Image.new("RGB", (self.width, self.height), config.render.background)
        self.draw = ImageDraw.Draw(self.image)
        self.config = config

    def box(self, bounds, fill, outline=None, radius=16):
        self.draw.rounded_rectangle([round(x * self.scale) for x in bounds],
                                    radius=round(radius * self.scale), fill=fill,
                                    outline=outline, width=max(1, round(2 * self.scale)))

    def line(self, points, fill, width=3):
        self.draw.line([(round(x * self.scale), round(y * self.scale)) for x, y in points],
                       fill=fill, width=max(1, round(width * self.scale)))

    def ellipse(self, bounds, fill, outline=None):
        self.draw.ellipse([round(x * self.scale) for x in bounds], fill=fill,
                          outline=outline, width=max(1, round(2 * self.scale)))

    def text(self, text, bounds, size=26, color=None, bold=False):
        left, top, right, bottom = [round(x * self.scale) for x in bounds]
        size = round(size * self.scale)
        for candidate in range(size, max(8, round(14 * self.scale)) - 1, -1):
            font = _font(candidate, bold)
            if self.config.render.font_path:
                from PIL import ImageFont
                font = ImageFont.truetype(self.config.render.font_path, candidate)
            lines = _pixel_wrap(self.draw, text, font, right - left)
            line_height = sum(font.getmetrics()) + round(5 * self.scale)
            if len(lines) * line_height <= bottom - top:
                for i, line in enumerate(lines):
                    self.draw.text((left, top + i * line_height), line, font=font,
                                   fill=color or self.config.render.ink)
                return
        raise ValueError(f"Text does not fit template; shorten it: {text[:80]}")


def render_frame(segment, scene_title, t, config, source=None):
    c = Canvas(config)
    shot = segment.shot
    c.text(shot.heading or scene_title, (48, 25, 1230, 96), 36, bold=True)
    c.line([(48, 104), (1232, 104)], "#D8DFD9", 2)
    times = stage_times(segment)
    stage = max(i for i, start in enumerate(times) if t >= start)
    duration = segment.end - segment.start
    accent = config.render.accent
    gold = config.render.second_accent
    if shot.template in {"photo", "video"}:
        if source is None:
            raise ValueError("Photo/video requires a source image")
        art = ImageOps.contain(source.convert("RGB"),
                               (round(1184 * c.scale), round(466 * c.scale)), Image.Resampling.LANCZOS)
        c.image.paste(art, ((c.width - art.width) // 2, round(120 * c.scale)))
    else:
        count = len(shot.steps)
        gap = 24
        cell = (1184 - gap * (count - 1)) / count
        for i, step in enumerate(shot.steps):
            left = 48 + i * (cell + gap)
            active = i == stage
            c.box((left, 220, left + cell, 490), "#FFFFFF", accent if active else "#D8DFD9")
            c.box((left + 18, 240, left + 62, 286), accent if active else "#788A80", radius=10)
            c.text(str(i + 1), (left + 30, 243, left + 60, 282), 23, "white", True)
            c.text(step, (left + 22, 314, left + cell - 22, 470), 27, bold=active)
            if shot.template == "process" and i < count - 1:
                x = left + cell + 12
                c.line([(x - 7, 354), (x + 7, 360), (x - 7, 366)], accent)
        c.text("Compare each idea" if shot.template == "comparison" else "Follow the sequence",
               (48, 144, 1200, 192), 24, accent)
    if config.render.burn_captions:
        cue = next((x for x in phrase_captions(segment) if x.start <= t < x.end), None)
        if cue:
            c.box((40, 630, 1240, 708), config.render.subtitle_bg, radius=12)
            c.text(cue.text, (62, 639, 1218, 705), 27, config.render.subtitle_fg)
    return c.image


def render_animation_clip(segment, title, clip, frame_count, config, source=None):
    from .renderer import _tool_path
    width, height = config.output_resolution.width, config.output_resolution.height
    if width * 9 != height * 16:
        raise ValueError("Educational templates currently require a 16:9 resolution")
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(config.fps), "-i", "pipe:0", "-an", "-frames:v", str(frame_count),
               "-c:v", "libx264", "-preset", config.render.video_preset,
               "-crf", str(config.render.video_crf), "-pix_fmt", "yuv420p", str(clip)]
    with tempfile.TemporaryFile() as errors:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, stdout=subprocess.DEVNULL)
        try:
            for i in range(frame_count):
                frame = render_frame(segment, title, i / config.fps, config, source)
                proc.stdin.write(frame.tobytes())
            proc.stdin.close()
            if proc.wait() != 0:
                errors.seek(0)
                raise RuntimeError(errors.read().decode("utf-8", "replace"))
        except BaseException:
            proc.kill()
            proc.wait()
            clip.unlink(missing_ok=True)
            raise
