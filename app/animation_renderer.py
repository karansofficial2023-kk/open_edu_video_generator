"""Deterministic educational templates. No generated Python is executed."""
from __future__ import annotations

import math
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
    count = len(shot.steps) if shot.template in {"process", "comparison", "classification", "agents", "pros_cons"} else 2
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


def flower(c: Canvas, x: float, y: float, highlight: str = ""):
    """Schematic longitudinal view, deliberately not species-specific anatomy."""
    green = c.config.render.accent
    gold = c.config.render.second_accent
    c.ellipse((x - 135, y - 100, x - 15, y + 125), "#EBD8E5")
    c.ellipse((x + 15, y - 100, x + 135, y + 125), "#EBD8E5")
    c.line([(x, y + 190), (x, y + 110)], green, 10)
    c.ellipse((x - 32, y + 60, x + 32, y + 130), "#BDD5C3", green)
    c.line([(x, y + 75), (x, y - 90)], green, 10)
    c.ellipse((x - 30, y - 105, x + 30, y - 78), green)
    for dx in [-80, 80]:
        c.line([(x + dx / 2, y + 100), (x + dx, y - 45)], green, 5)
        c.ellipse((x + dx - 22, y - 65, x + dx + 22, y - 40), gold)
    if highlight == "anther":
        c.ellipse((x - 111, y - 75, x - 49, y - 30), None, gold)
        c.line([(x - 104, y - 63), (x - 160, y - 120)], gold)
        c.text("Anther", (x - 215, y - 158, x - 100, y - 120), 21, gold, True)
    if highlight == "stigma":
        c.ellipse((x - 40, y - 115, x + 40, y - 68), None, green)
        c.line([(x + 35, y - 95), (x + 110, y - 130)], green)
        c.text("Stigma", (x + 90, y - 172, x + 210, y - 130), 21, green, True)


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
    elif shot.template in {"process", "comparison", "classification", "agents", "pros_cons"}:
        count = len(shot.steps)
        gap = 24
        cell = (1184 - gap * (count - 1)) / count
        if shot.template == "classification":
            _classification(c, shot, stage, accent, gold)
        elif shot.template == "agents":
            _agents(c, shot, stage, accent, gold)
        elif shot.template == "pros_cons":
            _pros_cons(c, shot, stage, accent, gold)
        else:
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
    elif shot.template == "life_cycle":
        _life_cycle(c, stage, accent, gold)
    elif shot.template == "wind":
        _wind(c, stage, accent, gold)
    elif shot.template == "water":
        _water(c, stage, accent, gold)
    elif shot.template == "insect":
        _insect(c, stage, accent, gold)
    elif shot.template == "pollination":
        flower(c, 300, 335, "anther")
        flower(c, 950, 335, "stigma")
        c.text("Pollen source", (220, 555, 470, 590), 23, accent, True)
        c.text("Receiving flower", (820, 555, 1120, 590), 23, accent, True)
        travel_start = times[1]
        progress = max(0, min(1, (t - travel_start) / max(.1, duration - travel_start - .5)))
        for i in range(8):
            p = max(0, min(1, progress * 1.25 - i * .035))
            x = 220 + (950 - 220) * p
            y = 283 + (243 - 283) * p - 125 * math.sin(math.pi * p)
            c.ellipse((x - 5, y - 5 + i % 3 * 6, x + 5, y + 5 + i % 3 * 6), gold)
    else:
        first = "anther" if shot.template == "protandry" else "stigma"
        second = "stigma" if first == "anther" else "anther"
        flower(c, 355, 340, first if stage == 0 else second)
        descriptions = {"anther": "Anthers release pollen", "stigma": "Stigma becomes receptive"}
        for i, part in enumerate([first, second]):
            y = 220 + i * 150
            c.box((685, y, 1210, y + 120), "#FFFFFF", accent if stage == i else "#D8DFD9")
            c.text(f"{i + 1}. {descriptions[part]}", (710, y + 25, 1180, y + 100), 28,
                   accent if stage == i else "#788A80", stage == i)
        c.text("Earlier", (695, 168, 920, 208), 22, accent)
        c.text("Later", (695, 500, 920, 540), 22, accent)
    if shot.template in {"pollination", "protandry", "protogyny"}:
        c.text("Simplified schematic", (48, 590, 600, 620), 17, "#65766A")
    if config.render.burn_captions:
        cue = next((x for x in phrase_captions(segment) if x.start <= t < x.end), None)
        if cue:
            c.box((40, 630, 1240, 708), config.render.subtitle_bg, radius=12)
            c.text(cue.text, (62, 639, 1218, 705), 27, config.render.subtitle_fg)
    return c.image


def _classification(c, shot, stage, accent, gold):
    c.text("Pollination", (520, 140, 780, 185), 27, accent, True)
    c.line([(640, 190), (380, 255)], accent, 3)
    c.line([(640, 190), (900, 255)], accent, 3)
    labels = [
        ("Self-pollination", "Same flower or same plant", 155, stage == 0),
        ("Cross-pollination", "Different plant of same species", 675, stage == 1),
    ]
    for title, body, left, active in labels:
        c.box((left, 255, left + 450, 485), "#FFFFFF", accent if active else "#CFD8D2")
        c.text(title, (left + 28, 290, left + 420, 340), 29, accent if active else "#4F6258", True)
        c.text(body, (left + 28, 365, left + 420, 435), 24)
    flower(c, 280, 530, "anther")
    flower(c, 1000, 530, "stigma")


def _agents(c, shot, stage, accent, gold):
    icons = [("Wind", "~", "#7CA7C7"), ("Water", "W", "#3D91C2"), ("Insects", "*", gold), ("Birds/Bats/Animals", "+", "#B45C45")]
    for i, (title, symbol, color) in enumerate(icons):
        left = 80 + (i % 2) * 585
        top = 185 + (i // 2) * 205
        active = i == stage
        c.box((left, top, left + 500, top + 155), "#FFFFFF", color if active else "#D1D9D4")
        c.ellipse((left + 30, top + 35, left + 105, top + 110), color)
        c.text(symbol, (left + 55, top + 50, left + 95, top + 96), 32, "white", True)
        c.text(title, (left + 140, top + 38, left + 460, top + 105), 27, color if active else "#4F6258", True)
    c.text("Different agents match different flower adaptations", (96, 565, 1120, 610), 24, accent)


def _pros_cons(c, shot, stage, accent, gold):
    columns = [("Advantages", ["Assured reproduction", "Useful traits preserved", "Genetic diversity in cross-pollination"], accent),
               ("Disadvantages", ["Less variation in selfing", "Pollen wastage", "Agent may be absent"], "#B45C45")]
    for i, (title, bullets, color) in enumerate(columns):
        left = 80 + i * 585
        active = i == stage
        c.box((left, 170, left + 520, 540), "#FFFFFF", color if active else "#D1D9D4")
        c.text(title, (left + 35, 210, left + 485, 265), 31, color, True)
        for j, bullet in enumerate(bullets):
            y = 305 + j * 70
            c.ellipse((left + 40, y + 10, left + 58, y + 28), color)
            c.text(bullet, (left + 75, y, left + 480, y + 48), 23)


def _life_cycle(c, stage, accent, gold):
    nodes = [("Flower", 210, 260), ("Pollination", 500, 185), ("Fertilization", 795, 260), ("Seeds & fruits", 640, 455)]
    for i, (label, x, y) in enumerate(nodes):
        color = gold if i <= stage + 1 else "#AAB8AF"
        c.ellipse((x - 58, y - 58, x + 58, y + 58), "#FFFFFF", color)
        c.text(label, (x - 115, y + 75, x + 120, y + 125), 23, accent, True)
    for a, b in zip(nodes, nodes[1:] + nodes[:1]):
        c.line([(a[1] + 65, a[2]), (b[1] - 65, b[2])], accent, 4)
    c.text("Pollination starts the path toward seeds and fruits", (120, 565, 1160, 610), 25, accent)


def _wind(c, stage, accent, gold):
    flower(c, 235, 365, "")
    c.text("Small, dull flowers", (95, 555, 420, 592), 22, accent, True)
    c.line([(420, 310), (1020, 240)], "#7CA7C7", 5)
    c.line([(420, 365), (1040, 350)], "#7CA7C7", 5)
    for i in range(14):
        x = 430 + i * 45 + stage * 10
        y = 260 + (i % 4) * 32
        c.ellipse((x, y, x + 13, y + 13), gold)
    c.line([(1040, 180), (1040, 500)], accent, 10)
    for y in [205, 245, 285, 325, 365, 405]:
        c.line([(1040, y), (1115, y - 22)], accent, 3)
        c.line([(1040, y), (1115, y + 22)], accent, 3)
    c.text("Feathery stigma catches airborne pollen", (740, 555, 1210, 600), 23, accent, True)


def _water(c, stage, accent, gold):
    c.box((60, 285, 1220, 545), "#D9F0F6", "#8BB9C9", 8)
    for y in [335, 405, 475]:
        c.line([(90, y), (1180, y + 20 * math.sin(y))], "#65A9BF", 4)
    c.text("Aquatic plant", (145, 210, 420, 255), 25, accent, True)
    c.text("Floating pollen", (505, 210, 780, 255), 25, gold, True)
    c.text("Receptive flower", (850, 210, 1160, 255), 25, accent, True)
    flower(c, 230, 420, "anther")
    flower(c, 1010, 420, "stigma")
    for i in range(10):
        x = 400 + i * 45 + stage * 15
        c.ellipse((x, 390 + (i % 3) * 20, x + 24, 402 + (i % 3) * 20), gold)


def _insect(c, stage, accent, gold):
    flower(c, 300, 365, "anther")
    flower(c, 960, 365, "stigma")
    bx = 500 if stage == 0 else 760
    by = 280 if stage == 0 else 250
    c.ellipse((bx - 55, by - 30, bx + 55, by + 30), gold, "#A56515")
    c.ellipse((bx - 15, by - 58, bx + 65, by + 5), "#F2F7FF", "#8EAAC0")
    c.ellipse((bx - 65, by - 58, bx + 15, by + 5), "#F2F7FF", "#8EAAC0")
    for i in range(6):
        c.ellipse((bx - 75 + i * 18, by + 35 + i % 2 * 10, bx - 65 + i * 18, by + 45 + i % 2 * 10), gold)
    c.text("Bright petals, nectar and scent attract insects", (180, 570, 1120, 615), 24, accent)


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
