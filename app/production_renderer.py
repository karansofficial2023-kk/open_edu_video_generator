"""Frame compositor for contract-driven storyboards: stills + Ken Burns, title cards, formulas, verified labels,
subtitle band, logo, crossfade. Everything that must be exact is drawn deterministically (never by the image model)."""
from __future__ import annotations

import math
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import labels as label_engine
from .animation_renderer import stage_times
from .charts import cards_frame, graph_frame, is_sequential, panels_frame, parse_function, parse_rows, plot_frame
from .captions import build_cues
from .config import AppConfig
from .formulas import render_line
from .renderer import _tool_path
from .schema import Segment, Shot
from .typography import (draw_lines, fit_text, font, hex_rgb, parse_style, set_font_override, subtitle_band, text_width,
                         wrap_text)


CARD_APPEAR_SECONDS = 0.4                  # a card lifts in over this long (charts.cards_frame)
FORMULA_FIRST, FORMULA_STEP, FORMULA_FADE = 0.1, 2.2, 0.25     # first equation at 0.1 s, one more every 2.2 s, each fading in over 0.25 s


@lru_cache(maxsize=4)
def panel_shadow_layer(size: tuple[int, int], panel: tuple[int, int, int, int], scale: float) -> Image.Image:
    """The blurred drop shadow of the equation card as a full-frame RGBA layer (it never moves, so it is built once per clip)."""
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((panel[0] + round(4 * scale), panel[1] + round(10 * scale), panel[2] + round(4 * scale),
                                              panel[3] + round(10 * scale)), radius=round(34 * scale), fill=(0, 0, 0, 46))
    return shadow.filter(ImageFilter.GaussianBlur(round(14 * scale)))


@dataclass(eq=False)
class ShotPlan:
    segment: Segment
    scene_title: str
    base: Image.Image | None = None          # 1920x1080 RGB (stills, title cards)
    anchors: list = field(default_factory=list)
    placed: list = field(default_factory=list)
    omitted: list = field(default_factory=list)
    label_starts: list = field(default_factory=list)
    video: object = None
    backdrop_path: str | None = None         # a photo of the same lesson shown softly behind concept cards (app.backdrops)
    backdrop: Image.Image | None = None
    steady: dict = field(default_factory=dict)       # frames that cannot change until the next reveal (cards, equations): drawn once, reused
    formula_step: float = 2.2                # seconds between equation lines; paced to the narration by render_clip (formula_pace)
    board: dict | None = None                # physics/chemistry board page: {theme, previous, current} (board.py)
    duration: float = 0.0                    # on-screen seconds of the shot being rendered (set by render_clip)


LONG_STATIC_SECONDS = 14.0     # a card or equation screen longer than this drifts very slowly so the board is never frozen
STATIC_DRIFT_SCALE = 1.035     # ... ending 3.5 % closer than it started
HOLD_DRIFT_SCALE = 1.06        # the last frame of a short motion clip keeps moving (slow push-in) for the rest of a longer shot


def slow_push(frame: Image.Image, progress: float, max_scale: float) -> Image.Image:
    """A centred, very slow push-in (progress 0..1 -> scale 1..max_scale); returns the frame unchanged at progress 0."""
    progress = min(1.0, max(0.0, progress))
    if progress <= 0:
        return frame
    width, height = frame.size
    scale = 1.0 + (max_scale - 1.0) * (0.5 - 0.5 * math.cos(math.pi * progress))       # eased: no jolt at the start or end
    w, h = width / scale, height / scale
    box = ((width - w) / 2, (height - h) / 2, (width + w) / 2, (height + h) / 2)
    return frame.resize((width, height), Image.Resampling.BICUBIC, box=box)


def formula_pace(lines: int, duration: float) -> float:
    """A teacher writes the next line of a derivation when the explanation reaches it, not all of them in the first seconds:
    the lines are spread over the first 75 % of the shot (at least 2.2 s apart, at most 8 s), so the last line still has time to be read."""
    if lines <= 1:
        return FORMULA_STEP
    return max(FORMULA_STEP, min(8.0, 0.75 * duration / lines))


def _safe_function(columns):
    try:
        return parse_function(columns)
    except Exception:
        return None


def size_of(config: AppConfig) -> tuple[int, int]:
    return config.output_resolution.width, config.output_resolution.height


def load_cover(path: str | Path, size: tuple[int, int]) -> Image.Image:
    with Image.open(path) as source:
        image = source.convert("RGB")
    scale = max(size[0] / image.width, size[1] / image.height)
    if abs(scale - 1) > 1e-3:
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
    left, top = (image.width - size[0]) // 2, (image.height - size[1]) // 2
    return image.crop((left, top, left + size[0], top + size[1]))


def camera_box(motion: str, t: float, duration: float, size: tuple[int, int], allow_move: bool) -> tuple[float, float, float, float]:
    """Sub-pixel crop window for a subtle Ken Burns move; identity when labels need fixed anchors."""
    width, height = size
    style = parse_style(motion)
    kind = style.get("camera", "static") if allow_move else "static"
    p = min(1.0, max(0.0, t / max(duration, 0.1)))
    p = p * p * (3 - 2 * p)
    zoom = {"slow_zoom_in": 1.0 + 0.07 * p, "slow_zoom_out": 1.07 - 0.07 * p}.get(kind, 1.0)
    w, h = width / zoom, height / zoom
    x, y = (width - w) / 2, (height - h) / 2
    if kind in {"pan_left", "pan_right", "slow_pan_left", "slow_pan_right"}:
        zoom = 1.06
        w, h = width / zoom, height / zoom
        span = width - w
        x = span * (1 - p) if "left" in kind else span * p
        y = (height - h) / 2
    return (x, y, x + w, y + h)


def gradient(size: tuple[int, int], strength: float = 0.62) -> Image.Image:
    width, height = size
    ramp = Image.linear_gradient("L").resize((width, height)).point(lambda v: round(v * strength))
    overlay = Image.new("RGBA", size, (5, 12, 18, 0))
    overlay.putalpha(ramp)
    return overlay


class ShotRenderer:
    def __init__(self, config: AppConfig, output_dir: Path):
        self.config = config
        self.size = size_of(config)
        self.style = config.production.subtitle
        self.output_dir = output_dir
        self.logo = self._load_logo()
        self._cue_cache: dict[tuple, Image.Image | None] = {}
        set_font_override(config.render.font_path)

    # ---------- preparation ----------
    def _load_logo(self) -> Image.Image | None:
        p = self.config.production
        if not p.logo_path or not Path(p.logo_path).is_file():
            return None
        logo = Image.open(p.logo_path).convert("RGBA")
        height = round(p.logo_height_px * self.size[1] / 1080)
        return logo.resize((round(logo.width * height / logo.height), height), Image.Resampling.LANCZOS)

    def keepout_zones(self) -> list[tuple[int, int, int, int]]:
        width, height = self.size
        zones = [(0, round(height * 0.80), width, height)]                 # subtitle band region
        if self.logo:
            zones.append((width - self.logo.width - round(60 * height / 1080), 0, width, self.logo.height + round(50 * height / 1080)))
        return zones

    def prepare(self, plan: ShotPlan) -> None:
        segment, shot = plan.segment, plan.segment.shot
        if shot.template == "formula" and not [line for line in segment.formula_lines if line.strip()]:
            from .assets import complete_cards           # last guard: an equation card with no equation would be a blank screen
            segment.shot = shot = Shot(template="process", heading=shot.heading or plan.scene_title, steps=complete_cards(segment.narration, shot.heading or plan.scene_title)[:4])
        if shot.template in {"photo", "title_card", "video", "split_screen"} and shot.asset_path:
            plan.base = load_cover(shot.asset_path, self.size)
        if shot.template == "process" and plan.backdrop_path and Path(plan.backdrop_path).is_file():
            from .backdrops import soft
            plan.backdrop = soft(plan.backdrop_path, self.size, hex_rgb(self.config.render.background))
        if shot.template in {"video", "photo", "title_card"} and shot.motion_asset and Path(shot.motion_asset).is_file() and not segment.labels:
            from .motion import VideoSource
            plan.video = VideoSource(shot.motion_asset, self.size)
        if shot.template == "photo" and segment.labels and plan.base is not None:
            cache_dir = self.output_dir / "assets"
            anchors, omitted = label_engine.locate(Path(shot.asset_path), segment, self.config, cache_dir)
            plan.anchors, plan.omitted = anchors, omitted
            style = parse_style(segment.label_style)
            plan.placed = label_engine.layout(anchors, self.size, self.keepout_zones(), self.config, style,
                                              label_engine.saliency_map(plan.base))
            for anchor in anchors[len(plan.placed):]:
                plan.omitted.append({"label": anchor.label, "reason": "no collision-free placement"})
            plan.label_starts = [0.8 + i * self.config.production.label_stagger_seconds for i in range(len(plan.placed))]

    def required_seconds(self, plan: ShotPlan) -> float:
        """Minimum on-screen time so every reveal is followed by the reading hold."""
        shot, p = plan.segment.shot, self.config.production
        if shot.template == "title_card":
            return 3.0
        if shot.template == "formula":
            return 1.0 + 2.2 * max(1, len(plan.segment.formula_lines)) + 1.5
        if shot.template == "graph":
            if _safe_function(plan.segment.columns):
                return 6.5
            return 1.0 + 0.6 * len(parse_rows(plan.segment.columns)) + 2.5
        if shot.template == "split_screen":
            return 1.0 + 0.5 * len(plan.segment.panel_assets) + 3.0
        if shot.template == "circuit":
            return 4.5 + 0.6 * len(plan.segment.columns)
        if plan.placed:
            return plan.label_starts[-1] + 0.9 + p.label_hold_seconds
        return 0.0

    # ---------- frames ----------
    def steady_key(self, plan: ShotPlan, t: float):
        """Identifies a card / equation screen that has finished animating (it cannot change until the next reveal), else None."""
        segment, shot = plan.segment, plan.segment.shot
        if plan.board is not None:
            if (plan.board.get("anchor") or {}).get("kind") in {"molecule", "diagram"}:
                return None                          # the molecule keeps turning: every frame is drawn
            return ("board",) if t >= plan.board["done"] else None
        if shot.template in {"process", "comparison"}:
            times = stage_times(segment)
            stage = max(i for i, start in enumerate(times) if t >= start)
            if t - times[stage] >= CARD_APPEAR_SECONDS + 0.05:        # the card has finished lifting in
                return ("cards", stage)
        elif shot.template == "formula":
            step = plan.formula_step
            shown = sum(1 for i in range(len(segment.formula_lines)) if t >= FORMULA_FIRST + i * step)
            if shown > 0 and t >= FORMULA_FIRST + (shown - 1) * step + FORMULA_FADE + 0.05:     # the newest line has finished fading in
                return ("formula", shown, self.note_index(plan, t, shown))
        return None

    def drifting(self, plan: ShotPlan) -> bool:
        """Long card / equation screens drift slowly (see LONG_STATIC_SECONDS); shorter ones are pixel-identical to before."""
        return plan.duration > LONG_STATIC_SECONDS and plan.segment.shot.template in {"process", "comparison", "formula"}

    def note_index(self, plan: ShotPlan, t: float, shown: int) -> int:
        """Which explanation note sits under the equation. Normally the note of the newest line; when there are more notes than lines
        (a long talk about one equation), the extra notes follow one another through the time after the last line appeared."""
        notes, lines = len(plan.segment.explain_steps), len(plan.segment.formula_lines)
        if not notes:
            return 0
        index = min(shown - 1, notes - 1)
        if notes > lines and shown >= lines and plan.duration > 0:
            start = FORMULA_FIRST + (lines - 1) * plan.formula_step
            span = max(0.1, plan.duration - start)
            extra = notes - lines + 1
            index = min(notes - 1, lines - 1 + int(extra * min(0.999, max(0.0, (t - start) / span))))
        return index

    def base_frame(self, plan: ShotPlan, t: float, duration: float) -> Image.Image:
        segment, shot = plan.segment, plan.segment.shot
        if plan.board is not None:
            from . import board as teacher_board
            key = self.steady_key(plan, t)
            if key is not None and key in plan.steady:
                return plan.steady[key].copy()
            image = teacher_board.frame(self.size, plan.board["theme"], plan.scene_title, plan.board["previous"], plan.board["current"],
                                        t, plan.board["reveal_end"], plan.board.get("anchor"))
            if key is not None:
                plan.steady[key] = image
            return image.copy()
        if plan.video is not None:
            frame = plan.video.frame_at(t).convert("RGB")
            clip_end = plan.video.duration
            if t > clip_end and duration - clip_end > 1.0:          # the clip is over: keep moving gently instead of freezing
                frame = slow_push(frame, (t - clip_end) / (duration - clip_end), HOLD_DRIFT_SCALE)
            return self._title_overlay(frame, plan, t) if shot.template == "title_card" else frame     # the title is drawn over the moving picture
        frame = self._board_frame(plan, t, duration)
        if frame is not None:
            return slow_push(frame, t / duration, STATIC_DRIFT_SCALE) if self.drifting(plan) and duration > 0 else frame
        return self._picture_frame(plan, t, duration)

    def _board_frame(self, plan: ShotPlan, t: float, duration: float) -> Image.Image | None:
        """Equation and card screens (cached once they stop changing); None for every other template."""
        segment, shot = plan.segment, plan.segment.shot
        if shot.template == "formula":
            key = self.steady_key(plan, t)
            cached = plan.steady.get(key) if key is not None else None
            if cached is not None:
                return cached.copy()
            frame = self._formula_frame(plan, t, duration)
            if key is not None:
                plan.steady[key] = frame
            return frame
        if shot.template in {"process", "comparison"}:
            times = stage_times(segment)
            stage = max(i for i, start in enumerate(times) if t >= start)
            age = t - times[stage]
            steady = self.steady_key(plan, t) is not None
            cached = plan.steady.get(("cards", stage)) if steady else None
            if cached is not None:
                return cached.copy()
            frame = cards_frame(self.size, self.config, shot.heading or plan.scene_title, shot.steps, stage, age,
                                shot.template == "process" and is_sequential(segment.narration), plan.backdrop)
            if steady:
                plan.steady[("cards", stage)] = frame
            return frame
        return None

    def _picture_frame(self, plan: ShotPlan, t: float, duration: float) -> Image.Image:
        segment, shot = plan.segment, plan.segment.shot
        if shot.template == "graph":
            function = _safe_function(segment.columns)
            if function:
                equation, x_range, y_range, func = function
                return plot_frame(self.size, self.config, shot.heading or plan.scene_title, equation, x_range, y_range, func, t)
            note = segment.explain_steps[0] if segment.explain_steps else ""
            return graph_frame(self.size, self.config, shot.heading or plan.scene_title, parse_rows(segment.columns), note, t)
        if shot.template == "circuit":
            if not hasattr(plan, "_circuits"):
                from .circuit import draw as draw_circuit
                drawn = [draw_circuit(row) for row in segment.columns[:3]]
                plan._circuits = ([image for _caption, image in drawn], [caption for caption, _image in drawn])
            panels, captions = plan._circuits
            return panels_frame(self.size, self.config, shot.heading, panels, captions, t, contain=True)
        if shot.template == "split_screen":
            panels = [load_cover(path, (self.size[0] // 2, self.size[1] // 2)) for path in segment.panel_assets]                 if not hasattr(plan, "_panels") else plan._panels
            plan._panels = panels
            return panels_frame(self.size, self.config, shot.heading, panels, shot.steps, t)
        if plan.base is None and shot.template == "title_card":
            plan.base = self._gradient_background()
        if plan.base is None:
            raise ValueError(f"shot {segment.shot_id}: no image asset available")
        allow_move = not plan.placed
        x0, y0, x1, y1 = camera_box(segment.motion, t, duration, self.size, allow_move)
        frame = plan.base if (x0, y0, x1, y1) == (0, 0, *self.size) else \
            plan.base.resize(self.size, Image.Resampling.BICUBIC, box=(x0, y0, x1, y1))
        frame = frame.copy()
        if shot.template == "title_card":
            frame = self._title_overlay(frame, plan, t)
        return frame

    def _gradient_background(self) -> Image.Image:
        """Deterministic title background (theme colours, soft light) used when no verified photograph exists."""
        width, height = self.size
        top, bottom = hex_rgb(self.config.render.accent, (46, 125, 99)), (14, 28, 36)
        base = Image.new("RGB", self.size, bottom)
        ramp = Image.linear_gradient("L").resize(self.size)
        base.paste(Image.new("RGB", self.size, top), (0, 0), ramp.point(lambda v: 255 - v))
        glow = Image.new("RGBA", self.size, (0, 0, 0, 0))
        ImageDraw.Draw(glow).ellipse((width * 0.55, -height * 0.25, width * 1.25, height * 0.75), fill=(255, 255, 255, 46))
        glow = glow.filter(ImageFilter.GaussianBlur(height // 6))
        return Image.alpha_composite(base.convert("RGBA"), glow).convert("RGB")

    def _title_overlay(self, frame: Image.Image, plan: ShotPlan, t: float) -> Image.Image:
        width, height = self.size
        scale = height / 1080
        title = (plan.segment.shot.heading or plan.scene_title).strip()
        canvas = Image.alpha_composite(frame.convert("RGBA"), gradient(self.size, 0.55))
        overlay = Image.new("RGBA", self.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        box = (round(width * 0.08), round(height * 0.30), round(width * 0.92), round(height * 0.62))
        fnt, lines, line_h = fit_text(draw, title, box, round(118 * scale), round(48 * scale), bold=True, max_lines=3)
        block_h = len(lines) * line_h
        rise = round(28 * scale * (1 - min(1.0, t / 0.8)))
        alpha = round(255 * min(1.0, t / 0.8))
        top = box[1] + (box[3] - box[1] - block_h) // 2 + rise
        draw_lines(draw, lines, fnt, line_h, box[0] + 3, top + 3, (0, 0, 0, round(alpha * 0.55)), "center", box[2] - box[0])
        draw_lines(draw, lines, fnt, line_h, box[0], top, (255, 255, 255, alpha), "center", box[2] - box[0])
        bar_w = round(min(width * 0.18, max(text_width(draw, l, fnt) for l in lines) * 0.5))
        accent = hex_rgb(self.config.render.second_accent, (209, 139, 50))
        cx = width // 2
        draw.rounded_rectangle((cx - bar_w // 2, top + block_h + round(14 * scale), cx + bar_w // 2, top + block_h + round(20 * scale)),
                               radius=round(3 * scale), fill=accent + (alpha,))
        return Image.alpha_composite(canvas, overlay).convert("RGB")

    def _formula_frame(self, plan: ShotPlan, t: float, duration: float) -> Image.Image:
        width, height = self.size
        scale = height / 1080
        segment = plan.segment
        bg, ink, accent = self.config.render.background, self.config.render.ink, self.config.render.accent
        bg_rgb, ink_rgb, accent_rgb = hex_rgb(bg), hex_rgb(ink), hex_rgb(accent)
        px = lambda v: round(v * scale)
        frame = Image.new("RGB", self.size, bg_rgb)
        draw = ImageDraw.Draw(frame)
        # notebook page: a very faint grid, so equations sit on "paper" and not on a void
        grid = tuple(round(b + (i - b) * 0.05) for b, i in zip(bg_rgb, ink_rgb))
        for gx in range(0, width, px(54)):
            draw.line((gx, 0, gx, height), fill=grid, width=1)
        for gy in range(0, height, px(54)):
            draw.line((0, gy, width, gy), fill=grid, width=1)
        heading = segment.shot.heading or plan.scene_title
        fnt, lines, line_h = fit_text(draw, heading, (px(96), px(48), width - px(96), px(170)), px(58), px(30), bold=True, max_lines=2)
        draw_lines(draw, lines, fnt, line_h, px(96), px(48), ink_rgb)
        draw.rounded_rectangle((px(96), px(178), px(96) + px(220), px(184)), radius=px(3), fill=accent_rgb)

        formulas = segment.formula_lines
        step_time, first, fade_seconds = plan.formula_step, FORMULA_FIRST, FORMULA_FADE   # the first equation is on screen almost at once
        shown = [i for i in range(len(formulas)) if t >= first + i * step_time]
        panel = (px(110), px(214), width - px(110), px(716))
        frame = Image.alpha_composite(frame.convert("RGBA"), panel_shadow_layer(self.size, panel, scale)).convert("RGB")
        draw = ImageDraw.Draw(frame)
        draw.rounded_rectangle(panel, radius=px(34), fill=(255, 255, 255), outline=tuple(round(b + (i - b) * 0.10) for b, i in zip(bg_rgb, ink_rgb)), width=max(1, px(2)))
        draw.rounded_rectangle((panel[0], panel[1] + px(26), panel[0] + px(12), panel[3] - px(26)), radius=px(6), fill=accent_rgb)

        inner_top, inner_bottom = panel[1] + px(22), panel[3] - px(22)
        # a long derivation keeps its newest 3 lines on the board (older ones move up and off, their step numbers stay), so every
        # line is large enough to read instead of five tiny rows; short derivations get large rows
        window = 3 if len(formulas) > 4 else max(1, len(formulas))
        row_h = min(px(230), (inner_bottom - inner_top) // window)
        top = inner_top + max(0, ((inner_bottom - inner_top) - row_h * window) // 2)        # centre the block vertically
        number_font = font(px(30), True)
        for row, i in enumerate(shown[-window:]):
            current = i == shown[-1]
            y0 = top + row * row_h
            if current and len(formulas) > 1:            # the line being explained now sits on a soft tinted band
                tint = Image.new("RGBA", self.size, (0, 0, 0, 0))
                ImageDraw.Draw(tint).rounded_rectangle((panel[0] + px(34), y0 + px(6), panel[2] - px(34), y0 + row_h - px(6)), radius=px(22),
                                                      fill=(*accent_rgb, 30))
                frame = Image.alpha_composite(frame.convert("RGBA"), tint).convert("RGB")
                draw = ImageDraw.Draw(frame)
            if len(formulas) > 1:                        # numbered steps of a derivation
                cx, cy, r = panel[0] + px(84), y0 + row_h // 2, px(24)
                draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=accent_rgb if current else (214, 222, 216))
                label = str(i + 1)
                box = draw.textbbox((0, 0), label, font=number_font)
                draw.text((cx - (box[2] - box[0]) / 2 - box[0], cy - (box[3] - box[1]) / 2 - box[1]), label, font=number_font,
                          fill=(255, 255, 255) if current else (90, 104, 96))
            image = render_line(formulas[i], accent if current else ink, round(96 * scale))
            left = panel[0] + (px(150) if len(formulas) > 1 else px(60))
            max_w, max_h = panel[2] - px(60) - left, row_h - px(46)           # margin inside the highlight band
            ratio = min(max_w / image.width, max_h / image.height, 2.6)
            image = image.resize((max(1, round(image.width * ratio)), max(1, round(image.height * ratio))), Image.Resampling.LANCZOS)
            fade = min(1.0, (t - (first + i * step_time)) / fade_seconds)
            layer = image.copy()
            layer.putalpha(image.getchannel("A").point(lambda v: round(v * fade)))
            frame.paste(layer, (left + (max_w - image.width) // 2, y0 + (row_h - image.height) // 2), layer)
        if segment.explain_steps and shown:
            idx = self.note_index(plan, t, len(shown))
            note = segment.explain_steps[idx]
            draw = ImageDraw.Draw(frame)
            nfnt, nlines, nh = fit_text(draw, note, (px(190), px(752), width - px(190), px(836)), px(34), px(24))
            pill = (px(150), px(738), width - px(150), px(738) + max(px(66), len(nlines) * nh + px(30)))
            draw.rounded_rectangle(pill, radius=px(30), fill=tuple(round(255 + (a - 255) * 0.10) for a in accent_rgb))
            draw_lines(draw, nlines, nfnt, nh, px(190), pill[1] + px(15), hex_rgb("#33483D"), "center", width - px(380))
        return frame

    def band_opacity(self, base: Image.Image | None) -> float:
        """Keep subtitle contrast accessible on bright pictures: denser band when the area behind it is light."""
        floor = self.style.band_opacity
        if base is None:
            return floor
        w, h = base.size
        region = base.crop((0, round(h * 0.78), w, h)).convert("L").resize((48, 8))
        luminance = sum(region.getdata()) / (48 * 8)
        return max(floor, 0.82 if luminance > 170 else 0.7 if luminance > 120 else floor)

    def overlay_layer(self, plan: ShotPlan, t: float, cues, base: Image.Image | None = None) -> Image.Image:
        layer = Image.new("RGBA", self.size, (0, 0, 0, 0))
        if plan.placed:
            label_engine.draw(layer, plan.placed, t, plan.label_starts, self.config, parse_style(plan.segment.label_style))
        cue = next((c for c in cues if c.start <= t < c.end), cues[-1] if cues and t >= cues[-1].start else None)
        if cue is not None:
            opacity = self.band_opacity(base)
            key = (cue.text, round(opacity, 2))
            if key not in self._cue_cache:
                built = subtitle_band(self.size, cue.text, self.style, opacity)
                self._cue_cache[key] = built[0] if built else None
            band = self._cue_cache[key]
            if band is not None:
                layer = Image.alpha_composite(layer, band)
        if self.logo is not None:
            margin = round(30 * self.size[1] / 1080)
            layer.alpha_composite(self.logo, (self.size[0] - self.logo.width - margin, margin))
        return layer

    # ---------- clip ----------
    def render_clip(self, plan: ShotPlan, clip: Path, frame_count: int, previous_last: Image.Image | None,
                    previous_template: str | None = None) -> Image.Image:
        fps = self.config.fps
        duration = frame_count / fps
        plan.formula_step = formula_pace(len(plan.segment.formula_lines), duration)
        plan.duration = duration
        cues = build_cues(plan.segment, self.style, self.size)
        # Text is drawn into the picture of these screens, so blending two of them ghosts one heading over the other: cut instead.
        text_screens = {"process", "formula", "graph", "circuit", "title_card"}
        smooth = plan.segment.shot.template not in text_screens and previous_template not in text_screens
        fade_frames = round(self.config.production.crossfade_seconds * fps) if previous_last is not None and smooth else 0
        command = [_tool_path(self.config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
                   "-s", f"{self.size[0]}x{self.size[1]}", "-r", str(fps), "-i", "pipe:0", "-an", "-frames:v", str(frame_count),
                   "-c:v", "libx264", "-preset", self.config.render.video_preset, "-crf", str(self.config.render.video_crf),
                   "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(clip)]
        last = None
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, stdout=subprocess.DEVNULL)
            try:
                repeat_key, repeat_bytes = None, None
                for i in range(frame_count):
                    t = i / fps
                    # A finished card / equation screen with an unchanged subtitle cue (and no labels) is the same picture as the last
                    # frame: write the same bytes again instead of recompositing 1920x1080 layers.
                    key = None
                    if i >= fade_frames and not plan.placed and not self.drifting(plan):
                        steady = self.steady_key(plan, t)
                        if steady is not None:
                            cue = next((c for c in cues if c.start <= t < c.end), cues[-1] if cues and t >= cues[-1].start else None)
                            key = (steady, cue.start if cue is not None else None)
                    if key is not None and key == repeat_key:
                        proc.stdin.write(repeat_bytes)
                        continue
                    base = self.base_frame(plan, t, duration).convert("RGB")
                    last = base
                    if i < fade_frames:   # crossfade the pictures only; subtitles/labels never ghost across shots
                        base = Image.blend(previous_last, base, (i + 1) / (fade_frames + 1))
                    frame = Image.alpha_composite(base.convert("RGBA"), self.overlay_layer(plan, t, cues, base)).convert("RGB")
                    data = frame.tobytes()
                    proc.stdin.write(data)
                    repeat_key, repeat_bytes = key, (data if key is not None else None)
                proc.stdin.close()
                if proc.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode("utf-8", "replace"))
            except BaseException:
                proc.kill()
                proc.wait()
                clip.unlink(missing_ok=True)
                raise
        last.save(clip.with_suffix(".last.png"))
        return last
