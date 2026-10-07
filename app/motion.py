"""Short generated motion clips (LTX through ComfyUI) with QA, plus frame-accurate playback inside the compositor.

Policy (production spec sections 9 and 15): motion is only for natural action that a still cannot show; it never carries labels
or formulas; every clip is checked (frame count, flicker, semantic frame review) and a rejected clip falls back to the approved
still with Ken Burns. Generation runs after stills so only one heavy model is resident at a time.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from . import progress, vision
from .assets import semantic_qa, _accepts
from .comfyui_client import ComfyUIClient
from .config import AppConfig
from .resilience import atomic_write_text, read_json_or_none, retry
from .schema import Segment, Storyboard


class VideoSource:
    """Random-access frames of a short clip, blended between source frames and cover-cropped to the output size."""

    def __init__(self, path: str | Path, size: tuple[int, int]):
        self.path = str(path)
        capture = cv2.VideoCapture(self.path)
        self.fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
        count = 0
        while capture.grab():            # counted by decoding (webm headers do not carry a frame count); frames are not kept
            count += 1
        capture.release()
        if not count:
            raise ValueError(f"cannot decode motion clip {path}")
        self.count = count
        self.size = size
        self.duration = count / self.fps
        # 1080p frames are 6 MB each: keep only the two being blended, read forward as the renderer asks for later times
        self._capture: cv2.VideoCapture | None = None
        self._next = 0
        self._cache: dict[int, np.ndarray] = {}

    def close(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        self._cache, self._next = {}, 0

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _frame(self, index: int) -> np.ndarray:
        if index in self._cache:
            return self._cache[index]
        if self._capture is None or index < self._next - 2:
            if self._capture is not None:
                self._capture.release()
            self._capture, self._next, self._cache = cv2.VideoCapture(self.path), 0, {}
        last = None
        while self._next <= index:
            ok, frame = self._capture.read()
            if not ok:
                break
            last = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._cache[self._next] = last
            self._next += 1
        if last is None and not self._cache:
            raise ValueError(f"cannot decode motion clip {self.path}")
        for stale in [k for k in self._cache if k < index - 1]:
            del self._cache[stale]
        return self._cache.get(index, last if last is not None else next(iter(self._cache.values())))

    def frame_at(self, t: float) -> Image.Image:
        position = min(max(t * self.fps, 0.0), self.count - 1.0)
        low = int(position)
        high = min(low + 1, self.count - 1)
        mix = position - low
        frame = self._frame(low) if mix < 0.02 or low == high else cv2.addWeighted(self._frame(low), 1 - mix, self._frame(high), mix, 0)
        image = Image.fromarray(frame)
        scale = max(self.size[0] / image.width, self.size[1] / image.height)
        image = image.resize((round(image.width * scale), round(image.height * scale)), Image.Resampling.LANCZOS)
        left, top = (image.width - self.size[0]) // 2, (image.height - self.size[1]) // 2
        return image.crop((left, top, left + self.size[0], top + self.size[1])).filter(ImageFilter.UnsharpMask(1.2, 40, 3))


def clip_metrics(path: Path) -> dict:
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS) or 16.0
    previous, diffs, middle = None, [], None
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        small = cv2.cvtColor(cv2.resize(frame, (96, 54)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        if previous is not None:
            diffs.append(float(np.abs(small - previous).mean()))
        previous = small
        if index == count // 2:
            middle = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        index += 1
    capture.release()
    return {"frames": index, "fps": fps, "mean_change": float(np.mean(diffs)) if diffs else 0.0,
            "max_change": float(np.max(diffs)) if diffs else 0.0, "middle": middle}


def motion_ok(metrics: dict) -> list[str]:
    reasons = []
    if metrics["frames"] < 24:
        reasons.append("too_short")
    if metrics["mean_change"] < 0.15:
        reasons.append("static")
    if metrics["max_change"] > 30 or metrics["mean_change"] > 18:
        reasons.append("flicker_or_cut")
    return reasons


def _luma(image: np.ndarray) -> np.ndarray:
    """16:9 centre crop, 64x36, lightly blurred luminance: what the picture 'looks like' regardless of encoding."""
    height, width = image.shape[:2]
    if width / height > 16 / 9:
        crop = int(height * 16 / 9)
        image = image[:, (width - crop) // 2:(width - crop) // 2 + crop]
    else:
        crop = int(width * 9 / 16)
        image = image[(height - crop) // 2:(height - crop) // 2 + crop]
    gray = cv2.cvtColor(cv2.resize(image, (64, 36), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (0, 0), 1.0)


def similarity(still_bgr: np.ndarray, frame_bgr: np.ndarray) -> float:
    a, b = _luma(still_bgr).ravel(), _luma(frame_bgr).ravel()
    if a.std() < 1e-3 or b.std() < 1e-3:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def fidelity(still: Path, clip: Path) -> tuple[float, float]:
    """(first-frame, last-frame) similarity to the approved still: an image-to-video clip must start on it and must not wander off."""
    reference = cv2.imread(str(still))
    capture = cv2.VideoCapture(str(clip))
    first = last = None
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        first = frame if first is None else first
        last = frame
    capture.release()
    if reference is None or first is None:
        return 0.0, 0.0
    return similarity(reference, first), similarity(reference, last)


def derive_motion_prompt(seg: Segment) -> str:
    """What moves, for a picture that has no motion prompt of its own: the scene described by the shot, kept stable and gentle."""
    scene = (seg.motion_prompt or seg.image_prompt or seg.image_requirement or seg.visual or seg.narration).strip()
    return (scene.rstrip(".") + ". Gentle natural motion of the scene, slow smooth camera push-in, subjects keep their shape and position, "
            "realistic educational documentary footage, no text.")


def motion_candidates(board: Storyboard, config: AppConfig) -> list[Segment]:
    """Shots whose approved picture should move: the ones the storyboard marked (`animate`, or visual type short_motion_clip), in order.
    Never a shot that carries labels, formulas or several panels: those need a picture that stands still."""
    chosen = []
    for _scene, seg in board.all_segments():
        shot = seg.shot
        if shot is None or shot.template not in {"photo", "video", "title_card"} or not shot.asset_path or not Path(shot.asset_path).is_file():
            continue
        if seg.labels or seg.formula_lines or seg.panel_assets:
            continue
        if seg.animate or shot.template == "video" or seg.visual_type == "short_motion_clip":
            chosen.append(seg)
    return chosen[:max(0, config.production.motion_max_clips)]


class MotionProducer:
    def __init__(self, board: Storyboard, output_dir: Path, config: AppConfig):
        self.board, self.config = board, config
        self.dir = output_dir / "assets"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.client = ComfyUIClient(config)
        self.issues: list[dict] = []

    def _warn(self, seg: Segment, reason: str, repair: str, severity: str = "warning") -> None:
        self.issues.append({"shot_id": seg.shot_id, "field": "short_motion_clip", "severity": severity, "renderer": "ltx",
                            "reason": reason, "repair": repair})

    def _cache_key(self, seg: Segment, prompt: str) -> str:
        p = self.config.production
        workflow = Path(p.motion_workflow_path)
        workflow_text = workflow.read_text(encoding="utf-8") if workflow.is_file() else str(workflow)      # the workflow's content, not just its name
        still = Path(seg.shot.asset_path)
        blob = json.dumps([prompt, workflow_text, still.stat().st_size, hashlib.sha1(still.read_bytes()).hexdigest(),
                           p.motion_width, p.motion_height, p.motion_frames, p.motion_fps])
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def _review(self, seg: Segment, clip: Path) -> tuple[list[str], dict | None, dict]:
        """Technical checks, fidelity to the approved picture, then the vision model on the middle frame."""
        p = self.config.production
        metrics = clip_metrics(clip)
        reasons = motion_ok(metrics)
        first, last = fidelity(Path(seg.shot.asset_path), clip)
        metrics["start_match"], metrics["end_match"] = round(first, 3), round(last, 3)
        if first < p.motion_start_match:
            reasons.append("does_not_start_on_the_picture")
        if last < p.motion_end_match:
            reasons.append("drifted_from_the_picture")
        semantic = None
        if not reasons and metrics["middle"] is not None and p.semantic_qa and vision.available(self.config):
            frame_path = clip.with_suffix(".mid.png")
            Image.fromarray(metrics["middle"]).save(frame_path)
            try:
                semantic = semantic_qa(frame_path, seg, seg.shot.heading or "", self.config)
                if not _accepts(semantic, self.config):
                    reasons.append("semantic_mismatch")
            except (RuntimeError, ValueError, TypeError, KeyError) as exc:
                self._warn(seg, f"motion clip accepted without a content review ({str(exc)[:80]})",
                           "start the visual review model and re-run, or have a person check this clip")
        return reasons, semantic, metrics

    def run(self) -> list[dict]:
        if not self.config.production.motion_enabled:
            return []
        jobs = motion_candidates(self.board, self.config)
        if not jobs:
            return []
        if not self.client.is_available():
            for seg in jobs:
                self._warn(seg, "ComfyUI is not reachable; the approved picture is used without motion", "start ComfyUI and re-run")
            return self.issues
        vision.unload(self.config)
        generated: list[tuple[Segment, Path]] = []
        for seg in jobs:
            prompt = derive_motion_prompt(seg)
            clip = self.dir / f"motion_{seg.shot_id.replace('.', '_')}_{self._cache_key(seg, prompt)}.webm"
            record = clip.with_suffix(".json")
            known = read_json_or_none(record) if clip.exists() else None
            if known is not None and not known.get("reasons"):
                seg.shot.motion_asset = str(clip)              # accepted earlier for exactly this picture, prompt and workflow
                continue
            if known is not None and not self.config.production.retry_rejected:
                continue                                       # rejected earlier: keep the still instead of judging the same clip again
            generated.append((seg, clip))
        for seg, clip in generated:
            prompt = derive_motion_prompt(seg)
            last_reasons: list[str] = []
            for attempt in range(1, max(1, self.config.production.motion_attempts) + 1):
                try:
                    progress.note(f"animating shot {seg.shot_id} (attempt {attempt}; about 1.5 minutes)")
                    clip.unlink(missing_ok=True)
                    retry(lambda: self.client.generate_motion(seg.shot.asset_path, prompt, clip, f"edu_motion_{seg.shot_id.replace('.', '_')}",
                                                              seed=random.Random(f"{clip.stem}:{attempt}").randint(1, 2**31 - 1)),
                          attempts=2, base_delay=5.0, label=f"motion {seg.shot_id}")
                except Exception as exc:                       # ComfyUI hiccup or out of memory: the still plays instead, the lesson goes on
                    self._warn(seg, f"motion generation failed ({str(exc)[:120]}); the approved picture is used without motion",
                               "check ComfyUI/LTX health")
                    self.client.free_memory()
                    break
                self.client.free_memory()
                last_reasons, semantic, metrics = self._review(seg, clip)
                atomic_write_text(clip.with_suffix(".json"), json.dumps(
                    {"clip": clip.name, "attempt": attempt, "metrics": {k: v for k, v in metrics.items() if k != "middle"},
                     "reasons": last_reasons, "semantic": semantic}, indent=2))
                if not last_reasons:
                    seg.shot.motion_asset = str(clip)
                    break
            if last_reasons:
                self._warn(seg, "motion clip rejected (" + ", ".join(last_reasons) + "); the approved picture is used without motion",
                           "simplify the scene (one subject, one action)", "info")
        vision.unload(self.config)
        return self.issues
