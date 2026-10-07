"""Final video QA: container, blank/repeated frames, subtitle bounds, audio, A/V drift, semantic + metadata-leak review."""
from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from . import vision
from .captions import build_cues
from .config import AppConfig
from .renderer import _tool_path
from .router import METADATA_PATTERN
from .schema import Storyboard
from .typography import subtitle_band


def _probe(video: Path, config: AppConfig) -> dict:
    out = subprocess.run([_tool_path(config.ffprobe_path, "ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(video)],
                         capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def _dhash(frame: np.ndarray) -> int:
    small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


QUESTION_WORDS = {"what", "why", "how", "when", "where", "who", "which", "whose"}


def blank_screen(frame: np.ndarray) -> bool:
    """True when the middle of a screen (below the heading, above the subtitle band, inside the card or panel) shows nothing: an
    empty equation card or an empty board. Measured as the share of edge pixels; a photograph, an equation or card text is far above."""
    height, width = frame.shape[:2]
    inner = frame[round(height * 0.17):round(height * 0.78), round(width * 0.05):round(width * 0.95)]      # all content below the heading, above subtitles (a board fills from the top)
    gray = cv2.cvtColor(inner, cv2.COLOR_BGR2GRAY) if inner.ndim == 3 else inner
    edges = cv2.Canny(gray, 20, 60)
    # count small marks (letters, symbols, picture detail); the long straight edges of an empty card, panel or table frame do not count
    count, _labels, stats, _centroids = cv2.connectedComponentsWithStats((edges > 0).astype(np.uint8), connectivity=8)
    h, w = gray.shape[:2]
    marks = sum(1 for i in range(1, count) if stats[i, cv2.CC_STAT_WIDTH] < w * 0.25 and stats[i, cv2.CC_STAT_HEIGHT] < h * 0.25
                and stats[i, cv2.CC_STAT_AREA] >= 4)
    if marks < 6:
        return True
    # an empty white card or panel in the middle of the screen (the note line under it does not make it a full screen)
    middle = frame[round(height * 0.25):round(height * 0.62), round(width * 0.12):round(width * 0.88)]
    middle = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY) if middle.ndim == 3 else middle
    return bool(middle.mean() > 200 and middle.std() < 12)


def near_identical(a: np.ndarray, b: np.ndarray) -> bool:
    """Two frames a viewer would take for the same screen (same picture/board; subtitles aside)."""
    if bin(_dhash(a) ^ _dhash(b)).count("1") > 2:
        return False
    small = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY) if f.ndim == 3 else f, (192, 108), interpolation=cv2.INTER_AREA)
    top = lambda f: f[: round(f.shape[0] * 0.78)]          # ignore the subtitle band, which changes with the narration
    changed = np.abs(small(top(a)).astype(np.float32) - small(top(b)).astype(np.float32)) > 24
    return float(changed.mean()) < 0.002                   # a new equation line or a new card changes far more pixels than this


def fragment_card(step: str, narration: str) -> bool:
    """A card a teacher would not write: a lone question word, a phrase ending on a dangling word, or the start of the narration's
    sentence cut off before its end."""
    from .assets import _DANGLING
    text = (step or "").strip()
    words = re.findall(r"[\w'-]+", text.lower())
    if not words:
        return True
    if len(words) == 1 and words[0] in QUESTION_WORDS:
        return True
    complete = text.endswith((".", "?", "!", ":"))
    if not complete and len(words) > 1 and words[-1] in _DANGLING:
        return True
    norm = lambda t: " ".join(re.findall(r"[\w'-]+", (t or "").lower()))
    head, said = norm(text), norm(narration)
    first_sentence = norm(re.split(r"(?<=[.!?])\s+", (narration or "").strip())[0])
    return (not complete and len(words) >= 4 and said.startswith(head) and len(head) < len(first_sentence))


def _audio_stats(video: Path, config: AppConfig) -> dict:
    proc = subprocess.run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-hide_banner", "-i", str(video), "-af", "volumedetect,silencedetect=n=-45dB:d=2.5",
                           "-vn", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    text = proc.stderr
    mean = re.search(r"mean_volume:\s*(-?[\d.]+) dB", text)
    peak = re.search(r"max_volume:\s*(-?[\d.]+) dB", text)
    return {"mean_db": float(mean.group(1)) if mean else None, "max_db": float(peak.group(1)) if peak else None,
            "long_silences": len(re.findall(r"silence_start", text))}


def run_qa(video: Path, board: Storyboard, config: AppConfig, output_dir: Path, unresolved: list[dict]) -> dict:
    findings: list[dict] = []

    def add(level, gate, message, shot="*"):
        findings.append({"level": level, "gate": gate, "shot_id": shot, "message": message})

    info = _probe(video, config)
    vstream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    astream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if not vstream or (vstream["width"], vstream["height"]) != (1920, 1080) or vstream["codec_name"] != "h264":
        add("critical", "container", f"expected 1920x1080 h264, got {vstream and (vstream['codec_name'], vstream['width'], vstream['height'])}")
    if vstream:
        num, den = (int(x) for x in vstream["avg_frame_rate"].split("/"))
        if den and abs(num / den - config.fps) > 0.01:
            add("critical", "container", f"frame rate {num / den:.2f} != {config.fps}")
    if not astream:
        add("critical", "container", "no audio stream")
    duration = float(info["format"]["duration"])
    if astream and vstream and abs(float(astream.get("duration", duration)) - float(vstream.get("duration", duration))) > 0.5:
        add("critical", "av_sync", "audio and video stream durations differ by more than 0.5 s")
    expected = board.all_segments()[-1][1].end
    if abs(duration - expected) > 0.5:
        add("critical", "av_sync", f"master duration {duration:.2f}s vs storyboard {expected:.2f}s")

    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or config.fps
    seen: dict[int, str] = {}
    samples = []
    previous: tuple[str, str, np.ndarray] | None = None
    for scene, seg in board.all_segments():
        sid = seg.shot_id or f"{scene.scene_number}.{seg.segment_number}"
        length = seg.end - seg.start
        if seg.shot is not None and seg.shot.template not in {"photo", "video"} and length > 2.0:
            empty = 0                   # an empty board for more than 2 s (e.g. an equation card whose equation was lost)
            for share in (0.25, 0.55, 0.85):
                capture.set(cv2.CAP_PROP_POS_FRAMES, round((seg.start + length * share) * fps))
                ok, probe_frame = capture.read()
                empty += bool(ok and blank_screen(probe_frame))
            if empty >= 2:
                add("critical", "blank_screen", "the screen shows nothing but its heading for most of the shot", sid)
        mid = seg.start + (seg.end - seg.start) * 0.85
        capture.set(cv2.CAP_PROP_POS_FRAMES, round(mid * fps))
        ok, frame = capture.read()
        if not ok:
            add("critical", "frames", "cannot decode sampled frame", sid)
            continue
        if float(frame.std()) < 8:
            add("critical", "blank_frame", "blank or near-uniform frame", sid)
        if previous is not None and previous[1] != seg.narration and not getattr(seg, "continuation", False) \
                and near_identical(previous[2], frame):
            add("warning", "repeated_screen", f"shows the same screen as shot {previous[0]} while the narration moves on", sid)
        previous = (sid, seg.narration, frame)
        h = _dhash(frame)
        if getattr(seg, "continuation", False):
            samples.append((sid, seg, frame))   # a shot that deliberately continues the previous picture is not an accidental repeat
            continue
        if seg.shot is not None and seg.shot.template in {"process", "title_card", "formula", "graph", "circuit", "table"}:
            samples.append((sid, seg, frame))   # deterministic cards share a background by design; only photographs can be accidental repeats
            continue
        for other, other_sid in seen.items():
            if bin(h ^ other).count("1") <= 2 and other_sid != sid:
                add("warning", "repeated_frame", f"looks identical to shot {other_sid}", sid)
        seen[h] = sid
        samples.append((sid, seg, frame))
    capture.release()

    for scene, seg in board.all_segments():
        sid = seg.shot_id or f"{scene.scene_number}.{seg.segment_number}"
        if seg.shot is not None and seg.shot.template in {"process", "comparison"}:
            broken = [step for step in seg.shot.steps if fragment_card(step, seg.narration)]
            if broken:
                add("warning", "fragment_card", f"card(s) that are not a complete idea: {broken[:2]}", sid)
        size = (config.output_resolution.width, config.output_resolution.height)
        try:
            for cue in build_cues(seg, config.production.subtitle, size):
                subtitle_band(size, cue.text, config.production.subtitle)
        except ValueError as exc:
            add("critical", "subtitles", str(exc), sid)
        if METADATA_PATTERN.search(seg.subtitle or ""):
            add("critical", "metadata_leak", "subtitle contains production metadata", sid)

    audio = _audio_stats(video, config)
    if audio["max_db"] is not None and audio["max_db"] > -0.1:
        add("critical", "audio", f"peak {audio['max_db']} dB clips")
    if audio["mean_db"] is None or audio["mean_db"] < -50:
        # a narration track that is effectively silent is only acceptable in a preview that was made without voice on purpose
        add("warning" if video.name.startswith("preview") else "critical", "audio", f"narration track is silent (mean level {audio['mean_db']} dB)")
    if audio["long_silences"]:
        add("warning", "audio", f"{audio['long_silences']} silence(s) longer than 2.5 s")

    if config.production.semantic_qa and vision.available(config):
        for sid, seg, frame in samples:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            from PIL import Image
            try:
                # the reviewer reads English: a lesson in another script is judged against the planned English visual requirement
                native = sum(1 for c in seg.narration if c.isalpha() and ord(c) >= 0x0590) > len(seg.narration) * 0.3
                said = (seg.image_requirement or seg.image_prompt or "") if native else seg.narration
                verdict = vision.ask(Image.fromarray(rgb), (
                    f"What the narration says at this moment (the frame should support it): {said or seg.narration}\n"
                    "Review this video frame. A camera cannot show invisible processes (gas exchange, chemical change): a frame of the "
                    "right subject that illustrates the idea plausibly supports the narration; only a frame of a different subject does not. "
                    "Return JSON {\"matches\": bool (the visible content supports the narration), "
                    "\"score\": 0..1, \"visible_text\": [strings of any text on screen other than the subtitle band and title], "
                    "\"production_notes_visible\": bool (prompts, filenames, placement instructions or metadata on screen)}."),
                    config, purpose=f"final:{sid}")
            except RuntimeError as exc:
                add("warning", "semantic", f"vision review unavailable: {exc}", sid)
                break
            if verdict.get("production_notes_visible"):
                add("critical", "metadata_leak", "production notes visible on screen", sid)
            if not verdict.get("matches") or float(verdict.get("score") or 0) < 0.5:
                add("warning", "semantic", f"frame may not support narration (score {verdict.get('score')})", sid)
        vision.unload(config)

    for item in unresolved:
        add("critical", "unresolved", f"{item.get('field')}: {item.get('reason')}", item.get("shot_id", "*"))
    report = {"video": str(video), "duration": duration, "audio": audio, "findings": findings,
              "critical": sum(1 for f in findings if f["level"] == "critical"),
              "passed": not any(f["level"] == "critical" for f in findings)}
    (output_dir / "final_qa.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
