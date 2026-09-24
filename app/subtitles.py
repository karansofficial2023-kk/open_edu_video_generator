from __future__ import annotations

from pathlib import Path

from .schema import Caption, Segment, Storyboard


def phrase_captions(segment: Segment) -> list[Caption]:
    """Group speech boundaries; proportional timing is the preview/legacy fallback."""
    subtitle = None
    if segment.shot and "subtitle" in segment.shot.motion:
        subtitle = str(segment.shot.motion.get("subtitle") or "").strip()
        if not subtitle:
            return []
    words = segment.captions
    duration = segment.speech_duration or (segment.end - segment.start)
    if subtitle is not None and subtitle != segment.narration.strip():
        tokens = subtitle.split()
        words = [Caption(text=w, start=i * duration / len(tokens),
                         end=(i + 1) * duration / len(tokens))
                 for i, w in enumerate(tokens)] if duration > 0 else []
    if not words:
        tokens = (subtitle if subtitle is not None else segment.narration).split()
        words = [Caption(text=w, start=i * duration / len(tokens),
                         end=(i + 1) * duration / len(tokens))
                 for i, w in enumerate(tokens)] if duration > 0 else []
    phrases = []
    group = []
    for word in words:
        if group and (len(" ".join(x.text for x in group + [word])) > 64 or len(group) >= 9):
            phrases.append(Caption(text=" ".join(x.text for x in group),
                                   start=group[0].start, end=group[-1].end))
            group = []
        group.append(word)
        if word.text.endswith((".", "?", "!", ";")):
            phrases.append(Caption(text=" ".join(x.text for x in group),
                                   start=group[0].start, end=group[-1].end))
            group = []
    if group:
        phrases.append(Caption(text=" ".join(x.text for x in group),
                               start=group[0].start, end=group[-1].end))
    return phrases


def write_srt(storyboard: Storyboard, output_path: str | Path) -> None:
    lines: list[str] = []
    index = 1
    for _scene, segment in storyboard.all_segments():
        for cue in phrase_captions(segment):
            lines.extend([str(index),
                          f"{_srt_time(segment.start + cue.start)} --> {_srt_time(segment.start + cue.end)}",
                          cue.text, ""])
            index += 1
    Path(output_path).write_text("\n".join(lines), encoding="utf-8")


def _srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3_600_000
    millis %= 3_600_000
    minutes = millis // 60_000
    millis %= 60_000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
