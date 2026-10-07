"""Release package: SME review sheet, manifest with hashes, sign-off record (production spec sections 16, 20)."""
from __future__ import annotations

import base64
import hashlib
import html
import io
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import cv2
from PIL import Image

from .schema import Storyboard


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _thumbnail(video: Path, seconds: float, width: int = 480) -> str:
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS) or 30
    capture.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
    ok, frame = capture.read()
    capture.release()
    if not ok:
        return ""
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.thumbnail((width, width))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=82)
    return base64.b64encode(buffer.getvalue()).decode()


def build_release(board: Storyboard, video: Path, output_dir: Path, issues: list[dict], qa: dict) -> Path:
    flags: dict[str, list[str]] = {}
    for item in issues:
        flags.setdefault(str(item.get("shot_id")), []).append(f"{item.get('severity')}: {item.get('reason')}")
    cards = []
    for _scene, seg in board.all_segments():
        sid = seg.shot_id or f"{seg.segment_number}"
        shot_flags = flags.get(sid, [])
        thumb = _thumbnail(video, seg.start + (seg.end - seg.start) * 0.85)
        detail = []
        if seg.labels:
            detail.append("Labels: " + ", ".join(seg.labels))
        if seg.formula_lines:
            detail.append("Formulas: " + " | ".join(seg.formula_lines))
        if seg.columns:
            detail.append("Data/panels: " + " | ".join(seg.columns[:3]))
        provenance = _provenance(output_dir, sid)
        if provenance:
            detail.append("Source: " + provenance)
        cards.append(
            f'<section class="{ "flag" if shot_flags else "" }"><img src="data:image/jpeg;base64,{thumb}" alt="shot {html.escape(sid)}">'
            f"<div><h3>Shot {html.escape(sid)} <small>{html.escape(seg.visual_type)} &rarr; {html.escape(seg.shot.template if seg.shot else '')}"
            f" &middot; {seg.start:.1f}-{seg.end:.1f}s</small></h3><p>{html.escape(seg.narration)}</p>"
            + "".join(f"<p class='meta'>{html.escape(d)}</p>" for d in detail)
            + "".join(f"<p class='warn'>{html.escape(f)}</p>" for f in shot_flags)
            + "<label><input type='checkbox'> Subject accuracy approved</label></div></section>")
    page = f"""<!doctype html><meta charset="utf-8"><title>Review - {html.escape(board.title)}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;margin:24px;max-width:1100px}}section{{display:flex;gap:16px;border:1px solid #ccd;border-radius:10px;
padding:12px;margin:12px 0}}section.flag{{border-color:#c77;background:#fff6f4}}img{{width:320px;height:auto;border-radius:6px}}
h3{{margin:0 0 6px}}small{{color:#667;font-weight:400}}.meta{{color:#345;margin:2px 0}}.warn{{color:#a33;margin:2px 0}}</style>
<h1>{html.escape(board.title)}</h1><p>QA passed: <b>{qa.get('passed')}</b> &middot; critical findings: {qa.get('critical')} &middot; duration {qa.get('duration', 0):.1f}s</p>
{''.join(cards)}"""
    release = output_dir / "release"
    release.mkdir(exist_ok=True)
    (release / "review.html").write_text(page, encoding="utf-8")
    manifest = {
        "title": board.title, "language": board.language, "created": datetime.now(timezone.utc).isoformat(), "host": platform.platform(),
        "files": {name: {"sha256": _sha256(output_dir / name), "bytes": (output_dir / name).stat().st_size}
                  for name in ("final.mp4", "preview.mp4", "storyboard.json", "subtitles.srt", "final_qa.json", "licenses.json", "production_issues.json")
                  if (output_dir / name).is_file()},
        "qa": {"passed": qa.get("passed"), "critical": qa.get("critical")},
        "open_flags": sum(1 for v in flags.values() if v),
    }
    (release / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (release / "sign_off.md").write_text(
        "# Release sign-off\n\n| Approval | Evidence | Owner / date |\n|---|---|---|\n"
        "| Editorial | Narration, timing and learner level approved | |\n"
        "| Subject accuracy | Visual evidence, labels, formulas and claims approved (review.html) | |\n"
        "| Technical | 1080p master, audio, subtitles and final QA passed (final_qa.json) | |\n"
        "| Rights | Asset provenance complete (licenses.json) | |\n| Release | Master and supporting files accepted | |\n", encoding="utf-8")
    return release


def _provenance(output_dir: Path, shot_id: str) -> str:
    base = output_dir / "assets" / f"shot_{shot_id.replace('.', '_')}.json"
    if not base.is_file():
        return ""
    record = json.loads(base.read_text(encoding="utf-8"))
    from .production import _shipped_attempt
    info = (_shipped_attempt(record) or {}).get("provenance")
    if info and record.get("accepted"):
        return f"{info.get('title')} ({info.get('license')}, {info.get('author')})"
    return "FLUX generated" if record.get("attempts") else ""
