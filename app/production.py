"""Production orchestrator for contract-driven storyboards (preflight -> stills -> voice -> plan -> render -> QA)."""
from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import cv2
from PIL import Image

from . import progress, vision
from .resilience import atomic_write_text, retry
from .assets import StillProducer
from .captions import build_cues
from .config import AppConfig
from .input_reader import save_storyboard
from .preflight import run_preflight
from .production_renderer import ShotPlan, ShotRenderer, size_of
from .qa import run_qa
from .renderer import _ffconcat_path, _run, _tool_path, assign_timing, concat_audio
from .schema import Shot, Storyboard
from .subtitles import _srt_time
from .tts import synthesize_storyboard


def _shipped_attempt(record: dict) -> dict | None:
    """The attempt whose picture is used in the video: the one at the record's path, else the latest."""
    attempts = record.get("attempts") or []
    path = record.get("path")
    for attempt in reversed(attempts):
        if path and attempt.get("path") == path:
            return attempt
    return attempts[-1] if attempts else None


def _write_licenses(output_dir: Path) -> None:
    """Provenance package for every retrieved asset (URL, author, license, retrieval date)."""
    credits = []
    for record_file in sorted(Path(output_dir, "assets").glob("shot_*.json")):
        record = json.loads(record_file.read_text(encoding="utf-8"))
        if not record.get("accepted"):
            continue
        shipped = _shipped_attempt(record)
        if shipped and shipped.get("provenance"):       # credit the picture that is in the video, reviewed or not
            credits.append({"shot": record["shot_id"], **shipped["provenance"]})
    Path(output_dir, "licenses.json").write_text(json.dumps({"assets": credits}, indent=2, ensure_ascii=False), encoding="utf-8")


def _write_issues(output_dir: Path, issues: list[dict]) -> None:
    Path(output_dir, "production_issues.json").write_text(json.dumps({"issues": issues}, indent=2, ensure_ascii=False), encoding="utf-8")


def _silent_wav(path: Path, seconds: float) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(24000)
        handle.writeframes(b"\0\0" * round(seconds * 24000))


def _last_frame(clip: Path) -> Image.Image | None:
    capture = cv2.VideoCapture(str(clip))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, total - 1))
    ok, frame = capture.read()
    capture.release()
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ok else None


def _clip_signature(segment, frame_count: int, config: AppConfig, backdrop: str | None = None) -> str:
    # every picture or clip that ends up in this shot, not only the main one: a regenerated split-screen panel keeps its file name
    files = [segment.shot.asset_path, *(segment.panel_assets or []), segment.shot.motion_asset]
    stamp = "|".join(f"{Path(f).stat().st_size}:{Path(f).stat().st_mtime_ns}" for f in files if f and Path(f).is_file())
    blob = json.dumps([segment.model_dump(exclude={"audio_path"}), frame_count, stamp, config.production.model_dump(),
                       config.render.model_dump(), config.fps, "v17", backdrop or "", "formula-ui4" if segment.shot and segment.shot.template == "formula" else ""],
                      sort_keys=True, default=str)   # formula-ui2: notebook-style equation card (only formula clips re-render); v14: soft photo behind concept cards; v13: automatic labels; v12: natural-break card splits before the model; v11: LLM card splits only for 8+ words, no space before punctuation in cues; v10: formula appears at 0.1 s with a 0.25 s fade, colon-aware card splits; v6: cut (not crossfade) around text screens; v7: formula starts at 0.3 s; v8: sentence-aware subtitle cues; v9: colon sentences are not term cards
    return hashlib.sha1(blob.encode()).hexdigest()


def write_srt(board: Storyboard, path: Path, config: AppConfig) -> None:
    size = size_of(config)
    lines, index = [], 1
    for _scene, seg in board.all_segments():
        for cue in build_cues(seg, config.production.subtitle, size):
            lines += [str(index), f"{_srt_time(seg.start + cue.start)} --> {_srt_time(seg.start + cue.end)}", cue.text, ""]
            index += 1
    path.write_text("\n".join(lines), encoding="utf-8")


def _release_comfyui(config: AppConfig) -> None:
    try:
        from .comfyui_client import ComfyUIClient
        ComfyUIClient(config).free_memory()
    except Exception:       # best effort only; never mask the real result or error
        pass


def run_production(board: Storyboard, output_dir: Path, config: AppConfig, *, preview: bool = False, skip_tts: bool = False,
                   allow_incomplete: bool = False) -> dict:
    from .gpu_lease import gpu_lease
    with gpu_lease("generator"):
        try:
            return _run_production(board, output_dir, config, preview=preview, skip_tts=skip_tts, allow_incomplete=allow_incomplete)
        finally:
            _release_comfyui(config)        # a crash or early exit must not leave FLUX (about 18 GB of RAM) resident


def _run_production(board: Storyboard, output_dir: Path, config: AppConfig, *, preview: bool = False, skip_tts: bool = False,
                    allow_incomplete: bool = False) -> dict:
    output_dir = Path(output_dir)
    result = {"status": "started", "video": None, "issues": []}
    config.output_resolution.width, config.output_resolution.height = 1920, 1080   # production master profile
    config.fps = 30

    # 1. Preflight: services and contract, before any expensive work or voice synthesis.
    progress.stage(1, f"Checking services and the storyboard ({sum(len(sc.segments) for sc in board.scenes)} shot(s))")
    issues = run_preflight(board, config, output_dir)
    blocking = [i for i in issues if i["severity"] == "error"]
    if blocking:
        _write_issues(output_dir, blocking)
        result.update(status="blocked", issues=blocking)
        print("Preflight blocked rendering; see preflight.json:")
        for item in blocking:
            print(f"  [{item['shot_id']}] {item['field']}: {item['reason']}")
        return result

    # 2. Stills (ComfyUI FLUX + technical/semantic QA + retry rounds). One heavy model at a time.
    progress.stage(2, "Pictures: find or generate each picture, then review it against the narration")
    producer = StillProducer(board, output_dir, config)
    unresolved = producer.run()
    _write_licenses(output_dir)
    progress.stage(3, "Motion: bring approved pictures to life (only shots the storyboard marked)")
    from .motion import MotionProducer
    unresolved += MotionProducer(board, output_dir, config).run()
    from . import auto_labels
    labelled = auto_labels.apply(board, output_dir, config)       # parts named in the narration, verified later in the picture
    if labelled:
        print(f"Proposed labels for {labelled} photo shot(s).", flush=True)
    save_storyboard(board, output_dir / "storyboard.json")

    # 3. Voice (cached by narration+voice+rate), then timing.
    progress.stage(4, "Voice: narration for every shot")
    if preview or skip_tts:
        for _s, seg in board.all_segments():
            seg.audio_path, seg.captions = None, []
    else:
        synthesize_storyboard(board, output_dir, config)
    for _s, seg in board.all_segments():
        seg.extra_hold = 0.0
    assign_timing(board, config)
    if preview or skip_tts:
        for _s, seg in board.all_segments():   # estimate speech from word count instead of the 5 s default
            seg.speech_duration = seg.duration_hint or max(2.5, len(seg.narration.split()) * 0.42)
        _reflow(board, config)

    # 4. Shot plans: locate + verify label targets, compute reading holds, re-time.
    progress.stage(5, "Planning shots: checking label positions and reading time")
    renderer = ShotRenderer(config, output_dir)
    plans: list[ShotPlan] = []
    from .backdrops import choose as choose_backdrops
    from .subject import plain_cards, subject_of
    subject = subject_of(board)
    progress.note(f"subject: {subject}")
    # a soft photo of the lesson behind cards that replace a photograph; mathematics keeps the plain notebook page
    backdrops = {} if plain_cards(subject) else choose_backdrops(board)
    for scene, seg in board.all_segments():
        plan = ShotPlan(segment=seg, scene_title=scene.title, backdrop_path=backdrops.get(id(seg)))
        renderer.prepare(plan)
        for omission in plan.omitted:
            unresolved.append({"shot_id": seg.shot_id, "field": "labels", "severity": "info" if seg.labels_auto else "warning", "renderer": "labels",
                               "reason": f"label {omission['label']!r} omitted: {omission['reason']}", "repair": "provide a visible target or split the shot"})
        needed = renderer.required_seconds(plan)
        current = seg.end - seg.start
        seg.extra_hold = max(0.0, needed - current)
        plans.append(plan)
    vision.unload(config)
    if preview or skip_tts:
        _reflow(board, config)
    else:
        assign_timing(board, config)
    write_srt(board, output_dir / "subtitles.srt", config)
    save_storyboard(board, output_dir / "storyboard.json")

    # 5. Render clips (resumable by content signature; one failure never erases the rest).
    progress.stage(6, f"Rendering {len(plans)} clip(s) (finished clips are reused)")
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(exist_ok=True)
    clips: list[Path] = []
    previous_last: Image.Image | None = None
    previous_template: str | None = None
    cursor = 0
    for index, plan in enumerate(plans, start=1):
        seg = plan.segment
        end_frame = round(seg.end * config.fps)
        frame_count = max(1, end_frame - cursor)
        cursor += frame_count
        clip = clips_dir / f"clip_{index:04d}.mp4"
        sig_file = clip.with_suffix(".sig")
        signature = _clip_signature(seg, frame_count, config, plan.backdrop_path)
        try:
            if clip.exists() and sig_file.exists() and sig_file.read_text() == signature:
                last_png = clip.with_suffix(".last.png")
                previous_last = Image.open(last_png).convert("RGB") if last_png.exists() else _last_frame(clip)
            else:
                progress.step("clip", index, len(plans), f"shot {seg.shot_id or index} ({seg.shot.template}, {frame_count / config.fps:.1f}s)")
                # one more try for a transient fault (encoder crash, antivirus lock) before the shot is reported
                previous_last = retry(lambda: renderer.render_clip(plan, clip, frame_count, previous_last, previous_template),
                                      attempts=2, base_delay=3.0, label=f"render shot {index}")
                atomic_write_text(sig_file, signature)
            previous_template = seg.shot.template
            clips.append(clip)
        except Exception as exc:   # isolate the failed shot; keep rendering the rest
            try:       # safe stand-in: the shot's heading over the plain themed background, with its narration subtitles, so the video completes
                spare = seg.model_copy(update={"shot": Shot(template="title_card", heading=seg.shot.heading or plan.scene_title or "Lesson"),
                                               "labels": [], "label_placement": [], "formula_lines": [], "panel_assets": [], "animate": False})
                spare_plan = ShotPlan(segment=spare, scene_title=plan.scene_title)
                renderer.prepare(spare_plan)
                previous_last = renderer.render_clip(spare_plan, clip, frame_count, previous_last, previous_template)
                atomic_write_text(sig_file, "stand-in")        # never matches a real signature: the next run renders the real shot again
                previous_template = "title_card"
                clips.append(clip)
                unresolved.append({"shot_id": seg.shot_id or str(index), "field": "render", "severity": "warning", "renderer": seg.shot.template,
                                   "reason": f"shot could not be rendered ({str(exc)[:200]}); a plain heading screen was used",
                                   "repair": "fix the shot contract or asset and rerun; approved clips are cached"})
            except Exception as spare_exc:
                unresolved.append({"shot_id": seg.shot_id or str(index), "field": "render", "severity": "error", "renderer": seg.shot.template,
                                   "reason": f"{str(exc)[:200]} (stand-in also failed: {str(spare_exc)[:100]})",
                                   "repair": "fix the shot contract or asset and rerun; approved clips are cached"})
                previous_last = None

    for stale in clips_dir.glob("clip_*.*"):          # a re-run with fewer shots (merged remarks) must not leave old clips behind
        number = stale.name.split(".")[0].removeprefix("clip_")
        if number.isdigit() and int(number) > len(plans):
            stale.unlink(missing_ok=True)
    _write_issues(output_dir, unresolved)
    fatal = [u for u in unresolved if u.get("severity") == "error"]
    if len(clips) != len(plans):
        result.update(status="failed", issues=unresolved)
        return result

    # 6. Audio + mux (loudness normalised).
    progress.stage(7, "Joining clips and audio, then final quality check")
    if preview or skip_tts:
        audio = output_dir / "preview_silence.wav"
        _silent_wav(audio, board.all_segments()[-1][1].end)
    else:
        audio = concat_audio(board, output_dir, config)
    concat = output_dir / "video_concat.txt"
    concat.write_text("\n".join(f"file '{_ffconcat_path(c)}'" for c in clips), encoding="utf-8")     # quotes in file names are escaped
    # A silent video (--skip-tts) or one with fatal errors is a preview, never "final".
    name = "final.mp4" if (not fatal or allow_incomplete) and not preview and not skip_tts else "preview.mp4"
    video = output_dir / name
    stale_final = output_dir / "final.mp4"
    if name != "final.mp4" and stale_final.exists():
        stale_final.replace(output_dir / "final.previous.mp4")        # an older finished video must not sit beside a newer preview
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat), "-i", str(audio)]
    music = Path(config.production.music_path) if config.production.music_path else None
    if music and music.is_file() and not (preview or skip_tts):
        # Licensed background music is looped, kept well under the voice and ducked while narration plays.
        command += ["-stream_loop", "-1", "-i", str(music), "-filter_complex",
                    f"[2:a]volume={config.production.music_volume_db}dB[bg];[1:a]asplit=2[voice][key];"
                    "[bg][key]sidechaincompress=threshold=0.03:ratio=9:attack=30:release=600[ducked];"
                    "[voice][ducked]amix=inputs=2:duration=first:normalize=0,loudnorm=I=-16:TP=-1.5:LRA=11[a]",
                    "-map", "0:v:0", "-map", "[a]"]
    else:
        command += ["-map", "0:v:0", "-map", "1:a:0", "-af", "loudnorm=I=-16:TP=-1.5:LRA=11"]
    command += ["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", "-shortest", str(video)]
    _run(command)

    # 7. Final QA + release package (SME review sheet, manifest, sign-off record).
    report = run_qa(video, board, config, output_dir, fatal)
    try:
        from .release import build_release
        build_release(board, video, output_dir, unresolved, report)
    except Exception as exc:        # governance files must never fail an otherwise finished render
        print(f"Release package skipped: {exc}")
        result["release_error"] = str(exc)
    result.update(status="ok" if report["passed"] else "qa_failed", video=str(video), issues=unresolved, qa=report)
    return result


def _reflow(board: Storyboard, config: AppConfig) -> None:
    cursor = 0.0
    items = board.all_segments()
    for index, (scene, seg) in enumerate(items):
        duration = seg.speech_duration
        nxt = items[index + 1][0] if index + 1 < len(items) else None
        if nxt is not None:
            duration += config.voice.scene_gap_seconds if nxt.scene_number != scene.scene_number else config.voice.sentence_gap_seconds
        duration += seg.extra_hold
        seg.start, seg.end = cursor, cursor + duration
        cursor = seg.end
