"""Audio-only dubbing: replace source audio and preserve the original video stream."""
from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import difflib
from pathlib import Path

from .config import AppConfig
from .renderer import _run, _run_capture, _tool_path, audio_duration, concat_audio
from .schema import Scene, Segment, Storyboard
from .tts import synthesize_storyboard
from .translation_qa import repair_protected_content
from .voice_profile import source_voice_profile
from .visual_source_qa import audit_visual_source

MAX_SPEED_UP = 1.50
MIN_NATURAL_TEMPO = 0.85
QA_VERSION = 11


def _normalize_speech(value: str) -> str:
    return re.sub(r"[^\w\u0900-\u097f]+", " ", value.casefold()).strip()


def _cue_speech_checks(transcript: Path, timing_groups: list[dict], segments: list[Segment]) -> list[dict]:
    """Compare synthesized words with the approved Hindi within each source-aligned cue group."""
    data = json.loads(transcript.read_text(encoding="utf-8-sig"))
    heard_words = [(float(w["start"]), str(w["word"]))
                   for cue in data.get("segments", []) for w in cue.get("words", [])
                   if w.get("start") is not None and w.get("word")]
    by_id = {s.shot_id: s for s in segments}
    checks = []
    for group in timing_groups:
        start = group["start_seconds"]
        end = start + group["slot_seconds"]
        expected = " ".join(by_id[shot].narration for shot in group["segment_ids"] if shot in by_id)
        heard = " ".join(word for when, word in heard_words if start <= when < end)
        score = difflib.SequenceMatcher(None, _normalize_speech(expected), _normalize_speech(heard),
                                        autojunk=False).ratio() if expected else 1.0
        checks.append({"segment_ids": group["segment_ids"], "start_seconds": start,
                       "end_seconds": round(end, 3), "similarity": round(score, 3),
                       "status": "needs_review" if len(_normalize_speech(expected)) > 20 and score < 0.35 else "passed",
                       "expected": expected, "heard": heard})
    return checks


def _source_speech_gaps(audio: Path, video: Path, segments: list[Segment], config: AppConfig,
                        source_transcript: Path | None = None) -> list[dict]:
    """Find long Hindi silences inside source ASR cue intervals and measure visual change."""
    result = _run_capture([_tool_path(config.ffmpeg_path, "ffmpeg"), "-hide_banner", "-i", str(audio),
                           "-af", "silencedetect=noise=-35dB:d=1.8", "-f", "null", "-"])
    starts = re.findall(r"silence_start:\s*([\d.]+)", result.stderr)
    ends = re.findall(r"silence_end:\s*([\d.]+)", result.stderr)
    source_words = None
    if source_transcript and source_transcript.is_file():
        transcript = json.loads(source_transcript.read_text(encoding="utf-8-sig"))
        source_words = [(float(w["start"]), float(w["end"])) for cue in transcript.get("segments", [])
                        for w in cue.get("words", []) if w.get("start") is not None and w.get("end") is not None]
        if not source_words:
            source_words = None
    gaps = []
    for begin_text, end_text in zip(starts, ends):
        begin, end = float(begin_text), float(end_text)
        overlapping = [s for s in segments if s.source_start is not None and s.source_end is not None
                       and min(end, float(s.source_end)) - max(begin, float(s.source_start)) >= 1.0]
        if overlapping:
            word_overlap = (sum(max(0.0, min(end, word_end) - max(begin, word_start))
                                for word_start, word_end in source_words)
                            if source_words is not None else None)
            probe_seconds = min(end - begin, 5.0)
            motion = 0.0
            try:
                frames = subprocess.run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-ss", str(begin),
                                         "-t", str(probe_seconds), "-i", str(video), "-vf",
                                         "fps=2,scale=160:90,format=gray", "-f", "rawvideo", "-"],
                                        capture_output=True, timeout=60, check=False)
                pixels = 160 * 90
                raw = frames.stdout
                pairs = [(raw[i:i+pixels], raw[i+pixels:i+2*pixels])
                         for i in range(0, len(raw) - 2*pixels + 1, pixels)]
                if pairs:
                    motion = sum(sum(abs(a - b) for a, b in zip(left, right)) / pixels
                                 for left, right in pairs) / len(pairs)
            except (OSError, RuntimeError, TypeError, subprocess.TimeoutExpired):
                pass
            gaps.append({"start_seconds": round(begin, 3), "end_seconds": round(end, 3),
                         "source_speech_overlap_seconds": round(max(min(end, float(s.source_end))
                            - max(begin, float(s.source_start)) for s in overlapping), 3),
                         "segment_ids": [s.shot_id for s in overlapping],
                         "source_word_overlap_seconds": round(word_overlap, 3) if word_overlap is not None else None,
                         "visual_change_score": round(motion, 2),
                         "visual_motion_detected": motion >= 6.0})
    return gaps


def _speech_verification(audio: Path, expected: str, output: Path) -> dict:
    """ASR the synthesized Hindi speech and compare it with the approved narration."""
    python = Path(os.environ.get("DUB_ASR_PYTHON", r"D:\AI\ComfyUI_windows_portable\python_embeded\python.exe"))
    script = Path(os.environ.get("DUB_ASR_SCRIPT", r"D:\Python\claude version\transcribe\whisper_transcribe.py"))
    if not python.is_file() or not script.is_file():
        return {"status": "unavailable", "detail": "Hindi ASR Python or script is missing"}
    transcript_path = output / "post_tts_transcript.json"
    try:
        process = subprocess.run([str(python), str(script), str(audio), "--model", "large-v3",
                                  "--device", "cuda", "--language", "hi", "--output", str(transcript_path)],
                                 capture_output=True, text=True, errors="replace", timeout=3600)
        if process.returncode or not transcript_path.is_file():
            return {"status": "unavailable", "detail": (process.stderr or process.stdout)[-500:]}
        heard = json.loads(transcript_path.read_text(encoding="utf-8-sig")).get("text", "")
        normalize = lambda value: re.sub(r"[^\w\u0900-\u097f]+", " ", value.casefold()).strip()
        similarity = difflib.SequenceMatcher(None, normalize(expected), normalize(heard), autojunk=False).ratio()
        return {"status": "passed" if similarity >= 0.72 else "needs_teacher_review",
                "similarity": round(similarity, 3), "heard": heard[:1000], "transcript": str(transcript_path)}
    except (OSError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        return {"status": "unavailable", "detail": str(error)[:500]}


def _probe(path: Path, config: AppConfig) -> dict:
    proc = _run_capture([_tool_path(config.ffprobe_path, "ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if proc.returncode:
        raise RuntimeError(f"Could not inspect video {path}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FFprobe returned invalid metadata for {path}") from exc


def _fit_segment(source: Path, target: Path, seconds: float, config: AppConfig,
                 source_speech_seconds: float | None = None) -> float:
    """Keep a natural pace when speech fits; speed up only when needed and pad the rest of its slot with silence."""
    duration = audio_duration(source, config)
    if duration <= 0 or seconds <= 0:
        raise ValueError(f"Cannot fit empty narration audio into {seconds:.2f}s")
    speed = duration / seconds if duration > seconds else 1.0
    if duration <= seconds and source_speech_seconds and duration < source_speech_seconds - 0.35:
        speech_target = min(seconds, source_speech_seconds)
        speed = max(MIN_NATURAL_TEMPO, duration / speech_target)
    if speed > MAX_SPEED_UP:
        raise ValueError(
            f"Translated narration needs {duration:.1f}s but its video slot is {seconds:.1f}s. "
            f"Fitting would require {speed:.2f}x speed (limit {MAX_SPEED_UP:.2f}x); shorten/re-split the translation or allow a longer video."
        )
    filters = ([f"atempo={speed:.6f}"] if abs(speed - 1.0) > 0.005 else [])
    filters.extend([f"apad=whole_dur={seconds:.6f}", f"atrim=duration={seconds:.6f}"])
    _run([
        _tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source),
        "-af", ",".join(filters), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ])
    return speed


def _concat_tts(files: list[Path], target: Path, config: AppConfig) -> Path:
    """Join neighboring translated ASR chunks before fitting when an individual cue window is too short."""
    if len(files) == 1:
        return files[0]
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y"]
    for path in files:
        command.extend(["-i", str(path)])
    filters = []
    inputs = []
    for index in range(len(files)):
        filters.append(f"[{index}:a]aresample=24000,aformat=channel_layouts=mono[a{index}]")
        inputs.append(f"[a{index}]")
    filters.append(f"{''.join(inputs)}concat=n={len(files)}:v=0:a=1[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "24000", "-ac", "1",
                    "-c:a", "pcm_s16le", str(target)])
    _run(command)
    return target


def replace_audio(board: Storyboard, source_video: str | Path, output_dir: str | Path, config: AppConfig,
                  keep_original_audio: bool = False, contract_path: str | Path | None = None) -> Path:
    """Dub a translated Transcribe contract without rendering images or subtitles."""
    source = Path(source_video).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source video does not exist: {source}")
    segments = [segment for _scene, segment in board.all_segments()]
    if not segments:
        raise ValueError("The translated storyboard contract contains no narration segments")
    language = (board.language or {}).get("bcp47", "").strip()
    if not language:
        raise ValueError("Input must be a translated Transcribe contract with language.bcp47 set")

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    qa_repairs, qa_issues = repair_protected_content(board)
    contract_file = Path(contract_path) if contract_path else None
    contract_hash = hashlib.sha256(contract_file.read_bytes()).hexdigest() if contract_file and contract_file.is_file() else ""
    review_file = contract_file.with_name(contract_file.name.replace("_contract.json", "_translation_review.json")) if contract_file else None
    resolved_reviews = []
    review_items = []
    if review_file and review_file.is_file():
        review = json.loads(review_file.read_text(encoding="utf-8-sig"))
        review_items = review.get("review_items", [])
        for item in review_items:
            if item.get("where") in ("title", "auto_repair"):
                continue
            # Only structured audit evidence for the exact current text can resolve a
            # review. Historical prose saying "passed" is not sufficient evidence.
            match = re.search(r"scene (\d+\.\d+)", item.get("where", ""))
            current = next((s for s in segments if s.shot_id == (match.group(1) if match else None)), None) if match else None
            repaired_from_review = any(r.get("shot_id") == (current.shot_id if current else None)
                and r.get("kind") == "narration_corrected"
                and r.get("original_narration") == item.get("translation")
                and r.get("corrected_narration") == (current.narration if current else None)
                for r in qa_repairs)
            deterministic_numeric_verified = (current is not None
                and item.get("where", "").endswith(" narration")
                and item.get("problem", "").startswith("number ")
                and item.get("source") == current.source_narration
                and item.get("translation") == current.narration
                and not any(i.get("shot_id") == current.shot_id for i in qa_issues))
            if deterministic_numeric_verified:
                resolved_reviews.append({**item, "resolution": "source_aligned_numeric_verified"})
                continue
            if (item.get("resolution") == "verified_repair" and current is not None
                    and item.get("source") == current.source_narration
                    and (item.get("translation") == current.narration or repaired_from_review)
                    and not any(i.get("shot_id") == current.shot_id for i in qa_issues)):
                resolved_reviews.append(item)
                continue
            qa_issues.append({"shot_id": item.get("where", ""), "kind": "translation_review",
                              "source": item.get("source", ""), "translation": item.get("translation", ""),
                              "detail": item.get("problem", "Translation needs review")})
    qa_report = output / "translation_qa_report.json"
    qa_report.write_text(json.dumps({
        "qa_version": QA_VERSION, "contract_sha256": contract_hash,
        "translation_review_integrated": True,
        "status": "needs_teacher_review" if qa_issues else "passed_with_corrections" if qa_repairs else "passed",
        "reason": "Source-aligned formula, number, equation, and known terminology checks found items for teacher review; safe corrections were applied before TTS." if qa_issues else "Source-aligned formula, number, equation, and known terminology checks passed; corrections were applied before TTS." if qa_repairs else "Source-aligned formula, number, equation, and known terminology checks passed.",
        "checked_segments": len(segments),
        "repair_count": len(qa_repairs),
        "repairs": qa_repairs,
        "resolved_translation_reviews": resolved_reviews,
        "issue_count": len(qa_issues),
        "issues": qa_issues,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    voice_profile = source_voice_profile(source, config, temp_parent=output)
    if voice_profile.get("gender") not in {"male", "female"}:
        qa_issues.append({"shot_id": "source_voice", "kind": "voice_selection_unverified",
                          "source": "Source speaker gender could not be determined", "translation": "",
                          "detail": "Fallback dubbing voice requires verification or an explicit voice override."})
    board.language = dict(board.language or {})
    board.language["voice_gender"] = voice_profile["gender"]

    metadata = _probe(source, config)
    video_stream = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"Source has no video stream: {source}")
    video_seconds = float(video_stream.get("duration") or metadata.get("format", {}).get("duration") or 0)
    if video_seconds <= 0:
        raise ValueError("Could not determine a positive source-video duration")

    visual_report = None
    if config.production.source_visual_ocr:
        try:
            visual_report = audit_visual_source(
                board, source, output, config, review_items,
                max_cues=config.production.source_visual_ocr_max_cues,
                ollama_url=config.ollama_url, vision_model=config.production.vision_model)
            qa_issues.extend(visual_report["issues"])
        except Exception as error:
            qa_issues.append({"shot_id": "visual_ocr", "kind": "visual_ocr_error", "source": "",
                              "translation": "", "detail": str(error)[:500]})

    # Preserve the original audio and frame beside uncertain ASR/formula findings.
    # These are evidence for correction; a language model must not silently invent
    # a scientific symbol that the source audio or slide does not establish.
    evidence_dir = output / "source_review_evidence"
    for issue in qa_issues:
        if issue.get("kind") not in {"translation_review", "formula_mismatch", "number_mismatch"}:
            continue
        shot = re.search(r"(\d+\.\d+)", str(issue.get("shot_id", "")))
        match = next((s for s in segments if s.shot_id == shot.group(1)), None) if shot else None
        if match is None or match.source_start is None or match.source_end is None:
            continue
        begin = max(0.0, float(match.source_start) - 0.7)
        length = min(video_seconds - begin, float(match.source_end) - begin + 0.7)
        evidence_dir.mkdir(exist_ok=True)
        stem = f"shot_{match.shot_id.replace('.', '_')}"
        audio_clip = evidence_dir / f"{stem}.wav"
        frame = evidence_dir / f"{stem}.jpg"
        if not audio_clip.is_file():
            _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-ss", str(begin),
                  "-t", str(length), "-i", str(source), "-map", "0:a:0", "-vn", "-ar", "16000",
                  "-ac", "1", str(audio_clip)])
        if not frame.is_file():
            _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-ss",
                  str((begin + min(length / 2, 3.0))), "-i", str(source), "-frames:v", "1", str(frame)])
        issue["source_audio_evidence"] = str(audio_clip)
        issue["source_frame_evidence"] = str(frame)

    audio_dir = output / "audio_replacement"
    audio_dir.mkdir(parents=True, exist_ok=True)
    print(f"Synthesizing replacement narration ({language})…", flush=True)
    synthesize_storyboard(board, audio_dir, config)

    source_audio = []
    for index, segment in enumerate(segments, start=1):
        if not segment.audio_path or not Path(segment.audio_path).is_file():
            raise RuntimeError(f"TTS did not produce narration audio for shot {segment.shot_id or index}")
        source_audio.append(Path(segment.audio_path).resolve())
    actual_durations = [audio_duration(path, config) for path in source_audio]
    has_timing = any(segment.source_start is not None or segment.source_end is not None for segment in segments)
    timed = all(segment.source_start is not None and segment.source_end is not None
                and segment.source_start >= 0 and segment.source_end > segment.source_start
                for segment in segments)
    if has_timing and not timed:
        raise ValueError("Translation contract has incomplete or invalid source_start/source_end timing")
    if timed:
        starts = [float(segment.source_start) for segment in segments]
        detected_speech_start = voice_profile.get("speech_start_seconds")
        if (detected_speech_start is not None and len(starts) > 1
                and float(detected_speech_start) > starts[0] + 2.0
                and float(detected_speech_start) < starts[1]):
            starts[0] = float(detected_speech_start)
            segments[0].source_start = starts[0]
        if any(current >= following for current, following in zip(starts, starts[1:])):
            raise ValueError("Timed translation segments must have strictly increasing source_start values")
        if starts[-1] >= video_seconds or any(start >= video_seconds for start in starts) \
                or any(float(segment.source_end) > video_seconds for segment in segments):
            raise ValueError("A transcript segment starts at or beyond the source video duration")
        # Preserve original ASR cue positions. Each translation may use the silence up to the next source utterance.
        slots = [(starts[index + 1] if index + 1 < len(starts) else video_seconds) - start
                 for index, start in enumerate(starts)]
        if any(slot <= 0 for slot in slots):
            raise ValueError("Timed translation segments have no positive audio slot")
    else:
        weights = [segment.duration_hint if segment.duration_hint > 0 else actual_durations[i]
                   for i, segment in enumerate(segments)]
        total_weight = sum(weights)
        if total_weight <= 0:
            raise ValueError("No usable shot durations or narration durations are available for synchronization")
        # Backward compatibility for older storyboard contracts without ASR timing metadata.
        slots = [video_seconds * weight / total_weight for weight in weights]
    # Partition into cue groups. If the tail cannot fit, merge it backward with previous groups so
    # spare time earlier in the video can absorb unusually long translated utterances.
    group_bounds = []
    index = 0
    while index < len(segments):
        end_index = index + 1
        while True:
            group_start = starts[index] if timed else sum(slots[:index])
            group_end = (starts[end_index] if timed and end_index < len(starts)
                         else video_seconds if timed else group_start + sum(slots[index:end_index]))
            group_slot = group_end - group_start
            if group_slot <= 0:
                raise ValueError(f"Audio timing group starting at segment {index + 1} has no positive duration")
            group_duration = sum(actual_durations[index:end_index])
            if group_duration / group_slot <= MAX_SPEED_UP:
                group_bounds.append((index, end_index, group_start, group_slot))
                index = end_index
                break
            if end_index < len(segments):
                # Borrow time from following ASR cues instead of making one short translated line unnaturally fast.
                end_index += 1
                continue
            if group_bounds:
                # If the last group still cannot fit, use unused timing space from preceding groups too.
                index = group_bounds.pop()[0]
                break
            raise ValueError(
                f"Translated speech from segment {index + 1} onward needs {group_duration:.1f}s, but only "
                f"{group_slot:.1f}s remain in the source video, even after grouping; cannot preserve all speech "
                f"within the {MAX_SPEED_UP:.2f}x speed limit."
            )

    speed_factors = []
    timing_groups = []
    timeline_scenes = []
    for group_number, (index, end_index, group_start, group_slot) in enumerate(group_bounds, start=1):
        group_files = source_audio[index:end_index]
        group_raw = _concat_tts(group_files, audio_dir / f"group_{group_number:04d}_raw.wav", config)
        fitted = audio_dir / f"group_{group_number:04d}_fit.wav"
        source_speech_seconds = (max(float(segments[i].source_end) for i in range(index, end_index)) - group_start
                                 if timed else None)
        speed = _fit_segment(group_raw, fitted, group_slot, config, source_speech_seconds)
        speed_factors.append(speed)
        scene = Scene(scene_number=group_number, title="", narration="", segments=[
            Segment(segment_number=1, narration="", audio_path=str(fitted))
        ])
        timeline_scenes.append(scene)
        timing_groups.append({
            "segment_ids": [segments[i].shot_id or str(i + 1) for i in range(index, end_index)],
            "start_seconds": round(group_start, 3), "slot_seconds": round(group_slot, 3),
            "speed_factor": round(speed, 3),
            "source_speech_seconds": round(source_speech_seconds, 3) if source_speech_seconds else None,
        })

    # Fitted groups each fill their cue windows. Keep the original opening audio until the first translated speech cue.
    concat_board = Storyboard(title=board.title, language=board.language, scenes=timeline_scenes)
    if timed and starts[0] > 0.001:
        leading = audio_dir / "source_audio_prefix.wav"
        _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source),
              "-map", "0:a:0", "-vn", "-t", f"{starts[0]:.6f}", "-ar", "24000", "-ac", "1",
              "-c:a", "pcm_s16le", str(leading)])
        first_segment = concat_board.scenes[0].segments[0]
        first_with_leading = audio_dir / "group_0001_with_leading_silence.wav"
        _concat_tts([leading, Path(first_segment.audio_path)], first_with_leading, config)
        first_segment.audio_path = str(first_with_leading)

    # Fitted segment files already include their complete slots, so no inter-sentence gaps are added here.
    concat_config = config.model_copy(deep=True)
    concat_config.voice.sentence_gap_seconds = 0.0
    concat_config.voice.scene_gap_seconds = 0.0
    narration = concat_audio(concat_board, audio_dir, concat_config)
    speech_check = _speech_verification(narration, " ".join(s.narration for s in segments), output)
    cue_checks = (_cue_speech_checks(Path(speech_check["transcript"]), timing_groups, segments)
                  if speech_check.get("transcript") else [])
    transcript_name = (re.sub(r"_[a-z]{2,3}_contract\.json$", "_transcript.json", contract_file.name)
                       if contract_file else "")
    source_transcript = (contract_file.with_name(transcript_name)
                         if contract_file and transcript_name != contract_file.name else None)
    speech_gaps = (_source_speech_gaps(narration, source, segments, config, source_transcript)
                   if timed else [])
    for gap in speech_gaps:
        if (gap["visual_motion_detected"] and gap["source_speech_overlap_seconds"] >= 2.0
                and (gap["source_word_overlap_seconds"] is None or gap["source_word_overlap_seconds"] >= 0.8)):
            qa_issues.append({"shot_id": ",".join(gap["segment_ids"]), "kind": "source_speech_motion_gap",
                              "source": f"{gap['start_seconds']:.2f}-{gap['end_seconds']:.2f}s",
                              "translation": "Hindi narration is silent inside a source ASR cue while frames change",
                              "detail": "Check the source audio and visual action; the ASR cue may also include a pause or music."})
    for cue in cue_checks:
        if cue["status"] == "needs_review":
            qa_issues.append({"shot_id": ",".join(cue["segment_ids"]), "kind": "post_tts_cue_mismatch",
                              "source": cue["expected"], "translation": cue["heard"],
                              "detail": f"Synthesized speech similarity {cue['similarity']:.2f} is below 0.35"})
    if speech_check["status"] != "passed":
        qa_issues.append({"shot_id": "whole_audio", "kind": "post_tts_speech_check",
                          "source": "approved Hindi narration", "translation": "",
                          "detail": speech_check.get("detail", "Speech-to-text differs from approved narration")})
    report = json.loads(qa_report.read_text(encoding="utf-8"))
    report["post_tts_speech_check"] = speech_check
    report["visual_source_qa"] = ({"status": visual_report["status"],
                                    "checked_cues": visual_report["checked_cues"],
                                    "eligible_cues": visual_report["eligible_cues"],
                                    "report_path": visual_report.get("report_path")}
                                   if visual_report else None)
    report["post_tts_cue_checks"] = cue_checks
    report["source_speech_gaps"] = speech_gaps
    report["issues"] = qa_issues
    report["issue_count"] = len(qa_issues)
    report["status"] = "needs_teacher_review" if qa_issues else "passed_with_corrections" if qa_repairs else "passed"
    report["reason"] = ("Unresolved source, translation, or synthesized-speech findings require teacher review."
                        if qa_issues else "Source, translation, and synthesized Hindi speech checks passed.")
    qa_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    measured = audio_duration(narration, config)
    if abs(measured - video_seconds) > 0.15:
        raise RuntimeError(f"Synchronized narration is {measured:.2f}s; source video is {video_seconds:.2f}s")

    target_code = language.split("-")[0].lower()
    suffix = source.suffix.lower() if source.suffix.lower() in {".mp4", ".mov", ".m4v", ".mkv"} else ".mkv"
    track_mode = "dual_audio" if keep_original_audio else "replacement_audio"
    dubbed = output / f"{source.stem}_dub_{target_code}{'_dual' if keep_original_audio else ''}{suffix}"
    print("Muxing original video and " + ("English + Hindi audio tracks…" if keep_original_audio else "replacement Hindi audio…"), flush=True)
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source), "-i", str(narration),
               "-map", "0:v:0"]
    if keep_original_audio:
        command.extend(["-map", "0:a:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy",
                        "-c:a:0", "copy", "-c:a:1", "aac", "-b:a:1", "192k",
                        "-filter:a:1", "loudnorm=I=-16:TP=-1.5:LRA=11",
                        "-metadata:s:a:0", "language=eng", "-metadata:s:a:0", "title=English (Original)",
                        "-disposition:a:0", "default", "-metadata:s:a:1", "language=hin",
                        "-metadata:s:a:1", "title=Hindi", "-disposition:a:1", "0"])
    else:
        command.extend(["-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11"])
    command.extend(["-movflags", "+faststart", str(dubbed)])
    _run(command)
    result = _probe(dubbed, config)
    result_video = next((stream for stream in result.get("streams", []) if stream.get("codec_type") == "video"), None)
    result_audio = next((stream for stream in result.get("streams", []) if stream.get("codec_type") == "audio"), None)
    audio_streams = [stream for stream in result.get("streams", []) if stream.get("codec_type") == "audio"]
    if result_video is None or result_audio is None:
        raise RuntimeError("Dubbed output is missing its copied video or replacement audio stream")
    if keep_original_audio:
        languages = [str(stream.get("tags", {}).get("language", "")).lower() for stream in audio_streams]
        if len(audio_streams) < 2 or "eng" not in languages or "hin" not in languages:
            raise RuntimeError(f"Dual-audio output must contain English and Hindi selectable tracks; found {languages}")
        default_audio = next((stream for stream in audio_streams if "default" in stream.get("disposition", {})
                              and stream["disposition"]["default"] == 1), None)
        if default_audio is None or str(default_audio.get("tags", {}).get("language", "")).lower() != "eng":
            raise RuntimeError("Dual-audio output does not mark English as the default track")
    output_seconds = float(result.get("format", {}).get("duration") or 0)
    if abs(output_seconds - video_seconds) > 0.5:
        raise RuntimeError(f"Dubbed output duration {output_seconds:.2f}s differs from source {video_seconds:.2f}s")

    manifest = {
        "qa_version": QA_VERSION, "contract_sha256": contract_hash,
        "source_video": str(source), "output_video": str(dubbed), "target_language": language,
        "source_duration_seconds": round(video_seconds, 3), "output_duration_seconds": round(output_seconds, 3),
        "video_codec": result_video.get("codec_name"), "audio_codec": result_audio.get("codec_name"),
        "video_stream_copied": True, "original_audio_replaced": not keep_original_audio,
        "audio_track_mode": track_mode,
        "audio_tracks": ([{"language": "eng", "title": "English (Original)", "default": True},
                          {"language": "hin", "title": "Hindi", "default": False}]
                         if keep_original_audio else [{"language": target_code, "title": language, "default": True}]),
        "subtitle_track_created": False,
        "original_audio_preserved_before_first_cue_seconds": round(starts[0], 3) if timed else 0.0,
        "source_voice_profile": voice_profile,
        "selected_voice_model": config.piper_gender_models.get(voice_profile["gender"], config.piper_model),
        "synchronization_mode": "source_asr_segment_start_times" if timed else "contract_duration_weights",
        "shot_speed_factors": [round(value, 3) for value in speed_factors],
        "timing_groups": timing_groups,
    }
    (output / "audio_replacement_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return dubbed
