"""Audio-only dubbing: replace source audio and preserve the original video stream."""
from __future__ import annotations

import json
from pathlib import Path

from .config import AppConfig
from .renderer import _run, _run_capture, _tool_path, audio_duration, concat_audio
from .schema import Storyboard
from .tts import synthesize_storyboard

MAX_SPEED_UP = 1.35


def _probe(path: Path, config: AppConfig) -> dict:
    proc = _run_capture([_tool_path(config.ffprobe_path, "ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)])
    if proc.returncode:
        raise RuntimeError(f"Could not inspect video {path}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"FFprobe returned invalid metadata for {path}") from exc


def _fit_segment(source: Path, target: Path, seconds: float, config: AppConfig) -> float:
    """Keep a natural pace when speech fits; speed up only when needed and pad the rest of its slot with silence."""
    duration = audio_duration(source, config)
    if duration <= 0 or seconds <= 0:
        raise ValueError(f"Cannot fit empty narration audio into {seconds:.2f}s")
    speed = duration / seconds if duration > seconds else 1.0
    if speed > MAX_SPEED_UP:
        raise ValueError(
            f"Translated narration needs {duration:.1f}s but its video slot is {seconds:.1f}s. "
            f"Fitting would require {speed:.2f}x speed (limit {MAX_SPEED_UP:.2f}x); shorten/re-split the translation or allow a longer video."
        )
    filters = ([f"atempo={speed:.6f}"] if speed > 1.005 else [])
    filters.extend([f"apad=whole_dur={seconds:.6f}", f"atrim=duration={seconds:.6f}"])
    _run([
        _tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source),
        "-af", ",".join(filters), "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(target),
    ])
    return speed


def replace_audio(board: Storyboard, source_video: str | Path, output_dir: str | Path, config: AppConfig) -> Path:
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

    metadata = _probe(source, config)
    video_stream = next((stream for stream in metadata.get("streams", []) if stream.get("codec_type") == "video"), None)
    if video_stream is None:
        raise ValueError(f"Source has no video stream: {source}")
    video_seconds = float(video_stream.get("duration") or metadata.get("format", {}).get("duration") or 0)
    if video_seconds <= 0:
        raise ValueError("Could not determine a positive source-video duration")

    output = Path(output_dir).resolve()
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
    weights = [segment.duration_hint if segment.duration_hint > 0 else actual_durations[i]
               for i, segment in enumerate(segments)]
    total_weight = sum(weights)
    if total_weight <= 0:
        raise ValueError("No usable shot durations or narration durations are available for synchronization")

    # Scale the contract's shot slots to the source duration; preserve order and relative shot timing.
    slots = [video_seconds * weight / total_weight for weight in weights]
    speed_factors = []
    for index, (segment, source_audio_path, slot) in enumerate(zip(segments, source_audio, slots), start=1):
        fitted = audio_dir / f"fit_{index:04d}.wav"
        speed_factors.append(_fit_segment(source_audio_path, fitted, slot, config))
        segment.audio_path = str(fitted)
        segment.extra_hold = 0.0

    # Fitted segment files already include their complete slots, so no inter-sentence gaps are added here.
    concat_config = config.model_copy(deep=True)
    concat_config.voice.sentence_gap_seconds = 0.0
    concat_config.voice.scene_gap_seconds = 0.0
    narration = concat_audio(board, audio_dir, concat_config)
    measured = audio_duration(narration, config)
    if abs(measured - video_seconds) > 0.15:
        raise RuntimeError(f"Synchronized narration is {measured:.2f}s; source video is {video_seconds:.2f}s")

    target_code = language.split("-")[0].lower()
    suffix = source.suffix.lower() if source.suffix.lower() in {".mp4", ".mov", ".m4v", ".mkv"} else ".mkv"
    dubbed = output / f"{source.stem}_dub_{target_code}{suffix}"
    print("Muxing dubbed audio and copying the original video stream…", flush=True)
    _run([
        _tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source), "-i", str(narration),
        "-map", "0:v:0", "-map", "1:a:0", "-map_metadata", "0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11", "-movflags", "+faststart", str(dubbed),
    ])
    result = _probe(dubbed, config)
    result_video = next((stream for stream in result.get("streams", []) if stream.get("codec_type") == "video"), None)
    result_audio = next((stream for stream in result.get("streams", []) if stream.get("codec_type") == "audio"), None)
    if result_video is None or result_audio is None:
        raise RuntimeError("Dubbed output is missing its copied video or replacement audio stream")
    output_seconds = float(result.get("format", {}).get("duration") or 0)
    if abs(output_seconds - video_seconds) > 0.5:
        raise RuntimeError(f"Dubbed output duration {output_seconds:.2f}s differs from source {video_seconds:.2f}s")

    manifest = {
        "source_video": str(source), "output_video": str(dubbed), "target_language": language,
        "source_duration_seconds": round(video_seconds, 3), "output_duration_seconds": round(output_seconds, 3),
        "video_codec": result_video.get("codec_name"), "audio_codec": result_audio.get("codec_name"),
        "video_stream_copied": True, "original_audio_replaced": True, "subtitle_track_created": False,
        "shot_speed_factors": [round(value, 3) for value in speed_factors],
    }
    (output / "audio_replacement_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return dubbed