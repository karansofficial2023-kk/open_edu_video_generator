from __future__ import annotations

import subprocess
import shutil
from pathlib import Path

from .config import AppConfig
from .label_overlay import is_label_template
from .schema import Storyboard, Shot


def assign_timing(storyboard: Storyboard, config: AppConfig) -> None:
    cursor = 0.0
    items = storyboard.all_segments()
    for index, (scene, segment) in enumerate(items):
        duration = config.render.seconds_per_segment_without_audio
        if segment.audio_path and Path(segment.audio_path).exists():
            duration = audio_duration(segment.audio_path, config)
        segment.speech_duration = duration
        next_scene = items[index + 1][0] if index + 1 < len(items) else None
        if next_scene is not None:
            duration += config.voice.scene_gap_seconds if next_scene.scene_number != scene.scene_number else config.voice.sentence_gap_seconds
        segment.start = cursor
        segment.end = cursor + duration
        cursor = segment.end


def concat_audio(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> Path:
    output_dir = Path(output_dir).resolve()
    audio_ext = _audio_ext(storyboard)
    output = output_dir / f"narration{audio_ext}"
    sentence_silence = output_dir / f"silence_sentence{audio_ext}"
    scene_silence = output_dir / f"silence_scene{audio_ext}"
    _make_silence(sentence_silence, config.voice.sentence_gap_seconds, config)
    _make_silence(scene_silence, config.voice.scene_gap_seconds, config)

    audio_files: list[Path] = []
    items = storyboard.all_segments()
    for index, (scene, segment) in enumerate(items):
        if not segment.audio_path:
            continue
        audio_files.append(Path(segment.audio_path).resolve())
        next_scene = items[index + 1][0] if index + 1 < len(items) else None
        if next_scene is None:
            continue
        gap = config.voice.scene_gap_seconds if next_scene.scene_number != scene.scene_number else config.voice.sentence_gap_seconds
        if gap > 0:
            audio_files.append(scene_silence if next_scene.scene_number != scene.scene_number else sentence_silence)

    if not audio_files:
        raise ValueError("No narration audio is available")

    command = [
        _tool_path(config.ffmpeg_path, "ffmpeg"),
        "-y",
    ]
    for audio_file in audio_files:
        command.extend(["-i", str(audio_file)])
    filter_inputs = "".join(f"[{index}:a]" for index in range(len(audio_files)))
    command.extend(
        [
            "-filter_complex",
            f"{filter_inputs}concat=n={len(audio_files)}:v=0:a=1[a]",
            "-map",
            "[a]",
            "-c:a",
            "libmp3lame" if audio_ext == ".mp3" else "pcm_s16le",
            str(output),
        ]
    )
    _run(command)
    return output


def render_video(storyboard: Storyboard, frames: list[Path], audio: Path, output_dir: str | Path,
                 config: AppConfig, subtitles: Path | None = None) -> Path:
    output_dir = Path(output_dir).resolve()
    output = output_dir / "final.mp4"
    clips_dir = output_dir / "clips"
    clips_dir.mkdir(parents=True, exist_ok=True)
    clips: list[Path] = []

    items = storyboard.all_segments()
    segments = [segment for _scene, segment in items]
    if not segments or len(frames) != len(segments):
        raise ValueError("Every storyboard segment must have exactly one rendered frame.")
    frame_cursor = 0
    for index, (segment, frame) in enumerate(zip(segments, frames), start=1):
        end_frame = round(segment.end * config.fps)
        frame_count = max(1, end_frame - frame_cursor)
        frame_cursor += frame_count
        duration = frame_count / config.fps
        clip = clips_dir / f"clip_{index:04d}.mp4"
        if segment.shot and segment.shot.template in {"video", "video_broll", "short_motion_clip"}:
            _render_video_asset(frame, clip, duration, config)
        elif segment.shot and segment.shot.template == "photo":
            _render_fullscreen_image_clip(frame, clip, duration, config)
        elif segment.shot and segment.shot.template in {"title_card", "split_screen"}:
            # These frames are already composited with generated/reviewed assets and labels.
            # Re-rendering them would lose the source image and create blank local templates.
            _render_image_clip(frame, clip, duration, config)
        elif segment.shot and is_label_template(segment.shot.template):
            from PIL import Image
            from .animation_renderer import render_animation_clip
            with Image.open(frame) as im:
                source = im.convert("RGB")
            render_config = config
            if config.render.burn_captions:
                render_config = config.model_copy(deep=True)
                render_config.render.burn_captions = False
            render_animation_clip(segment, items[index - 1][0].title, clip, frame_count, render_config, source)
        elif config.render.layout == "modern" or segment.shot:
            from PIL import Image
            from .animation_renderer import render_animation_clip
            if segment.shot is None:
                segment.shot = Shot(template="photo")
            source = None
            if segment.shot.template == "photo":
                with Image.open(frame) as im:
                    source = im.convert("RGB")
            render_config = config
            if config.render.burn_captions:
                render_config = config.model_copy(deep=True)
                render_config.render.burn_captions = False
            render_animation_clip(segment, items[index - 1][0].title, clip, frame_count, render_config, source)
        else:
            _render_image_clip(frame, clip, duration, config)
        clips.append(clip)

    concat_file = output_dir / "video_concat.txt"
    concat_file.write_text("\n".join(f"file '{_ffconcat_path(clip)}'" for clip in clips), encoding="utf-8")
    command = [
        _tool_path(config.ffmpeg_path, "ffmpeg"),
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-i",
        str(Path(audio).resolve()),
    ]
    burn_subtitles = bool(config.render.burn_captions and subtitles and subtitles.exists())
    embed_subtitles = bool(config.render.embed_subtitles and not burn_subtitles and subtitles and subtitles.exists())
    if embed_subtitles:
        command.extend(["-i", str(Path(subtitles).resolve())])
    command.extend([
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
    ])
    if embed_subtitles:
        command.extend(["-map", "2:s:0"])
    if burn_subtitles:
        command.extend([
            "-vf",
            _subtitle_filter(Path(subtitles).resolve(), config),
            "-c:v",
            "libx264",
            "-preset",
            config.render.video_preset,
            "-crf",
            str(config.render.video_crf),
            "-pix_fmt",
            "yuv420p",
        ])
        _append_video_rate_options(command, config)
    else:
        command.extend([
            "-c:v",
            "copy",
        ])
    command.extend([
        "-movflags",
        "+faststart",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
    ])
    if embed_subtitles:
        command.extend(["-c:s", "mov_text", "-metadata:s:s:0", "language=eng"])
    if not embed_subtitles:
        command.append("-shortest")
    command.append(str(output))
    _run(command)
    return output


def _subtitle_filter(subtitles: Path, config: AppConfig) -> str:
    path = subtitles.as_posix().replace("\\", "/")
    path = path.replace(":", r"\:").replace("'", r"\'")
    style = ",".join([
        "FontName=Arial",
        "Fontsize=24",
        "PrimaryColour=&H00FFFFFF",
        "OutlineColour=&H00000000",
        "BackColour=&H00000000",
        "BorderStyle=1",
        "Outline=2",
        "Shadow=2",
        "Alignment=2",
        "MarginV=38",
    ])
    return f"subtitles='{path}':force_style='{style}'"


def _ass_color(hex_color: str, alpha: str = "00") -> str:
    value = (hex_color or "#000000").strip().lstrip("#")
    if len(value) != 6:
        value = "000000"
    rr, gg, bb = value[0:2], value[2:4], value[4:6]
    return f"&H{alpha}{bb}{gg}{rr}"


def _render_video_asset(source: Path, clip: Path, duration: float, config: AppConfig):
    """Imported or generated clips are silent; narration is muxed separately."""
    w, h = config.output_resolution.width, config.output_resolution.height
    vf = (
        f"scale={w}:{h}:force_original_aspect_ratio=increase,"
        f"crop={w}:{h},setsar=1,fps={config.fps},"
        f"tpad=stop_mode=clone:stop_duration={duration}"
    )
    _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source),
          "-an", "-vf", vf,
          "-frames:v", str(round(duration * config.fps)), "-c:v", "libx264",
          "-preset", config.render.video_preset, "-crf", str(config.render.video_crf),
          *_video_rate_options(config),
          "-pix_fmt", "yuv420p", str(clip)])


def _render_fullscreen_image_clip(frame: Path, clip: Path, duration: float, config: AppConfig) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    frame_count = max(1, round(duration * config.fps))
    # Real photos should fill the whole video frame. Use cover/crop to avoid letterbox borders.
    vf = f"scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},setsar=1"
    command = [
        _tool_path(config.ffmpeg_path, "ffmpeg"),
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(config.fps),
        "-i",
        str(frame),
        "-frames:v",
        str(frame_count),
        "-vf",
        vf,
        "-r",
        str(config.fps),
        "-c:v",
        "libx264",
        "-preset",
        config.render.video_preset,
        "-crf",
        str(config.render.video_crf),
        *_video_rate_options(config),
        "-pix_fmt",
        "yuv420p",
        str(clip),
    ]
    _run(command)


def _render_image_clip(frame: Path, clip: Path, duration: float, config: AppConfig) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    # Frames contain typography: keep it stable and avoid resampling every frame.
    vf = f"scale={width}:{height}:flags=lanczos,setsar=1"
    command = [
        _tool_path(config.ffmpeg_path, "ffmpeg"),
        "-y",
        "-loop",
        "1",
        "-framerate",
        str(config.fps),
        "-i",
        str(frame),
        "-frames:v",
        str(max(1, round(duration * config.fps))),
        "-vf",
        vf,
        "-r",
        str(config.fps),
        "-c:v",
        "libx264",
        "-preset",
        config.render.video_preset,
        "-crf",
        str(config.render.video_crf),
        *_video_rate_options(config),
        "-pix_fmt",
        "yuv420p",
        str(clip),
    ]
    _run(command)


def _make_silence(path: Path, duration: float, config: AppConfig) -> None:
    if duration <= 0:
        return
    command = [
        _tool_path(config.ffmpeg_path, "ffmpeg"),
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=22050:cl=mono",
        "-t",
        f"{duration:.3f}",
        "-c:a",
        "libmp3lame" if path.suffix.lower() == ".mp3" else "pcm_s16le",
        str(path),
    ]
    _run(command)


def audio_duration(path: str | Path, config: AppConfig) -> float:
    command = [
        _tool_path(config.ffprobe_path, "ffprobe"),
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = _run_capture(command)
    if proc.returncode != 0:
        raise RuntimeError(f"FFprobe failed for {path}:\n{proc.stderr}\n{proc.stdout}")
    return float(proc.stdout.strip())


def _audio_ext(storyboard: Storyboard) -> str:
    for _scene, segment in storyboard.all_segments():
        if segment.audio_path:
            suffix = Path(segment.audio_path).suffix.lower()
            if suffix:
                return suffix
    return ".wav"


def _ffconcat_path(path: Path) -> str:
    return path.resolve().as_posix().replace("'", "'\\''")


def _video_rate_options(config: AppConfig) -> list[str]:
    options: list[str] = []
    if config.render.video_bitrate:
        options.extend(["-b:v", config.render.video_bitrate])
        options.extend(["-minrate", config.render.video_bitrate])
    if config.render.video_maxrate:
        options.extend(["-maxrate", config.render.video_maxrate])
    if config.render.video_bufsize:
        options.extend(["-bufsize", config.render.video_bufsize])
    if config.render.video_bitrate:
        options.extend(["-x264-params", "nal-hrd=cbr:force-cfr=1:filler=1"])
    return options


def _append_video_rate_options(command: list[str], config: AppConfig) -> None:
    command.extend(_video_rate_options(config))


def _run(command: list[str]) -> None:
    proc = _run_capture(command)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + proc.stderr + "\n" + proc.stdout)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Tool not found: {command[0]}\n"
            "Fix config.yaml ffmpeg_path/ffprobe_path, or add FFmpeg bin folder to PATH.\n"
            "Example:\n"
            '  ffmpeg_path: "C:/ffmpeg/bin/ffmpeg.exe"\n'
            '  ffprobe_path: "C:/ffmpeg/bin/ffprobe.exe"'
        ) from exc


def _tool_path(configured: str, fallback_name: str) -> str:
    configured = configured.strip()
    if configured:
        configured_path = Path(configured)
        if configured_path.exists():
            return str(configured_path)
        found_configured = shutil.which(configured)
        if found_configured:
            return found_configured
    found_fallback = shutil.which(fallback_name)
    if found_fallback:
        return found_fallback
    return configured or fallback_name
