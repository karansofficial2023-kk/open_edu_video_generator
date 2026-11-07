from __future__ import annotations

import os
import subprocess
import shutil
import tempfile
from pathlib import Path

from .config import AppConfig
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
        duration += segment.extra_hold   # reading hold for labels/formulas; audio gets matching silence
        segment.start = cursor
        segment.end = cursor + duration
        cursor = segment.end


def concat_audio(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> Path:
    output_dir = Path(output_dir).resolve()
    audio_ext = _audio_ext(storyboard)
    output = output_dir / f"narration{audio_ext}"
    silences: dict[int, Path] = {}

    def silence(seconds: float) -> Path:
        key = round(seconds * 1000)
        if key not in silences:
            silences[key] = output_dir / f"silence_{key}ms{audio_ext}"
            _make_silence(silences[key], key / 1000, config)
        return silences[key]

    audio_files: list[Path] = []
    items = storyboard.all_segments()
    for index, (scene, segment) in enumerate(items):
        if not segment.audio_path:
            continue
        audio_files.append(Path(segment.audio_path).resolve())
        next_scene = items[index + 1][0] if index + 1 < len(items) else None
        gap = segment.extra_hold
        if next_scene is not None:
            gap += config.voice.scene_gap_seconds if next_scene.scene_number != scene.scene_number else config.voice.sentence_gap_seconds
        if gap > 0:
            audio_files.append(silence(gap))

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
    primary_karaoke = storyboard.is_primary()
    if primary_karaoke:
        for segment in segments:
            segment.subtitle_style = "primary_karaoke"
    frame_cursor = 0
    for index, (segment, frame) in enumerate(zip(segments, frames), start=1):
        end_frame = round(segment.end * config.fps)
        frame_count = max(1, end_frame - frame_cursor)
        frame_cursor += frame_count
        duration = frame_count / config.fps
        clip = clips_dir / f"clip_{index:04d}.mp4"
        if segment.shot and segment.shot.template == "video":
            _render_video_asset(frame, clip, duration, config)
        elif segment.shot and segment.shot.template == "photo":
            _render_fullscreen_image_clip(frame, clip, duration, config, primary=primary_karaoke)
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
        if primary_karaoke:
            # The primary-school subtitle is part of the picture: a cream card with
            # the currently spoken word highlighted. Apply it after whichever
            # legacy clip renderer was selected so photos and imported video get it too.
            _render_primary_overlay(clip, segment, items[index - 1][0].title,
                                    items[index - 1][0].scene_number, duration, config)
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
    # Primary captions are already drawn word-by-word into the clips above. Burning
    # the ordinary SRT here would cover that design with a second subtitle track.
    burn_subtitles = bool(not primary_karaoke and config.render.burn_captions and subtitles and subtitles.exists())
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
        "Fontsize=18",
        f"PrimaryColour={_ass_color(config.render.subtitle_fg)}",
        "OutlineColour=&H00FFFFFF",
        f"BackColour={_ass_color(config.render.subtitle_bg, alpha='80')}",
        "BorderStyle=3",
        "Outline=1",
        "Shadow=0",
        "Alignment=2",
        "MarginV=48",
    ])
    return f"subtitles='{path}':force_style='{style}'"


def _render_primary_overlay(clip: Path, segment, scene_title: str, scene_number: int,
                            duration: float, config: AppConfig) -> None:
    """Burn the reference-style primary karaoke card over a legacy rendered clip."""
    import cv2
    from PIL import Image
    from .captions import build_cues
    from .typography import primary_karaoke_band, primary_scene_banner

    width, height = config.output_resolution.width, config.output_resolution.height
    size = (width, height)
    style = config.production.subtitle
    cues = build_cues(segment, style, size)
    fps = config.fps
    frame_count = max(1, round(duration * fps))
    temp_clip = clip.with_name(f"{clip.stem}.primary.mp4")
    capture = cv2.VideoCapture(str(clip))
    if not capture.isOpened():
        raise RuntimeError(f"Could not read rendered clip for primary subtitles: {clip}")
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
               "-r", str(fps), "-i", "pipe:0", "-an", "-frames:v", str(frame_count),
               "-c:v", "libx264", "-preset", config.render.video_preset,
               "-crf", str(config.render.video_crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart",
               str(temp_clip)]
    try:
        with tempfile.TemporaryFile() as errors:
            proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors, stdout=subprocess.DEVNULL)
            try:
                for i in range(frame_count):
                    ok, bgr = capture.read()
                    if not ok:
                        break
                    if bgr.shape[1] != width or bgr.shape[0] != height:
                        bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
                    t = i / fps
                    cue = next((c for c in cues if c.start <= t < c.end),
                               cues[-1] if cues and t >= cues[-1].start else None)
                    frame = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).convert("RGBA")
                    if cue is not None:
                        overlay = primary_karaoke_band(size, cue, t, style,
                                                       topic_style=getattr(segment, "topic_style", "general"))
                        if overlay:
                            frame = Image.alpha_composite(frame, overlay[0])
                    if scene_title:
                        banner = primary_scene_banner(size, f"{scene_number}. {scene_title}",
                                                       topic_style=getattr(segment, "topic_style", "general"))
                        if banner:
                            frame = Image.alpha_composite(frame, banner)
                    proc.stdin.write(frame.convert("RGB").tobytes())
                proc.stdin.close()
                if proc.wait() != 0:
                    errors.seek(0)
                    raise RuntimeError(errors.read().decode("utf-8", "replace"))
            except BaseException:
                proc.kill()
                proc.wait()
                raise
        capture.release()
        temp_clip.replace(clip)
    except BaseException:
        capture.release()
        temp_clip.unlink(missing_ok=True)
        raise


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
          "-pix_fmt", "yuv420p", str(clip)])


def _render_fullscreen_image_clip(frame: Path, clip: Path, duration: float, config: AppConfig, primary: bool = False) -> None:
    width = config.output_resolution.width
    height = config.output_resolution.height
    frame_count = max(1, round(duration * config.fps))
    # Fill the frame; primary storybook stills get a gentle push-in like the reference video.
    if primary:
        zoom_width, zoom_height = round(width * 1.06), round(height * 1.06)
        vf = (f"scale={zoom_width}:{zoom_height}:force_original_aspect_ratio=increase,"
              f"crop={zoom_width}:{zoom_height},zoompan=z='min(zoom+0.00014,1.04)':d=1:"
              f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s={width}x{height}:fps={config.fps},setsar=1")
    else:
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


FFMPEG_TIMEOUT_SECONDS = int(os.environ.get("EDU_FFMPEG_TIMEOUT", "1800"))


def _run(command: list[str]) -> None:
    proc = _run_capture(command)
    if proc.returncode != 0:
        raise RuntimeError("Command failed:\n" + " ".join(command) + "\n" + proc.stderr + "\n" + proc.stdout)


def _run_capture(command: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=FFMPEG_TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired as exc:       # a hung encoder must fail this step (which is retried), not freeze the run forever
        raise RuntimeError(f"Command timed out after {FFMPEG_TIMEOUT_SECONDS}s: {' '.join(command)[:300]}") from exc
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
