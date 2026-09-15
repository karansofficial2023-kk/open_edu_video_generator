from __future__ import annotations

import asyncio
from pathlib import Path

import edge_tts

from .config import AppConfig
from .schema import Caption, Storyboard


def synthesize_storyboard(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> None:
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for scene, segment in storyboard.all_segments():
        print(f"Synthesizing voice {scene.scene_number}.{segment.segment_number}...", flush=True)
        out = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.mp3"
        segment.captions = synthesize_text(segment.narration, out, config)
        # Measure decoded PCM, avoiding MP3 encoder-padding drift between sentences.
        from .renderer import _run, _tool_path
        wav = out.with_suffix(".wav")
        _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(out),
              "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        segment.audio_path = str(wav)


def synthesize_text(text: str, output_path: str | Path, config: AppConfig) -> list[Caption]:
    return asyncio.run(_synthesize_text(text, Path(output_path), config))


async def _synthesize_text(text: str, output_path: Path, config: AppConfig) -> list[Caption]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    communicate = edge_tts.Communicate(
        text,
        config.edge_tts.voice,
        rate=config.edge_tts.rate,
        boundary="WordBoundary",
    )
    words = []
    temporary = output_path.with_suffix(".partial")
    try:
        with temporary.open("wb") as audio:
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio.write(chunk["data"])
                elif chunk["type"] == "WordBoundary":
                    start = chunk["offset"] / 10_000_000
                    end = start + chunk["duration"] / 10_000_000
                    if end > start:
                        words.append(Caption(text=chunk["text"], start=start, end=end))
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return words
