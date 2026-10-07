from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path

import edge_tts

from .config import AppConfig
from .schema import Caption, Storyboard


def config_for_language(storyboard: Storyboard, config: AppConfig) -> AppConfig:
    """Pick the Edge voice/rate for the lesson language (language package from Transcribe); English keeps the configured voice."""
    language = (storyboard.language or {}).get("bcp47", "") or ""
    prefix = language.split("-")[0].lower()
    if not prefix or prefix == "en":
        return config
    preferred = (storyboard.language or {}).get("voice_preference", "") or ""
    voice = preferred if preferred.lower().startswith(prefix) else config.production.indic_voices.get(prefix, "")
    if not voice:
        raise ValueError(f"No Edge TTS voice is configured for language {language!r}; set production.indic_voices.{prefix}")
    updated = config.model_copy(deep=True)
    updated.edge_tts.voice, updated.edge_tts.rate = voice, config.production.indic_rate
    return updated


def synthesize_storyboard(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> None:
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    lang = ((storyboard.language or {}).get("bcp47", "") or "en").split("-")[0].lower()
    if chosen_provider(lang, config) == "open":
        _synthesize_open(storyboard, audio_dir, config, lang)           # decided first: an open voice needs no Edge voice for the language
        return
    config = config_for_language(storyboard, config)

    for scene, segment in storyboard.all_segments():
        out = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.mp3"
        # Audio is cached by narration + voice + rate + provider so reruns never resynthesize (spec section 11).
        signature = hashlib.sha1(f"edge2|{config.edge_tts.voice}|{config.edge_tts.rate}|{segment.narration}".encode()).hexdigest()   # edge2: silence trimmed
        meta = out.with_suffix(".json")
        wav = out.with_suffix(".wav")
        if meta.exists() and wav.exists():
            cached = json.loads(meta.read_text(encoding="utf-8"))
            if cached.get("signature") == signature:
                segment.captions = [Caption(**c) for c in cached["captions"]]
                segment.audio_path = str(wav)
                continue
        print(f"Synthesizing voice {scene.scene_number}.{segment.segment_number}...", flush=True)
        meta.unlink(missing_ok=True)            # the cache entry is invalid until the new audio is fully converted
        segment.captions = _retry(lambda: synthesize_text(segment.narration, out, config))
        # Measure decoded PCM, avoiding MP3 encoder-padding drift between sentences.
        from .renderer import _run, _tool_path
        wav = out.with_suffix(".wav")
        _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(out),
              "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(wav)])
        removed = trim_silence(wav, config)
        if removed:
            segment.captions = [Caption(text=c.text, start=max(0.0, c.start - removed), end=max(0.0, c.end - removed)) for c in segment.captions]
        segment.audio_path = str(wav)
        meta.write_text(json.dumps({"signature": signature, "captions": [c.model_dump() for c in segment.captions]}), encoding="utf-8")


KEEP_LEAD, KEEP_TAIL = 0.08, 0.15       # natural breathing room kept around each sentence after trimming


def trim_silence(wav: Path, config: AppConfig) -> float:
    """Cut the TTS engine's leading/trailing silence in place (it adds ~0.3 s + ~1.1 s, which makes every pause ~1.6 s).
    Returns the seconds removed from the START so word timings can be shifted by the same amount."""
    import re
    import subprocess
    from .renderer import _run, _tool_path

    ffmpeg = _tool_path(config.ffmpeg_path, "ffmpeg")
    probe = subprocess.run([ffmpeg, "-hide_banner", "-i", str(wav), "-af", "silencedetect=noise=-45dB:d=0.05", "-f", "null", "-"],
                           capture_output=True, text=True).stderr
    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", probe)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", probe)]
    lead = ends[0] if starts and starts[0] <= 0.05 and ends else 0.0
    removed = max(0.0, lead - KEEP_LEAD)
    trimmed = wav.with_suffix(".trim.wav")
    _run([ffmpeg, "-v", "error", "-y", "-i", str(wav), "-af",
          f"silenceremove=start_periods=1:start_threshold=-45dB:start_silence={KEEP_LEAD},areverse,"
          f"silenceremove=start_periods=1:start_threshold=-45dB:start_silence={KEEP_TAIL},areverse",
          "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le", str(trimmed)])
    if trimmed.exists() and trimmed.stat().st_size > 2000:        # never replace the audio with an empty result
        trimmed.replace(wav)
        return removed
    trimmed.unlink(missing_ok=True)
    return 0.0


def chosen_provider(lang: str, config: AppConfig) -> str:
    """'open' when the open-source voice is requested (or auto and installed); otherwise 'edge'. Strict mode never falls back to a cloud voice."""
    from . import tts_indic

    wanted = config.production.tts_provider
    if wanted == "edge" or lang == "en":
        return "edge"
    if tts_indic.available(lang, config):
        return "open"
    if wanted == "open":
        raise ValueError(f"tts_provider is 'open' but no open-source voice is installed for '{lang}' (see docs/OPEN_VOICES.md)")
    return "edge"


def _synthesize_open(storyboard: Storyboard, audio_dir: Path, config: AppConfig, lang: str) -> None:
    from . import tts_indic
    from .renderer import _run, _tool_path

    table = config.production.pronunciations.get(lang, {})
    speaker = config.production.open_tts.speaker
    voice = None
    try:
        for scene, segment in storyboard.all_segments():
            base = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}"
            text = tts_indic.prepare_text(segment.narration, lang, table)
            signature = hashlib.sha1(f"open2|{lang}|{speaker}|{text}".encode()).hexdigest()
            meta, wav = base.with_suffix(".json"), base.with_suffix(".wav")
            if meta.exists() and wav.exists():
                cached = json.loads(meta.read_text(encoding="utf-8"))
                if cached.get("signature") == signature:
                    segment.captions = [Caption(**c) for c in cached["captions"]]
                    segment.audio_path = str(wav)
                    continue
            if voice is None:
                voice = tts_indic.OpenVoice(config)
            print(f"Synthesizing open voice {scene.scene_number}.{segment.segment_number}...", flush=True)
            raw = base.with_suffix(".raw.wav")
            info = voice.speak(lang, text, raw)
            segment.captions = tts_indic.distribute_captions(segment.narration, info["seconds"], info["lead"], info["tail"], lang, table)
            _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(raw), "-ar", "24000", "-ac", "1",
                  "-c:a", "pcm_s16le", str(wav)])
            raw.unlink(missing_ok=True)
            meta.write_text(json.dumps({"signature": signature, "captions": [c.model_dump() for c in segment.captions]}), encoding="utf-8")
            segment.audio_path = str(wav)
    finally:
        if voice is not None:
            voice.close()


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


def _retry(action, attempts: int = 8):
    """Network TTS with exponential backoff (2, 4, 8, 16, 32, 60, 60 s): a dropped connection or a busy service for a few minutes
    must not end a lesson, whose voice cannot be replaced by silence."""
    from .resilience import retry
    return retry(action, attempts=attempts, base_delay=2.0, label="voice synthesis")
