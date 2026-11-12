from __future__ import annotations

import subprocess
import hashlib
import json
import wave
from pathlib import Path

from .config import AppConfig
from .schema import Storyboard


def synthesize_storyboard(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> None:
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    voice_gender = str((storyboard.language or {}).get("voice_gender", "")).strip().lower()
    selected_model = config.piper_gender_models.get(voice_gender, "")
    selected_config = config.piper_gender_configs.get(voice_gender, "")
    updates = {}
    if selected_model:
        updates["piper_model"] = selected_model
    if selected_config:
        updates["piper_config"] = selected_config
    voice_config = config.model_copy(update=updates) if updates else config
    language = str((storyboard.language or {}).get("bcp47", "")).split("-")[0].lower()
    global_pronunciations = config.production.pronunciations.get(language, {})

    for scene, segment in storyboard.all_segments():
        out = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.wav"
        pronunciations = {**global_pronunciations, **(segment.pronunciations or {})}
        spoken = segment.narration
        if pronunciations:
            from .tts_indic import apply_pronunciations
            spoken = apply_pronunciations(spoken, pronunciations)
        model = Path(voice_config.piper_model)
        model_config = Path(voice_config.piper_config) if voice_config.piper_config else None
        cache_key = hashlib.sha256(json.dumps({
            "spoken": spoken, "model": str(model.resolve()),
            "model_mtime": model.stat().st_mtime_ns if model.is_file() else None,
            "config": str(model_config.resolve()) if model_config else "",
            "config_mtime": model_config.stat().st_mtime_ns if model_config and model_config.is_file() else None,
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        marker = out.with_suffix(".sha256")
        if not (out.is_file() and out.stat().st_size > 44 and marker.is_file()
                and marker.read_text(encoding="ascii").strip() == cache_key):
            marker.unlink(missing_ok=True)
            synthesize_text(spoken, out, voice_config)
            marker.write_text(cache_key, encoding="ascii")
        segment.audio_path = str(out)


def synthesize_text(text: str, output_path: str | Path, config: AppConfig) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        config.piper_path,
        "--model",
        config.piper_model,
        "--output_file",
        str(output),
    ]
    if config.piper_config:
        command.extend(["--config", config.piper_config])
    proc = subprocess.run(
        command,
        input=text,
        text=True,
        encoding="utf-8",
        capture_output=True,
        timeout=300,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Piper failed for {output}:\n{proc.stderr}\n{proc.stdout}")


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate)

