from __future__ import annotations

import subprocess
import wave
from pathlib import Path

from .config import AppConfig
from .schema import Storyboard


def synthesize_storyboard(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> None:
    audio_dir = Path(output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    for scene, segment in storyboard.all_segments():
        out = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.wav"
        synthesize_text(segment.narration, out, config)
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

