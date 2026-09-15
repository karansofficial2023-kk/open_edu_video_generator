from __future__ import annotations

from pathlib import Path

from .config import AppConfig
from .schema import Storyboard


def synthesize_storyboard(storyboard: Storyboard, output_dir: str | Path, config: AppConfig) -> None:
    provider = config.tts_provider.strip().lower()
    if provider == "edge":
        from .tts_edge import synthesize_storyboard as edge_synthesize

        edge_synthesize(storyboard, output_dir, config)
        return
    if provider == "piper":
        from .tts_piper import synthesize_storyboard as piper_synthesize

        piper_synthesize(storyboard, output_dir, config)
        return
    raise ValueError(f"Unsupported TTS provider: {config.tts_provider}")
