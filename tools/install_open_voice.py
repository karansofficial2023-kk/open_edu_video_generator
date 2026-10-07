"""Installs an open-source Indic voice (AI4Bharat Indic-TTS, MIT licence): downloads the language archive and unpacks it.

    python tools/install_open_voice.py ta te            # download (about 1.5 GB each) and unpack into the models folder
    python tools/install_open_voice.py ta --zip D:/downloads/ta.zip

The voice engine itself lives in its own environment (see docs/OPEN_VOICES.md); this only installs voice files.
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "https://github.com/AI4Bharat/Indic-TTS/releases/download/v1-checkpoints-release/{lang}.zip"
LANGUAGES = {"as", "bn", "brx", "gu", "hi", "kn", "ml", "mni", "mr", "or", "pa", "raj", "ta", "te"}


def install(lang: str, models_dir: Path, archive: Path | None = None) -> Path:
    if lang not in LANGUAGES:
        raise SystemExit(f"no open voice for '{lang}' (available: {sorted(LANGUAGES)})")
    target = models_dir / lang
    if (target / "fastpitch" / "best_model.pth").is_file():
        print(f"{lang}: already installed at {target}")
        return target
    models_dir.mkdir(parents=True, exist_ok=True)
    if archive is None:
        archive = models_dir / f"{lang}.zip"
        print(f"{lang}: downloading {RELEASE.format(lang=lang)} ...", flush=True)
        urllib.request.urlretrieve(RELEASE.format(lang=lang), archive)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(models_dir)               # the archive's top-level folder is the language code
    if not (target / "fastpitch" / "best_model.pth").is_file():
        raise SystemExit(f"{archive} did not contain {lang}/fastpitch/best_model.pth")
    print(f"{lang}: installed at {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("languages", nargs="+")
    parser.add_argument("--models-dir", default="D:/AI/models/indic-tts")
    parser.add_argument("--zip", default=None, help="use an archive you already downloaded (single language)")
    arguments = parser.parse_args()
    if arguments.zip and len(arguments.languages) != 1:
        sys.exit("--zip works with exactly one language")
    for language in arguments.languages:
        install(language, Path(arguments.models_dir), Path(arguments.zip) if arguments.zip else None)
