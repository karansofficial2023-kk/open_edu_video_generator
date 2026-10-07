"""Open-source Indic text-to-speech worker (AI4Bharat Indic-TTS: FastPitch + HiFi-GAN, MIT licence), run in an isolated environment.

    <venv>/python tools/indic_tts_worker.py --models-dir D:/AI/models/indic-tts

Protocol (one JSON object per line, UTF-8):
    -> {"lang": "ta", "speaker": "female", "text": "...", "out": "x.wav"}
    <- {"ok": true, "seconds": 7.6, "lead": 0.12, "tail": 0.20, "sample_rate": 22050}   or   {"ok": false, "error": "..."}
    -> {"command": "quit"}
Models are loaded on first use of a language and kept. Runs on the CPU (several times faster than real time).
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path


def model_root(models_dir: Path, lang: str) -> Path | None:
    """Folder that holds fastpitch/ and hifigan/ for a language, in the layouts the installer and the release zips produce."""
    for candidate in (models_dir / lang, models_dir / f"{lang}_extracted" / lang):
        if (candidate / "fastpitch" / "best_model.pth").is_file() and (candidate / "hifigan" / "best_model.pth").is_file():
            return candidate
    return None


def absolute_config(config_path: Path, root: Path) -> Path:
    """The release configs point at 'models/v1/<lang>/...' on the authors' machine; rewrite those to this install."""
    raw = json.loads(config_path.read_text(encoding="utf-8"))

    def fix(value):
        if isinstance(value, str) and value.startswith("models/v1/"):
            return str(root.parent / value.replace("models/v1/", "", 1)).replace("\\", "/")
        if isinstance(value, dict):
            return {key: fix(item) for key, item in value.items()}
        if isinstance(value, list):
            return [fix(item) for item in value]
        return value

    target = config_path.with_name("config_abs.json")
    target.write_text(json.dumps(fix(raw), ensure_ascii=False), encoding="utf-8")
    return target


def silence_edges(samples, sample_rate: int, threshold: float = 0.01) -> tuple[float, float]:
    """Seconds of near-silence before the first and after the last audible sample."""
    loud = [i for i, value in enumerate(samples) if abs(value) > threshold]
    if not loud:
        return 0.0, 0.0
    return loud[0] / sample_rate, (len(samples) - 1 - loud[-1]) / sample_rate


class Voices:
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.loaded = {}

    def get(self, lang: str):
        if lang not in self.loaded:
            from TTS.utils.synthesizer import Synthesizer

            root = model_root(self.models_dir, lang)
            if root is None:
                raise FileNotFoundError(f"no Indic-TTS voice for '{lang}' under {self.models_dir}")
            print(f"Loading Indic-TTS voice '{lang}' from {root}", file=sys.stderr, flush=True)
            self.loaded[lang] = Synthesizer(
                tts_checkpoint=str(root / "fastpitch" / "best_model.pth"),
                tts_config_path=str(absolute_config(root / "fastpitch" / "config.json", root)),
                tts_speakers_file=str(root / "fastpitch" / "speakers.pth"),
                vocoder_checkpoint=str(root / "hifigan" / "best_model.pth"),
                vocoder_config=str(root / "hifigan" / "config.json"), use_cuda=False)
        return self.loaded[lang]

    def speak(self, request: dict) -> dict:
        synthesizer = self.get(request["lang"])
        samples = synthesizer.tts(request["text"], speaker_name=request.get("speaker", "female"))
        out = Path(request["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        synthesizer.save_wav(samples, str(out))
        with wave.open(str(out), "rb") as handle:
            rate, frames = handle.getframerate(), handle.getnframes()
        lead, tail = silence_edges([float(x) for x in samples], rate)
        return {"ok": True, "seconds": frames / rate, "lead": round(lead, 3), "tail": round(tail, 3), "sample_rate": rate}


def serve(voices, stdin, stdout) -> None:
    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "quit":
                break
            answer = voices.speak(request)
        except Exception as error:      # one bad sentence must not kill the worker
            answer = {"ok": False, "error": f"{type(error).__name__}: {error}"}
        stdout.write(json.dumps(answer, ensure_ascii=False) + "\n")
        stdout.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models-dir", required=True)
    arguments = parser.parse_args()
    sys.stdin.reconfigure(encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    serve(Voices(Path(arguments.models_dir)), sys.stdin, sys.stdout)
