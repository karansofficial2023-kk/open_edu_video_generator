"""Make the English voice (formulas, labels) sound as close as possible to the Hindi narrator.

The strict formula rule needs a second, English voice. Piper has no single speaker for both languages, so the two voices are
brought together instead: the English part is pitch-shifted to the Hindi voice's pitch and level-matched to the Hindi speech
of the same line, and every part gets a tiny fade so joins do not click.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

_REFERENCE_HI = "सल्फ्यूरिक अम्ल एक प्रबल अम्ल है और यह जल में घुलकर आयन बनाता है"
_REFERENCE_EN = "H two S O four, N A O H, C double bond O, S O four"
MAX_PITCH_RATIO = 1.20           # never bend the voice further than this either way
MAX_GAIN_DB = 6.0
FADE_SECONDS = 0.010


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as handle:
        samples = np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        return samples, handle.getframerate()


def median_f0(samples: np.ndarray, sample_rate: int) -> float:
    """Median pitch of the voiced, active frames (plain autocorrelation, enough to compare two TTS voices)."""
    frame = int(0.04 * sample_rate)
    low, high = int(sample_rate / 400), int(sample_rate / 70)
    pitches = []
    for start in range(0, len(samples) - frame, frame):
        chunk = samples[start:start + frame]
        chunk = chunk - chunk.mean()
        if float(np.sqrt((chunk * chunk).mean())) < 0.01:
            continue
        autocorrelation = np.correlate(chunk, chunk, "full")[frame - 1:]
        if autocorrelation[0] <= 0:
            continue
        lag = int(np.argmax(autocorrelation[low:high])) + low
        if autocorrelation[lag] / autocorrelation[0] > 0.35:
            pitches.append(sample_rate / lag)
    return float(np.median(pitches)) if pitches else 0.0


def active_rms(samples: np.ndarray) -> float:
    """RMS of the non-silent part, so pauses do not change the loudness estimate."""
    active = samples[np.abs(samples) > 0.01]
    return float(np.sqrt((active * active).mean())) if active.size else 0.0


def gain_db(reference_rms: float, part_rms: float) -> float:
    if reference_rms <= 0 or part_rms <= 0:
        return 0.0
    return float(np.clip(20.0 * np.log10(reference_rms / part_rms), -MAX_GAIN_DB, MAX_GAIN_DB))


def pitch_ratio(hindi_f0: float, english_f0: float) -> float:
    """Factor that moves the English voice to the Hindi voice's pitch (1.0 when unknown or already within 2%)."""
    if hindi_f0 <= 0 or english_f0 <= 0:
        return 1.0
    ratio = float(np.clip(hindi_f0 / english_f0, 1.0 / MAX_PITCH_RATIO, MAX_PITCH_RATIO))
    return 1.0 if abs(ratio - 1.0) < 0.02 else ratio


def english_pitch_ratio(piper_path: str, hindi_model: str, english_model: str, english_config: str = "") -> float:
    """Measured once per voice pair with two short reference phrases, then cached on disk."""
    key = hashlib.sha256("|".join(str(Path(m).resolve()) + str(Path(m).stat().st_mtime_ns)
                                  for m in (hindi_model, english_model)).encode("utf-8")).hexdigest()[:16]
    cache = Path(tempfile.gettempdir()) / "dub_voice_match.json"
    try:
        known = json.loads(cache.read_text(encoding="utf-8")) if cache.is_file() else {}
    except (OSError, ValueError):
        known = {}
    if key in known:
        return float(known[key])
    pitches = []
    with tempfile.TemporaryDirectory(prefix="voice_match_") as folder:
        for name, model, text, conf in (("hi", hindi_model, _REFERENCE_HI, ""), ("en", english_model, _REFERENCE_EN, english_config)):
            out = Path(folder) / f"{name}.wav"
            command = [piper_path, "--model", model, "--output_file", str(out)] + (["--config", conf] if conf else [])
            subprocess.run(command, input=text, text=True, encoding="utf-8", capture_output=True, timeout=300, check=False)
            if not out.is_file() or out.stat().st_size <= 44:
                return 1.0
            samples, rate = read_wav(out)
            pitches.append(median_f0(samples, rate))
    ratio = pitch_ratio(pitches[0], pitches[1])
    known[key] = ratio
    try:
        cache.write_text(json.dumps(known), encoding="utf-8")
    except OSError:
        pass
    return ratio


def match_filter(ratio: float, sample_rate: int, db: float) -> str:
    """ffmpeg audio filter: shift pitch by `ratio` without changing tempo, then apply the level gain."""
    filters = []
    if abs(ratio - 1.0) >= 0.02:
        filters.append(f"asetrate={sample_rate * ratio:.2f},aresample={sample_rate},atempo={1.0 / ratio:.6f}")
    if abs(db) >= 0.3:
        filters.append(f"volume={db:.2f}dB")
    return ",".join(filters)
