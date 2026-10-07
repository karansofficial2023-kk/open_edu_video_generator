"""Open-source narration for Indian languages (AI4Bharat Indic-TTS, MIT licence) as an alternative to the cloud Edge voices.

The voice engine needs its own Python environment (PyTorch + Coqui TTS), so it runs as a worker process that this module
talks to over a pipe (see tools/indic_tts_worker.py and docs/OPEN_VOICES.md). The engine reads native script only, so digits are
spoken as words, Latin terms can be replaced through a pronunciation table, and word timings for captions are derived from the
audio (these voices report none).
"""
from __future__ import annotations

import json
import re
import subprocess
import unicodedata
from pathlib import Path

from .config import AppConfig
from .schema import Caption

LANGUAGES = {"as", "bn", "brx", "gu", "hi", "kn", "ml", "mni", "mr", "or", "pa", "raj", "ta", "te"}

# Tamil number words. Compound numbers need sandhi rules, so only these exact forms are spoken as words; other numbers are read digit by digit.
_TAMIL_WORDS = {
    0: "பூஜ்ஜியம்", 1: "ஒன்று", 2: "இரண்டு", 3: "மூன்று", 4: "நான்கு", 5: "ஐந்து", 6: "ஆறு", 7: "ஏழு", 8: "எட்டு", 9: "ஒன்பது",
    10: "பத்து", 11: "பதினொன்று", 12: "பன்னிரண்டு", 13: "பதிமூன்று", 14: "பதினான்கு", 15: "பதினைந்து", 16: "பதினாறு",
    17: "பதினேழு", 18: "பதினெட்டு", 19: "பத்தொன்பது", 20: "இருபது", 30: "முப்பது", 40: "நாற்பது", 50: "ஐம்பது", 60: "அறுபது",
    70: "எழுபது", 80: "எண்பது", 90: "தொண்ணூறு", 100: "நூறு"}
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_POINT = {"ta": "புள்ளி", "te": "బిందువు", "kn": "ಬಿಂದು", "bn": "দশমিক"}


def _spell(number: str, lang: str) -> str:
    if "." in number:
        whole, _, fraction = number.partition(".")
        return f"{_spell(whole, lang)} {_POINT.get(lang, '.')} " + " ".join(_spell(d, lang) for d in fraction)
    value = int(number)
    if lang == "ta":
        if value in _TAMIL_WORDS:
            return _TAMIL_WORDS[value]
        return " ".join(_TAMIL_WORDS[int(d)] for d in number)           # digit by digit
    try:
        from num2words import num2words

        return num2words(value, lang=lang)
    except Exception:
        return " ".join(number)                                          # unsupported language: digits are left to the voice


def expand_numbers(text: str, lang: str) -> str:
    """Digits become words in the lesson language; without this the character-based voice would skip them silently."""
    return _NUMBER.sub(lambda match: _spell(match.group(), lang), text)


def apply_pronunciations(text: str, table: dict[str, str] | None) -> str:
    """Replaces terms (longest first) with how they should be spoken, e.g. {'CO2': 'கார்பன் டை ஆக்சைடு'}."""
    for term in sorted(table or {}, key=len, reverse=True):
        text = text.replace(term, table[term])
    return text


def prepare_text(text: str, lang: str, pronunciations: dict[str, str] | None = None) -> str:
    return re.sub(r"\s+", " ", expand_numbers(apply_pronunciations(text, pronunciations), lang)).strip()


def _speech_weight(word: str, lang: str | None, table: dict[str, str] | None) -> float:
    """Speaking effort of one caption word: letters AND combining marks (vowel signs, viramas) of what the voice really says.
    Digits and mapped terms count as their spoken form; Latin letters the voice cannot read count as almost silent."""
    spoken = prepare_text(word, lang, table) if lang else word
    marks = [c for c in spoken if unicodedata.category(c)[0] in "LM"]
    if lang and lang != "en" and marks and all(c.isascii() for c in marks):
        return 1.0
    return float(max(1, len(marks)))


def distribute_captions(text: str, duration: float, lead: float = 0.0, tail: float = 0.0, lang: str | None = None,
                        pronunciations: dict[str, str] | None = None) -> list[Caption]:
    """Word timings from the audio length: speaking time is shared by spoken length, with a pause after each phrase mark."""
    words = text.split()
    if not words:
        return []
    start, end = lead, max(lead + 0.05, duration - tail)
    weights = []
    for word in words:
        weight = _speech_weight(word, lang, pronunciations)
        if word[-1] in ",;:":
            weight += 2.0
        elif word[-1] in ".!?।۔":
            weight += 4.0
        weights.append(weight)
    scale = (end - start) / sum(weights)
    captions, cursor = [], start
    for word, weight in zip(words, weights):
        spoken = scale * (weight - (2.0 if word[-1] in ",;:" else 4.0 if word[-1] in ".!?।۔" else 0.0))
        captions.append(Caption(text=word, start=round(cursor, 3), end=round(cursor + max(spoken, 0.04), 3)))
        cursor += scale * weight
    return captions


def available(lang: str, config: AppConfig) -> bool:
    """True when an open voice for the language is installed and the worker's environment exists."""
    settings = config.production.open_tts
    models = Path(settings.models_dir)
    return (lang in LANGUAGES and Path(settings.python).is_file()
            and any((models / sub / "fastpitch" / "best_model.pth").is_file() for sub in (lang, f"{lang}_extracted/{lang}")))


class OpenVoice:
    """Long-lived worker process; the voice models load once per language."""

    def __init__(self, config: AppConfig):
        settings = config.production.open_tts
        worker = Path(__file__).resolve().parents[1] / "tools" / "indic_tts_worker.py"
        self.process = subprocess.Popen([settings.python, str(worker), "--models-dir", settings.models_dir], stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, text=True, encoding="utf-8", bufsize=1)
        self.speaker = settings.speaker

    def speak(self, lang: str, text: str, out: Path) -> dict:
        request = {"lang": lang, "speaker": self.speaker, "text": text, "out": str(out)}
        self.process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        self.process.stdin.flush()
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("the open voice worker ended unexpectedly")
        answer = json.loads(line)
        if not answer.get("ok"):
            raise RuntimeError(f"open voice failed: {answer.get('error')}")
        return answer

    def close(self) -> None:
        try:
            self.process.stdin.write(json.dumps({"command": "quit"}) + "\n")
            self.process.stdin.flush()
            self.process.wait(timeout=30)
        except Exception:
            self.process.kill()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
