from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import wave
from pathlib import Path

import numpy as np

from .config import AppConfig
from .schema import Storyboard
from .spoken_forms import Span, is_formula_token, split_spans

TTS_VERSION = "v2.2-subject-formula-speech"
_PART_PAUSE_SECONDS = 0.06


def _voice_files(config: AppConfig, gender: str, english: bool) -> tuple[Path | None, Path | None]:
    if english:
        model = (config.piper_english_gender_models.get(gender) or config.piper_english_model
                 or os.environ.get("PIPER_EN_MODEL_" + gender.upper(), "") or os.environ.get("PIPER_EN_MODEL", ""))
        conf = (config.piper_english_gender_configs.get(gender) or config.piper_english_config
                or os.environ.get("PIPER_EN_CONFIG_" + gender.upper(), "") or os.environ.get("PIPER_EN_CONFIG", ""))
        if model and not conf and Path(model + ".json").is_file():
            conf = model + ".json"
    else:
        model, conf = config.piper_model, config.piper_config
    return (Path(model) if model else None), (Path(conf) if conf else None)


def _pronunciations_without_formulas(pronunciations: dict[str, str]) -> dict[str, str]:
    """STRICT RULE: a formula is never replaced by a Hindi spoken name."""
    return {key: value for key, value in pronunciations.items() if not is_formula_token(key)}


def _formula_speech_mode(storyboard: Storyboard, config: AppConfig) -> str:
    """Resolve the YAML policy; auto changes only clearly identified mathematics lessons."""
    mode = os.environ.get("FORMULA_SPEECH_MODE", str(config.formula_speech_mode or "auto")).lower()
    if mode not in {"auto", "english_stem", "hindi_math"}:
        raise ValueError("formula_speech_mode must be auto, english_stem, or hindi_math")
    if mode != "auto":
        return mode
    text = " ".join([storyboard.title or ""] + [scene.title or "" for scene in storyboard.scenes] +
                    [segment.heading or "" for _scene, segment in storyboard.all_segments()]).lower()
    math_subject = re.search(r"\b(mathematics|maths?|algebra|geometry|trigonometry|calculus|arithmetic|statistics|"
                             r"probability|quadratic|polynomial|coordinate geometry)\b|"
                             r"गणित|बीजगणित|ज्यामिति|त्रिकोणमिति|कलन|सांख्यिकी|प्रायिकता", text)
    return "hindi_math" if math_subject else "english_stem"


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
    global_pronunciations = _pronunciations_without_formulas(config.production.pronunciations.get(language, {}))
    english_model, english_config = _voice_files(config, voice_gender, english=True)
    formula_speech_mode = _formula_speech_mode(storyboard, config)

    for scene, segment in storyboard.all_segments():
        out = audio_dir / f"scene_{scene.scene_number:02d}_segment_{segment.segment_number:02d}.wav"
        pronunciations = {**global_pronunciations, **_pronunciations_without_formulas(segment.pronunciations or {})}
        narration = segment.narration
        if pronunciations:
            from .tts_indic import apply_pronunciations
            narration = apply_pronunciations(narration, pronunciations)
        spans = split_spans(narration, formula_speech_mode)
        if any(span.lang == "en" for span in spans) and not (english_model and english_model.is_file()):
            raise RuntimeError(
                "Strict dubbing rule: chemical formulas, equations and labels must be spoken in English, but no "
                "English Piper voice is configured. Set piper_english_model (or piper_english_gender_models) in the "
                "config, or the PIPER_EN_MODEL environment variable, to an installed en_US/en_GB .onnx voice.")
        model = Path(voice_config.piper_model)
        model_config = Path(voice_config.piper_config) if voice_config.piper_config else None
        cache_key = hashlib.sha256(json.dumps({
            "tts_version": TTS_VERSION,
            "formula_speech_mode": formula_speech_mode,
            "spans": [[span.lang, span.text] for span in spans],
            "length_scale": config.piper_length_scale, "piper": str(config.piper_path),
            "english_length_scale": config.piper_english_length_scale, "match_voice": config.piper_english_match_voice,
            "model": str(model.resolve()),
            "model_mtime": model.stat().st_mtime_ns if model.is_file() else None,
            "config": str(model_config.resolve()) if model_config else "",
            "config_mtime": model_config.stat().st_mtime_ns if model_config and model_config.is_file() else None,
            "english_model": str(english_model.resolve()) if english_model and any(s.lang == "en" for s in spans) else "",
            "english_mtime": (english_model.stat().st_mtime_ns
                              if english_model and english_model.is_file() and any(s.lang == "en" for s in spans) else None),
        }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        marker = out.with_suffix(".sha256")
        if not (out.is_file() and out.stat().st_size > 44 and marker.is_file()
                and marker.read_text(encoding="ascii").strip() == cache_key):
            marker.unlink(missing_ok=True)
            synthesize_spans(spans, out, voice_config, english_model, english_config)
            marker.write_text(cache_key, encoding="ascii")
        segment.audio_path = str(out)


def _write_silence(path: Path, seconds: float = 0.4, rate: int = 22050) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(rate * seconds))


def synthesize_spans(spans: list[Span], output_path: str | Path, config: AppConfig,
                     english_model: Path | None, english_config: Path | None) -> None:
    """Hindi spans use the Hindi voice; English spans (formulas, equations, labels) use the English voice."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    spans = [span for span in spans if re.search(r"\w", span.text)]
    if not spans:                                   # empty/symbol-only narration -> short silence, never a crash
        _write_silence(output)
        return
    if len(spans) == 1 and spans[0].lang == "hi":
        synthesize_text(spans[0].text, output, config)
        return
    parts: list[Path] = []
    for index, span in enumerate(spans):
        part = output.with_name(f"{output.stem}.part{index:02d}.wav")
        if span.lang == "en":
            synthesize_text(span.text, part, config, model=str(english_model),
                            model_config=str(english_config) if english_config else "",
                            length_scale=config.piper_english_length_scale)
        else:
            synthesize_text(span.text, part, config)
        parts.append(part)
    if config.piper_english_match_voice:
        _match_english_parts(spans, parts, config, english_model, english_config)
    _join_wavs(parts, output, config)
    for part in parts:
        part.unlink(missing_ok=True)


def _match_english_parts(spans: list[Span], parts: list[Path], config: AppConfig,
                         english_model: Path | None, english_config: Path | None) -> None:
    """Pitch-shift the English parts to the Hindi narrator and match their level to the Hindi speech of the same line."""
    from . import voice_match
    from .renderer import _run, _tool_path
    english_indexes = [i for i, span in enumerate(spans) if span.lang == "en"]
    if not english_indexes or english_model is None:
        return
    hindi_levels = [voice_match.active_rms(voice_match.read_wav(parts[i])[0]) for i, span in enumerate(spans) if span.lang == "hi"]
    reference = float(np.mean(hindi_levels)) if hindi_levels else 0.0
    ratio = voice_match.english_pitch_ratio(config.piper_path, config.piper_model, str(english_model),
                                            str(english_config) if english_config else "")
    for index in english_indexes:
        samples, rate = voice_match.read_wav(parts[index])
        chain = voice_match.match_filter(ratio, rate, voice_match.gain_db(reference, voice_match.active_rms(samples)))
        if not chain:
            continue
        matched = parts[index].with_name(parts[index].stem + ".matched.wav")
        _run([_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(parts[index]), "-af", chain,
              "-ar", str(rate), "-ac", "1", "-c:a", "pcm_s16le", str(matched)])
        matched.replace(parts[index])


def _join_wavs(parts: list[Path], output: Path, config: AppConfig) -> None:
    from .renderer import _run, _tool_path
    command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y"]
    for part in parts:
        command.extend(["-i", str(part)])
    filters = [f"[{i}:a]aresample=22050,aformat=channel_layouts=mono,afade=t=in:d=0.01,areverse,afade=t=in:d=0.01,areverse,apad=pad_dur={_PART_PAUSE_SECONDS}[a{i}]"
               for i in range(len(parts))]
    filters.append("".join(f"[a{i}]" for i in range(len(parts))) + f"concat=n={len(parts)}:v=0:a=1[out]")
    command.extend(["-filter_complex", ";".join(filters), "-map", "[out]", "-ar", "22050", "-ac", "1",
                    "-c:a", "pcm_s16le", str(output)])
    _run(command)


def synthesize_text(text: str, output_path: str | Path, config: AppConfig, model: str | None = None,
                    model_config: str | None = None, length_scale: float | None = None) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = re.sub(r"\s+", " ", text).strip()          # one utterance per call; newlines would split it
    command = [
        config.piper_path,
        "--model",
        model or config.piper_model,
        "--output_file",
        str(output),
    ]
    chosen_config = config.piper_config if model_config is None else model_config
    if chosen_config:
        command.extend(["--config", chosen_config])
    scale = config.piper_length_scale if length_scale is None else length_scale
    if abs(scale - 1.0) > 1e-6:
        command.extend(["--length_scale", f"{scale:g}"])
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
    if not output.is_file() or output.stat().st_size <= 44:
        raise RuntimeError(f"Piper produced no audio for {output} (text: {text[:80]!r})")


def wav_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as handle:
        frames = handle.getnframes()
        rate = handle.getframerate()
        return frames / float(rate)
