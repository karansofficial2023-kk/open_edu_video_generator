"""Best-effort perceived voice pitch profiling for selecting a matching dub voice.

This estimates acoustic presentation from the source narration. It does not infer a
speaker's identity or gender identity; ambiguous/mixed audio is reported as unknown.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np

from .config import AppConfig
from .renderer import _tool_path


def source_voice_profile(source_video: str | Path, config: AppConfig, temp_parent: str | Path | None = None) -> dict:
    source = Path(source_video)
    override_path = source.with_suffix(".voice.json")
    override_gender = None
    if override_path.is_file():
        try:
            override = json.loads(override_path.read_text(encoding="utf-8-sig"))
            gender = str(override.get("gender", "")).strip().lower()
            if gender in {"male", "female"}:
                override_gender = gender
        except (OSError, json.JSONDecodeError, AttributeError):
            pass

    sample_rate = 16000
    try:
        with tempfile.TemporaryDirectory(prefix="source_voice_", dir=str(temp_parent) if temp_parent else None) as temp_dir:
            wav_path = Path(temp_dir) / "sample.wav"
            command = [
                _tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-i", str(source),
                "-map", "0:a:0", "-vn", "-ac", "1", "-ar", str(sample_rate), "-t", "300",
                "-c:a", "pcm_s16le", str(wav_path),
            ]
            result = subprocess.run(command, capture_output=True, text=True, timeout=180, check=False)
            if result.returncode != 0 or not wav_path.is_file():
                return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate",
                        "reason": (result.stderr or "could not extract source audio")[-500:]}
            with wave.open(str(wav_path), "rb") as wav:
                if wav.getnchannels() != 1 or wav.getsampwidth() != 2 or wav.getframerate() != sample_rate:
                    return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate",
                            "reason": "unexpected audio sample format"}
                samples = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
        try:
            from faster_whisper.vad import VadOptions, get_speech_timestamps
            speech_chunks = get_speech_timestamps(
                samples, VadOptions(threshold=0.5, min_speech_duration_ms=400,
                                    min_silence_duration_ms=300, speech_pad_ms=80),
                sampling_rate=sample_rate,
            )
            speech_ranges = [(chunk["start"] / sample_rate, chunk["end"] / sample_rate) for chunk in speech_chunks]
        except Exception as exc:
            speech_ranges = []
            vad_error = str(exc)
        else:
            vad_error = ""
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate", "reason": str(exc)}

    frame_length = 640  # 40 ms
    hop = 640
    if samples.size < frame_length:
        return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate",
                "reason": "source audio sample is too short"}
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_length)[::hop]
    frame_times = np.arange(frames.shape[0], dtype=np.float32) * hop / sample_rate
    # Ignore the opening 20 seconds when estimating voice; intro music skewed the previous pitch estimate.
    # Use Silero VAD speech chunks when available so instrumental music and pauses do not vote on the voice.
    profile_ranges = [(start, end) for start, end in speech_ranges if end > 20.0 and end - max(start, 20.0) >= 0.4]
    if not profile_ranges:
        profile_ranges = [(start, end) for start, end in speech_ranges if end - start >= 0.4]
    if speech_ranges:
        speech_mask = np.zeros(frame_times.shape, dtype=bool)
        for start, end in profile_ranges:
            speech_mask |= (frame_times >= max(start, 20.0) if start < 20.0 else frame_times >= start) & (frame_times < end)
        frames = frames[speech_mask]
        frame_times = frame_times[speech_mask]
    else:
        # Keep a conservative fallback for installations without the bundled VAD runtime.
        frames = frames[frame_times >= 20.0]
        frame_times = frame_times[frame_times >= 20.0]
    if frames.shape[0] > 2250:  # at most 90 seconds of 40 ms samples
        frames = frames[:2250]
    frames = frames - frames.mean(axis=1, keepdims=True)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    active = rms >= max(0.003, float(rms.max()) * 0.025)
    frames = frames[active]
    if frames.shape[0] < 20:
        return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate",
                "reason": "too few voiced/active speech frames after intro/music filtering",
                "voiced_frames": int(frames.shape[0]), "speech_start_seconds": round(speech_ranges[0][0], 2) if speech_ranges else None,
                "vad_error": vad_error or None}

    frames = frames * np.hamming(frame_length).astype(np.float32)
    n_fft = 2048
    spectrum = np.fft.rfft(frames, n=n_fft, axis=1)
    autocorrelation = np.fft.irfft(spectrum * spectrum.conj(), n=n_fft, axis=1).real
    zero_lag = np.maximum(autocorrelation[:, 0], 1e-9)
    min_lag = int(sample_rate / 350.0)
    max_lag = int(sample_rate / 65.0)
    search = autocorrelation[:, min_lag:max_lag + 1] / zero_lag[:, None]
    best_lags = np.argmax(search, axis=1) + min_lag
    periodicity = search[np.arange(search.shape[0]), best_lags - min_lag]
    pitches = sample_rate / best_lags[periodicity >= 0.30]
    if pitches.size < 20:
        return {"gender": "unknown", "confidence": 0.0, "method": "pitch_estimate",
                "reason": "source has too few stable pitched speech frames", "voiced_frames": int(pitches.size),
                "speech_start_seconds": round(speech_ranges[0][0], 2) if speech_ranges else None,
                "vad_error": vad_error or None}

    median_hz = float(np.median(pitches))
    # With music excluded, use a broad male range; higher-pitched male narration
    # around 165–185 Hz was previously left unknown and silently defaulted female.
    if median_hz <= 185.0:
        gender = "male"
        confidence = min(0.95, 0.65 + (185.0 - median_hz) / 100.0)
    elif median_hz >= 190.0:
        gender = "female"
        confidence = min(0.95, 0.65 + (median_hz - 190.0) / 100.0)
    else:
        gender = "unknown"
        confidence = 0.0
    return {"gender": override_gender or gender, "confidence": 1.0 if override_gender else round(confidence, 3),
            "method": "sidecar_override" if override_gender else "vad_filtered_source_voice_pitch",
            "median_f0_hz": round(median_hz, 1), "voiced_frames": int(pitches.size),
            "sample_seconds": round(min(300.0, samples.size / sample_rate), 1),
            "speech_start_seconds": round(speech_ranges[0][0], 2) if speech_ranges else None,
            "profile_speech_chunks": len(profile_ranges), "vad_error": vad_error or None,
            "sidecar": str(override_path) if override_gender else None}
