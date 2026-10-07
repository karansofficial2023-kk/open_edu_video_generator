"""Qwen Vision (Ollama) client: constrained JSON, malformed-output repair, raw-response logging."""
from __future__ import annotations

import base64
import io
import json
import re
from pathlib import Path

import requests
from PIL import Image

from .config import AppConfig

LOG: list[dict] = []   # raw responses kept for the QA manifest


def _encode(image: Image.Image | str | Path, max_side: int = 1024) -> str:
    if not isinstance(image, Image.Image):
        image = Image.open(image)
    image = image.convert("RGB")
    image.thumbnail((max_side, max_side))
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=90)
    return base64.b64encode(buffer.getvalue()).decode()


def extract_json(text: str) -> dict:
    """Parse a JSON object from raw model output (fenced, prefixed, or trailing-comma variants)."""
    text = (text or "").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in vision response")
        candidate = text[start:end + 1]
    candidate = re.sub(r",\s*([}\]])", r"\1", candidate)
    return json.loads(candidate)


def ask(image, prompt: str, config: AppConfig, purpose: str = "", keep_alive: str | int = "2m") -> dict:
    """Ask the vision model about one image. Retries once with a stricter instruction, then falls back."""
    models = [config.production.vision_model, config.production.vision_fallback_model]
    last_error = None
    for model in [m for m in models if m]:
        for attempt, suffix in enumerate(["", "\nReturn ONLY one valid JSON object. No prose, no markdown."]):
            try:
                response = requests.post(
                    f"{config.ollama_url}/api/chat",
                    json={"model": model, "stream": False, "format": "json", "keep_alive": keep_alive,
                          "options": {"temperature": 0.05, "num_predict": 900},
                          "messages": [{"role": "user", "content": prompt + suffix, "images": [_encode(image)]}]},
                    timeout=600,
                )
                response.raise_for_status()
                raw = response.json()["message"]["content"]
                LOG.append({"purpose": purpose, "model": model, "attempt": attempt, "raw": raw})
                return extract_json(raw)
            except (ValueError, requests.RequestException, KeyError) as exc:
                last_error = exc
    raise RuntimeError(f"Vision model unavailable or returned invalid JSON ({purpose}): {last_error}")


def unload(config: AppConfig) -> None:
    """Release VRAM so ComfyUI can use the GPU (12 GB profile: one heavy model at a time)."""
    for model in {config.production.vision_model, config.production.vision_fallback_model, config.production.director_model}:
        if not model:
            continue
        try:
            requests.post(f"{config.ollama_url}/api/generate", json={"model": model, "keep_alive": 0}, timeout=30)
        except requests.RequestException:
            pass


def available(config: AppConfig) -> bool:
    try:
        tags = requests.get(f"{config.ollama_url}/api/tags", timeout=5).json()
        names = {m["name"] for m in tags.get("models", [])}
        return config.production.vision_model in names or config.production.vision_fallback_model in names
    except (requests.RequestException, ValueError):
        return False
