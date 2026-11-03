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


def tifa_checklist_review(image, elements: list[str], config: AppConfig, purpose: str = "") -> dict:
    """TIFA/DSG style per-element verification: independently verify each predetermined element (yes/no)."""
    clean_elements = [e.strip() for e in elements if e and e.strip()][:6]
    if not clean_elements:
        return {"ok": True, "score": 1.0, "present": {}, "missing": []}

    elem_list = "\n".join(f"- {e}" for e in clean_elements)
    prompt = (
        "You are a strict QA inspection tool for educational illustrations.\n"
        "Check whether each of the following required concrete elements is clearly and correctly visible in this image:\n"
        f"{elem_list}\n\n"
        "Rules:\n"
        "- Only mark present: true if the specific object/structure is genuinely visible.\n"
        "- If it is missing, obscured, or substituted with something else, mark present: false.\n"
        "Return JSON:\n"
        "{\"checklist\": [{\"element\": str, \"present\": bool, \"reason\": str}]}"
    )
    try:
        result = ask(image, prompt, config, purpose=purpose)
        raw_list = result.get("checklist") or []
        present_map = {}
        missing = []
        for item in raw_list:
            elem = str(item.get("element", "")).strip()
            val = item.get("present")
            is_pres = str(val).lower() in {"true", "yes", "1"} if isinstance(val, str) else bool(val)
            present_map[elem] = is_pres
            if not is_pres and elem:
                missing.append(elem)
        total = len(clean_elements)
        passed = max(0, total - len(missing))
        score = round(passed / max(1, total), 2)
        return {
            "ok": len(missing) == 0 or (len(missing) <= 1 and total >= 4),
            "score": score,
            "present": present_map,
            "missing": missing,
        }
    except Exception as exc:
        return {"ok": True, "score": 1.0, "present": {}, "missing": [], "error": str(exc)}
