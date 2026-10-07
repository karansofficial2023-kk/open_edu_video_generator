"""Labels for photographs whose contract asked for none.

A biology lesson is far easier to follow when "stigma" and "anther" are written on the flower. The storyboard planner often types every
photograph as plain, so this step proposes labels itself: a language model picks up to three physical parts that the NARRATION names
(each word must really occur in the narration - nothing is invented), and the existing label engine then has the vision model find each part
in the accepted picture and verify it with an independent crop. A part that cannot be verified is simply left out; no guessed arrows.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import requests

from . import vision
from .config import AppConfig
from .schema import Segment, Storyboard

MAX_LABELS = 3
_VERSION = "autolabels2"          # bump when the proposal logic changes: stored proposals are keyed by it
# A label names a thing. Single adjectives / quantifiers lifted out of "small, inconspicuous flowers" are not things.
_ADJECTIVE_SUFFIXES = ("ous", "ful", "ive", "less", "able", "ible", "ical")
_MODIFIERS = set("small large big tiny huge long short tall wide narrow thin thick red yellow green blue white black pink orange purple brown "
                 "many few several some more most other another same different various typical average sticky bright dark pale young old "
                 "new first second third last early late simple common main".split())
_STOP = set("the a an of and or to in on at for from with by as is are was were be this that these those its their it which "
            "process type types example examples part parts structure structures way kind form".split())


def eligible(segment: Segment) -> bool:
    """Only an accepted photograph with no labels, no equations and real narration is a candidate."""
    shot = segment.shot
    return bool(shot and shot.template == "photo" and shot.asset_path and not shot.motion_asset and not segment.animate and not segment.labels and not segment.formula_lines
                and not segment.continuation and len((segment.narration or "").split()) >= 5)


def valid_labels(candidates, narration: str) -> list[str]:
    """Keep only short noun phrases whose every content word occurs in the narration; drop duplicates, cap at three."""
    text = (narration or "").lower()
    words = set(re.findall(r"[^\W\d_]+", text))
    stems = {w[:5] for w in words}
    kept: list[str] = []
    for item in candidates or []:
        label = re.sub(r"\s+", " ", str(item)).strip(" .,:;-")
        parts = re.findall(r"[^\W\d_]+", label.lower())
        content = [p for p in parts if p not in _STOP]
        if not (1 <= len(parts) <= 3) or not content or len(label) > 28:
            continue
        if len(content) == 1 and (content[0] in _MODIFIERS or content[0].endswith(_ADJECTIVE_SUFFIXES)):
            continue                                                         # a lone adjective is not a part
        if not all(p in words or p[:5] in stems for p in content):          # every word must come from the narration
            continue
        if any(label.lower() == k.lower() or label.lower() in k.lower() or k.lower() in label.lower() for k in kept):
            continue
        kept.append(label[:1].upper() + label[1:])
        if len(kept) == MAX_LABELS:
            break
    return kept


def _ask_model(narration: str, config: AppConfig) -> list[str]:
    response = requests.post(f"{config.ollama_url}/api/generate", json={
        "model": config.production.director_model, "stream": False, "format": "json", "think": False, "keep_alive": "2m",
        "system": "You choose on-screen labels for a photograph in an educational video. From the narration, list up to 3 concrete PHYSICAL PARTS "
                  "or structures that are explicitly named in it and would be visible in a close-up photograph of the subject (for example the "
                  "parts of a flower, an organ, a piece of apparatus). Copy each name exactly as written in the narration, 1-3 words. Give NOUNS "
                  "only: never an adjective or adverb (not 'small', 'inconspicuous', 'sticky'). Do NOT "
                  "list processes, ideas, quantities, places or anything not named in the narration. If nothing qualifies, "
                  "return an empty list. Return JSON {\"labels\": [\"...\"]}.",
        "prompt": narration, "options": {"temperature": 0, "num_predict": 120}}, timeout=300)
    response.raise_for_status()
    return list(vision.extract_json(response.json()["response"]).get("labels", []))


def propose(segment: Segment, cache_dir: Path, config: AppConfig) -> list[str]:
    """Stored per shot, keyed by narration: a re-run shows the same labels."""
    narration = segment.narration or ""
    key = hashlib.sha1(f"{_VERSION}|{narration}".encode()).hexdigest()[:16]
    path = cache_dir / f"autolabels_{(segment.shot_id or 'x').replace('.', '_')}.json"
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved.get("key") == key:
                return list(saved.get("labels", []))
        except (ValueError, OSError):
            pass
    try:
        labels = valid_labels(_ask_model(narration, config), narration)
    except (requests.RequestException, ValueError, KeyError, AttributeError, TypeError):
        return []                                  # the model is unavailable: no labels this time, and nothing is stored
    path.write_text(json.dumps({"key": key, "narration": narration, "labels": labels}, ensure_ascii=False, indent=1), encoding="utf-8")
    return labels


def apply(board: Storyboard, output_dir: Path, config: AppConfig) -> int:
    """Add proposed labels to eligible photo shots. Never raises: a lesson must render with or without them. Returns the number of shots labelled."""
    if not getattr(config.production, "auto_labels", False):
        return 0
    cache_dir = Path(output_dir) / "assets"
    cache_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for _scene, segment in board.all_segments():
        try:
            if not eligible(segment):
                continue
            labels = propose(segment, cache_dir, config)
            if labels:
                segment.labels = labels
                segment.label_placement = list(labels)         # target description = the label itself; the vision model finds and verifies it
                segment.labels_auto = True
                count += 1
        except Exception as exc:                              # noqa: BLE001 - labels are an enhancement, never a reason to fail a render
            print(f"Auto labels skipped for {segment.shot_id}: {exc}", flush=True)
    vision.unload(config)
    return count
