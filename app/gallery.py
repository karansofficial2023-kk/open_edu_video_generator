"""Small pictures for a "uses" list on the teacher's board (paints, varnishes, perfumes ...), like the thumbnail rows of good classroom
videos. Each item named in the storyboard's `gallery_items` gets one text-free picture (FLUX), checked by OCR (small labels are painted
out) and by the vision model ("does this show <item>?"). An item whose picture is not approved stays a plain chip - a wrong thumbnail
is worse than none. Pictures are cached per item name in `<output>/gallery/`, so a re-run does not draw them again.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

from PIL import Image

from . import ocr, progress, vision
from .config import AppConfig
from .resilience import atomic_write_text, read_json_or_none
from .schema import Storyboard

MAX_ITEMS = 4
_ABSTRACT = re.compile(r"(?i)\b(process|method|measurement|determination|calculation|analysis|study|theory|law|principle|concept|"
                       r"property|properties|value|ratio|effect|industry|industries|application|applications)\b")


def slug(item: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", item.lower()).strip("_")[:48] or "item"


def drawable(item: str) -> bool:
    """A thing a camera can show: one to four words, no abstract idea ("measurement of conductivity" stays a chip)."""
    words = item.split()
    return 1 <= len(words) <= 4 and not _ABSTRACT.search(item) and not re.search(r"\d|=", item)


def thumbnail_for(output_dir: Path, item: str) -> str | None:
    index = read_json_or_none(Path(output_dir) / "gallery" / "index.json") or {}
    entry = index.get(slug(item)) or {}
    path = entry.get("path")
    return path if entry.get("approved") and path and Path(path).is_file() else None


def items_needed(board: Storyboard) -> list[str]:
    from .board import layout_of
    from .production import BOARD_TEMPLATES
    found: list[str] = []
    for _scene, seg in board.all_segments():
        if seg.shot and seg.shot.template in BOARD_TEMPLATES and layout_of(seg) == "gallery":
            for item in [str(i).strip() for i in seg.gallery_items[:6]][:MAX_ITEMS]:
                if drawable(item) and item.lower() not in {f.lower() for f in found}:
                    found.append(item)
    return found


class GalleryProducer:
    def __init__(self, board: Storyboard, output_dir: Path, config: AppConfig) -> None:
        self.board, self.config = board, config
        self.dir = Path(output_dir) / "gallery"
        self.index_path = self.dir / "index.json"

    def run(self) -> list[dict]:
        items = items_needed(self.board)
        if not items or not self.config.production.gallery_thumbnails:
            return []
        self.dir.mkdir(parents=True, exist_ok=True)
        index = read_json_or_none(self.index_path) or {}
        todo = [i for i in items if slug(i) not in index]
        if not todo:
            return []
        from .assets import NEGATIVE, STYLE_SUFFIX, finish_image
        from .comfyui_client import ComfyUIClient
        client = ComfyUIClient(self.config)
        p = self.config.production
        issues = []
        for n, item in enumerate(todo, start=1):
            progress.note(f"gallery picture {n}/{len(todo)}: {item}")
            entry = {"item": item, "approved": False}
            for attempt in range(2):
                path = self.dir / f"{slug(item)}.png"
                seed = random.Random(f"{item}:{attempt}").randint(1, 2**31 - 1)
                prompt = (f"A clear photograph of {item}, a typical real example, centered on a plain softly lit background, "
                          f"no packaging text, no labels, no writing. {STYLE_SUFFIX}")
                try:
                    raw = client.generate_still(prompt, NEGATIVE, self.dir / f"{slug(item)}_raw.png", seed, p.image_workflow_path,
                                                p.image_checkpoint, p.image_width, p.image_height, p.image_steps)
                    with Image.open(raw) as image:
                        finish_image(image).save(path)
                    Path(raw).unlink(missing_ok=True)
                except Exception as exc:             # ComfyUI down or the workflow failed: this item stays a plain chip
                    entry["error"] = str(exc)[:160]
                    break
                words = ocr.stray_text(path)
                if ocr.has_stray_text(words) and not ocr.erase_text(path):
                    entry["rejected"] = f"lettering: {', '.join(words[:3])}"
                    continue
                if self._shows(path, item):
                    entry.update(approved=True, path=str(path))
                    entry.pop("rejected", None)
                    break
                entry["rejected"] = "the picture does not clearly show the item"
            index[slug(item)] = entry
            atomic_write_text(self.index_path, json.dumps(index, indent=2, ensure_ascii=False))
            if not entry["approved"]:
                issues.append({"shot_id": "gallery", "field": "gallery_items", "severity": "info", "renderer": "gallery",
                               "reason": f"no picture for {item!r} ({entry.get('rejected') or entry.get('error', 'not generated')}); shown as a chip",
                               "repair": "optional: supply a picture"})
        vision.unload(self.config)
        return issues

    def _shows(self, path: Path, item: str) -> bool:
        try:
            answer = vision.ask(path, f"Does this photograph clearly and correctly show: {item}? A viewer must recognise it at a glance. "
                                      'Return JSON {"shows": true or false, "reason": "short"}.', self.config, purpose="gallery")
        except RuntimeError:
            return False                                 # unreviewed thumbnails are never shown
        return answer.get("shows") is True
