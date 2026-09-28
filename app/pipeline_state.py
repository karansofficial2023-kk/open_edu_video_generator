from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def stable_signature(*values: Any) -> str:
    encoded = json.dumps(values, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_signature(path: str | Path) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class PipelineState:
    def __init__(self, output_dir: str | Path) -> None:
        self.path = Path(output_dir) / "pipeline_state.json"
        self.data: dict[str, Any] = {"version": 1, "stages": {}}
        if self.path.exists():
            try:
                loaded = json.loads(self.path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    self.data.update(loaded)
                    self.data.setdefault("stages", {})
            except (OSError, json.JSONDecodeError):
                pass

    def reusable(self, stage: str, key: str, signature: str, outputs: list[str | Path]) -> dict | None:
        record = self.data.get("stages", {}).get(stage, {}).get(key)
        if not isinstance(record, dict) or record.get("status") != "approved":
            return None
        if record.get("signature") != signature:
            return None
        if not all(Path(path).is_file() for path in outputs):
            return None
        return record

    def approved(self, stage: str, key: str, signature: str, outputs: list[str | Path], **details: Any) -> None:
        record = {
            "status": "approved",
            "signature": signature,
            "outputs": [str(Path(path).resolve()) for path in outputs],
            **details,
        }
        self.data.setdefault("stages", {}).setdefault(stage, {})[key] = record
        self._save()

    def failed(self, stage: str, key: str, signature: str, error: str) -> None:
        self.data.setdefault("stages", {}).setdefault(stage, {})[key] = {
            "status": "failed",
            "signature": signature,
            "error": error,
        }
        self._save()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.data, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(self.path)
