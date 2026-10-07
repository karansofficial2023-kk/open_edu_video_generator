"""Repair derivation steps that do not follow from the previous equation (symbolic check + local LLM retry).

    python tools/repair_derivations.py <lesson_contract.json> [--model qwen3:14b] [--attempts 3]

For each flagged formula line the model is given the narration, the last good equation and the failure, must return one corrected
line, and the line is accepted only if sympy confirms it follows. Unrepairable steps are left unchanged and reported for SME review.
The contract is rewritten in place (a .bak copy is kept).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.derivation import equivalent, to_equation  # noqa: E402


def propose(narration: str, previous: str, bad: str, model: str, url: str) -> str:
    system = ("You correct one step of an algebraic derivation. Apply exactly ONE algebraic operation to BOTH sides of the previous "
              "equation, as the narration describes. Write both sides in full using plain math (x^2, a/b, sqrt(x)). "
              'Return JSON {"line": "<the corrected equation>"} and nothing else.')
    prompt = json.dumps({"narration": narration, "previous_equation": previous, "rejected_line": bad})
    response = requests.post(f"{url}/api/generate", json={"model": model, "system": system, "prompt": prompt, "stream": False,
                                                         "format": "json", "think": False, "options": {"temperature": 0.1}}, timeout=600)
    return str(json.loads(response.json()["response"]).get("line", "")).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract")
    parser.add_argument("--model", default="qwen3:14b")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--attempts", type=int, default=3)
    args = parser.parse_args()
    path = Path(args.contract)
    data = json.loads(path.read_text(encoding="utf-8"))
    shots = [shot for scene in data["scenes"] for shot in scene["shots"]]
    previous_line, previous = "", None
    repaired, unresolved = 0, []
    for shot in shots:
        for index, line in enumerate(shot.get("formula_lines", [])):
            current = to_equation(line)
            if current is None:
                continue
            introduces = previous is not None and bool(current.free_symbols - previous.free_symbols)
            if previous is not None and not introduces and not current.free_symbols < previous.free_symbols and not equivalent(previous, current):
                fixed = None
                for _ in range(args.attempts):
                    candidate = propose(shot["narration"], previous_line, line, args.model, args.url)
                    parsed = to_equation(candidate) if candidate else None
                    if parsed is not None and equivalent(previous, parsed):
                        fixed = candidate
                        break
                if fixed:
                    shot["formula_lines"][index] = fixed
                    current, line = to_equation(fixed), fixed
                    repaired += 1
                    print(f"[{shot['shot_id']}] repaired: {fixed}")
                else:
                    unresolved.append(shot["shot_id"])
                    print(f"[{shot['shot_id']}] could not be repaired automatically: {line}")
            previous, previous_line = current, line
    if repaired:
        shutil.copyfile(path, path.with_suffix(".json.bak"))
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"repaired {repaired}; unresolved {unresolved}")
    return 1 if unresolved else 0


if __name__ == "__main__":
    raise SystemExit(main())
