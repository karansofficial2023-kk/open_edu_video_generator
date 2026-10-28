"""SME review sheet for one lesson: export the contract to a spreadsheet, let a subject expert edit it, apply the edits to a COPY.

    python tools/review_sheet.py export "C:/Users/User/output/Male Sex Hormones_contract.json"
        -> "<lesson>_review.csv" next to the contract (opens in Excel; UTF-8 with BOM)
    python tools/review_sheet.py apply  "C:/Users/User/output/Male Sex Hormones_contract.json" --reviewer "Dr A" [--sheet other.csv]
        -> "<lesson>_contract.reviewed.json" (the original contract is never changed) + "<lesson>_sign_off.json"

Columns the expert fills in (decision, one of: ok / edit / picture / remove; blank = ok):
  edit     -> `new_narration` replaces the narration (and the subtitle) of that shot
  picture  -> `new_asset_path` is the approved picture to use for that shot (an absolute path to a PNG/JPG)
  remove   -> the shot is dropped from the lesson
The `check` column carries what the pipeline already flagged (the narration sanity check of Transcribe, if its review file exists).
Re-render only after `apply`: the generator takes the .reviewed.json as its input, and unchanged shots reuse their cached clips.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
from pathlib import Path

COLUMNS = ["shot_id", "scene", "narration", "visual_type", "picture_asset", "check", "decision", "new_narration", "new_asset_path", "notes"]
DECISIONS = {"", "ok", "edit", "picture", "remove"}


def _paths(contract: Path):
    stem = contract.name[: -len("_contract.json")] if contract.name.endswith("_contract.json") else contract.stem
    return (contract.with_name(f"{stem}_review.csv"), contract.with_name(f"{stem}_contract.reviewed.json"),
            contract.with_name(f"{stem}_sign_off.json"), contract.with_name(f"{stem}_narration_review.json"))


def export(contract_path: Path, sheet: Path | None = None) -> Path:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    csv_path, _reviewed, _sign, narr = _paths(contract_path)
    flags: dict[str, list[str]] = {}
    if narr.exists():
        for item in json.loads(narr.read_text(encoding="utf-8")):
            flags.setdefault(str(item.get("shot_id")), []).append(", ".join(item.get("signals") or []) or str(item.get("llm_verdict", "")))
    sheet = sheet or csv_path
    with open(sheet, "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for scene in contract.get("scenes", []):
            for shot in scene.get("shots", []):
                writer.writerow({"shot_id": shot.get("shot_id", ""), "scene": scene.get("scene_title", ""), "narration": shot.get("narration", ""),
                                 "visual_type": shot.get("visual_type", ""), "picture_asset": shot.get("asset_path", ""),
                                 "check": " | ".join(flags.get(str(shot.get("shot_id")), [])), "decision": "", "new_narration": "",
                                 "new_asset_path": "", "notes": ""})
    return sheet


def apply(contract_path: Path, reviewer: str, sheet: Path | None = None) -> tuple[Path, dict]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    csv_path, reviewed_path, sign_path, _narr = _paths(contract_path)
    sheet = sheet or csv_path
    with open(sheet, encoding="utf-8-sig", newline="") as handle:
        rows = {str(r["shot_id"]): r for r in csv.DictReader(handle)}
    counts = {"edited": 0, "picture": 0, "removed": 0, "unchanged": 0}
    problems: list[str] = []
    for scene in contract.get("scenes", []):
        kept = []
        for shot in scene.get("shots", []):
            row = rows.get(str(shot.get("shot_id")))
            decision = ((row or {}).get("decision") or "").strip().lower()
            if decision not in DECISIONS:
                problems.append(f"shot {shot.get('shot_id')}: unknown decision {decision!r} (use ok, edit, picture or remove)")
                decision = ""
            if decision == "remove":
                counts["removed"] += 1
                continue
            if decision == "edit":
                text = (row.get("new_narration") or "").strip()
                if text:
                    shot["narration"] = text
                    shot["subtitle"] = text
                    counts["edited"] += 1
                else:
                    problems.append(f"shot {shot.get('shot_id')}: decision edit without new_narration")
            elif decision == "picture":
                path = (row.get("new_asset_path") or "").strip()
                if path and Path(path).is_file():
                    shot["asset_path"] = path
                    counts["picture"] += 1
                else:
                    problems.append(f"shot {shot.get('shot_id')}: decision picture but new_asset_path is missing or not a file")
            else:
                counts["unchanged"] += 1
            kept.append(shot)
        scene["shots"] = kept
        scene["narration"] = " ".join(s.get("narration", "") for s in kept)
    if problems:
        raise SystemExit("Nothing written. Fix the sheet first:\n  " + "\n  ".join(problems))
    reviewed_path.write_text(json.dumps(contract, indent=2, ensure_ascii=False), encoding="utf-8")
    sign = {"lesson": contract.get("title"), "reviewer": reviewer, "date": datetime.date.today().isoformat(), "source_contract": str(contract_path),
            "reviewed_contract": str(reviewed_path), "changes": counts, "status": "reviewed - release only after the reviewer confirms the final video"}
    sign_path.write_text(json.dumps(sign, indent=2, ensure_ascii=False), encoding="utf-8")
    return reviewed_path, sign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", choices=["export", "apply"])
    parser.add_argument("contract", type=Path)
    parser.add_argument("--reviewer", default="", help="name of the subject expert (required for apply)")
    parser.add_argument("--sheet", type=Path, default=None)
    args = parser.parse_args()
    if args.action == "export":
        print(f"Review sheet: {export(args.contract, args.sheet)}")
    else:
        if not args.reviewer.strip():
            raise SystemExit("apply needs --reviewer \"<name>\" so the sign-off records who checked the lesson")
        path, sign = apply(args.contract, args.reviewer.strip(), args.sheet)
        print(f"Reviewed contract: {path}\nSign-off: {sign['changes']}")


if __name__ == "__main__":
    main()
