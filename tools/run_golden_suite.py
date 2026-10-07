"""Golden-suite gate: score every lesson contract in a folder and (optionally) render them.

    python tools/run_golden_suite.py <folder with *_contract.json> [--min-score 85] [--render --config config.12gb.yaml --output outputs/golden]

Release criteria (production spec): every storyboard scores >= min-score, has zero renderer-readiness errors and zero
derivation warnings; with --render every master passes final QA with no critical finding.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.storyboard_scorecard import score  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("folder")
    parser.add_argument("--min-score", type=float, default=85.0)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--config", default="config.12gb.yaml")
    parser.add_argument("--output", default="outputs/golden")
    parser.add_argument("--preview", action="store_true", help="silent draft renders (no TTS)")
    args = parser.parse_args()

    rows, failed = [], False
    for contract in sorted(Path(args.folder).glob("*_contract.json")):
        transcript = contract.with_name(contract.name.replace("_contract.json", "_transcript.txt"))
        report = score(contract, transcript.read_text(encoding="utf-8") if transcript.is_file() else "")
        render = "-"
        if args.render:
            out = Path(args.output) / contract.name.replace("_contract.json", "")
            command = [sys.executable, "-m", "app.main", "--config", args.config, "--input", str(contract), "--output", str(out)]
            if args.preview:
                command.append("--preview")
            qa = out / "final_qa.json"
            qa.unlink(missing_ok=True)          # never read the previous run's verdict if this run crashes
            finished = subprocess.run(command, check=False)
            if qa.is_file():
                data = json.loads(qa.read_text(encoding="utf-8"))
                render = "pass" if data.get("passed") else f"{data.get('critical')} critical"
            elif finished.returncode not in (0, 3):
                render = f"crashed (exit {finished.returncode})"
            else:
                render = "no master"
            failed |= render != "pass"
        ok = report["overall"] >= args.min_score and report["preflight_errors"] == 0 and report["metrics"].get("derivation_consistency", 100.0) >= 100.0
        failed |= not ok
        rows.append((contract.name.replace("_contract.json", ""), report["shots"], report["overall"], report["preflight_errors"],
                     report["metrics"].get("derivation_consistency", "-"), render, "PASS" if ok else "FAIL"))

    lines = ["| Lesson | Shots | Score | Preflight errors | Derivation | Render | Gate |", "|---|---|---|---|---|---|---|"]
    lines += [f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} |" for r in rows]
    text = "\n".join(lines)
    print(text)
    Path(args.folder, "golden_report.md").write_text(text + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
