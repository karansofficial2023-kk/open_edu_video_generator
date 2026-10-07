"""Storyboard release scorecard (Transcribe contract JSON v2).

    python tools/storyboard_scorecard.py <lesson_contract.json> [--transcript lesson_transcript.txt] [--min-score 85]

Scores completeness, renderer readiness (the generator's own preflight), source fidelity, subject-appropriate visual
routing and pacing. No lesson-specific rules: every check is derived from the contract fields.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import AppConfig  # noqa: E402
from app.input_reader import read_input  # noqa: E402
from app.assets import _words, clean_prompt  # noqa: E402
from app.preflight import check_storyboard  # noqa: E402

STOP = set("the a an of and to in on with for from is are was were that this these those it its by as at or be can which their "
           "have has had not but also such than then there where when while into over under between within".split())
MATH = re.compile(r"(?i)(=|\b(equals?|equal to|plus|minus|times|multiplied by|divided by|squared|cubed|square root|proportional to|"
                  r"reciprocal|derivative|integral|discriminant)\b)")
MATH_SUBJECTS = ("math", "phys", "chem")
WORDS_PER_SECOND = 2.3


_PUNCT = " \t\n\"'.,;:!?()[]{}।۔*_-"


def content_words(text: str) -> set[str]:
    """Word stems (6 chars) of 4+ char tokens; split on whitespace so combining marks stay inside Indic words."""
    tokens = (w.strip(_PUNCT) for w in (text or "").lower().split())
    return {w[:6] for w in tokens if len(w) >= 4 and w not in STOP and not w.isdigit()}


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?।])\s+", text or "") if s.strip()]


def score(contract_path: Path, transcript: str = "") -> dict:
    _text, board = read_input(contract_path)
    config = AppConfig()
    segments = [(scene, seg) for scene, seg in board.all_segments()]
    n = len(segments)
    findings: list[str] = []
    metrics: dict[str, float] = {}

    # 1. renderer readiness: generator preflight on the contract itself
    issues = check_storyboard(board, config)
    errors = [i for i in issues if i["severity"] == "error"]
    metrics["renderer_ready"] = 100.0 if not errors else max(0.0, 100.0 - 15.0 * len(errors))
    findings += [f"[{i['shot_id']}] {i['field']}: {i['reason']}" for i in errors]

    # 2. completeness
    complete = 0
    for _scene, seg in segments:
        ok = bool(seg.narration.strip()) and bool(seg.visual_type)
        if seg.shot and seg.shot.template in {"photo", "title_card"}:
            ok = ok and bool((seg.image_prompt or seg.image_requirement).strip())
        if seg.shot and seg.shot.template == "formula":
            ok = ok and bool(seg.formula_lines)
        complete += 1 if ok else 0
    metrics["completeness"] = 100.0 * complete / max(1, n)

    # 3. source fidelity / coverage (needs the transcript)
    if transcript.strip():
        source_words = content_words(transcript)
        narration = " ".join(seg.narration for _s, seg in segments)
        narration_words = content_words(narration)
        recall = len(source_words & narration_words) / max(1, len(source_words))
        unsupported = [seg.shot_id for _s, seg in segments
                       if seg.visual_type != "title_card" and len(content_words(seg.narration) & (source_words | content_words(board.title))) < 0.2 * max(1, len(content_words(seg.narration)))]
        metrics["source_coverage"] = min(100.0, 100.0 * recall / 0.8)
        metrics["source_support"] = 100.0 * (1 - len(unsupported) / max(1, n))
        findings += [f"[{sid}] narration has little overlap with the source transcript" for sid in unsupported]

    # 4. subject-appropriate routing: math-bearing sentences should become formula/graph shots
    subject = str(json.loads(contract_path.read_text(encoding="utf-8")).get("subject", "")).lower()
    math_shots = [seg for _s, seg in segments if MATH.search(seg.narration)] if any(k in subject for k in MATH_SUBJECTS) else []
    routed = [seg for seg in math_shots if seg.shot and seg.shot.template in {"formula", "graph"}]
    if len(math_shots) >= 2:
        metrics["math_routing"] = 100.0 * len(routed) / len(math_shots)
        findings += [f"[{seg.shot_id}] narration states mathematics but is shown as {seg.visual_type}" for seg in math_shots if seg not in routed][:6]

    # 4b. language integrity: non-English lessons keep narration in the lesson script and carry a voice
    raw = json.loads(contract_path.read_text(encoding="utf-8"))
    language = raw.get("language") or {}
    bcp47 = str(language.get("bcp47", "en"))
    if not bcp47.lower().startswith("en"):
        letters = [ch for _s, seg in segments for ch in seg.narration if ch.isalpha()]
        native = sum(1 for ch in letters if ord(ch) >= 0x0590)
        share = native / max(1, len(letters))
        voice_ok = bool(language.get("voice_preference"))
        metrics["language_integrity"] = (70.0 if share >= 0.8 else 100.0 * share * 0.7 / 0.8) + (30.0 if voice_ok else 0.0)
        if share < 0.8:
            findings.append(f"narration is only {share:.0%} in the lesson script (language {bcp47}); it may have been translated to English")
        if not voice_ok:
            findings.append(f"language package for {bcp47} has no TTS voice")

    # 4c. derivation consistency (symbolic): every formula step must follow from the previous equation
    from app.preflight import check_derivations
    derivation = check_derivations(board)
    if any(seg.formula_lines for _s, seg in segments):
        metrics["derivation_consistency"] = max(0.0, 100.0 - 34.0 * len(derivation))
        findings += [f"[{i['shot_id']}] {i['reason']}" for i in derivation]

    # 5. visual variety (non-title shots)
    kinds = Counter(seg.visual_type for _s, seg in segments if seg.visual_type != "title_card")
    top_share = max(kinds.values()) / max(1, sum(kinds.values())) if kinds else 1
    metrics["visual_variety"] = 100.0 if (n < 6 or len(kinds) >= 2) and top_share < 0.9 else 55.0 if len(kinds) >= 2 else 40.0

    # 6. exactly one title card first; no generic prompts; pacing
    title_ok = sum(1 for _s, seg in segments if seg.visual_type == "title_card") == 1 and segments[0][1].visual_type == "title_card"
    def specific(seg) -> bool:       # judge the lesson-specific part of the prompt, not the gate's boilerplate contract
        if any(ord(ch) >= 0x0590 for ch in seg.narration):      # non-English narration: prompts are English by design
            return len(_words(clean_prompt(seg.image_prompt))) >= 8
        return len(_words(clean_prompt(seg.image_prompt)) & _words(seg.narration + " " + " ".join(seg.labels) + " " + board.title)) >= 2
    generic = sum(1 for _s, seg in segments if seg.shot and seg.shot.template in {"photo", "title_card"} and not specific(seg))
    metrics["structure"] = (50.0 if title_ok else 0.0) + 50.0 * (1 - generic / max(1, n))
    pacing_bad = [seg.shot_id for _s, seg in segments if seg.duration_hint and
                  seg.duration_hint < 0.6 * len(seg.narration.split()) / WORDS_PER_SECOND]
    metrics["pacing"] = 100.0 * (1 - len(pacing_bad) / max(1, n))
    findings += [f"[{sid}] duration shorter than the narration needs" for sid in pacing_bad[:6]]

    overall = sum(metrics.values()) / len(metrics)
    return {"contract": str(contract_path), "title": board.title, "shots": n, "visual_mix": dict(kinds),
            "metrics": {k: round(v, 1) for k, v in metrics.items()}, "overall": round(overall, 1),
            "preflight_errors": len(errors), "findings": findings[:25]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("contract", nargs="+")
    parser.add_argument("--transcript", help="source transcript text (same order as contracts if several, else a folder lookup)")
    parser.add_argument("--min-score", type=float, default=85.0)
    args = parser.parse_args()
    failed = False
    for path in map(Path, args.contract):
        transcript_path = Path(args.transcript) if args.transcript else Path(str(path).replace("_contract.json", "_transcript.txt"))
        transcript = transcript_path.read_text(encoding="utf-8") if transcript_path.is_file() else ""
        report = score(path, transcript)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        path.with_name(path.name.replace("_contract.json", "_scorecard.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        failed |= report["overall"] < args.min_score
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
