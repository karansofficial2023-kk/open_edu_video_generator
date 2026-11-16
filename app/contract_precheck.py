"""Dubbing v2 (deep dive): inspect and repair a translated Transcribe contract BEFORE any dubbing.

Run automatically by ``replace_audio`` and by the queue (``python -m app.contract_precheck``).
Everything here is deterministic: it fixes what is provably wrong, and reports (never invents)
what needs a human or the source audio.

Fixes
  * foreign-script debris from ASR/translation (CJK, Cyrillic, Arabic, ...) removed
  * broken spacing ("left -hand", "O -N -E", " ,") normalised
  * leftover untranslated English function words ("the", "of") removed from Hindi lines
  * known science mistranslations repaired from the source term (app/data/hi_dub_glossary.json)
  * bare numbering cues ("1.", "2.") spoken as Hindi ordinals instead of noise
  * Devanagari-spelled formulas restored to the Latin formula (strict English-formula rule)
  * timing: cues sorted, zero/negative durations and duplicate starts repaired
Review items that are demonstrably low-risk are downgraded to advisory notes instead of
inflating the teacher-review count; real findings stay as issues.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path

from .schema import Storyboard
from .spoken_forms import hindi_number, normalize_chemistry_text
from .translation_qa import _LETTER_NAMES_HI

PRECHECK_VERSION = "v2-deep-dive-1"
_FOREIGN_SCRIPT = re.compile(
    "[Ѐ-ӿ԰-֏֐-ࣿ฀-໿ᄀ-ᇿ⺀-⿟　-ヿ"
    "㄀-ㄯ㆐-鿿ꥠ-꥿가-퟿豈-﫿︰-﹏＀-￯]")
_ORDINALS_HI = ["", "पहला", "दूसरा", "तीसरा", "चौथा", "पाँचवाँ", "छठा", "सातवाँ", "आठवाँ", "नौवाँ", "दसवाँ"]
_NUMBERING_CUE = re.compile(r"^\s*\(?(\d{1,3})[.):]?\s*$")


@lru_cache(maxsize=1)
def _glossary() -> dict:
    return json.loads((Path(__file__).parent / "data" / "hi_dub_glossary.json").read_text(encoding="utf-8"))


def _strip_foreign(text: str) -> tuple[str, list[str]]:
    removed = _FOREIGN_SCRIPT.findall(text or "")
    if not removed:
        return text, []
    return re.sub(r"[ \t]{2,}", " ", _FOREIGN_SCRIPT.sub("", text)).strip(), removed


_SPLIT_FORMULA = re.compile(r"(?<![A-Za-z0-9])([A-Z][a-z]?\d*(?:\s*-\s*[A-Z][a-z]?\d*){2,})(?![A-Za-z0-9-])")


def _join_split_formulas(text: str) -> str:
    """Whisper writes RCONH2 as "R -C -O -N -H2" and RCOCl as "R-C-O-C-L"; rejoin them so they are one formula.

    Only chains that contain an R/X/Y group or a digit are joined; a plain letter chain such as the IUPAC ending
    "O-N-E" is left alone because it is a spelled word, not a formula.
    """
    def join(match: re.Match) -> str:
        parts = re.split(r"\s*-\s*", match.group(1))
        if not (parts[0] in {"R", "X", "Y"} or any(re.search(r"\d", part) for part in parts)):
            return match.group(1)
        joined = "".join(parts)
        return joined[:-2] + "Cl" if parts[-2:] == ["C", "L"] else joined
    return _SPLIT_FORMULA.sub(join, text)


def _tidy_spacing(text: str) -> str:
    text = re.sub(r"(?<=[A-Za-z0-9]) +-(?=[A-Za-z0-9])", "-", text)       # "left -hand", "O -N -E"
    text = re.sub(r"\s+([,.;:!?।])", r"\1", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return _join_split_formulas(text.strip())


def _drop_latin_function_words(text: str) -> str:
    stop = "|".join(map(re.escape, _glossary()["latin_stopwords"]))
    cleaned = re.sub(rf"(?<![A-Za-zऀ-ॿ])(?:{stop})(?![A-Za-zऀ-ॿ])", "", text)
    return re.sub(r"[ \t]{2,}", " ", cleaned).replace(" ,", ",").strip()


def _is_hindi(language: str) -> bool:
    return (language or "").lower().startswith("hi")


_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+")


def _count_words(text: str) -> int:
    """Whitespace words that contain a letter or digit (a regex \w would split Devanagari at every vowel sign)."""
    return sum(1 for w in text.split() if any(c.isalnum() for c in w))


def _missing_content(source: str, narration: str) -> str:
    """Reason when the Hindi covers far less than the English (a dropped sentence), else ''. Hindi runs about as long as English in words."""
    src_words, hi_words = _count_words(source), _count_words(narration)
    if src_words < 12 or not narration.strip():
        return ""
    src_sentences = [x for x in _SENTENCE_END.split(source.strip()) if _count_words(x) >= 3]
    hi_sentences = [x for x in _SENTENCE_END.split(narration.strip()) if x.strip(" .।")]
    ratio = hi_words / src_words
    if len(src_sentences) >= 2 and (ratio < 0.6 or len(hi_sentences) < len(src_sentences) - 1):
        return (f"The English has {len(src_sentences)} sentences ({src_words} words) but the Hindi has {len(hi_sentences)} "
                f"({hi_words} words): a sentence of the source is probably missing from the dub.")
    if ratio < 0.4:
        return f"The Hindi has only {hi_words} words for {src_words} English words: part of the source is probably missing."
    return ""


def precheck_shot(shot_id: str, source: str, narration: str, language: str) -> dict:
    """Return corrected source/narration plus repairs and issues for one timed cue."""
    repairs: list[dict] = []
    issues: list[dict] = []
    original_source, original_narration = source or "", narration or ""
    source = normalize_chemistry_text(original_source)
    source, removed_source = _strip_foreign(source)
    narration, removed_narration = _strip_foreign(original_narration)
    if removed_source:
        issues.append({"shot_id": shot_id, "kind": "source_asr_garbled", "source": original_source,
                       "translation": narration,
                       "detail": f"Foreign-script ASR debris {''.join(removed_source)!r} was removed from the source; "
                                 "verify the spoken English against the source audio."})
    if removed_source or removed_narration:
        repairs.append({"shot_id": shot_id, "kind": "foreign_script_removed",
                        "from": (original_source if removed_source else original_narration)[:200],
                        "to": (source if removed_source else narration)[:200]})
    source, narration = _tidy_spacing(source), _tidy_spacing(narration)

    if _is_hindi(language) and narration:
        pruned = _drop_latin_function_words(narration)
        if pruned != narration and re.search(r"[ऀ-ॿ]", pruned):
            repairs.append({"shot_id": shot_id, "kind": "untranslated_function_word_removed",
                            "from": narration[:200], "to": pruned[:200]})
            narration = pruned
        for rule in _glossary()["rules"]:
            if not re.search(rule["source"], source, re.IGNORECASE):
                continue
            for wrong in rule["wrong"]:
                fixed = re.sub(wrong, rule["right"], narration)
                if fixed != narration:
                    repairs.append({"shot_id": shot_id, "kind": "science_terminology", "rule": rule["id"],
                                    "from": narration[:200], "to": fixed[:200]})
                    narration = fixed
        for label in dict.fromkeys(re.findall(r"\b[Aa]rms?\s+([A-Z]{1,2})\b", source)):
            letters = r"\s*".join(re.escape(_LETTER_NAMES_HI[c]) for c in label)
            pattern = rf"(?:हाथ|आर्म|आर्मस|भुजा)\s+(?:{re.escape(label)}|{letters})(?![\wऀ-ॿ])"
            fixed = re.sub(pattern, "भुजा " + label, narration)
            if fixed != narration:
                repairs.append({"shot_id": shot_id, "kind": "science_terminology", "rule": "arm_label",
                                "from": narration[:200], "to": fixed[:200]})
                narration = fixed
        # The speaker may SAY "C double bond O"; the strict rule wants the formula itself (spoken in English as a bond).
        for first, kind, second in dict.fromkeys(re.findall(r"\b([A-Z])\s+(double|triple)\s+bond\s+([A-Z])\b", source)):
            hindi_kind = "डबल" if kind == "double" else "ट्रिपल"
            fixed = re.sub(rf"{first}\s*{hindi_kind}\s*बॉन्ड\s*{second}", first + ("=" if kind == "double" else "≡") + second, narration)
            if fixed != narration:
                repairs.append({"shot_id": shot_id, "kind": "bond_notation_restored", "from": narration[:200], "to": fixed[:200]})
                narration = fixed
        cue = _NUMBERING_CUE.match(source)
        if cue and narration.strip(" .।)(") != "":
            number = int(cue.group(1))
            spoken = (_ORDINALS_HI[number] if 1 <= number <= 10 else hindi_number(str(number))) + "।"
            if narration != spoken:
                repairs.append({"shot_id": shot_id, "kind": "numbering_cue_spoken", "from": narration, "to": spoken})
                narration = spoken
        if re.search(r"(?:[a-z]{2,}\s+){2,}[a-z]{2,}", narration):
            issues.append({"shot_id": shot_id, "kind": "latin_prose_in_hindi", "source": source, "translation": narration,
                           "detail": "Untranslated English words remain in the Hindi line (formulas and labels are fine)."})
    if _is_hindi(language):
        gap = _missing_content(source, narration)
        if gap:
            issues.append({"shot_id": shot_id, "kind": "hindi_missing_content", "source": source, "translation": narration, "detail": gap})
    if source.strip() and not narration.strip():
        issues.append({"shot_id": shot_id, "kind": "missing_translation", "source": source, "translation": "",
                       "detail": "Source narration has no Hindi translation."})
    return {"source": source, "narration": narration, "repairs": repairs, "issues": issues}


def precheck_board(board: Storyboard) -> tuple[list[dict], list[dict]]:
    """Fix every segment in place. Returns (repairs, issues)."""
    language = (board.language or {}).get("bcp47", "")
    all_repairs: list[dict] = []
    all_issues: list[dict] = []
    segments = [segment for _scene, segment in board.all_segments()]
    for segment in segments:
        shot_id = segment.shot_id or str(segment.segment_number)
        before = segment.narration
        result = precheck_shot(shot_id, segment.source_narration, segment.narration, language)
        segment.source_narration, segment.narration = result["source"], result["narration"]
        all_repairs.extend(result["repairs"])
        all_issues.extend(result["issues"])
        if result["narration"] != before:
            all_repairs.append({"shot_id": shot_id, "kind": "narration_corrected",
                                "original_narration": before, "corrected_narration": result["narration"],
                                "source": "contract_precheck"})
    all_repairs.extend(_timing_repairs(segments))
    return all_repairs, all_issues


def _timing_repairs(segments) -> list[dict]:
    repairs = []
    previous = None
    for segment in segments:
        shot_id = segment.shot_id or str(segment.segment_number)
        if segment.source_start is None or segment.source_end is None:
            continue
        if segment.source_end <= segment.source_start:
            segment.source_end = round(float(segment.source_start) + 0.3, 3)
            repairs.append({"shot_id": shot_id, "kind": "timing_end_extended", "to": segment.source_end})
        if previous is not None and float(segment.source_start) <= float(previous.source_start):
            segment.source_start = round(float(previous.source_start) + 0.01, 3)
            repairs.append({"shot_id": shot_id, "kind": "timing_start_nudged", "to": segment.source_start})
        previous = segment
    return repairs


def review_item_is_advisory(item: dict, shot_has_other_findings: bool) -> str | None:
    """Return a reason when a stored review item is low-risk noise, otherwise None.

    A semantic-audit or ASR-uncertainty note stays an ISSUE whenever the same cue also has a deterministic
    finding (formula, number, foreign script, untranslated words, missing translation).
    """
    problem = str(item.get("problem", ""))
    if problem.startswith("source_wording_issue:") or problem.startswith("source_formula_ambiguity:"):
        return "Qwen classified this as a source defect/ambiguity, not a Hindi translation error"
    if re.search("[⺀-鿿぀-ヿ가-퟿]", str(item.get("source", "")) + str(item.get("translation", ""))
                 + str(item.get("problem", ""))):
        return "foreign-script debris was removed by the contract precheck; tracked once as source_asr_garbled"
    if ("C=O" in str(item.get("problem", "")) and re.search(r"-one\b|\bO\s*-\s*N\s*-\s*E\b", str(item.get("source", "")))
            and "C=O" not in str(item.get("source", ""))):
        return "audit confuses the IUPAC suffix -one (O-N-E) with the C=O group; the translation keeps the suffix correctly"
    if shot_has_other_findings:
        return None
    where = str(item.get("where", ""))
    if where.endswith("source_asr") or "needs source-audio verification" in problem:
        match = re.search(r"uncertain words:\s*(.+)$", problem)
        if not match:
            return None
        words = [re.sub(r"[^\w]", "", w).lower() for w in match.group(1).split(",")]
        words = [w for w in words if w]
        function_words = set(_glossary()["asr_function_words"])
        if words and all(w in function_words for w in words):
            return "ASR uncertainty only on function words; no formula, number or term involved"
        return None
    if "semantic_audit" in where or problem.startswith("semantic audit"):
        source, translation = str(item.get("source", "")), str(item.get("translation", ""))
        if (re.search(r"[一-鿿]", source + translation) or re.search(r"\d|[A-Z][a-z]?\d", source)):
            return None
        return "unverified language-model concern; deterministic formula/number/term checks passed for this cue"
    return None


def _shot_dicts(data: dict):
    for scene in data.get("scenes", []):
        for shot in scene.get("shots", []):
            yield scene, shot


def precheck_contract_dict(data: dict) -> dict:
    """Repair a raw contract dict in place and return the pre-dub report."""
    language = (data.get("language") or {}).get("bcp47", "")
    repairs: list[dict] = []
    issues: list[dict] = []
    shots = [shot for _scene, shot in _shot_dicts(data)]
    if not language:
        issues.append({"shot_id": "contract", "kind": "missing_language", "source": "", "translation": "",
                       "detail": "language.bcp47 is missing"})
    if not shots:
        issues.append({"shot_id": "contract", "kind": "no_shots", "source": "", "translation": "",
                       "detail": "The contract has no narration shots."})
    previous_start = None
    for shot in shots:
        shot_id = str(shot.get("shot_id", ""))
        result = precheck_shot(shot_id, shot.get("source_narration", ""), shot.get("narration", ""), language)
        before = shot.get("narration", "")
        shot["source_narration"], shot["narration"] = result["source"], result["narration"]
        repairs.extend(result["repairs"])
        issues.extend(result["issues"])
        if result["narration"] != before:
            repairs.append({"shot_id": shot_id, "kind": "narration_corrected", "original_narration": before,
                            "corrected_narration": result["narration"], "source": "contract_precheck"})
        start, end = shot.get("source_start"), shot.get("source_end")
        if start is not None and end is not None:
            if end <= start:
                shot["source_end"] = round(start + 0.3, 3)
                repairs.append({"shot_id": shot_id, "kind": "timing_end_extended", "from": end, "to": shot["source_end"]})
            if previous_start is not None and shot["source_start"] <= previous_start:
                shot["source_start"] = round(previous_start + 0.01, 3)
                repairs.append({"shot_id": shot_id, "kind": "timing_start_nudged", "from": start, "to": shot["source_start"]})
            previous_start = shot["source_start"]
        elif shot.get("source_narration", "").strip():
            issues.append({"shot_id": shot_id, "kind": "missing_timing", "source": shot["source_narration"],
                           "translation": shot["narration"], "detail": "Cue has no source_start/source_end."})
    for scene in data.get("scenes", []):
        scene["narration"] = " ".join(str(s.get("narration", "")).strip() for s in scene.get("shots", [])).strip()
    return {"precheck_version": PRECHECK_VERSION, "checked_shots": len(shots), "repair_count": len(repairs),
            "repairs": repairs, "issue_count": len(issues), "issues": issues}


def precheck_contract_file(path: str | Path, out_path: str | Path | None = None) -> dict:
    """Repair a contract file. Keeps the first original as ``<name>.v1.bak`` and writes ``<name>.precheck.json``."""
    source_path = Path(path)
    target = Path(out_path) if out_path else source_path
    data = json.loads(source_path.read_text(encoding="utf-8-sig"))
    original_hash = hashlib.sha256(source_path.read_bytes()).hexdigest()
    report = precheck_contract_dict(data)
    report["original_contract_sha256"] = original_hash
    if report["repair_count"]:
        backup = source_path.with_name(source_path.name + ".v1.bak")
        if target == source_path and not backup.exists():
            shutil.copy2(source_path, backup)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    report["fixed_contract_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest() if target.is_file() else ""
    report_path = target.with_name(target.stem + ".precheck.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def load_precheck_repairs(contract_path: Path | None) -> list[dict]:
    """Narration repairs recorded when the contract file was fixed ahead of dubbing (keeps stale review items resolvable)."""
    if not contract_path:
        return []
    sidecar = contract_path.with_name(contract_path.stem + ".precheck.json")
    if not sidecar.is_file():
        return []
    try:
        return [r for r in json.loads(sidecar.read_text(encoding="utf-8-sig")).get("repairs", [])
                if r.get("kind") == "narration_corrected"]
    except (OSError, ValueError):
        return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check and repair translated contracts before dubbing (v2 deep dive)")
    parser.add_argument("contracts", nargs="+", help="contract .json files or folders containing *_contract.json")
    parser.add_argument("--out-dir", help="write repaired contracts here instead of fixing in place")
    args = parser.parse_args(argv)
    files: list[Path] = []
    for item in args.contracts:
        path = Path(item)
        files.extend(sorted(path.glob("*_contract.json")) if path.is_dir() else [path])
    if not files:
        print("No contract files found", file=sys.stderr)
        return 2
    worst = 0
    for file in files:
        out = Path(args.out_dir) / file.name if args.out_dir else None
        report = precheck_contract_file(file, out)
        print(f"PRECHECK {file.name}: {report['checked_shots']} cues | {report['repair_count']} repairs | "
              f"{report['issue_count']} findings for review")
        for repair in report["repairs"]:
            if repair["kind"] != "narration_corrected":
                print(f"   fixed {repair['shot_id']} [{repair['kind']}]")
        for issue in report["issues"]:
            print(f"   REVIEW {issue['shot_id']} [{issue['kind']}] {issue['detail'][:140]}")
        worst = max(worst, 1 if any(i["kind"] in {"missing_language", "no_shots"} for i in report["issues"]) else 0)
    return worst


if __name__ == "__main__":
    raise SystemExit(main())
