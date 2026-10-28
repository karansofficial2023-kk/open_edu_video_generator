"""Scores every lesson's first and latest passed render so that each change to the pipeline can be measured with the SAME rules.

    python tools/score_lessons.py [--contracts C:/Users/User/output] [--outputs outputs] [--json scores.json] [--markdown scores.md]

Per render, seven numbers from 0 to 10: safety, visual quality, content, support for understanding, audio, structure; the quality score
combines them (0.25 visual + 0.25 content + 0.2 understanding + 0.1 audio + 0.2 structure) and the RELEASE score caps it by safety evidence
(no final-frame safety check -> at most 6). Everything except content/structure/visual adjustments is measured from final_qa.json and
production_issues.json. Content is a placeholder (7.5) until an expert reviews it; structure is by design era; VISUAL_ADJ / CONTENT_ADJ are
hand notes from frame reviews (keep them next to the evidence).  The scoring rules were used for the October 2026 scorecard.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import glob
import json
import os
import re
import statistics
from pathlib import Path

GATE = "2026-10-26 16:00"          # the final-frame safety check was first applied to renders after this moment
BODY = {"Adolescence", "Anatomy and Physiology of Male Reproductive System", "Female Reproductive Anatomy",
        "Male Reproductive System in Humans", "Male Sex Hormones", "Modes of Reproduction", "Puberty, Menstruation and Menopause",
        "Regeneration In Real Life", "Role of Hormones in Reproductive Functions", "Sexual Reproduction in Animals",
        "Sexually Transmitted Disease \u2013 AIDS", "The Process of Fertilisation"}
FORMULA_REVIEWED = {"A Simple Question Based on Minima of a Function", "Applications of Kohlrausch\u2019s Law", "All About Electroplating",
                    "Conceptual Problem on Resistance and Resistivity"}
VISUAL_ADJ = {("Female Reproductive Anatomy", "safe_v3"): -1.0, ("Male Reproductive System in Humans", "safe_v3"): -0.5,
              ("Male Sex Hormones", "safe_v3"): -0.5}
CONTENT_ADJ = {("Male Reproductive System in Humans", "safe_v3"): -2.0, ("Male Reproductive System in Humans", "queue20"): -2.0}
REVIEWED_BY_EYE = {("Female Reproductive Anatomy", "safe_v3"), ("Male Reproductive System in Humans", "safe_v3"), ("Male Sex Hormones", "safe_v3")}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower().replace("\u2019", ""))


def load(path):
    try:
        return json.load(open(path, encoding="utf-8"))
    except Exception:
        return None


def collect(contracts_dir: Path, outputs: Path) -> dict:
    contracts = {}
    for f in glob.glob(str(contracts_dir / "*_contract.json")):
        name = os.path.basename(f)[: -len("_contract.json")]
        contracts[norm(name)] = (name, f)
    renders = collections.defaultdict(list)
    for mp4 in glob.glob(str(outputs / "*" / "final.mp4")) + glob.glob(str(outputs / "*" / "*" / "final.mp4")):
        d = os.path.dirname(mp4)
        leaf = norm(os.path.basename(d))
        src = load(os.path.join(d, "source.json")) or {}
        key_src = norm(os.path.basename(str(src.get("input", ""))).replace("_contract.json", "")) if isinstance(src, dict) else ""
        if leaf not in contracts:
            if key_src in contracts:
                leaf = key_src
            else:
                continue
        qa = load(os.path.join(d, "final_qa.json")) or {}
        issues = (load(os.path.join(d, "production_issues.json")) or {}).get("issues", [])
        sb = load(os.path.join(d, "storyboard.json")) or {}
        shots = sum(len(sc.get("segments") or sc.get("shots") or []) for sc in sb.get("scenes", []))
        gates = collections.Counter(f.get("gate", "?") for f in qa.get("findings", []) if f.get("level") != "critical")
        renders[leaf].append({
            "folder": os.path.relpath(d, outputs), "when": datetime.datetime.fromtimestamp(os.path.getmtime(mp4)).strftime("%Y-%m-%d %H:%M"),
            "passed": bool(qa.get("passed")), "critical": qa.get("critical"), "duration_s": round(qa.get("duration") or 0, 1), "audio": qa.get("audio"),
            "warnings_by_gate": dict(gates), "shots_in_storyboard": shots,
            "card_fallbacks": sum(1 for i in issues if i.get("renderer") in ("concept_cards", "diagram") and i.get("field") in ("image", "visual")),
            "labels_dropped": sum(1 for i in issues if i.get("field") == "labels")})
    result = {}
    for leaf, (name, path) in contracts.items():
        c = load(path) or {}
        result[name] = {"subject": c.get("subject"), "renders": sorted(renders.get(leaf, []), key=lambda r: r["when"])}
    return result


NEW_STYLE_FOLDERS = ("safe_v3", "rerender_old", "test_pilot")      # renders made with the organ-model style, diagrams and self-repair


def era(r):
    if r["folder"].startswith(NEW_STYLE_FOLDERS):
        return "safe"
    return "gate" if r["when"] >= GATE else "old"


def safety(name, r):
    f, e = r["folder"], era(r)
    if e in ("safe", "gate"):
        return 9.0
    if f.startswith("queue20"):
        return {"Female Reproductive Anatomy": 4.0, "Adolescence": 6.0, "Male Reproductive System in Humans": 6.0,
                "Anatomy and Physiology of Male Reproductive System": 6.0}.get(name, 8.0)
    if f.startswith("queue10"):
        return 8.0
    if f.startswith("queue5"):
        return 5.0 if name in BODY else 7.0
    return 8.0 if name in FORMULA_REVIEWED else 7.0


def clamp(x, lo=0.0, hi=10.0):
    return max(lo, min(hi, x))


def half(x):
    return round(x * 2) / 2


def score(name, r):
    shots = max(1, r["shots_in_storyboard"])
    cards = r["card_fallbacks"] / shots
    w = r["warnings_by_gate"]
    sem = w.get("semantic", 0) + w.get("repeated_frame", 0)
    key = (name, r["folder"].split("\\")[0].split("/")[0])
    visual = clamp(9.5 - 7 * cards - min(3, 0.3 * sem) + VISUAL_ADJ.get(key, 0))
    content = 7.5 + (0.5 if name in FORMULA_REVIEWED else 0) - min(2, 0.5 * w.get("fragment_card", 0)) + CONTENT_ADJ.get(key, 0)
    under = clamp(9 - 4 * min(1, r["labels_dropped"] / shots) - 2 * cards - min(2, 0.2 * sem))
    a = r.get("audio") or {}
    audio = clamp(9 - min(3, a.get("long_silences", 0)) - (1.5 if a.get("max_db", -3) > -1.0 else 0) - (0.5 if not -20 <= a.get("mean_db", -16) <= -14 else 0))
    structure = 4.0 if r["when"] < "2026-10-22" else 5.5 if r["when"] < GATE else 6.0
    if r["folder"].startswith(NEW_STYLE_FOLDERS):
        structure = 6.5
    safe = safety(name, r)
    quality = 0.25 * visual + 0.25 * content + 0.2 * under + 0.1 * audio + 0.2 * structure
    release = min(quality, 4.0) if safe < 5 else min(quality, 6.0) if safe < 9 else quality
    return {"safety": safe, "visual": half(visual), "content": half(content), "under": half(under), "audio": half(audio), "structure": half(structure),
            "quality": half(quality), "release": half(release), "cards": r["card_fallbacks"], "shots": shots, "labels_dropped": r["labels_dropped"],
            "era": era(r), "folder": r["folder"], "when": r["when"], "confidence": "H" if key in REVIEWED_BY_EYE else ("M" if era(r) != "old" else "L")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--contracts", type=Path, default=Path("C:/Users/User/output"))
    parser.add_argument("--outputs", type=Path, default=Path(__file__).resolve().parents[1] / "outputs")
    parser.add_argument("--json", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args()
    evidence = collect(args.contracts, args.outputs)
    rows = {}
    for name, v in sorted(evidence.items()):
        passed = [x for x in v["renders"] if x["passed"]]
        if passed:
            rows[name] = {"subject": v["subject"], "first": score(name, passed[0]), "latest": score(name, passed[-1])}
    mean = lambda xs: round(statistics.mean(xs), 2) if xs else None
    for label, pick in (("ALL", lambda n: True), ("BODY", lambda n: n in BODY), ("OTHER", lambda n: n not in BODY)):
        sel = [rows[n] for n in rows if pick(n)]
        print(f"{label:5s} n={len(sel):2d}  quality {mean([r['first']['quality'] for r in sel])} -> {mean([r['latest']['quality'] for r in sel])}   "
              f"release {mean([r['first']['release'] for r in sel])} -> {mean([r['latest']['release'] for r in sel])}   "
              f"text-board share {sum(r['latest']['cards'] for r in sel) / max(1, sum(r['latest']['shots'] for r in sel)):.1%}")
    if args.json:
        args.json.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    if args.markdown:
        lines = ["| Lesson | Latest render | Safety | Visual | Content | Understanding | Audio | Structure | Quality | Release | Conf. |", "| --- " * 11 + "|"]
        for n, r in sorted(rows.items(), key=lambda kv: (-kv[1]["latest"]["release"], kv[0])):
            l = r["latest"]
            lines.append(f"| {n} | {l['era']} | {l['safety']:.1f} | {l['visual']:.1f} | {l['content']:.1f} | {l['under']:.1f} | {l['audio']:.1f} | "
                         f"{l['structure']:.1f} | {l['quality']:.1f} | {l['release']:.1f} | {l['confidence']} |")
        args.markdown.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
