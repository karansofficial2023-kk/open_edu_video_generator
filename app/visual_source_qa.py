"""Read printed science facts from the original video before dubbing.

OCR and the vision model provide evidence, never an automatic replacement for
what the teacher said. A frame can be transitional or contain an unrelated
formula, so disagreements remain review findings with saved frames.
"""
from __future__ import annotations

import base64
import json
import re
import subprocess
from collections import Counter
from pathlib import Path

from .renderer import _tool_path
from .translation_qa import _formula_tokens, _SPOKEN_EQUATION

VISUAL_QA_VERSION = 1
_CHEMISTRY = re.compile(r"\b(formula|equation|reaction|compound|bleaching|sodium|calcium|hydroxide|"
                        r"chloride|oxide|carbonate|acid|alkali|ketone|ethanol|ethanoic)\b", re.I)
_MATH = re.compile(r"\b(plus|minus|times|multiply|divid|subtract|equals?|equal to|calculate|solve)\b|[+×÷=]", re.I)
_EQUATION = re.compile(r"(?<!\w)\(?\s*(-?\d+(?:\.\d+)?)\s*([+\-×x*÷/])\s*(-?\d+(?:\.\d+)?)\s*\)?\s*=\s*(-?\d+(?:\.\d+)?)(?!\w)")
_SUBSCRIPTS = str.maketrans("₀₁₂₃₄₅₆₇₈₉−–", "0123456789--")


def _engine():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except (ImportError, OSError, RuntimeError) as error:
        return error


def _sample_times(start: float, end: float) -> list[float]:
    duration = end - start
    return list(dict.fromkeys(round(start + duration * fraction, 2) for fraction in (0.2, 0.5, 0.8)))


def _exact_token_present(token: str, lines: list[str]) -> bool:
    normalized = token.translate(_SUBSCRIPTS)
    pattern = re.compile(r"(?<![A-Za-z0-9])" + re.escape(normalized) + r"(?![A-Za-z0-9])", re.I)
    return any(pattern.search(line.translate(_SUBSCRIPTS)) for line in lines)


def _math_equations(lines: list[str]) -> list[dict]:
    from decimal import Decimal, InvalidOperation
    results = []
    for line in lines:
        normalized = line.translate(_SUBSCRIPTS).replace("＝", "=").replace("∗", "*")
        for match in _EQUATION.finditer(normalized):
            left, operator, right, stated = match.groups()
            zero_division = operator in "÷/" and Decimal(right) == 0
            try:
                a, b, expected = Decimal(left), Decimal(right), Decimal(stated)
                actual = (a + b if operator == "+" else a - b if operator == "-" else
                          a * b if operator in "×x*" else a / b if b else None)
            except (InvalidOperation, ZeroDivisionError):
                actual = None
            results.append({"expression": f"{left}{operator}{right}={stated}", "ocr_text": match.group(),
                            "calculated": str(actual) if actual is not None else
                            "undefined (division by zero)" if zero_division else None,
                            "correct": actual == expected if actual is not None else
                            False if zero_division else None})
    return results


def _spoken_equation(source: str) -> tuple[str, str, str, str] | None:
    match = _SPOKEN_EQUATION.search(source)
    if not match:
        return None
    left, spoken_operator, right, result = match.groups()
    operator = {"plus": "+", "minus": "-", "times": "*", "multiplied by": "*", "divided by": "/"}[spoken_operator.lower()]
    return left, operator, right, result


def _visual_equation_parts(expression: str) -> tuple[str, str, str, str] | None:
    match = _EQUATION.search(expression.translate(_SUBSCRIPTS))
    if not match:
        return None
    left, operator, right, result = match.groups()
    return left, ("*" if operator in "×x*" else "/" if operator in "÷/" else operator), right, result


def _read_frame(engine, image: Path) -> list[dict]:
    import cv2
    import numpy as np
    pixels = cv2.imdecode(np.fromfile(str(image), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pixels is None:
        return []
    result, _ = engine(pixels)
    return [{"text": str(text).strip(), "confidence": round(float(confidence), 3), "box": box}
            for box, text, confidence in result or [] if str(text).strip()]


def _vision_read(image: Path, source_line: str, model: str, ollama_url: str) -> dict:
    import requests
    schema = {"type": "object", "properties": {
        "visible_labels": {"type": "array", "items": {"type": "string"}},
        "visible_equations": {"type": "array", "items": {"type": "string"}},
        "contradicts_spoken_line": {"type": "boolean"}, "reason": {"type": "string"}},
        "required": ["visible_labels", "visible_equations", "contradicts_spoken_line", "reason"]}
    response = requests.post(ollama_url.rstrip("/") + "/api/generate", json={
        "model": model, "stream": False, "think": False, "format": schema,
        "options": {"temperature": 0, "num_predict": 350},
        "prompt": ("Read only printed formulae, identifiers and arithmetic equations visible in this frame. "
                   "Count duplicate labels. Do not infer hidden or future animation frames. "
                   "Does the visible text directly contradict this spoken line? A different but related diagram "
                   "does not count as a contradiction. Spoken line: " + source_line),
        "images": [base64.b64encode(image.read_bytes()).decode("ascii")],
    }, timeout=120)
    response.raise_for_status()
    return json.loads(response.json()["response"])


def _candidate_priority(source: str, flagged: bool) -> int:
    if _formula_tokens(source) or _SPOKEN_EQUATION.search(source):
        return 3
    if _MATH.search(source) and re.search(r"\d", source):
        return 2
    if flagged or _CHEMISTRY.search(source):
        return 1
    return 0


def audit_visual_source(board, source_video: Path, output_dir: Path, config,
                        review_items: list[dict] | None = None, max_cues: int = 24,
                        ollama_url: str = "http://127.0.0.1:11434",
                        vision_model: str = "qwen3-vl:8b-instruct") -> dict:
    """Return an evidence report and conservative issues from sampled source frames."""
    visual_dir = output_dir / "visual_source_qa"
    visual_dir.mkdir(parents=True, exist_ok=True)
    report_path = visual_dir / "report.json"
    engine = _engine()
    flagged_ids = {match.group(1) for item in review_items or []
                   if (match := re.search(r"scene (\d+\.\d+)", str(item.get("where", ""))))}
    candidates = []
    for _scene, segment in board.all_segments():
        if segment.source_start is None or segment.source_end is None or not segment.source_narration:
            continue
        priority = _candidate_priority(segment.source_narration, segment.shot_id in flagged_ids)
        if priority:
            candidates.append((priority, segment))
    candidates.sort(key=lambda pair: (-pair[0], float(pair[1].source_start)))
    high_priority = [pair for pair in candidates if pair[0] >= 2]
    lower_priority = [pair for pair in candidates if pair[0] < 2]
    selected = high_priority + lower_priority[:max(0, max_cues - len(high_priority))]
    report = {"visual_qa_version": VISUAL_QA_VERSION, "source_video": str(source_video),
              "ocr_engine": "rapidocr_onnxruntime", "vision_model": vision_model,
              "status": "completed", "eligible_cues": len(candidates), "checked_cues": 0,
              "cues": [], "issues": []}
    if isinstance(engine, Exception):
        report["status"] = "ocr_unavailable"
        report["detail"] = str(engine)
        if candidates:
            report["issues"].append({"shot_id": "visual_ocr", "kind": "visual_ocr_unavailable",
                                     "source": "", "translation": "", "detail": "Source-video OCR could not run: " + str(engine)})
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    for _priority, segment in selected:
        shot_id = segment.shot_id or str(segment.segment_number)
        expected = list(dict.fromkeys(token for token, _a, _b in _formula_tokens(segment.source_narration)))
        samples = []
        token_frames = Counter()
        visible_token_frames = Counter()
        equation_frames = Counter()
        for number, when in enumerate(_sample_times(float(segment.source_start), float(segment.source_end)), start=1):
            image = visual_dir / f"shot_{shot_id.replace('.', '_')}_{number}.jpg"
            command = [_tool_path(config.ffmpeg_path, "ffmpeg"), "-v", "error", "-y", "-ss", str(when),
                       "-i", str(source_video), "-frames:v", "1", "-q:v", "2", str(image)]
            process = subprocess.run(command, capture_output=True, timeout=45, check=False)
            if process.returncode or not image.is_file():
                continue
            try:
                lines = _read_frame(engine, image)
            except (ValueError, OSError, RuntimeError):
                lines = []
            confident = [line["text"] for line in lines if line["confidence"] >= 0.5]
            for token in expected:
                if _exact_token_present(token, confident):
                    token_frames[token] += 1
            for token in {value for line in confident for value, _a, _b in _formula_tokens(line)}:
                visible_token_frames[token] += 1
            for equation in _math_equations(confident):
                equation_frames[equation["expression"]] += 1
            samples.append({"time_seconds": when, "image": str(image), "ocr_lines": lines,
                            "equations": _math_equations(confident)})
        entry = {"shot_id": shot_id, "start_seconds": segment.source_start, "end_seconds": segment.source_end,
                 "source_narration": segment.source_narration, "expected_tokens": expected,
                 "token_frame_counts": dict(token_frames), "visible_token_frame_counts": dict(visible_token_frames),
                 "samples": samples, "status": "no_visible_protected_text"}
        if _priority >= 2 and (not samples or all(not sample["ocr_lines"] for sample in samples)):
            entry["status"] = "visual_evidence_unavailable"
            report["issues"].append({"shot_id": shot_id, "kind": "visual_evidence_unavailable",
                                     "source": segment.source_narration, "translation": "",
                                     "detail": "No readable source-video frame was available for this formula or arithmetic cue; check the original slide and audio.",
                                     "evidence": [sample["image"] for sample in samples]})
        if expected and all(token_frames[token] > 0 for token in expected):
            entry["status"] = "visually_corroborated"
        elif expected and any(token_frames.values()):
            entry["status"] = "partial_visual_evidence"
        for expression, count in equation_frames.items():
            if count < 2:
                continue
            equation = next((equation for sample in samples for equation in sample["equations"]
                             if equation["expression"] == expression), None)
            if equation and equation["correct"] is False:
                report["issues"].append({"shot_id": shot_id, "kind": "visual_equation_inconsistent",
                                         "source": segment.source_narration, "translation": expression,
                                         "detail": f"Printed arithmetic evaluates to {equation['calculated']}, not the displayed result; inspect the source slide.",
                                         "evidence": [sample["image"] for sample in samples]})
            spoken = _spoken_equation(segment.source_narration)
            printed = _visual_equation_parts(expression)
            if spoken and printed and spoken != printed:
                entry["possible_equation_disagreement"] = {"spoken": spoken, "printed": printed}
        # Vision is a second reader only for a possible disagreement, not a substitute
        # for OCR or audio. A transient frame may omit a label that appears later.
        needs_formula_crosscheck = bool(expected) and not all(token_frames[token] > 0 for token in expected)
        needs_math_crosscheck = "possible_equation_disagreement" in entry
        if samples and (needs_formula_crosscheck or needs_math_crosscheck):
            text_samples = [sample for sample in samples if any(line["confidence"] >= 0.5 for line in sample["ocr_lines"])]
            if text_samples:
                if needs_math_crosscheck:
                    text_samples = [sample for sample in text_samples if sample["equations"]] or text_samples
                chosen = max(text_samples, key=lambda sample: sum(line["confidence"] for line in sample["ocr_lines"]))
                try:
                    entry["vision_crosscheck"] = _vision_read(Path(chosen["image"]),
                                                                  segment.source_narration, vision_model, ollama_url)
                except Exception as error:  # local Ollama outage or malformed model response stays visible in the report
                    entry["vision_crosscheck_error"] = str(error)[:300]
                    report["issues"].append({"shot_id": shot_id, "kind": "vision_crosscheck_unavailable",
                                             "source": segment.source_narration, "translation": "",
                                             "detail": "Visual OCR suggested a possible disagreement, but Qwen vision could not verify it: " + str(error)[:200],
                                             "evidence": [chosen["image"]]})
                vision = entry.get("vision_crosscheck", {})
                visible = [str(token) for token in vision.get("visible_labels", [])]
                # A conflict needs independent OCR evidence from at least two frames
                # plus a consistent vision-model statement; otherwise record uncertainty.
                alternative_tokens = [token for token, count in visible_token_frames.items()
                                      if count >= 2 and token not in expected and _exact_token_present(token, visible)]
                if (needs_formula_crosscheck and vision.get("contradicts_spoken_line") is True and
                    any(not _exact_token_present(token, visible) for token in expected) and alternative_tokens):
                    report["issues"].append({"shot_id": shot_id, "kind": "visual_formula_needs_review",
                                             "source": segment.source_narration, "translation": "",
                                             "detail": "Printed tokens " + ", ".join(alternative_tokens)
                                             + " may conflict with the spoken formula or identifier. Review all sampled frames; no narration was changed automatically.",
                                             "evidence": [sample["image"] for sample in samples]})
                    entry["status"] = "needs_review"
                disagreement = entry.get("possible_equation_disagreement")
                if (needs_math_crosscheck and vision.get("contradicts_spoken_line") is True
                        and disagreement and any(_visual_equation_parts(str(value)) == tuple(disagreement["printed"])
                                                 for value in vision.get("visible_equations", []))):
                    report["issues"].append({"shot_id": shot_id, "kind": "visual_equation_source_mismatch",
                                             "source": segment.source_narration,
                                             "translation": " ".join(disagreement["printed"]),
                                             "detail": "A printed equation differs from the spoken source. Review the frame and audio before correcting either.",
                                             "evidence": [sample["image"] for sample in samples]})
                    entry["status"] = "needs_review"
        report["cues"].append(entry)
    report["checked_cues"] = len(report["cues"])
    if len(selected) < len(candidates):
        report["status"] = "partial_low_priority"
        report["detail"] = (f"All {len(high_priority)} formula/equation cues were checked; "
                            f"{len(lower_priority) - (len(selected) - len(high_priority))} lower-priority cues were skipped.")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report
