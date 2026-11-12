from __future__ import annotations

import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app.visual_source_qa import (_candidate_priority, _exact_token_present, _math_equations, _vision_read,
                                  _sample_times, audit_visual_source, load_visual_report,
                                  visual_input_signature, visual_report_output)


class VisualSourceQaTests(unittest.TestCase):
    def test_parenthesized_arithmetic_is_read_and_checked(self):
        equations = _math_equations(["(13 - 6) = 7", "(13 - 6) = 8", "23 - 13 = 10", "1 / 0 = 1"])
        self.assertEqual([item["expression"] for item in equations], ["13-6=7", "13-6=8", "23-13=10", "1/0=1"])
        self.assertEqual([item["correct"] for item in equations], [True, False, True, False])
        self.assertEqual(equations[-1]["calculated"], "undefined (division by zero)")

    def test_formula_requires_exact_boundary_and_accepts_subscripts(self):
        self.assertTrue(_exact_token_present("CaOCl2", ["Bleaching powder: CaOCl₂"]))
        self.assertFalse(_exact_token_present("CaOCl2", ["CaOCl20 is different"]))

    def test_protected_cues_take_priority_over_general_chemistry(self):
        self.assertEqual(_candidate_priority("The formula is CaOCl2.", False), 3)
        self.assertEqual(_candidate_priority("Bleaching powder is used here.", False), 1)
        self.assertEqual(_candidate_priority("Look at this.", True), 3)
        self.assertEqual(_sample_times(10, 20), [12.0, 15.0, 18.0])

    def test_vision_retries_malformed_json_once(self):
        bad = Mock()
        bad.json.return_value = {"response": '{"visible_labels": ["B1"], "reason": "'}
        good = Mock()
        good.json.return_value = {"response": '{"visible_labels": ["B1"], "visible_equations": [], "contradicts_spoken_line": false, "reason": "No conflict"}'}
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "frame.jpg"
            image.write_bytes(b"frame")
            with patch("requests.post", side_effect=[bad, good]) as post:
                self.assertEqual(_vision_read(image, "Brush B1.", "qwen3-vl:8b-instruct",
                                              "http://127.0.0.1:11434")["visible_labels"], ["B1"])
            self.assertEqual(post.call_count, 2)

    def test_formula_conflict_requires_repeated_ocr_and_vision_agreement(self):
        segment = SimpleNamespace(source_start=1, source_end=4, source_narration="CaOCl2 is the formula.",
                                  shot_id="1.1", segment_number=1)
        board = SimpleNamespace(all_segments=lambda: [(None, segment)])
        config = SimpleNamespace(ffmpeg_path="ffmpeg")

        def capture(command, **_kwargs):
            Path(command[-1]).write_bytes(b"frame")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            with (patch("app.visual_source_qa._engine", return_value=object()),
                  patch("app.visual_source_qa.subprocess.run", side_effect=capture),
                  patch("app.visual_source_qa._read_frame", return_value=[
                      {"text": "CaCO3", "confidence": 0.95, "box": []}]),
                  patch("app.visual_source_qa._vision_read", return_value={
                      "visible_labels": ["CaCO3"], "visible_equations": [],
                      "contradicts_spoken_line": True, "reason": "different formula"})):
                report = audit_visual_source(board, Path("source.mp4"), Path(directory), config)
            self.assertEqual([issue["kind"] for issue in report["issues"]], ["visual_formula_needs_review"])
            self.assertEqual(report["checked_cues"], 1)

    def test_pre_dubbing_report_is_reused_only_for_matching_inputs(self):
        segment = SimpleNamespace(source_start=1, source_end=4, source_narration="CaOCl2 is the formula.",
                                  shot_id="1.1", segment_number=1)
        board = SimpleNamespace(all_segments=lambda: [(None, segment)])
        config = SimpleNamespace(ffmpeg_path="ffmpeg")

        def capture(command, **_kwargs):
            Path(command[-1]).write_bytes(b"frame")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, contract, review = root / "lesson.mp4", root / "lesson_hi_contract.json", root / "review.json"
            video.write_bytes(b"video")
            contract.write_text("contract", encoding="utf-8")
            review.write_text("review", encoding="utf-8")
            output = visual_report_output(contract)
            signature = visual_input_signature(video, contract, review, 24, "qwen3-vl:8b-instruct")
            with (patch("app.visual_source_qa._engine", return_value=object()),
                  patch("app.visual_source_qa.subprocess.run", side_effect=capture),
                  patch("app.visual_source_qa._read_frame", return_value=[
                      {"text": "CaOCl2", "confidence": 0.95, "box": []}])):
                audit_visual_source(board, video, output, config, input_signature=signature)
            self.assertEqual(load_visual_report(output, signature)["reused"], True)
            review.write_text("changed", encoding="utf-8")
            changed = visual_input_signature(video, contract, review, 24, "qwen3-vl:8b-instruct")
            self.assertIsNone(load_visual_report(output, changed))

            saved = output / "visual_source_qa" / "report.json"
            import json
            data = json.loads(saved.read_text(encoding="utf-8"))
            data["issues"] = [{"kind": "vision_crosscheck_unavailable"}]
            saved.write_text(json.dumps(data), encoding="utf-8")
            self.assertIsNotNone(load_visual_report(output, signature))
            self.assertIsNone(load_visual_report(output, signature, retry_incomplete=True))

    def test_screening_finds_visual_equation_when_spoken_line_has_no_math(self):
        segment = SimpleNamespace(source_start=1, source_end=4, source_narration="Look at the screen.",
                                  shot_id="2.1", segment_number=1)
        board = SimpleNamespace(all_segments=lambda: [(None, segment)])
        config = SimpleNamespace(ffmpeg_path="ffmpeg")
        reads = []

        def capture(command, **_kwargs):
            Path(command[-1]).write_bytes(b"frame")
            return SimpleNamespace(returncode=0)

        def read(_engine, image):
            reads.append(image.name)
            return [{"text": "(13 - 6) = 7", "confidence": 0.95, "box": []}]

        with tempfile.TemporaryDirectory() as directory:
            with (patch("app.visual_source_qa._engine", return_value=object()),
                  patch("app.visual_source_qa.subprocess.run", side_effect=capture),
                  patch("app.visual_source_qa._read_frame", side_effect=read)):
                report = audit_visual_source(board, Path("source.mp4"), Path(directory), config)
            self.assertEqual(report["screened_unflagged_cues"], 1)
            self.assertEqual(report["checked_cues"], 1)
            self.assertEqual(len(reads), 3)  # midpoint OCR is reused, not read twice
            self.assertEqual([item["expression"] for item in report["cues"][0]["samples"][1]["equations"]],
                             ["13-6=7"])

    def test_semantic_review_without_printed_text_is_not_an_ocr_failure(self):
        segment = SimpleNamespace(source_start=1, source_end=4,
                                  source_narration="Now, let us see the schematic diagram of the motor.",
                                  shot_id="11.1", segment_number=1)
        board = SimpleNamespace(all_segments=lambda: [(None, segment)])
        config = SimpleNamespace(ffmpeg_path="ffmpeg")

        def capture(command, **_kwargs):
            Path(command[-1]).write_bytes(b"frame")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as directory:
            with (patch("app.visual_source_qa._engine", return_value=object()),
                  patch("app.visual_source_qa.subprocess.run", side_effect=capture),
                  patch("app.visual_source_qa._read_frame", return_value=[])):
                report = audit_visual_source(board, Path("source.mp4"), Path(directory), config,
                                             review_items=[{"where": "scene 11.1 semantic_audit"}])
            self.assertEqual(report["checked_cues"], 1)
            self.assertEqual(report["issues"], [])


if __name__ == "__main__":
    unittest.main()
