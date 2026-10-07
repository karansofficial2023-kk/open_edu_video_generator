import json
import tempfile
import unittest
from pathlib import Path

from app.resilience import atomic_write_text, read_json_or_none, retry, wait_for


class ResilienceTests(unittest.TestCase):
    def test_retry_recovers_after_transient_faults(self):
        calls, pauses = [], []

        def flaky():
            calls.append(1)
            if len(calls) < 3:
                raise OSError("busy")
            return "ok"

        self.assertEqual(retry(flaky, attempts=3, base_delay=1, sleep=pauses.append), "ok")
        self.assertEqual(pauses, [1, 2])

    def test_retry_gives_up_with_last_error(self):
        with self.assertRaises(ValueError):
            retry(lambda: (_ for _ in ()).throw(ValueError("bad")), attempts=2, sleep=lambda s: None)

    def test_corrupt_json_counts_as_not_cached(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "r.json")
            path.write_text('{"half": ', encoding="utf-8")
            self.assertIsNone(read_json_or_none(path))
            self.assertIsNone(read_json_or_none(Path(tmp, "missing.json")))

    def test_atomic_write_replaces_and_leaves_no_temp(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp, "r.json")
            atomic_write_text(path, json.dumps({"a": 1}))
            atomic_write_text(path, json.dumps({"a": 2}))
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), {"a": 2})
            self.assertEqual([p.name for p in Path(tmp).iterdir()], ["r.json"])

    def test_wait_for_service_coming_back(self):
        state = {"n": 0}

        def check():
            state["n"] += 1
            return state["n"] >= 3

        self.assertTrue(wait_for(check, timeout=30, interval=1, sleep=lambda s: None))
        self.assertFalse(wait_for(lambda: False, timeout=3, interval=1, sleep=lambda s: None))


if __name__ == "__main__":
    unittest.main()


class SelfHealingIntegrationTests(unittest.TestCase):
    def test_comfyui_client_waits_and_retries_but_not_for_rejected_workflows(self):
        import requests
        from app.comfyui_client import ComfyUIClient, WorkflowRejected
        from app.config import AppConfig
        import app.resilience as resilience

        client = ComfyUIClient(AppConfig())
        client.is_available = lambda: True
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise requests.ConnectionError("refused")
            return "image"

        original = resilience.time.sleep
        resilience.time.sleep = lambda s: None
        try:
            self.assertEqual(client._resilient(flaky, "still"), "image")
            self.assertEqual(calls["n"], 3)
            rejected = {"n": 0}

            def bad():
                rejected["n"] += 1
                raise WorkflowRejected("HTTP 400")

            with self.assertRaises(WorkflowRejected):
                client._resilient(bad, "still")
            self.assertEqual(rejected["n"], 1)               # a rejected workflow is not repeated
        finally:
            resilience.time.sleep = original

    def test_tts_retry_window_covers_a_few_minutes(self):
        import app.resilience as resilience
        from app.tts_edge import _retry

        pauses, calls = [], {"n": 0}
        original = resilience.time.sleep
        resilience.time.sleep = pauses.append

        def down():
            calls["n"] += 1
            raise OSError("no network")

        try:
            with self.assertRaises(OSError):
                _retry(down)
        finally:
            resilience.time.sleep = original
        self.assertEqual(calls["n"], 8)
        self.assertGreaterEqual(sum(pauses), 150)


class TeacherReviewFixTests(unittest.TestCase):
    """The five findings of the 6 Oct review of five rendered lessons, each as a rule (no topic is named in the code)."""

    def _board(self, rows, subject=""):
        from app.schema import Scene, Segment, Shot, Storyboard
        segments = []
        for number, (narration, kind, extra) in enumerate(rows, start=1):
            segments.append(Segment(segment_number=number, narration=narration, shot_id=f"1.{number}",
                                    shot=Shot(template=kind, heading="H", **({"steps": ["A b c", "D e f"]} if kind == "process" else {})), **extra))
        return Storyboard(title="T", subject=subject, scenes=[Scene(scene_number=1, title="S", narration="n", segments=segments)])

    def test_an_equation_card_without_an_equation_becomes_cards_not_a_blank_screen(self):
        from app.storyboard_cleanup import normalize_storyboard
        board = self._board([("Through experiments the law was found.", "formula", {"formula_lines": ["="]}),
                             ("Kohlrausch's law states this.", "formula", {"formula_lines": ["Λ₀ = Λ⁺ + Λ⁻"]})])
        normalize_storyboard(board)
        first, second = board.scenes[0].segments
        self.assertEqual(first.shot.template, "process")                        # "=" alone is not an equation
        self.assertEqual(second.shot.template, "formula")                       # Greek letters and sub/superscripts are
        self.assertEqual(second.formula_lines, ["Λ₀ = Λ⁺ + Λ⁻"])

    def test_cards_are_complete_ideas(self):
        from app.assets import complete_cards
        self.assertEqual(complete_cards("We are asked to find the value of x."), ["Value", "We are asked to find the value of x."])
        self.assertEqual(complete_cards("Let the given number be x. This means we are adding 1 over x to x."),
                         ["Let the given number be x.", "This means we are adding 1 over x to x."])
        self.assertEqual(complete_cards("What draws a pollinator to a flower?")[0], "Pollinator")

    def test_remarks_are_spoken_over_the_previous_picture(self):
        from app.storyboard_cleanup import merge_filler_shots
        board = self._board([("The wire is stretched to twice its length.", "photo", {}), ("Clear?", "photo", {}),
                             ("Keep learning.", "photo", {}), ("Now the resistance doubles.", "photo", {}),
                             ("Fascinating, isn't it?", "formula", {"formula_lines": ["R = 2R_0"]})])
        self.assertEqual(merge_filler_shots(board), 2)
        shots = board.scenes[0].segments
        self.assertEqual([s.shot_id for s in shots], ["1.1", "1.4", "1.5"])      # a shot with its own equation is never merged away
        self.assertEqual(shots[0].narration, "The wire is stretched to twice its length. Clear? Keep learning.")

    def test_subject_comes_from_the_storyboard_then_from_its_words(self):
        from app.subject import BIOLOGY, MATHS, PHYSICS, subject_of
        self.assertEqual(subject_of(self._board([("x", "photo", {})], subject="Mathematics")), MATHS)
        words = self._board([("The current in the wire depends on its resistance and the voltage.", "photo", {}),
                             ("A thicker wire has lower resistance, so more current flows.", "photo", {})])
        self.assertEqual(subject_of(words), PHYSICS)
        self.assertEqual(subject_of(self._board([("x", "photo", {})], subject="Botany")), BIOLOGY)

    def test_derivation_lines_follow_the_explanation(self):
        from app.production_renderer import formula_pace
        self.assertEqual(formula_pace(1, 30), 2.2)
        self.assertAlmostEqual(formula_pace(5, 40), 6.0)          # 5 lines over 75 % of a 40 s explanation
        self.assertEqual(formula_pace(3, 4), 2.2)                 # never faster than one line per 2.2 s
        self.assertEqual(formula_pace(2, 60), 8.0)                # never a line waiting more than 8 s

    def test_the_last_subtitle_leaves_during_a_silent_hold(self):
        from app.captions import _extend_to_next
        from app.schema import Caption, Segment
        seg = Segment(segment_number=1, narration="a b", start=0, end=20)
        cues = _extend_to_next([Caption(text="a", start=0, end=2), Caption(text="b", start=2, end=5)], seg)
        self.assertEqual([c.end for c in cues], [2, 6.5])


class SubtitlePacingTests(unittest.TestCase):
    def test_a_slow_long_sentence_changes_subtitle_at_its_commas(self):
        from app.captions import build_cues
        from app.config import AppConfig
        from app.schema import Caption, Segment
        words = ("Through experiments, Kohlrausch discovered that the limiting molar conductivity of any electrolyte, "
                 "measured at infinite dilution, equals the sum of the limiting molar conductivities of its individual ions.").split()
        captions = [Caption(text=w, start=i * 0.6, end=(i + 1) * 0.6) for i, w in enumerate(words)]   # slow speech: 0.6 s per word
        seg = Segment(segment_number=1, narration=" ".join(words), captions=captions, speech_duration=len(words) * 0.6,
                      start=0, end=len(words) * 0.6 + 1)
        cues = build_cues(seg, AppConfig().production.subtitle, (1920, 1080))
        self.assertGreaterEqual(len(cues), 3)
        self.assertTrue(all(c.end - c.start <= 9.0 for c in cues), [(c.text, c.end - c.start) for c in cues])
