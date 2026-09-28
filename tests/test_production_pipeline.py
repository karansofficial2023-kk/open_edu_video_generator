from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from app.animation_renderer import render_frame
from app.config import AppConfig
from app.final_qa import _contrast_ratio, _review_subtitles, _same_visual_intent
from app.pipeline_state import PipelineState, stable_signature
from app.schema import Scene, Segment, Shot, Storyboard
from app.subtitles import _two_line_text, phrase_captions


class ProductionPipelineTests(unittest.TestCase):
    def test_subtitles_are_two_lines_and_metadata_is_rejected(self):
        wrapped = _two_line_text("Pollen moves from the anther to the receptive stigma of a flower.")
        self.assertLessEqual(len(wrapped.splitlines()), 2)
        segment = Segment(
            segment_number=1,
            narration="Visible explanation.",
            visual="",
            image_prompt="",
            shot=Shot(template="photo", motion={"subtitle": "Image requirement: hidden production note"}),
            speech_duration=3,
        )
        with self.assertRaisesRegex(ValueError, "metadata"):
            phrase_captions(segment)

    def test_subtitle_contrast_and_srt_line_gate(self):
        config = AppConfig()
        self.assertGreaterEqual(_contrast_ratio("#FFFFFF", "#101820"), 4.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "subtitles.srt"
            path.write_text("1\n00:00:00,000 --> 00:00:02,000\nLine one\nLine two\nLine three\n", encoding="utf-8")
            result = _review_subtitles(path, config)
        self.assertTrue(any("two lines" in item for item in result["failures"]))

    def test_pipeline_state_reuses_only_matching_approved_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip.mp4"
            output.write_bytes(b"approved")
            state = PipelineState(directory)
            signature = stable_signature("shot", 1)
            state.approved("clips", "clip_0001", signature, [output])
            restored = PipelineState(directory)
            self.assertIsNotNone(restored.reusable("clips", "clip_0001", signature, [output]))
            self.assertIsNone(restored.reusable("clips", "clip_0001", stable_signature("changed"), [output]))

    def test_same_visual_intent_allows_a_deliberate_hold(self):
        first = Segment(segment_number=1, narration="First", visual="Stable view", image_prompt="Same exact view")
        second = Segment(segment_number=2, narration="Second", visual="Stable view", image_prompt="Same exact view")
        self.assertTrue(_same_visual_intent(first, second))
        second.image_prompt = "Different apparatus"
        self.assertFalse(_same_visual_intent(first, second))

    def test_three_subject_render_trial(self):
        config = AppConfig()
        config.render.burn_captions = False
        source = Image.effect_noise((1280, 720), 45).convert("RGB")

        biology = Segment(
            segment_number=1,
            narration="Identify the two visible structures.",
            visual="Stable labeled specimen",
            image_prompt="Stable labeled specimen",
            shot=Shot(
                template="realistic_labeled_image",
                labels=[
                    {"text": "Structure A", "target_xy": "0.36,0.42"},
                    {"text": "Structure B", "target_xy": "0.64,0.42"},
                ],
                motion={"type": "arrow_draw_then_label_fade"},
            ),
            end=8,
            speech_duration=8,
        )
        chemistry = Segment(
            segment_number=1,
            narration="Apply the ionic contribution equation.",
            visual="Exact equation",
            image_prompt="",
            shot=Shot(template="formula", formula_lines=["Λ°ₘ = λ°₊ + λ°₋", "CH₃COOH ⇌ H⁺ + CH₃COO⁻"]),
            end=8,
            speech_duration=8,
        )
        mathematics = Segment(
            segment_number=1,
            narration="Differentiate and test the stationary point.",
            visual="Exact derivation",
            image_prompt="",
            shot=Shot(template="formula", formula_lines=["f(x) = x + 1/x", "f′(x) = 1 − 1/x²", "f′(x) = 0 ⇒ x = ±1"]),
            end=8,
            speech_duration=8,
        )

        images = [
            render_frame(biology, "Biology", 7, config, source),
            render_frame(chemistry, "Chemistry", 7, config),
            render_frame(mathematics, "Mathematics", 7, config),
        ]
        self.assertTrue(all(image.size == (1280, 720) for image in images))
        self.assertTrue(all(ImageChops.difference(images[0], image).getbbox() for image in images[1:]))


if __name__ == "__main__":
    unittest.main()
