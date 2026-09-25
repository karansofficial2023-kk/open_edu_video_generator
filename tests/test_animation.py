import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image, ImageChops
from pydantic import ValidationError

from app.animation_renderer import _safe_display_heading, render_frame, stage_times
from app.config import AppConfig, load_config
from app.coverage import coverage_report
from app.label_overlay import (
    allows_precise_arrows,
    arrow_label_visibility,
    build_label_plans,
    draw_label_overlay,
    _segments_intersect,
)
from app.schema import Caption, Shot, Storyboard
from app.review import review_storyboard
from app.storyboard_cleanup import normalize_storyboard
from app.visuals import _apply_verified_target_coordinates, _casefold_target
from app.vision_qa import _target_collisions
from app.subtitles import phrase_captions
from app.tts_edge import _synthesize_text

ROOT = Path(__file__).resolve().parents[1]


class AnimationTests(unittest.TestCase):
    def setUp(self):
        self.board = Storyboard.model_validate_json((ROOT / 'samples/pollination_animated.json').read_text())
        self.segment = self.board.all_segments()[0][1]
        self.segment.end = self.segment.speech_duration = 8

    def test_pollen_moves_and_text_remains_stable(self):
        config = AppConfig()
        config.render.burn_captions = False
        a = render_frame(self.segment, 'Pollination', 0, config)
        b = render_frame(self.segment, 'Pollination', 6, config)
        self.assertIsNotNone(ImageChops.difference(a, b).getbbox())
        self.assertIsNone(ImageChops.difference(a.crop((0, 0, 1280, 104)), b.crop((0, 0, 1280, 104))).getbbox())

    def test_real_speech_cues_override_approximate_times(self):
        self.segment.captions = [Caption(text='Anthers produce pollen', start=.2, end=1.5),
                                 Caption(text='During pollination', start=3.7, end=4.6)]
        self.assertEqual(stage_times(self.segment), [0, 3.7])

    def test_missing_cue_fails(self):
        self.segment.shot.cues[1] = 'not in narration'
        with self.assertRaisesRegex(ValueError, 'cue not found'):
            stage_times(self.segment)

    def test_invalid_templates_and_stage_order_rejected(self):
        for data in [{'template': 'execute_python'}, {'template': 'process', 'steps': ['only one']},
                     {'template': 'protandry', 'stage_fractions': [0, 1.5]}]:
            with self.assertRaises(ValidationError):
                Shot.model_validate(data)

    def test_label_contract_maps_manual_coordinate_targets(self):
        shot = Shot(
            template="realistic_labeled_image",
            labels=[{"text": "Anther", "target_xy": "0.25,0.40", "placement": "left side"}],
        )
        plans, review = build_label_plans(shot, (1280, 720))
        self.assertFalse(review)
        self.assertEqual(plans[0].target, (320, 288))
        self.assertGreaterEqual(plans[0].confidence, 0.99)

    def test_low_confidence_target_is_rejected(self):
        shot = Shot(template="realistic_labeled_image", labels=[{"text": "Unknown gland"}])
        plans, review = build_label_plans(shot, (1280, 720))
        self.assertIsNone(plans[0].target)
        self.assertTrue(review)

    def test_semantic_description_never_guesses_a_scientific_coordinate(self):
        shot = Shot(template="realistic_labeled_image", labels=[{
            "text": "Anther",
            "placement": "target the pollen-bearing terminal part on the upper left",
        }])
        plans, review = build_label_plans(shot, (1280, 720))
        self.assertIsNone(plans[0].target)
        self.assertEqual(plans[0].reason, "verified image-specific coordinate required")
        self.assertTrue(review)

    def test_verified_vision_target_becomes_label_coordinate(self):
        segment = self.segment.model_copy(deep=True)
        segment.shot = Shot(
            template="realistic_labeled_image",
            labels=[{"text": "Anther", "placement": "pollen-bearing terminal structure"}],
        )
        _apply_verified_target_coordinates(segment, {
            "Anther": {"x": 0.375, "y": 0.42, "confidence": 0.93},
        })
        label = segment.shot.labels[0]
        self.assertEqual(label["target_xy"], "0.375000,0.420000")
        self.assertEqual(label["target_source"], "vision_verified")

    def test_unverified_vision_target_is_not_applied(self):
        segment = self.segment.model_copy(deep=True)
        segment.shot = Shot(template="realistic_labeled_image", labels=[{"text": "Stigma"}])
        _apply_verified_target_coordinates(segment, {
            "Stigma": {"x": 0.5, "y": 0.4, "confidence": 0.7},
        })
        self.assertNotIn("target_xy", segment.shot.labels[0])

    def test_review_manifest_proposal_lookup_is_case_insensitive(self):
        point = {"x": 0.24, "y": 0.42, "confidence": 0.9}
        self.assertEqual(_casefold_target({"Anther": point}, "anther"), point)

    def test_distinct_scientific_targets_cannot_share_one_endpoint(self):
        collisions = _target_collisions({
            "Anther": {"x": 0.55, "y": 0.46, "confidence": 0.9},
            "Stigma": {"x": 0.551, "y": 0.455, "confidence": 0.9},
        })
        self.assertEqual(collisions, [("Anther", "Stigma")])

    def test_well_separated_scientific_targets_pass_collision_gate(self):
        collisions = _target_collisions({
            "Anther": {"x": 0.29, "y": 0.42, "confidence": 0.9},
            "Stigma": {"x": 0.35, "y": 0.46, "confidence": 0.9},
        })
        self.assertEqual(collisions, [])

    def test_label_boxes_avoid_subtitle_safe_area_and_overlap(self):
        shot = Shot(
            template="realistic_labeled_image",
            labels=[
                {"text": "Anther", "target_xy": "0.35,0.42", "placement": "left side"},
                {"text": "Stigma", "target_xy": "0.58,0.36", "placement": "right side"},
            ],
        )
        plans, review = build_label_plans(shot, (1280, 720), subtitle_safe=(0, 600, 1280, 720))
        self.assertFalse(review)
        boxes = [plan.box for plan in plans]
        self.assertTrue(all(box and box[3] <= 600 for box in boxes))
        self.assertTrue(boxes[0][2] <= boxes[1][0] or boxes[1][2] <= boxes[0][0] or boxes[0][3] <= boxes[1][1] or boxes[1][3] <= boxes[0][1])

    def test_leader_line_crossing_detection(self):
        self.assertTrue(_segments_intersect((0, 0), (100, 100), (0, 100), (100, 0)))
        self.assertFalse(_segments_intersect((0, 0), (100, 0), (0, 40), (100, 40)))

    def test_arrow_first_then_label_fade_timing(self):
        arrow, label = arrow_label_visibility(0, 2, 0.35, 4.0, "arrow_draw_then_label_fade")
        self.assertGreater(arrow, 0)
        self.assertEqual(label, 0)
        arrow, label = arrow_label_visibility(0, 2, 1.25, 4.0, "arrow_draw_then_label_fade")
        self.assertEqual(arrow, 1)
        self.assertGreater(label, 0)

    def test_precise_arrows_are_blocked_for_motion_footage(self):
        self.assertFalse(allows_precise_arrows("short_motion_clip"))
        self.assertFalse(allows_precise_arrows("video_broll"))

    def test_metadata_never_becomes_visible_label_text(self):
        self.assertEqual(_safe_display_heading("Image requirement: macro flower with no text"), "")
        self.assertEqual(_safe_display_heading("Label placement: anther left side"), "")
        config = AppConfig()
        config.render.burn_captions = False
        segment = self.segment.model_copy(deep=True)
        segment.shot = Shot(
            template="realistic_labeled_image",
            heading="Image requirement: macro flower with no text",
            labels=[{"text": "Anther", "target_xy": "0.35,0.42", "placement": "left side"}],
            motion={"type": "arrow_draw_then_label_fade"},
        )
        segment.end = segment.speech_duration = 4
        image = render_frame(segment, "Pollination", 2.5, config)
        plans, _ = build_label_plans(segment.shot, image.size)
        overlay = draw_label_overlay(image, plans, 2.5, 4, "arrow_draw_then_label_fade")
        self.assertEqual(overlay.size, image.size)

    def test_labeled_photo_does_not_render_heading_over_image(self):
        config = AppConfig()
        config.render.burn_captions = False
        source = Image.new("RGB", (1280, 720), "#7BA36A")
        segment = self.segment.model_copy(deep=True)
        segment.shot = Shot(template="realistic_labeled_image", heading="Narration repeated at top")
        segment.end = segment.speech_duration = 4
        with_heading = render_frame(segment, "Pollination", 2, config, source)
        segment.shot.heading = ""
        without_heading = render_frame(segment, "Pollination", 2, config, source)
        self.assertIsNone(ImageChops.difference(with_heading, without_heading).getbbox())

    def test_caption_text_and_timing_preserved(self):
        self.segment.captions = [Caption(text=w, start=i, end=i + .8)
                                 for i, w in enumerate(self.segment.narration.split())]
        cues = phrase_captions(self.segment)
        self.assertEqual(' '.join(x.text for x in cues), self.segment.narration)
        self.assertEqual(cues[-1].end, self.segment.captions[-1].end)

    def test_explicit_blank_subtitle_suppresses_narration_caption(self):
        self.segment.shot.motion["subtitle"] = ""
        self.assertEqual(phrase_captions(self.segment), [])

    def test_explicit_subtitle_is_used_instead_of_narration(self):
        self.segment.shot.motion["subtitle"] = "Viewer-facing summary only."
        text = " ".join(cue.text for cue in phrase_captions(self.segment))
        self.assertEqual(text, "Viewer-facing summary only.")

    def test_review_rejects_known_bad_term(self):
        self.segment.narration = 'This is glystogamy.'
        with tempfile.TemporaryDirectory() as folder:
            findings = review_storyboard(self.board, folder)
            self.assertTrue(any(x['level'] == 'error' and 'terminology' in x['message'] for x in findings))

    def test_legacy_storyboard_cleanup_fixes_terms_and_adds_shots(self):
        self.segment.narration = (
            'The male anther curls in on itself to deposit pollen on the sticky female stigma. '
            'This is glystogamy and hetrostyle.'
        )
        self.segment.visual = 'Show Glystogamy.'
        self.segment.image_prompt = 'A diagram of Glystogamy.'
        self.segment.keywords = ['glystogamy', 'hetrostyle']
        self.board.title = 'Gystogamy'
        self.board.scenes[0].title = 'Protogenic and Gystogamy'
        self.segment.shot = None
        normalize_storyboard(self.board)
        text = ' '.join([
            self.board.title,
            self.board.scenes[0].title,
            self.segment.narration,
            self.segment.visual,
            self.segment.image_prompt,
        ]).lower()
        self.assertNotRegex(text, r'glystogamy|gystogamy|hetrostyle|protogenic|anther curls')
        self.assertIn('cleistogamy', text)
        self.assertTrue(self.segment.source_references)
        self.assertIsNotNone(self.segment.shot)

    def test_ambiguous_follow_up_visual_inherits_previous_subject(self):
        first = self.board.all_segments()[0][1]
        first.narration = "A sunflower head contains many small florets."
        second = first.model_copy(deep=True)
        second.segment_number = 2
        second.narration = "At this scale, we can observe them closely."
        second.image_prompt = "Source-faithful visible subject and action."
        second.shot = Shot(template="photo", heading="Close view")
        self.board.scenes[0].segments = [first, second]
        normalize_storyboard(self.board)
        self.assertIn("sunflower head", second.image_prompt.lower())
        self.assertIn("no unrelated substitute subject", second.image_prompt.lower())

    def test_concrete_follow_up_does_not_inherit_unrelated_previous_agent(self):
        first = self.board.all_segments()[0][1]
        first.narration = "A bee visits a sunflower."
        second = first.model_copy(deep=True)
        second.segment_number = 2
        second.narration = "Pollen from the anther reaches the stigma during self-pollination."
        second.image_prompt = "Source-faithful visible subject and action."
        second.shot = Shot(template="photo", heading="Self-pollination")
        self.board.scenes[0].segments = [first, second]
        normalize_storyboard(self.board)
        self.assertNotIn("previous narration: a bee", second.image_prompt.lower())
        self.assertIn("no bee", second.image_prompt.lower())

    def test_dangling_strategy_teaser_becomes_one_visible_idea(self):
        segment = self.board.all_segments()[0][1]
        segment.narration = (
            "While sunflowers are typically pollinated by bees, they have a clever strategy "
            "to ensure pollination when bees are absent."
        )
        segment.visual = segment.narration
        segment.image_prompt = segment.narration
        segment.shot = Shot(template="photo", heading="Pollination strategy")
        normalize_storyboard(self.board)
        self.assertEqual(segment.narration, "Sunflowers are typically pollinated by bees.")
        self.assertNotIn("strategy", segment.image_prompt.lower())

    def test_transfer_sentence_becomes_stable_labeled_image(self):
        segment = self.board.all_segments()[0][1]
        segment.narration = "Pollen grains from the anther are transferred to the receptive stigma."
        segment.shot = Shot(template="photo", heading="Pollen transfer")
        normalize_storyboard(self.board)
        self.assertEqual(segment.shot.template, "realistic_labeled_image")
        self.assertEqual(
            [item["text"] for item in segment.shot.labels],
            ["Anther", "Receptive Stigma"],
        )
        self.assertEqual(segment.shot.motion["type"], "arrow_draw_then_label_fade")
        self.assertIn("simultaneously visible", segment.image_prompt.lower())
        self.assertIn("anther", segment.image_prompt.lower())

    def test_user_voice_preserved_in_both_configs(self):
        for filename in ['config.yaml', 'config.12gb.yaml']:
            config = load_config(ROOT / filename)
            self.assertEqual(config.tts_provider, 'edge')
            self.assertEqual(config.edge_tts.voice, 'en-IN-NeerjaNeural')
            self.assertEqual(config.edge_tts.rate, '-20%')

    def test_types_of_pollination_coverage_flags_missing_topics(self):
        self.board.title = 'Types of Pollination'
        self.board.source = 'Pollination by sunflower only. It mentions pollen and stigma.'
        with tempfile.TemporaryDirectory() as folder:
            report = coverage_report(self.board, folder)
        self.assertLess(report['coverage_ratio'], .75)
        self.assertIn('wind pollination', report['missing_topics'])

    def test_edge_stream_writes_audio_and_captures_word_offsets(self):
        class FakeStream:
            async def stream(self):
                yield {'type': 'audio', 'data': b'audio'}
                yield {'type': 'WordBoundary', 'offset': 10000000, 'duration': 5000000, 'text': 'Pollen'}
        with tempfile.TemporaryDirectory() as folder, patch('app.tts_edge.edge_tts.Communicate', return_value=FakeStream()) as mocked:
            output = Path(folder) / 'voice.mp3'
            words = asyncio.run(_synthesize_text('Pollen', output, AppConfig()))
            self.assertEqual(output.read_bytes(), b'audio')
            self.assertEqual((words[0].start, words[0].end), (1, 1.5))
            self.assertEqual(mocked.call_args.kwargs['boundary'], 'WordBoundary')


if __name__ == '__main__':
    unittest.main()
