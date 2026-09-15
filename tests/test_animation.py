import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import ImageChops
from pydantic import ValidationError

from app.animation_renderer import render_frame, stage_times
from app.config import AppConfig, load_config
from app.coverage import coverage_report
from app.schema import Caption, Shot, Storyboard
from app.review import review_storyboard
from app.storyboard_cleanup import normalize_storyboard
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

    def test_caption_text_and_timing_preserved(self):
        self.segment.captions = [Caption(text=w, start=i, end=i + .8)
                                 for i, w in enumerate(self.segment.narration.split())]
        cues = phrase_captions(self.segment)
        self.assertEqual(' '.join(x.text for x in cues), self.segment.narration)
        self.assertEqual(cues[-1].end, self.segment.captions[-1].end)

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
