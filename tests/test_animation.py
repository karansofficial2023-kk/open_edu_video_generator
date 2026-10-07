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
        self.board = Storyboard.model_validate_json((ROOT / 'samples/example_lesson.json').read_text())
        self.segment = self.board.all_segments()[0][1]
        self.segment.end = self.segment.speech_duration = 8

    def test_active_stage_changes_and_heading_remains_stable(self):
        config = AppConfig()
        config.render.burn_captions = False
        a = render_frame(self.segment, 'Lesson', 0, config)
        b = render_frame(self.segment, 'Lesson', 6, config)
        self.assertIsNotNone(ImageChops.difference(a, b).getbbox())
        self.assertIsNone(ImageChops.difference(a.crop((0, 0, 1280, 104)), b.crop((0, 0, 1280, 104))).getbbox())

    def test_real_speech_cues_override_approximate_times(self):
        self.segment.captions = [Caption(text='First the input', start=.2, end=1.5),
                                 Caption(text='Then it is processed', start=3.7, end=4.6)]
        self.assertEqual(stage_times(self.segment), [0, 3.7])

    def test_missing_cue_fails(self):
        self.segment.shot.cues[1] = 'not in narration'
        with self.assertRaisesRegex(ValueError, 'cue not found'):
            stage_times(self.segment)

    def test_invalid_templates_and_stage_order_rejected(self):
        for data in [{'template': 'execute_python'}, {'template': 'process', 'steps': ['only one']},
                     {'template': 'process', 'steps': ['a', 'b'], 'stage_fractions': [0, 1.5]},
                     {'template': 'pollination'}]:
            with self.assertRaises(ValidationError):
                Shot.model_validate(data)

    def test_caption_text_and_timing_preserved(self):
        self.segment.captions = [Caption(text=w, start=i, end=i + .8)
                                 for i, w in enumerate(self.segment.narration.split())]
        cues = phrase_captions(self.segment)
        self.assertEqual(' '.join(x.text for x in cues), self.segment.narration)
        self.assertEqual(cues[-1].end, self.segment.captions[-1].end)

    def test_review_uses_configured_terminology_only(self):
        self.segment.narration = 'This is a misspelt word.'
        with tempfile.TemporaryDirectory() as folder:
            self.assertFalse(any('terminology' in x['message'] for x in review_storyboard(self.board, folder)))
            config = AppConfig(text_replacements={'misspelt': 'misspelled'})
            findings = review_storyboard(self.board, folder, config)
            self.assertTrue(any(x['level'] == 'error' and x['code'] == 'terminology' for x in findings))

    def test_cleanup_applies_configured_replacements_and_adds_generic_shot(self):
        config = AppConfig(text_replacements={'misspelt': 'misspelled'})
        self.segment.narration = 'A Misspelt word, and another idea.'
        self.segment.shot = None
        normalize_storyboard(self.board, config)
        self.assertIn('Misspelled', self.segment.narration)
        self.assertTrue(self.segment.source_references)
        self.assertEqual(self.segment.shot.template, 'process')
        self.assertEqual(len(self.segment.shot.steps), 2)

    def test_contrast_narration_becomes_comparison(self):
        self.segment.narration = 'Copper conducts well, whereas rubber insulates.'
        self.segment.shot = None
        normalize_storyboard(self.board)
        self.assertEqual(self.segment.shot.template, 'comparison')
        self.assertEqual(self.segment.shot.steps, ['Copper conducts well', 'rubber insulates'])

    def test_user_voice_preserved_in_both_configs(self):
        for filename in ['config.yaml', 'config.12gb.yaml']:
            config = load_config(ROOT / filename)
            self.assertEqual(config.tts_provider, 'edge')
            self.assertEqual(config.edge_tts.voice, 'en-IN-NeerjaNeural')
            self.assertEqual(config.edge_tts.rate, '-20%')

    def test_no_coverage_topics_means_no_coverage_check(self):
        with tempfile.TemporaryDirectory() as folder:
            report = coverage_report(self.board, folder)
            findings = review_storyboard(self.board, folder, strict_coverage=True)
        self.assertFalse(report['checked'])
        self.assertFalse(any(x['code'] == 'coverage_incomplete' for x in findings))

    def test_coverage_checklist_comes_from_config_or_storyboard(self):
        self.board.required_topics = {'output': ['result']}
        config = AppConfig(coverage_topics={'missing one': ['zzz'], 'missing two': ['yyy']})
        with tempfile.TemporaryDirectory() as folder:
            report = coverage_report(self.board, folder, config.coverage_topics)
            self.assertEqual(report['missing_topics'], ['missing one', 'missing two'])
            warn = review_storyboard(self.board, folder, config)
            strict = review_storyboard(self.board, folder, config, strict_coverage=True)
        self.assertEqual([x['level'] for x in warn if x['code'] == 'coverage_incomplete'], ['warning'])
        self.assertEqual([x['level'] for x in strict if x['code'] == 'coverage_incomplete'], ['error'])

    def test_edge_stream_writes_audio_and_captures_word_offsets(self):
        class FakeStream:
            async def stream(self):
                yield {'type': 'audio', 'data': b'audio'}
                yield {'type': 'WordBoundary', 'offset': 10000000, 'duration': 5000000, 'text': 'Word'}
        with tempfile.TemporaryDirectory() as folder, patch('app.tts_edge.edge_tts.Communicate', return_value=FakeStream()) as mocked:
            output = Path(folder) / 'voice.mp3'
            words = asyncio.run(_synthesize_text('Word', output, AppConfig()))
            self.assertEqual(output.read_bytes(), b'audio')
            self.assertEqual((words[0].start, words[0].end), (1, 1.5))
            self.assertEqual(mocked.call_args.kwargs['boundary'], 'WordBoundary')


if __name__ == '__main__':
    unittest.main()
