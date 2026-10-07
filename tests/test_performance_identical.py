import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from app.charts import card_shadow, cards_frame
from app.config import AppConfig
from app.production_renderer import ShotPlan, ShotRenderer
from app.schema import Segment, Shot


def legacy_shadow(size, box, s):
    """The original code: full-frame layer, blurred (twice) for every card in every frame."""
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle((box[0] + 6, box[1] + 10, box[2] + 6, box[3] + 10), radius=round(28 * s), fill=(0, 0, 0, 40))
    return shadow.filter(ImageFilter.GaussianBlur(round(12 * s)))


class FasterButIdenticalTests(unittest.TestCase):
    """Speed-ups must not change a single pixel of the output."""

    def test_card_shadow_patch_equals_the_full_frame_shadow(self):
        size = (1920, 1080)
        for box in [(96, 250, 900, 700), (960, 250, 1824, 700), (96, 274, 650, 724), (500, 300, 1400, 700), (96, 250, 1824, 700)]:
            background = Image.new("RGB", size, (244, 246, 242))
            old = background.copy()
            layer = legacy_shadow(size, box, 1.0)
            old.paste(layer, (0, 0), layer)
            new = background.copy()
            patch, origin = card_shadow(box, 1.0)
            new.paste(patch, origin, patch)
            self.assertIsNone(ImageChops.difference(old, new).getbbox(), f"shadow differs for {box}")

    def test_card_shadow_is_cached(self):
        card_shadow.cache_clear()
        card_shadow((96, 250, 900, 700), 1.0)
        card_shadow((96, 250, 900, 700), 1.0)
        self.assertEqual(card_shadow.cache_info().hits, 1)

    def renderer(self, tmp):
        config = AppConfig()
        config.output_resolution.width, config.output_resolution.height, config.fps = 1920, 1080, 30
        return ShotRenderer(config, Path(tmp))

    def test_steady_state_card_frames_are_reused_and_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.renderer(tmp)
            seg = Segment(segment_number=1, narration="Alpha beta gamma delta.", start=0.0, end=6.0, speech_duration=6.0,
                          shot=Shot(template="process", heading="Heading", steps=["First card", "Second card"], stage_fractions=[0, 0.5]))
            plan = ShotPlan(segment=seg, scene_title="s")
            first = r.base_frame(plan, 1.0, 6.0).convert("RGB")        # stage 0, well after the 0.4 s appear animation
            self.assertIn(("cards", 0), plan.steady)                      # ...so it was stored
            later = r.base_frame(plan, 1.9, 6.0).convert("RGB")
            self.assertIsNone(ImageChops.difference(first, later).getbbox())
            from app.animation_renderer import stage_times
            times = stage_times(seg)
            direct = cards_frame(r.size, r.config, "Heading", seg.shot.steps, 0, 1.0 - times[0], False, None).convert("RGB")
            self.assertIsNone(ImageChops.difference(first, direct).getbbox(), "cached frame differs from a fresh render")
            animating = r.base_frame(plan, times[1] + 0.1, 6.0).convert("RGB")              # the second card is still lifting in
            settled = r.base_frame(plan, times[1] + 1.0, 6.0).convert("RGB")
            self.assertIsNotNone(ImageChops.difference(animating, settled).getbbox())        # animation frames are never replaced by the cache

    def test_steady_state_equation_frames_are_reused_and_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            r = self.renderer(tmp)
            seg = Segment(segment_number=1, narration="x", formula_lines=["a = b", "c = d"], explain_steps=["one", "two"],
                          shot=Shot(template="formula", heading="Heading"))
            plan = ShotPlan(segment=seg, scene_title="s")
            a = r.base_frame(plan, 3.0, 8.0).convert("RGB")          # both lines shown and faded in
            self.assertIn(("formula", 2, 1), plan.steady)      # key: lines shown, explanation note shown
            b = r.base_frame(plan, 3.5, 8.0).convert("RGB")
            self.assertIsNone(ImageChops.difference(a, b).getbbox())
            fresh = r._formula_frame(ShotPlan(segment=seg, scene_title="s"), 3.0, 8.0).convert("RGB")
            self.assertIsNone(ImageChops.difference(a, fresh).getbbox())
            fading = r.base_frame(plan, 2.3 + 0.1, 8.0).convert("RGB")    # line 2 is fading in at 0.1 + 2.2 s
            self.assertIsNotNone(ImageChops.difference(fading, a).getbbox())
            self.assertNotIn(("formula", 2), {k for k in plan.steady if k == ("formula", 3)})      # nothing stored for a frame still animating
            fresh_fade = r._formula_frame(ShotPlan(segment=seg, scene_title="s"), 2.4, 8.0).convert("RGB")
            self.assertIsNone(ImageChops.difference(fading, fresh_fade).getbbox())                 # animation frames are never taken from the cache


class CapturedFrames(unittest.TestCase):
    """The whole clip, byte for byte, with the repeat-frame shortcut and without it."""

    class FakeStdin:
        def __init__(self):
            self.chunks = []

        def write(self, data):
            self.chunks.append(bytes(data))

        def close(self):
            pass

    class FakeProc:
        def __init__(self, sink):
            self.stdin = sink

        def wait(self):
            return 0

        def kill(self):
            pass

    def capture(self, template, shortcut):
        import subprocess
        from app import production_renderer as pr
        config = AppConfig()
        config.output_resolution.width, config.output_resolution.height, config.fps = 960, 540, 30
        sink = self.FakeStdin()
        original = pr.subprocess.Popen
        pr.subprocess.Popen = lambda *a, **k: self.FakeProc(sink)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                renderer = ShotRenderer(config, Path(tmp))
                if template == "process":
                    shot = Shot(template="process", heading="Heading", steps=["First card", "Second card"], stage_fractions=[0, 0.5])
                    seg = Segment(segment_number=1, narration="Alpha beta gamma delta epsilon zeta eta theta iota kappa.", start=0.0, end=3.0,
                                  speech_duration=3.0, shot=shot)
                else:
                    seg = Segment(segment_number=1, narration="Alpha beta gamma delta epsilon zeta eta theta.", start=0.0, end=3.0, speech_duration=3.0,
                                  formula_lines=["a = b", "c = d"], explain_steps=["one", "two"], shot=Shot(template="formula", heading="Heading"))
                plan = ShotPlan(segment=seg, scene_title="s")
                if not shortcut:
                    renderer.steady_key = lambda plan, t: None          # every frame is drawn from scratch
                renderer.render_clip(plan, Path(tmp) / "c.mp4", 90, None, None)
        finally:
            pr.subprocess.Popen = original
        return b"".join(sink.chunks), len(sink.chunks)

    def test_card_clip_is_byte_identical_with_the_shortcut(self):
        fast, frames = self.capture("process", True)
        slow, _ = self.capture("process", False)
        self.assertEqual(frames, 90)
        self.assertEqual(fast, slow)

    def test_equation_clip_is_byte_identical_with_the_shortcut(self):
        fast, frames = self.capture("formula", True)
        slow, _ = self.capture("formula", False)
        self.assertEqual(frames, 90)
        self.assertEqual(fast, slow)


if __name__ == "__main__":
    unittest.main()
