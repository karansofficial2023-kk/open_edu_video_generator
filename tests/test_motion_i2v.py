import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.comfyui_client import ComfyUIClient
from app.config import AppConfig
from app.motion import VideoSource, derive_motion_prompt, fidelity, motion_candidates, similarity
from app.schema import Scene, Segment, Shot, Storyboard


def _picture(path: Path, shift: int = 0, seed: int = 1) -> Path:
    rng = np.random.default_rng(seed)
    image = cv2.GaussianBlur((rng.random((270, 480, 3)) * 255).astype(np.uint8), (0, 0), 6)
    image = np.roll(image, shift, axis=1)
    cv2.imwrite(str(path), image)
    return path


def _clip(path: Path, frames: list[np.ndarray]) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        writer.write(frame)
    writer.release()
    return path


def _segment(tmp: Path, template="photo", animate=True, labels=None, number=1, **kw) -> Segment:
    still = _picture(tmp / f"s{number}.png", seed=number)
    return Segment(segment_number=number, narration="A calm wide scene.", shot_id=f"1.{number}", animate=animate, labels=labels or [],
                   shot=Shot(template=template, heading="H", asset_path=str(still)), **kw)


class MotionTests(unittest.TestCase):
    def test_only_marked_label_free_pictures_are_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            board = Storyboard(title="T", scenes=[Scene(scene_number=1, title="S", narration="n", segments=[
                _segment(tmp, "title_card", number=1),
                _segment(tmp, "photo", animate=False, number=2),
                _segment(tmp, "photo", labels=["Anther"], number=3),
                _segment(tmp, "photo", number=4, visual_type="realistic_image"),
            ])])
            config = AppConfig()
            ids = [s.shot_id for s in motion_candidates(board, config)]
            self.assertEqual(ids, ["1.1", "1.4"])
            config.production.motion_max_clips = 1
            self.assertEqual([s.shot_id for s in motion_candidates(board, config)], ["1.1"])

    def test_fidelity_accepts_a_clip_that_starts_on_the_picture_and_rejects_a_different_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            still = _picture(tmp / "a.png", seed=1)
            base = cv2.imread(str(still))
            good = _clip(tmp / "good.mp4", [np.roll(base, i, axis=1) for i in range(0, 12, 2)])
            other = cv2.imread(str(_picture(tmp / "b.png", seed=7)))
            bad = _clip(tmp / "bad.mp4", [other] * 6)
            first, last = fidelity(still, good)
            self.assertGreater(first, 0.9)
            self.assertGreater(last, 0.55)
            self.assertLess(fidelity(still, bad)[0], 0.5)
            self.assertAlmostEqual(similarity(base, base), 1.0, places=3)

    def test_workflow_is_filled_by_node_type_and_links(self):
        config = AppConfig()
        config.production.motion_width, config.production.motion_height, config.production.motion_frames = 1920, 1088, 97
        client = ComfyUIClient(config)
        workflow = {
            "10": {"class_type": "LoadImage", "inputs": {"image": "x.png"}},
            "6": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
            "7": {"class_type": "CLIPTextEncode", "inputs": {"text": "oldneg"}},
            "11": {"class_type": "LTXVImgToVideo", "inputs": {"positive": ["6", 0], "negative": ["7", 0], "width": 8, "height": 8, "length": 9}},
            "69": {"class_type": "LTXVConditioning", "inputs": {"frame_rate": 1}},
            "72": {"class_type": "SamplerCustom", "inputs": {"noise_seed": 1}},
            "77": {"class_type": "SaveWEBM", "inputs": {"filename_prefix": "a", "fps": 1}},
        }
        out = client.patch_motion_workflow(workflow, "up.png", "POS", "NEG", "pre", 42)
        self.assertEqual(out["10"]["inputs"]["image"], "up.png")
        self.assertEqual((out["11"]["inputs"]["width"], out["11"]["inputs"]["height"], out["11"]["inputs"]["length"]), (1920, 1088, 97))
        self.assertEqual((out["6"]["inputs"]["text"], out["7"]["inputs"]["text"]), ("POS", "NEG"))
        self.assertEqual((out["69"]["inputs"]["frame_rate"], out["72"]["inputs"]["noise_seed"], out["77"]["inputs"]["fps"]), (24, 42, 24))
        self.assertEqual(workflow["6"]["inputs"]["text"], "old")           # the template itself is untouched

    def test_bundled_workflow_has_the_nodes_the_patcher_fills(self):
        import json
        text = json.loads(Path("workflows/ltx_i2v_1080_api.json").read_text(encoding="utf-8"))
        kinds = {n["class_type"] for n in text.values()}
        self.assertTrue({"LoadImage", "LTXVImgToVideo", "LTXVConditioning", "SamplerCustom", "SaveWEBM"} <= kinds)

    def test_video_source_reads_lazily_and_holds_the_last_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            frames = [np.full((90, 160, 3), 20 * i, np.uint8) for i in range(10)]
            source = VideoSource(_clip(tmp / "c.mp4", frames), (160, 90))
            self.assertEqual(source.count, 10)
            means = [np.asarray(source.frame_at(i / 24)).mean() for i in range(10)]
            self.assertTrue(all(b >= a - 1 for a, b in zip(means, means[1:])))
            held = np.asarray(source.frame_at(60.0)).mean()                 # long after the clip ended
            self.assertAlmostEqual(held, means[-1], delta=3)
            self.assertLess(np.asarray(source.frame_at(0.0)).mean(), 10)    # seeking backwards works
            self.assertLessEqual(len(source._cache), 3)                     # never the whole clip in memory
            source.close()

    def test_prompt_uses_the_scene_description_and_asks_for_gentle_motion(self):
        seg = Segment(segment_number=1, narration="n", image_prompt="a field of wheat at dawn")
        text = derive_motion_prompt(seg)
        self.assertTrue(text.startswith("a field of wheat at dawn."))
        self.assertIn("Gentle natural motion", text)


if __name__ == "__main__":
    unittest.main()
