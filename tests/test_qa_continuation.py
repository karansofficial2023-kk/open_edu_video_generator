import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from app.config import AppConfig
from app.qa import run_qa
from app.schema import Scene, Segment, Shot, Storyboard


class RepeatedFrameQaTests(unittest.TestCase):
    """A shot that deliberately continues the previous picture is not an accidental repeat."""

    def build(self, tmp: Path, continuation: bool):
        noise = (np.random.default_rng(3).random((1080, 1920, 3)) * 255).astype("uint8")
        Image.fromarray(noise).save(tmp / "still.png")
        video = tmp / "v.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "30", "-i", str(tmp / "still.png"), "-f", "lavfi", "-i",
                        "sine=frequency=300:duration=3", "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-r", "30", "-c:a", "aac", "-shortest",
                        str(video)], check=True)
        segs = []
        for n in range(3):
            seg = Segment(segment_number=n + 1, shot_id=f"1.{n + 1}", narration=f"Sentence number {n + 1} is spoken here.", start=float(n), end=float(n + 1),
                          shot=Shot(template="photo", heading="x", asset_path="a.png"))
            seg.continuation = continuation and n > 0
            segs.append(seg)
        return video, Storyboard(title="t", scenes=[Scene(scene_number=1, title="s", narration="n", segments=segs)])

    def qa_findings(self, continuation: bool):
        config = AppConfig()
        config.fps = 30
        config.production.semantic_qa = False                   # no vision model in this test
        with tempfile.TemporaryDirectory() as tmp:
            video, board = self.build(Path(tmp), continuation)
            return run_qa(video, board, config, Path(tmp), [])

    def test_identical_photos_are_flagged_unless_the_shot_is_a_continuation(self):
        flagged = [f for f in self.qa_findings(False)["findings"] if f["gate"] == "repeated_frame"]
        self.assertGreaterEqual(len(flagged), 1)
        self.assertEqual([f for f in self.qa_findings(True)["findings"] if f["gate"] == "repeated_frame"], [])


if __name__ == "__main__":
    unittest.main()
