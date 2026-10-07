import tempfile
import unittest
from pathlib import Path

from PIL import Image

from app.backdrops import choose, soft
from app.config import AppConfig
from app.charts import cards_frame
from app.schema import Scene, Segment, Shot, Storyboard


def seg(number, template, visual_type, asset=None):
    shot = Shot(template=template, heading="x", asset_path=asset,
                steps=["a card", "another card"] if template == "process" else [],
                stage_fractions=[0, 0.5] if template == "process" else [])
    return Segment(segment_number=number, narration=f"Narration number {number} is here.", visual_type=visual_type, shot=shot)


class BackdropTests(unittest.TestCase):
    """Pollination: 13 of 46 clips (the closing 'Keep learning.' lines among them) were text on a plain page."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.paths = []
        for index, colour in enumerate(("#d9a400", "#c0392b", "#27ae60")):
            path = Path(self.tmp.name) / f"p{index}.png"
            Image.new("RGB", (400, 300), colour).save(path)
            self.paths.append(str(path))

    def tearDown(self):
        self.tmp.cleanup()

    def test_cards_take_the_nearest_photo_of_their_own_scene(self):
        a, card_a, b = seg(1, "photo", "realistic_image", self.paths[0]), seg(2, "process", "realistic_image"), seg(3, "photo", "realistic_image", self.paths[1])
        c, card_c = seg(4, "photo", "realistic_image", self.paths[2]), seg(5, "process", "realistic_image")
        board = Storyboard(title="t", scenes=[Scene(scene_number=1, title="s1", narration="n", segments=[a, card_a, b]),
                                              Scene(scene_number=2, title="s2", narration="n", segments=[c, card_c])])
        chosen = choose(board)
        self.assertEqual(chosen[id(card_a)], self.paths[0])      # tie between photos 1 and 3: the one already seen
        self.assertEqual(chosen[id(card_c)], self.paths[2])      # the photo of its own scene
        self.assertNotIn(id(a), chosen)                           # photos need no backdrop

    def test_a_real_process_diagram_keeps_its_plain_page(self):
        photo, diagram = seg(1, "photo", "realistic_image", self.paths[0]), seg(2, "process", "process_steps")
        board = Storyboard(title="t", scenes=[Scene(scene_number=1, title="s", narration="n", segments=[photo, diagram])])
        self.assertEqual(choose(board), {})                       # visual_type process_steps was not a missing photograph

    def test_no_photos_means_no_backdrops(self):
        card = seg(1, "process", "realistic_image")
        board = Storyboard(title="t", scenes=[Scene(scene_number=1, title="s", narration="n", segments=[card])])
        self.assertEqual(choose(board), {})

    def test_the_wash_is_light_enough_for_dark_text(self):
        page = (244, 246, 242)
        image = soft(self.paths[1], (640, 360), page)             # a strong red photo
        self.assertEqual(image.size, (640, 360))
        mean = sum(sum(px) / 3 for px in image.resize((16, 9)).getdata()) / 144
        self.assertGreater(mean, 150)                              # still a light page: the heading stays readable
        r, g, b = image.getpixel((320, 180))
        self.assertGreater(r, g + 15)                              # but the photo's colour is visible

    def test_cards_draw_over_a_backdrop(self):
        config = AppConfig()
        plain = cards_frame((1920, 1080), config, "Closing", ["Keep learning", "We believe in you"], 1, 1.0, False)
        washed = cards_frame((1920, 1080), config, "Closing", ["Keep learning", "We believe in you"], 1, 1.0, False,
                             soft(self.paths[0], (1920, 1080), (244, 246, 242)))
        self.assertEqual(washed.size, (1920, 1080))
        self.assertNotEqual(plain.getpixel((1700, 950)), washed.getpixel((1700, 950)))        # the page behind the cards differs


if __name__ == "__main__":
    unittest.main()
