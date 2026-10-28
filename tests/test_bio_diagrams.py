import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from app import bio_diagrams, diagrams
from app.assets import StillProducer
from app.config import AppConfig
from app.schema import Scene, Segment, Shot, Storyboard


class BioDiagramTests(unittest.TestCase):
    def test_concept_is_chosen_from_the_words_not_the_topic(self):
        cases = {"The uterus connects to the fallopian tubes, and the ovary releases an egg.": "female_system",
                 "Sperm pass from the testis through the epididymis and the vas deferens.": "male_system",
                 "The pituitary gland releases a hormone into the bloodstream and a feedback signal returns.": "hormone_loop",
                 "Ovulation happens about day 14 of the menstrual cycle.": "menstrual_cycle",
                 "The anther holds pollen and the stigma receives it; the petal attracts insects.": "flower",
                 "After fertilisation the zygote divides to form an embryo.": "fertilisation",
                 "During germination the radicle emerges first, then the plumule.": "seed_germination"}
        for text, concept in cases.items():
            self.assertEqual(bio_diagrams.concept_for(text), concept, text)
        self.assertIsNone(bio_diagrams.concept_for("Plants need sunlight."))
        self.assertEqual(diagrams.concept_for("The uterus and the ovary.", biology=True), "female_system")

    def test_every_concept_renders_at_the_requested_size_and_reveals_over_time(self):
        for concept in bio_diagrams.BIO_CONCEPTS:
            early = diagrams.render(concept, (710, 662), 0.0)
            late = diagrams.render(concept, (710, 662), 9.0)
            self.assertEqual(late.size, (710, 662), concept)
            self.assertNotEqual(early.tobytes(), late.tobytes(), concept)       # parts appear one after the other

    def test_body_shots_are_explained_by_a_diagram_without_asking_the_image_model(self):
        uterus = Segment(segment_number=1, shot_id="2.1", narration="The uterus is connected to the ovary by the fallopian tube.",
                         image_prompt="x", shot=Shot(template="photo"))
        cervix = Segment(segment_number=2, shot_id="2.2", narration="The cervix leads into the uterus.", image_prompt="x", shot=Shot(template="photo"))
        bee = Segment(segment_number=1, shot_id="3.1", narration="A bee collects nectar from the flower.", image_prompt="x", shot=Shot(template="photo"))
        board = Storyboard(title="Reproduction", subject="Biology", scenes=[
            Scene(scene_number=2, title="Female system", narration="", segments=[uterus, cervix]),
            Scene(scene_number=3, title="Pollination", narration="", segments=[bee])])
        with tempfile.TemporaryDirectory() as tmp:
            StillProducer(board, Path(tmp), AppConfig())._diagrams_for_body_shots()
        self.assertNotIn(uterus.shot.template, {"photo", "video"})
        self.assertNotIn(cervix.shot.template, {"photo", "video"})
        self.assertEqual(bee.shot.template, "photo")

    def test_a_label_is_asked_for_again_when_the_model_returns_an_unusable_box(self):
        from app import labels
        bad = {"visible": True, "box": "none", "confidence": 0.9}
        good = {"visible": True, "box": [100, 100, 300, 300], "point": [200, 200], "confidence": 0.9}
        verdict = {"shows_target": True, "confidence": 0.9, "what_is_shown": "uterus"}
        spot = {"under_cross_is_target": True, "under_cross": "uterus"}
        with mock.patch("app.labels.vision.ask", side_effect=[bad, good, verdict, spot]) as ask:
            anchor = labels._locate_one(Image.new("RGB", (400, 400), "black"), "Uterus", "the uterus", AppConfig())
        self.assertNotIsInstance(anchor, str)
        self.assertEqual(ask.call_count, 4)


if __name__ == "__main__":
    unittest.main()
