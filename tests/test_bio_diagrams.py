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

    def test_the_part_named_in_the_spoken_sentence_is_lit(self):
        plain = bio_diagrams.render("female_system", (710, 662), 9.0)
        cervix = bio_diagrams.render("female_system", (710, 662), 9.0, "The cervix leads into the cervical canal.")
        ovary = bio_diagrams.render("female_system", (710, 662), 9.0, "The ovaries are oval-shaped.")
        self.assertNotEqual(plain.tobytes(), cervix.tobytes())
        self.assertNotEqual(cervix.tobytes(), ovary.tobytes())          # a different sentence lights a different part
        self.assertEqual(plain.tobytes(), bio_diagrams.render("female_system", (710, 662), 9.0).tobytes())     # nothing sticks between renders

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


class PlantDiagramTests(unittest.TestCase):
    def test_plant_text_gets_the_pollen_tube_figure_never_sperm_and_egg(self):
        from app import bio_diagrams as b
        text = "The pollen tube grows down the style and carries the male gamete to the ovule for fertilisation of the egg cell."
        self.assertEqual(b.concept_for(text), "plant_fertilisation")
        self.assertEqual(b.render("plant_fertilisation", (700, 600), 9.0).size, (700, 600))
        self.assertEqual(b.concept_for("The sperm fertilises the ovum and forms a zygote."), "fertilisation")


class PlantMicroTests(unittest.TestCase):
    def test_pollen_tube_steps_get_the_diagram_but_a_pollinator_shot_keeps_its_picture(self):
        from app import bio_diagrams as b
        for line in ("Once pollen lands on the stigma, it can germinate.",
                     "When the pollen lands on the stigma, a tube-like structure grows from it.",
                     "Upon reaching the ovary, the pollen tube fuses with an ovule and fertilizes it."):
            self.assertTrue(b.PLANT_MICRO.search(line), line)
            self.assertGreaterEqual(len(b.parts_named("plant_fertilisation", line)), 2, line)
        for line in ("The merging of pollen and ovule in the ovary is referred to as fertilization.",
                     "The pollen tube travels downward, enters the ovary and fuses with the ovule."):
            self.assertTrue(b.PLANT_MICRO.search(line), line)
        for line in ("The ovary appears swollen because it contains an ovule.",          # anatomy and aftermath keep their pictures
                     "Once the ovule has been fertilized, the ovary begins to grow and develops into a fruit.",
                     "It is attached to a long tube-like structure called the style, which leads to the ovary that contains ovules."):
            self.assertFalse(b.PLANT_MICRO.search(line), line)
        carry = "Pollen grains from the anther transfer to the stigma of another flower."
        self.assertFalse(b.PLANT_MICRO.search(carry))        # a bee carrying pollen is drawn, not diagrammed
        self.assertEqual(len(b.parts_named("plant_fertilisation", "The sperm fertilises the ovum.")), 0)      # animal text never gets the plant figure


    def test_only_micro_process_shots_of_a_plant_scene_become_diagrams(self):
        import tempfile
        from app.assets import StillProducer
        from app.config import AppConfig
        from app.schema import Scene, Segment, Shot, Storyboard
        micro = Segment(segment_number=1, shot_id="1.1", image_prompt="x", shot=Shot(template="photo"),
                        narration="When the pollen lands on the stigma, a tube-like structure grows from it.")
        bee = Segment(segment_number=2, shot_id="1.2", image_prompt="a bee", shot=Shot(template="photo"),
                      narration="Pollen from the anther sticks to the bee and reaches the stigma of the next flower.")
        scene = Scene(scene_number=1, title="Fertilisation", narration="", segments=[micro, bee])
        board_ = Storyboard(title="Pollination and Fertilisation in Flowering Plants", scenes=[scene])
        with tempfile.TemporaryDirectory() as tmp:
            StillProducer(board_, Path(tmp), AppConfig())._diagrams_for_body_shots()
        self.assertNotIn(micro.shot.template, {"photo", "video"})
        self.assertEqual(bee.shot.template, "photo")
