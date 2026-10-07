import unittest
from pathlib import Path

from app import board
from app.schema import Segment, Shot


def _seg(n, narration, template="process", **extra):
    shot = Shot(template=template, heading="H", steps=["A b", "C d"] if template == "process" else [])
    return Segment(segment_number=n, narration=narration, shot_id=f"1.{n}", shot=shot, **extra)


class _Plan:
    def __init__(self, segment, scene_title="Conductors"):
        self.segment, self.scene_title, self.video, self.board = segment, scene_title, None, None


class BoardTests(unittest.TestCase):
    def test_each_layout_adds_the_right_items(self):
        defn = board.items_for(_seg(1, "Substances that allow current to pass are called conductors."))
        self.assertEqual([i["kind"] for i in defn], ["box"])
        bullets = board.items_for(_seg(2, "Conduction increases with temperature. Ions carry the charge. Clear?"))
        self.assertEqual([i["text"] for i in bullets], ["Conduction increases with temperature", "Ions carry the charge"])
        table = board.items_for(_seg(3, "Metals use electrons whereas electrolytes use ions.", layout="table",
                                     table_rows=["Metallic | Electrolytic", "Electrons | Ions"]))
        self.assertEqual([(i["kind"], i["header"]) for i in table], [("row", True), ("row", False)])
        derivation = board.items_for(_seg(4, "Ohm's law gives the current.", "formula", formula_lines=["V = IR"], explain_steps=["Voltage"]))
        self.assertEqual([i["kind"] for i in derivation], ["lead", "eq"])
        self.assertEqual(derivation[1]["note"], "Voltage")
        gallery = board.items_for(_seg(5, "It is used in paints, varnishes and perfumes.", layout="gallery",
                                       gallery_items=["Paints", "Varnishes", "Perfumes"]))
        self.assertEqual(gallery[-1], {"kind": "chips", "items": ["Paints", "Varnishes", "Perfumes"]})
        self.assertEqual(board.layout_of(_seg(6, "x", layout="table", table_rows=["only one | row"])), "bullets")

    def test_a_part_builds_up_on_one_board_and_a_table_keeps_one_header(self):
        from app.production import _plan_board_pages
        plans = [_Plan(_seg(1, "Metals use electrons whereas electrolytes use ions.", layout="table",
                            table_rows=["Metallic | Electrolytic", "Electrons | Ions"])),
                 _Plan(_seg(2, "There is no chemical change in metals.", layout="table",
                            table_rows=["Metallic | Electrolytic", "No chemical change | Chemical decomposition"])),
                 _Plan(_seg(3, "A photo.", "photo")),
                 _Plan(_seg(4, "Ohm's law holds.", "formula", formula_lines=["V = IR"]))]
        for p in plans:
            p.segment.start, p.segment.end, p.segment.speech_duration = 0, 6, 5
        _plan_board_pages(plans, "Conductance")
        self.assertEqual(len(plans[1].board["previous"]), 2)                       # the first shot's table stays on the board
        self.assertEqual([i["header"] for i in plans[1].board["current"]], [False])  # header written once
        self.assertIsNone(plans[2].board)                                          # a picture is not a board
        self.assertEqual(plans[3].board["previous"], [])                           # and starts a new page after it

    def test_a_frame_is_drawn_and_reveals_progressively(self):
        items = board.items_for(_seg(1, "Conduction increases with temperature. Ions carry the charge."))
        theme = board.theme_for("Lesson")
        early = board.frame((1280, 720), theme, "Conductors", [], items, 0.4, 4.0)
        late = board.frame((1280, 720), theme, "Conductors", [], items, 4.2, 4.0)
        self.assertEqual(early.size, (1280, 720))
        import numpy as np
        self.assertGreater(np.asarray(late).astype(int).sum(), np.asarray(early).astype(int).sum())   # more text written later


if __name__ == "__main__":
    unittest.main()


class BoardVisualTests(unittest.TestCase):
    def test_concept_diagrams_are_chosen_from_the_narration_and_drawn(self):
        from app import diagrams
        self.assertEqual(diagrams.concept_for("The resistance of a wire depends on its length and its cross-section."), "conductor")
        self.assertEqual(diagrams.concept_for("An electromagnetic wave: the electric and magnetic fields oscillate."), "wave")
        self.assertIsNone(diagrams.concept_for("Plants make food from sunlight."))
        self.assertEqual(diagrams.concept_for("The bulb glows when the switch closes the circuit with a battery."), "circuit")
        self.assertEqual(diagrams.concept_for("A convex lens bends parallel rays to its focus by refraction."), "lens")
        self.assertEqual(diagrams.concept_for("The field lines of a bar magnet run from the north pole to the south pole."), "magnet")
        self.assertEqual(diagrams.concept_for("Like charges repel and unlike charges attract."), "charge")
        self.assertEqual(diagrams.concept_for("Friction opposes the applied force on the block."), "force")
        self.assertEqual(diagrams.concept_for("The time period of a simple pendulum depends on its length; the bob swings."), "pendulum")
        self.assertEqual(diagrams.concept_for("An electromagnetic wave has electric and magnetic fields; the wave travels."), "wave")
        for concept in diagrams.CONCEPTS:
            for t in (0.0, 1.3, 7.9):
                image = diagrams.render(concept, (700, 600), t, "electric and magnetic")
                self.assertEqual(image.size, (700, 600))
                self.assertGreater(image.getchannel("A").getbbox()[2], 300, concept)

    def test_compound_names_follow_naming_rules(self):
        from app import molecules
        self.assertEqual(molecules.candidates("a weak electrolyte like acetic acid"), ["acetic acid"])
        self.assertIn("sodium chloride", molecules.candidates("strong electrolytes such as sodium chloride"))
        self.assertEqual(molecules.candidates("The resistance of the material depends on its nature."), [])

    def test_a_molecule_is_drawn_when_rdkit_is_available(self):
        from app import molecules
        if molecules.model_for("CC(=O)O") is None:
            self.skipTest("RDKit not installed")
        image = molecules.render("CC(=O)O", (600, 500), 0.5)
        self.assertGreater(len(set(image.convert("RGB").getdata())), 50)          # shaded spheres, not an empty picture

    def test_the_main_visual_takes_the_left_and_the_board_the_right(self):
        items = board.items_for(_seg(1, "Conduction increases with temperature."))
        frame = board.frame((1920, 1080), board.theme_for("x"), "Wires", [], items, 3.0, 2.0,
                            {"kind": "diagram", "concept": "conductor", "text": ""})
        import numpy as np
        pixels = np.asarray(frame).astype(int)
        self.assertGreater(pixels[250:700, 60:770].std(), 10)                       # the diagram is on the left
        self.assertTrue(board.fits(items, split=True))


class QualityGapTests(unittest.TestCase):
    def test_an_empty_scene_in_a_contract_is_skipped_not_fatal(self):
        from app.contract import read_contract_json
        shot = {"shot_id": "1.1", "visual_type": "process_steps", "narration": "Ions carry the charge in a solution."}
        board_ = read_contract_json({"schema_version": "2.0", "title": "T", "scenes": [
            {"scene_number": 1, "scene_title": "Ions", "shots": [shot]},
            {"scene_number": 2, "scene_title": "", "narration": "", "shots": []}]})
        self.assertIsNotNone(board_)
        self.assertEqual(len(board_.scenes), 1)
        self.assertEqual(board.layout_of(_seg(1, "Therefore rho remains constant.", layout="summary")), "bullets")

    def test_chemistry_figures_and_long_equations_get_room(self):
        from app import diagrams
        self.assertEqual(diagrams.concept_for("This graph shows molar conductivity of a weak electrolyte against concentration; "
                                              "strong electrolytes fall in a line.", chemistry=True), "conductivity_graph")
        self.assertEqual(diagrams.concept_for("Electrolysis is used in electroplating.", chemistry=True), "electrolysis")
        self.assertIsNone(diagrams.concept_for("Weak acids are weak electrolytes.", chemistry=True))
        for concept in diagrams.CHEM_CONCEPTS:
            self.assertEqual(diagrams.render(concept, (700, 600), 2.0).size, (700, 600))
        self.assertTrue(board.wide([{"kind": "eq", "text": "\\Lambda(CH3COOH) = \\Lambda(CH3COONa) + \\Lambda(HCl) - \\Lambda(NaCl)"}]))
        self.assertFalse(board.wide([{"kind": "eq", "text": "V = IR"}]))

    def test_a_shot_about_a_graph_never_gets_a_generated_picture(self):
        import tempfile
        from app.assets import StillProducer
        from app.config import AppConfig
        from app.schema import Scene, Storyboard
        graph = Segment(segment_number=1, shot_id="3.1", animate=True, image_prompt="A blue curve rising on white",
                        narration="For weak electrolytes, the graph approaches an asymptote at lower concentrations.",
                        shot=Shot(template="video"))
        beaker = Segment(segment_number=2, shot_id="3.2", narration="A conductivity cell dips into the solution in the beaker.",
                         image_prompt="A conductivity cell in a beaker", shot=Shot(template="photo"))
        board_ = Storyboard(title="Conductance", scenes=[Scene(scene_number=3, title="Graphs", narration="", segments=[graph, beaker])])
        with tempfile.TemporaryDirectory() as tmp:
            producer = StillProducer(board_, Path(tmp), AppConfig())
            producer._no_drawn_graphs()
        self.assertNotIn(graph.shot.template, {"photo", "video"})
        self.assertFalse(graph.animate)
        self.assertEqual(beaker.shot.template, "photo")

    def test_board_points_are_the_narrations_own_short_words(self):
        self.assertEqual(board.key_point("So, now we can see that the resistance increases with the length of the wire."),
                         "The resistance increases with the length of the wire")
        long = ("The resistivity of a material depends only on the nature of the material and its temperature, "
                "which is why copper is used for wires in our homes and in most electrical appliances.")
        self.assertEqual(board.key_point(long), "The resistivity of a material depends only on the nature of the material and its temperature")
        self.assertEqual(board.key_point("In metals, current is carried by free electrons."), "In metals, current is carried by free electrons")

    def test_gallery_pictures_replace_chips_only_when_two_or_more_exist(self):
        import json
        import tempfile
        from app import gallery
        from app.production import _plan_board_pages
        from PIL import Image
        self.assertTrue(gallery.drawable("Paints"))
        self.assertFalse(gallery.drawable("Measurement of conductivity"))
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "gallery"
            folder.mkdir()
            index = {}
            for name in ("Paints", "Varnishes"):
                path = folder / f"{gallery.slug(name)}.png"
                Image.new("RGB", (64, 48), (200, 90, 40)).save(path)
                index[gallery.slug(name)] = {"item": name, "approved": True, "path": str(path)}
            (folder / "index.json").write_text(json.dumps(index), encoding="utf-8")
            plan = _Plan(_seg(1, "It is used in paints, varnishes and perfumes.", layout="gallery",
                              gallery_items=["Paints", "Varnishes", "Perfumes"]))
            plan.segment.start, plan.segment.end, plan.segment.speech_duration = 0, 6, 5
            _plan_board_pages([plan], "Uses", output_dir=Path(tmp))
            thumbs = plan.board["current"][-1]
            self.assertEqual(thumbs["kind"], "thumbs")
            self.assertEqual([bool(p) for p in thumbs["images"]], [True, True, False])
            frame = board.frame((1920, 1080), board.theme_for("x"), "Uses", [], plan.board["current"], 5.0, 4.0)
            self.assertEqual(frame.size, (1920, 1080))

    def test_a_small_label_is_painted_out_but_a_page_of_text_is_not(self):
        import tempfile
        from app import ocr
        from PIL import ImageDraw
        if ocr._get_engine() is None:
            self.skipTest("OCR not installed")
        from app.typography import font
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "label.png"
            image = Image.new("RGB", (1024, 576), (90, 140, 170))
            ImageDraw.Draw(image).text((380, 250), "ASPIRIN", font=font(48, True), fill=(255, 255, 255))
            image.save(label)
            self.assertTrue(ocr.has_stray_text(ocr.stray_text(label)))
            self.assertTrue(ocr.erase_text(label))
            self.assertFalse(ocr.stray_text(label, min_conf=0.6, min_height=0.02))
            self.assertTrue((Path(tmp) / "label.with_text.png").is_file())
            page = Path(tmp) / "page.png"
            image = Image.new("RGB", (1024, 576), (240, 240, 235))
            draw = ImageDraw.Draw(image)
            for row in range(8):
                draw.text((80, 40 + row * 62), "Chapter text line number %d here" % row, font=font(34), fill=(20, 20, 20))
            image.save(page)
            self.assertIsNone(ocr.erase_text(page))


from PIL import Image  # noqa: E402
