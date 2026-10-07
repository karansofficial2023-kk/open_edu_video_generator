import tempfile
import unittest
from pathlib import Path

from app import auto_labels
from app.config import AppConfig
from app.schema import Scene, Segment, Shot, Storyboard


def photo(narration: str, shot_id: str = "2.1") -> Segment:
    return Segment(segment_number=1, narration=narration, shot_id=shot_id, shot=Shot(template="photo", heading="x", asset_path="a.png"))


class AutoLabelTests(unittest.TestCase):
    """Biology photographs should carry the labels of the parts the narration names (real Pollination lines)."""

    def test_only_parts_named_in_the_narration_survive(self):
        narration = "In this process, the pollen grains from the anther land on the stigma of the same flower."
        proposed = ["stigma", "anther", "ovary", "pollen grains", "pollination process", "the stigma", "flower of the sunflower plant"]
        kept = auto_labels.valid_labels(proposed, narration)
        self.assertEqual(kept, ["Stigma", "Anther", "Pollen grains"])           # ovary / process / duplicate / too long: dropped

    def test_label_shape_rules(self):
        narration = "The stigma is sticky and the style leads to the ovary."
        self.assertEqual(auto_labels.valid_labels(["Stigma", "Style"], narration), ["Stigma", "Style"])
        self.assertEqual(auto_labels.valid_labels(["", "  ", "a", "The"], narration), [])
        self.assertEqual(auto_labels.valid_labels(None, narration), [])
        self.assertEqual(len(auto_labels.valid_labels(["stigma", "style", "ovary", "sticky"], narration)), 3)    # at most three

    def test_adjectives_are_not_labels(self):
        """Real Pollination shot: the model labelled the clover with 'Small', 'Inconspicuous' and 'Flowers'."""
        narration = "Some plants, like wood sorrel, produce small, inconspicuous flowers that lack fragrance."
        self.assertEqual(auto_labels.valid_labels(["Small", "Inconspicuous", "Flowers"], narration), ["Flowers"])
        self.assertEqual(auto_labels.valid_labels(["small flowers", "Sticky"], "The small flowers are sticky."), ["Small flowers"])
        self.assertEqual(auto_labels.valid_labels(["petal", "Sepal"], "A petal and a sepal."), ["Petal", "Sepal"])         # nouns ending like adjectives survive

    def test_eligibility(self):
        ok = photo("The stigma receives the pollen from the anther.")
        self.assertTrue(auto_labels.eligible(ok))
        self.assertFalse(auto_labels.eligible(photo("Fascinating, isn't it?")))                          # too short to name anything
        labelled = photo("The stigma receives the pollen from the anther.")
        labelled.labels = ["Stigma"]
        self.assertFalse(auto_labels.eligible(labelled))                                                    # the contract already labelled it
        formula = photo("The stigma receives the pollen from the anther.")
        formula.formula_lines = ["x = 1"]
        self.assertFalse(auto_labels.eligible(formula))
        card = Segment(segment_number=1, narration="The stigma receives the pollen from the anther.",
                       shot=Shot(template="process", heading="x", steps=["a", "b"], stage_fractions=[0, 0.5]))
        self.assertFalse(auto_labels.eligible(card))                                                        # a concept card has no picture to label

    def test_apply_is_off_by_default_and_never_fails_a_render(self):
        original = auto_labels._ask_model
        try:
            with tempfile.TemporaryDirectory() as tmp:
                seg = photo("The stigma receives the pollen from the anther.")
                board = Storyboard(title="t", scenes=[Scene(scene_number=1, title="s", narration="n", segments=[seg])])
                config = AppConfig()
                auto_labels._ask_model = lambda narration, cfg: ["stigma", "anther"]
                self.assertEqual(auto_labels.apply(board, Path(tmp), config), 0)                           # switched off
                self.assertEqual(seg.labels, [])

                config.production.auto_labels = True
                self.assertEqual(auto_labels.apply(board, Path(tmp), config), 1)
                self.assertEqual(seg.labels, ["Stigma", "Anther"])
                self.assertEqual(seg.label_placement, ["Stigma", "Anther"])                                 # one target description per label
                self.assertTrue(seg.labels_auto)

                def boom(narration, cfg):
                    raise RuntimeError("model crashed")
                auto_labels._ask_model = boom
                other = photo("The petals protect the stamen and the carpel inside.", "3.1")
                board.scenes[0].segments.append(other)
                auto_labels.apply(board, Path(tmp), config)                                                # must not raise
                self.assertEqual(other.labels, [])
        finally:
            auto_labels._ask_model = original

    def test_proposals_are_stored_so_a_rerun_shows_the_same_labels(self):
        original = auto_labels._ask_model
        calls = []
        try:
            with tempfile.TemporaryDirectory() as tmp:
                auto_labels._ask_model = lambda narration, cfg: calls.append(1) or ["stigma"]
                seg = photo("The stigma receives the pollen from the anther.")
                first = auto_labels.propose(seg, Path(tmp), AppConfig())
                auto_labels._ask_model = lambda narration, cfg: calls.append(1) or ["anther"]
                again = auto_labels.propose(seg, Path(tmp), AppConfig())
        finally:
            auto_labels._ask_model = original
        self.assertEqual(first, again)
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
