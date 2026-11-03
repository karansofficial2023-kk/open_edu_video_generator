import sys
import tempfile
import unittest
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(root), str(root / ".runtime-deps")]

from PIL import Image

from app.mermaid_diagram import parse_mermaid, render_mermaid_diagram
from app.manim_anim import render_manim_animation
from app.contract import segment_from_fields


class TestManimAndMermaidTools(unittest.TestCase):

    def test_mermaid_parser_extracts_nodes_and_edges(self):
        code = """
        graph TD
            A[Evaporation] -->|Heated| B[Condensation]
            B --> C[Precipitation]
            C --> A
        """
        nodes, edges, orientation = parse_mermaid(code)
        self.assertEqual(orientation, "TD")
        self.assertIn("A", nodes)
        self.assertEqual(nodes["A"], "Evaporation")
        self.assertEqual(nodes["B"], "Condensation")
        self.assertEqual(nodes["C"], "Precipitation")
        self.assertEqual(len(edges), 3)
        self.assertEqual(edges[0], ("A", "B", "Heated"))

    def test_mermaid_renderer_produces_valid_image(self):
        code = """
        graph LR
            Start[Nitrogen in Air] --> Bacteria[Nitrogen-fixing Bacteria]
            Bacteria --> Plants[Plant Proteins]
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_png = Path(tmp_dir) / "test_diagram.png"
            res = render_mermaid_diagram(code, out_png, heading="Nitrogen Cycle")
            self.assertTrue(res.is_file())
            self.assertGreater(res.stat().st_size, 1000)
            with Image.open(res) as img:
                self.assertEqual(img.size, (1920, 1080))

    def test_manim_renderer_produces_valid_video(self):
        snippet = r"""
        t = MathTex(r"f(x) = \sin(x)")
        self.play(Write(t))
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            out_mp4 = Path(tmp_dir) / "test_math.mp4"
            res = render_manim_animation(snippet, out_mp4, duration=1.0, fps=12, heading="Calculus Proof")
            self.assertTrue(res.is_file())
            self.assertGreater(res.stat().st_size, 500)

    def test_contract_roundtrip_includes_manim_and_mermaid_fields(self):
        fields = {
            "shot_id": "3.1",
            "visual_type": "math_animation",
            "manim_code": "class Wave(Scene): pass",
            "mermaid_code": "graph TD; A-->B",
            "reasoning": "Mathematical proof requires dynamic animation",
            "heading": "Proof",
        }
        seg = segment_from_fields(fields)
        self.assertEqual(seg.shot_id, "3.1")
        self.assertEqual(seg.visual_type, "math_animation")
        self.assertEqual(seg.manim_code, "class Wave(Scene): pass")
        self.assertEqual(seg.mermaid_code, "graph TD; A-->B")
        self.assertEqual(seg.reasoning, "Mathematical proof requires dynamic animation")


if __name__ == "__main__":
    unittest.main()
