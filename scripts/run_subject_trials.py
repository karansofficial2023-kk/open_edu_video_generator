from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from app.animation_renderer import render_animation_clip, render_frame
from app.config import load_config
from app.schema import Segment, Shot, Storyboard


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    output = ROOT / "outputs" / "production_subject_trials"
    output.mkdir(parents=True, exist_ok=True)
    config = load_config(ROOT / "config.pro_style.12gb.yaml")
    config.output_resolution.width = 1280
    config.output_resolution.height = 720
    config.fps = 15
    config.render.video_preset = "ultrafast"
    config.render.video_bitrate = ""
    config.render.video_maxrate = ""
    config.render.video_bufsize = ""
    config.render.burn_captions = False

    reviewed = Storyboard.model_validate_json(
        (ROOT / "samples" / "reviewed_primula_heterostyly_smoke.json").read_text(encoding="utf-8")
    )
    biology = next(segment for _, segment in reviewed.all_segments() if segment.shot and segment.shot.labels)
    biology.start = 0
    biology.end = biology.speech_duration = 4
    biology.captions = []
    biology_source = Image.open(biology.shot.asset_path).convert("RGB")

    chemistry = _formula_segment(
        "Apply Kohlrausch's law using exact ionic contributions.",
        "Kohlrausch's Law",
        ["Λ°ₘ = λ°₊ + λ°₋", "CH₃COOH ⇌ H⁺ + CH₃COO⁻"],
    )
    mathematics = _formula_segment(
        "Differentiate the function and test its stationary point.",
        "Minimum of a Function",
        ["f(x) = x + 1/x", "f′(x) = 1 − 1/x²", "f′(x) = 0 ⇒ x = ±1"],
    )

    trials = [
        ("biology_labels", biology, biology_source),
        ("chemistry_formula", chemistry, None),
        ("mathematics_derivation", mathematics, None),
    ]
    report = []
    for name, segment, source in trials:
        frame = render_frame(segment, name.replace("_", " ").title(), 3.8, config, source)
        image_path = output / f"{name}.png"
        video_path = output / f"{name}.mp4"
        frame.save(image_path)
        render_animation_clip(segment, name.replace("_", " ").title(), video_path, 60, config, source)
        report.append({"subject": name, "image": str(image_path), "video": str(video_path)})
    (output / "trial_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)


def _formula_segment(narration: str, heading: str, lines: list[str]) -> Segment:
    return Segment(
        segment_number=1,
        narration=narration,
        visual="Exact deterministic derivation",
        image_prompt="",
        shot=Shot(template="formula", heading=heading, formula_lines=lines),
        start=0,
        end=4,
        speech_duration=4,
    )


if __name__ == "__main__":
    main()
