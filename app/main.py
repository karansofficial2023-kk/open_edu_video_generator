from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .docx_exporter import export_storyboard_docx
from .input_reader import read_input, save_storyboard
from .llm_planner import plan_storyboard, write_prompt_pack
from .renderer import assign_timing, concat_audio, render_video
from .subtitles import write_srt
from .tts import synthesize_storyboard
from .visuals import render_segment_frames
from .visual_planner import prepare_visual_prompts
from .review import review_storyboard
from .storyboard_cleanup import normalize_storyboard, promote_real_image_shots, promote_video_shots


def _is_deferred_render_error(message: str, config) -> bool:
    return bool(
        config.vision_qa.enabled
        and "needs a verified coordinate from the exact approved image" in message
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate an educational video from text, DOCX, or storyboard JSON.")
    parser.add_argument("--input", required=True, help="Path to .txt, .docx, or storyboard .json")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--title", default="Educational Video", help="Fallback title for plain text input")
    parser.add_argument("--skip-tts", action="store_true", help="Build visuals and storyboard without TTS audio")
    parser.add_argument("--limit-segments", type=int, help="Render only the first N segments for a quality preview")
    parser.add_argument("--plan-only", action="store_true", help="Save storyboard and review findings without TTS or image generation")
    parser.add_argument("--preview", action="store_true", help="Render a silent MP4 with approximate timing; no TTS required")
    parser.add_argument("--burn-captions", action="store_true", help="Burn captions into video pixels")
    parser.add_argument("--no-embed-subtitles", action="store_true", help="Write subtitles.srt without embedding a subtitle track in MP4")
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Create a safe draft despite incomplete coverage or unresolved visual QA; rejected visuals and labels are omitted",
    )
    parser.add_argument("--strict-coverage", action="store_true", help="Block rendering when coverage lint says the topic is incomplete")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.allow_incomplete:
        config.vision_qa.block_on_failure = False
    if args.burn_captions:
        config.render.burn_captions = True
    if args.no_embed_subtitles:
        config.render.embed_subtitles = False
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    text, existing_storyboard = read_input(args.input)
    storyboard = existing_storyboard or plan_storyboard(text, config, title=args.title)
    normalize_storyboard(storyboard)
    promote_real_image_shots(storyboard, config)
    promote_video_shots(storyboard, config)
    if not storyboard.all_segments():
        parser.error("The storyboard contains no narration segments")
    if args.limit_segments is not None:
        if args.limit_segments < 1:
            parser.error("--limit-segments must be positive")
        remaining = args.limit_segments
        scenes = []
        for scene in storyboard.scenes:
            if remaining <= 0:
                break
            scene.segments = scene.segments[:remaining]
            remaining -= len(scene.segments)
            scene.narration = " ".join(segment.narration for segment in scene.segments)
            scenes.append(scene)
        storyboard.scenes = scenes
    storyboard_path = output_dir / "storyboard.json"
    save_storyboard(storyboard, storyboard_path)
    export_storyboard_docx(storyboard, output_dir / "storyboard.docx")
    expected_topic = Path(args.input).stem.replace("_", " ").replace("-", " ")
    findings = review_storyboard(storyboard, output_dir, expected_topic)
    if args.plan_only:
        print(f"Plan: {storyboard_path}; review: {output_dir / 'review.json'}")
        return
    errors = [item for item in findings if item["level"] == "error"]
    blocking_errors = [
        item for item in errors
        if not _is_deferred_render_error(item["message"], config)
        and (args.strict_coverage or "does not cover the full Types of Pollination topic" not in item["message"])
    ]
    if blocking_errors and not args.allow_incomplete:
        messages = "\n".join(f"- {item['message']}" for item in blocking_errors)
        coverage_path = output_dir / "coverage.json"
        if coverage_path.exists():
            coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
            missing = coverage.get("missing_topics") or []
            if missing:
                messages += "\nMissing topics: " + ", ".join(missing)
        parser.exit(
            2,
            f"Storyboard review blocked rendering.\n{messages}\n"
            f"Review file: {output_dir / 'review.json'}\n"
            f"Use the teacher-approved storyboard for final output, or pass --allow-incomplete for a draft render.\n"
        )
    prepare_visual_prompts(storyboard, output_dir, config)
    write_prompt_pack(storyboard, output_dir)

    if not args.skip_tts and not args.preview:
        synthesize_storyboard(storyboard, output_dir, config)
    else:
        for _, segment in storyboard.all_segments():
            segment.audio_path = None
            segment.captions = []

    assign_timing(storyboard, config)
    save_storyboard(storyboard, storyboard_path)
    subtitles_path = output_dir / "subtitles.srt"
    write_srt(storyboard, subtitles_path)

    frames = render_segment_frames(storyboard, output_dir, config)
    # Vision QA resolves image-specific scientific targets during rendering. Persist
    # those verified coordinates and refresh lint artifacts so they describe the
    # delivered frames instead of the pre-render storyboard.
    save_storyboard(storyboard, storyboard_path)
    export_storyboard_docx(storyboard, output_dir / "storyboard.docx")
    review_storyboard(storyboard, output_dir, expected_topic)

    if args.skip_tts and not args.preview:
        print(f"Storyboard and frames created in {output_dir}")
        print("Run again without --skip-tts after configuring TTS and FFmpeg to create final.mp4.")
        return

    if args.preview:
        import wave
        audio = output_dir / "preview_silence.wav"
        with wave.open(str(audio), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(24000)
            handle.writeframes(b"\0\0" * round(storyboard.all_segments()[-1][1].end * 24000))
    else:
        audio = concat_audio(storyboard, output_dir, config)
    video = render_video(storyboard, frames, audio, output_dir, config, subtitles_path)
    print(f"Done: {video}")


if __name__ == "__main__":
    main()
