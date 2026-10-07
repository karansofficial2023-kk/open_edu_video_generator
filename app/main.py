from __future__ import annotations

import argparse
import json
import traceback
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
from .storyboard_cleanup import normalize_storyboard


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an educational video from text, DOCX, or storyboard JSON.")
    parser.add_argument("--input", required=True, nargs="+",
                        help="One or more .txt/.docx/.json files or folders (folders run every storyboard inside, isolated per lesson)")
    parser.add_argument("--output", required=True, help="Output folder (one sub-folder per lesson when several inputs are given)")
    parser.add_argument("--config", default=None, help="Path to config YAML (default: config.yaml when present)")
    parser.add_argument("--title", default="Educational Video", help="Fallback title for plain text input")
    parser.add_argument("--skip-tts", action="store_true", help="Build visuals and storyboard without TTS audio")
    parser.add_argument("--limit-segments", type=int, help="Render only the first N segments for a quality preview")
    parser.add_argument("--plan-only", action="store_true", help="Save storyboard and review findings without TTS or image generation")
    parser.add_argument("--preview", action="store_true", help="Render a silent MP4 with approximate timing; no TTS required")
    parser.add_argument("--burn-captions", action="store_true", help="Burn captions into video pixels (legacy path)")
    parser.add_argument("--no-embed-subtitles", action="store_true", help="Write subtitles.srt without embedding a subtitle track in MP4")
    parser.add_argument("--allow-incomplete", action="store_true", help="Render even when review/QA reports unresolved issues (draft)")
    parser.add_argument("--strict-coverage", action="store_true", help="Treat an unmet coverage checklist as blocking")
    parser.add_argument("--preflight-only", action="store_true", help="Validate the storyboard contract and services, then stop")
    return parser


def expand_inputs(paths: list[str]) -> list[Path]:
    result: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            contracts = sorted(path.glob("*_contract.json"))
            covered = {c.name.replace("_contract.json", "") for c in contracts}
            docs = [d for d in sorted(path.glob("*_storyboard.docx")) if d.name.replace("_storyboard.docx", "") not in covered]
            result.extend(contracts + docs)
        else:
            result.append(path)
    return result


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.config is not None and not Path(args.config).is_file():
        # a mistyped --config must stop the run: silently rendering with default settings wastes an hour of GPU time
        parser.error(f"config file not found: {args.config} (e.g. --config config.realistic.12gb.yaml)")
    config = load_config(args.config or "config.yaml")
    if args.burn_captions:
        config.render.burn_captions = True
    if args.no_embed_subtitles:
        config.render.embed_subtitles = False
    inputs = expand_inputs(args.input)
    if not inputs:
        parser.error("No storyboard inputs found")
    output_root = Path(args.output)
    if len(inputs) == 1:
        run_job(parser, args, config, inputs[0], output_root)
        return
    # Queue: one failed lesson never blocks later lessons; each keeps a diagnostic package.
    summary = []
    for path in inputs:
        job_dir = output_root / path.stem.replace("_contract", "").replace("_storyboard", "")
        job_dir.mkdir(parents=True, exist_ok=True)
        try:
            run_job(parser, args, config, path, job_dir)
            summary.append({"input": str(path), "output": str(job_dir), "status": "done"})
        except (Exception, SystemExit) as exc:
            (job_dir / "production_issues.json").write_text(json.dumps(
                {"error": str(exc), "trace": traceback.format_exc()[-2000:]}, indent=2), encoding="utf-8")
            summary.append({"input": str(path), "output": str(job_dir), "status": "unresolved", "error": str(exc)[:300]})
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "queue_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def run_job(parser, args, config, input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    try:        # which contract this lesson came from (the verifier looks for its storyboard DOCX beside it)
        (output_dir / "source.json").write_text(json.dumps({"input": str(Path(input_path).resolve())}, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass
    text, existing_storyboard = read_input(input_path)
    storyboard = existing_storyboard or plan_storyboard(text, config, title=args.title)
    normalize_storyboard(storyboard, config)
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
    findings = review_storyboard(storyboard, output_dir, config, strict_coverage=args.strict_coverage)
    if args.plan_only:
        print(f"Plan: {storyboard_path}; review: {output_dir / 'review.json'}")
        return
    errors = [item for item in findings if item["level"] == "error"]
    if errors and not args.allow_incomplete:
        messages = "\n".join(f"- {item['message']}" for item in errors)
        coverage_path = output_dir / "coverage.json"
        if coverage_path.exists():
            missing = json.loads(coverage_path.read_text(encoding="utf-8")).get("missing_topics") or []
            if missing:
                messages += "\nMissing topics: " + ", ".join(missing)
        parser.exit(2, f"Storyboard review blocked rendering.\n{messages}\nReview file: {output_dir / 'review.json'}\n"
                       f"Fix the storyboard, or pass --allow-incomplete for a draft render.\n")

    if any(segment.visual_type for _, segment in storyboard.all_segments()):
        if args.preflight_only:
            from .preflight import run_preflight
            print(json.dumps(run_preflight(storyboard, config, output_dir), indent=2))
            return
        from .production import run_production
        outcome = run_production(storyboard, output_dir, config, preview=args.preview, skip_tts=args.skip_tts,
                                 allow_incomplete=args.allow_incomplete)
        if outcome["status"] in {"blocked", "failed"}:
            parser.exit(2, f"Production {outcome['status']}. See {output_dir / 'production_issues.json'}\n")
        qa = outcome.get("qa", {})
        print(f"{outcome['status'].upper()}: {outcome['video']} (critical QA findings: {qa.get('critical', '?')}) - {output_dir / 'final_qa.json'}")
        if outcome["status"] == "qa_failed":
            parser.exit(3, "Final QA found critical problems; this video must not be released.\n")      # scripts and queues must see a failure
        return

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
