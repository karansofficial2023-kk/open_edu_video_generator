"""Run source-slide OCR on an existing timed translation contract without TTS."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / ".runtime-deps")]

from app.config import load_config
from app.input_reader import read_input
from app.visual_source_qa import (audit_visual_source, load_visual_report,
                                  visual_input_signature, visual_report_output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path,
                        help="Evidence root; defaults to one folder beside the translated contract")
    parser.add_argument("--config", type=Path, default=ROOT / "config.realistic.12gb.yaml")
    args = parser.parse_args()
    _text, board = read_input(args.contract)
    if board is None:
        parser.error("--contract must be a timed storyboard JSON contract")
    review_path = args.contract.with_name(args.contract.name.replace("_contract.json", "_translation_review.json"))
    review_items = (json.loads(review_path.read_text(encoding="utf-8-sig")).get("review_items", [])
                    if review_path.is_file() else [])
    config = load_config(args.config)
    output = args.output or visual_report_output(args.contract)
    signature = visual_input_signature(args.video, args.contract, review_path,
                                       config.production.source_visual_ocr_max_cues,
                                       config.production.vision_model)
    report = load_visual_report(output, signature, retry_incomplete=True)
    if report is None:
        report = audit_visual_source(board, args.video, output, config, review_items,
                                     max_cues=config.production.source_visual_ocr_max_cues,
                                     ollama_url=config.ollama_url, vision_model=config.production.vision_model,
                                     input_signature=signature)
    print(f"OCR {report['status']}: checked {report['checked_cues']}/{report['eligible_cues']} cues, "
          f"{len(report['issues'])} visual findings, reused={report.get('reused', False)}. "
          f"Report: {report.get('report_path')}")


if __name__ == "__main__":
    main()
