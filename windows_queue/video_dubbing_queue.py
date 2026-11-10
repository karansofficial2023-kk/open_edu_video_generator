#!/usr/bin/env python3
"""Visible, resumable launcher for the Hindi dubbing PowerShell queue."""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path


INPUT_ROOT = Path(r"D:\Python\video_dubbing")
OUTPUT_ROOT = Path(r"D:\Python\video_dubbing_output")
GENERATOR_ROOT = Path(r"D:\Python\claude version\open_edu_video_generator")
TRANSCRIBE_ROOT = Path(r"D:\Python\claude version\transcribe")
QUEUE_PS1 = GENERATOR_ROOT / "run_audio_dub_queue.ps1"
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".wmv", ".webm", ".flv", ".m4v", ".mpeg", ".mpg"}
FINAL_OK = {"completed", "completed_existing"}
REVIEW = {"completed_needs_teacher_review"}
FAILURE = {"probe_error", "duplicate_stem", "transcribe_failed", "dub_failed", "qa_blocked"}
ACTIVE = {"queued", "transcribing", "ready_to_dub", "dubbing", "audio_found"}


def say(message: str, log=None) -> None:
    print(message, flush=True)
    if log:
        log.write(message + "\n")
        log.flush()


def scan_videos() -> list[Path]:
    return sorted((p.resolve() for p in INPUT_ROOT.rglob("*")
                   if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS), key=lambda p: str(p).casefold())


def load_or_assign_ids(videos: list[Path]) -> dict[str, str]:
    """Persist sequential teacher-facing IDs; IDs do not change when new videos are added."""
    manifest = INPUT_ROOT / "video_ids.csv"
    mapping: dict[str, str] = {}
    if manifest.exists():
        with manifest.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                path = os.path.normcase(os.path.abspath(row.get("source_path", "")))
                video_id = row.get("video_id", "").strip()
                if path and video_id:
                    mapping[path] = video_id
    max_id = max((int(v[1:]) for v in mapping.values() if v.startswith("V") and v[1:].isdigit()), default=0)
    for video in videos:
        key = os.path.normcase(str(video))
        if key not in mapping:
            max_id += 1
            mapping[key] = f"V{max_id:06d}"
    rows = sorted(((video_id, path) for path, video_id in mapping.items()), key=lambda x: int(x[0][1:]))
    with manifest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "source_path"])
        writer.writerows(rows)
    return mapping


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return []


def queue_counts() -> dict[str, int]:
    results = read_csv(OUTPUT_ROOT / "reports" / "queue_results.csv")
    no_audio = len(read_csv(OUTPUT_ROOT / "reports" / "no_audio.csv"))
    ok = review = failed = pending = 0
    for row in results:
        status = row.get("status", "")
        if status in FINAL_OK:
            try:
                flags = int(row.get("translation_review_flags") or 0)
            except ValueError:
                flags = 0
            if flags == 0:
                ok += 1
            else:
                review += 1
        elif status in REVIEW:
            review += 1
        elif status in FAILURE:
            failed += 1
        elif status in ACTIVE or not status:
            pending += 1
    return {"completed": ok, "review": review, "failed": failed, "pending": pending, "no_audio": no_audio}


def run_queue(mode: str, videos: list[Path], ids: dict[str, str]) -> int:
    if not videos:
        say("No matching videos were found.")
        return 2
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_dir = OUTPUT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{stamp}_{mode}.log"
    out_root = OUTPUT_ROOT
    command = ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(QUEUE_PS1),
               "-InputPath", str(videos[0] if len(videos) == 1 else INPUT_ROOT),
               "-OutputRoot", str(out_root), "-TranscribeRoot", str(TRANSCRIBE_ROOT),
               "-BaseConfig", str(GENERATOR_ROOT / "config.realistic.12gb.yaml"),
               "-GradeLevel", "10th standard", "-AcademicLevel", "secondary", "-TargetLanguage", "hi"]
    if mode == "fix":
        # Resume: completed dubs are skipped and unfinished items are retried.
        pass
    elif mode == "dual":
        # Resume in dual-track mode; replacement-mode outputs are rebuilt by mode mismatch.
        command.append("-KeepOriginalAudio")
    elif mode == "teacher-rerun":
        command.append("-RerunDubbing")

    lines: queue.Queue[str | None] = queue.Queue()
    log = log_path.open("w", encoding="utf-8", newline="\n")
    say(f"Mode: {mode} | videos selected: {len(videos)}", log)
    say(f"Input: {INPUT_ROOT if len(videos) != 1 else videos[0]}", log)
    say(f"Output: {OUTPUT_ROOT}", log)
    say(f"Live log: {log_path}", log)
    say("IDs: " + ", ".join(f"{ids[os.path.normcase(str(v))]}={v.name}" for v in videos), log)
    say("Starting Transcribe/video-generator queue…", log)
    run_started = time.time()
    try:
        process = subprocess.Popen(command, cwd=GENERATOR_ROOT, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace",
                                   bufsize=1, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        say(f"Could not start PowerShell queue: {exc}", log)
        log.close()
        return 2

    def read_output() -> None:
        assert process.stdout is not None
        for line in process.stdout:
            lines.put(line.rstrip("\r\n"))
        lines.put(None)

    threading.Thread(target=read_output, daemon=True).start()
    last_dashboard = 0.0
    ended = False
    while not ended or process.poll() is None:
        try:
            line = lines.get(timeout=1)
            if line is None:
                ended = True
            else:
                say(line, log)
        except queue.Empty:
            pass
        now = time.monotonic()
        if now - last_dashboard >= 10:
            report_file = OUTPUT_ROOT / "reports" / "queue_results.csv"
            report_is_current = report_file.exists() and report_file.stat().st_mtime >= run_started - 2
            c = queue_counts() if report_is_current else {
                "completed": 0, "review": 0, "failed": 0, "pending": len(videos), "no_audio": 0
            }
            total = len(videos)
            say(f"STATUS total={total} | completed_QA_ok={c['completed']} | needs_teacher_review={c['review']} | "
                f"failed={c['failed']} | pending={c['pending']} | no_audio={c['no_audio']} (not counted as failure)", log)
            last_dashboard = now
    return_code = process.wait()
    report_file = OUTPUT_ROOT / "reports" / "queue_results.csv"
    report_is_current = report_file.exists() and report_file.stat().st_mtime >= run_started - 2
    c = queue_counts() if report_is_current else {
        "completed": 0, "review": 0, "failed": 1 if return_code else 0,
        "pending": max(len(videos) - (1 if return_code else 0), 0), "no_audio": 0
    }
    say("=" * 78, log)
    say(f"FINAL total={len(videos)} | completed_QA_ok={c['completed']} | needs_teacher_review={c['review']} | "
        f"failed={c['failed']} | pending={c['pending']} | no_audio={c['no_audio']} (not counted as failure)", log)
    say(f"Results CSV: {OUTPUT_ROOT / 'reports' / 'queue_results.csv'}", log)
    say(f"No-audio list: {OUTPUT_ROOT / 'reports' / 'no_audio.csv'}", log)
    say(f"Full log: {log_path}", log)
    log.close()
    return return_code


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Hindi dubbing queue with visible logs and per-video IDs.")
    parser.add_argument("--mode", choices=("fix", "dual", "teacher-rerun", "list"), required=True)
    parser.add_argument("--video-id", help="Persistent ID from video_ids.csv; required for teacher-rerun.")
    args = parser.parse_args()
    if not QUEUE_PS1.is_file():
        print(f"Queue script is missing: {QUEUE_PS1}", file=sys.stderr)
        return 2
    videos = scan_videos()
    ids = load_or_assign_ids(videos)
    if args.mode == "list":
        print(f"Video ID manifest: {INPUT_ROOT / 'video_ids.csv'}")
        for video in videos:
            print(f"{ids[os.path.normcase(str(video))]}  {video}")
        return 0
    if args.mode == "teacher-rerun":
        if not args.video_id:
            print("Teacher rerun needs an ID, for example V000001.", file=sys.stderr)
            return 2
        found = [p for p in videos if ids[os.path.normcase(str(p))].casefold() == args.video_id.casefold()]
        if not found:
            print(f"Unknown video ID: {args.video_id}. Find valid IDs in {INPUT_ROOT / 'video_ids.csv'}", file=sys.stderr)
            return 2
        videos = found
        # A one-video run rewrites the queue runner's summary CSVs. Preserve the last whole-folder report.
        report_dir = OUTPUT_ROOT / "reports"
        names = ("queue_results.csv", "inventory.csv", "no_audio.csv", "probe_failures.csv", "duplicate_names.csv")
        previous = {name: (report_dir / name).read_bytes() if (report_dir / name).exists() else None for name in names}
        code = run_queue(args.mode, videos, ids)
        report_dir.mkdir(parents=True, exist_ok=True)
        last_result = report_dir / f"teacher_{args.video_id.upper()}_last_result.csv"
        current = report_dir / "queue_results.csv"
        if current.exists():
            last_result.write_bytes(current.read_bytes())
        for name, contents in previous.items():
            path = report_dir / name
            if contents is None:
                path.unlink(missing_ok=True)
            else:
                path.write_bytes(contents)
        print(f"Single-video result saved: {last_result}", flush=True)
        return code
    return run_queue(args.mode, videos, ids)


if __name__ == "__main__":
    raise SystemExit(main())
