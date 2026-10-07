"""Console progress for long runs: which stage we are in, how far along, how long it has taken, and a rough time left.

    [00:12:34] STAGE 2/7 - Pictures: 44 shot(s) need a picture
    [00:15:02]   pictures 12/44 (27%)  shot 2.3  | stage 02:28, about 06:20 left
"""
from __future__ import annotations

import time

TOTAL_STAGES = 7
_start = time.monotonic()
_stage_start = _start


def _clock(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}" if seconds >= 3600 else f"{seconds // 60:02d}:{seconds % 60:02d}"


def elapsed() -> str:
    return time.strftime("%H:%M:%S", time.gmtime(time.monotonic() - _start))


def stage(number: int, text: str) -> None:
    global _stage_start
    _stage_start = time.monotonic()
    print(f"[{elapsed()}] STAGE {number}/{TOTAL_STAGES} - {text}", flush=True)


def step(kind: str, done: int, total: int, label: str = "") -> None:
    """One line per finished item; the time left is the stage's average pace so far times what remains."""
    spent = time.monotonic() - _stage_start
    left = f", about {_clock(spent / done * (total - done))} left" if 0 < done < total and spent > 5 else ""
    percent = round(100 * done / total) if total else 100
    print(f"[{elapsed()}]   {kind} {done}/{total} ({percent}%)  {label}  | stage {_clock(spent)}{left}".rstrip(), flush=True)


def note(text: str) -> None:
    print(f"[{elapsed()}]   {text}", flush=True)
