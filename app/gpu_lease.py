"""Cross-process GPU lease shared with Transcribe (same lock file protocol): only one heavy GPU workload at a time.

Both projects use heavyweight models (Ollama Qwen, Whisper, FLUX, LTX). On a 12 GB card two of them at once starve each other
(timeouts, out-of-memory). A job takes the lease for its whole run; a second job waits instead of thrashing.
"""
from __future__ import annotations

import contextlib
import json
import os
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

LOCK = Path(tempfile.gettempdir()) / "open_edu_gpu.lock"


def _alive(pid: int) -> bool:
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True).stdout
            return str(pid) in out
        os.kill(pid, 0)
        return True
    except (OSError, ValueError):
        return False


def _owner_pid() -> int | None:
    """PID written in the lock file, or None when the file is missing or not (yet) readable."""
    try:
        return int(json.loads(LOCK.read_text(encoding="utf-8")).get("pid"))
    except (OSError, ValueError, TypeError):
        return None


def _stale() -> bool:
    """A lock is stale only when its owner process no longer exists. A file that is empty or half written is a lock that is
    being created right now, so it counts as stale only after it has stayed unreadable for a while."""
    pid = _owner_pid()
    if pid is None:
        try:
            return time.time() - LOCK.stat().st_mtime > 30
        except OSError:
            return False                    # it just disappeared: nothing to remove
    return not _alive(pid)


def _steal_stale_lock() -> None:
    """Remove a stale lock without ever deleting a fresh one that another waiter created a moment ago (atomic rename, then verify)."""
    grave = LOCK.with_name(f"{LOCK.name}.stale.{os.getpid()}")
    try:
        LOCK.replace(grave)
    except OSError:
        return                              # another waiter got there first
    try:
        info = json.loads(grave.read_text(encoding="utf-8"))
        if _alive(int(info.get("pid", -1))):
            with contextlib.suppress(OSError):
                grave.replace(LOCK)         # it was a live lock after all: put it back
            return
    except (OSError, ValueError, TypeError):
        pass
    with contextlib.suppress(OSError):
        grave.unlink()


@contextlib.contextmanager
def gpu_lease(owner: str, wait_seconds: float = 4 * 3600, poll: float = 15.0):
    deadline = time.time() + wait_seconds
    announced = False
    while True:
        try:
            handle = os.open(str(LOCK), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(handle, json.dumps({"pid": os.getpid(), "owner": owner,
                                         "since": datetime.now(timezone.utc).isoformat()}).encode())
            os.close(handle)
            break
        except FileExistsError:
            if _stale():
                _steal_stale_lock()
                continue
            if time.time() > deadline:
                raise TimeoutError(f"GPU is busy ({LOCK.read_text(encoding='utf-8', errors='replace')}); waited {wait_seconds / 3600:.1f} h")
            if not announced:
                print(f"Waiting for the GPU lease held by another job ({LOCK})...", flush=True)
                announced = True
            time.sleep(poll)
    try:
        yield
    finally:
        if _owner_pid() == os.getpid():     # release only our own lock, never one taken over after a long job
            with contextlib.suppress(OSError):
                LOCK.unlink()
