"""Small self-healing helpers: a fault in one step must cost a retry, not the lesson."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def retry(action: Callable[[], T], *, attempts: int = 3, base_delay: float = 2.0, sleep: Callable[[float], None] | None = None,
          retry_on: tuple[type[BaseException], ...] = (Exception,), stop_on: tuple[type[BaseException], ...] = (), label: str = "step") -> T:
    """Run `action`; on a transient fault wait base*2^n seconds and try again, re-raising the last error after `attempts` tries."""
    last: BaseException | None = None
    for number in range(1, max(1, attempts) + 1):
        try:
            return action()
        except stop_on:
            raise                           # a fault that repeating cannot fix
        except retry_on as exc:
            last = exc
            if number >= attempts:
                break
            print(f"[retry] {label} failed ({str(exc)[:120]}); attempt {number + 1} of {attempts}", flush=True)
            (sleep or time.sleep)(min(60.0, base_delay * 2 ** (number - 1)))
    assert last is not None
    raise last


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write to a temp file and rename, so a crash or a full disk never leaves a half-written cache record."""
    path = Path(path)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(text, encoding=encoding)
    os.replace(temp, path)


def read_json_or_none(path: Path) -> Any | None:
    """A missing or corrupt cache file means 'not cached', never a crash."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def wait_for(check: Callable[[], bool], *, timeout: float, interval: float = 5.0, sleep: Callable[[float], None] | None = None) -> bool:
    """Poll until `check()` is true (service came back) or the time budget is used up."""
    waited = 0.0
    while True:
        try:
            if check():
                return True
        except Exception:
            pass
        if waited >= timeout:
            return False
        (sleep or time.sleep)(interval)
        waited += interval
