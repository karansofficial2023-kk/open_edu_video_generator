"""Check and repair translated Transcribe contracts before dubbing (v2 deep dive).

Usage: python scripts/precheck_contract.py CONTRACT.json [CONTRACT2.json | FOLDER ...] [--out-dir DIR]
Exit code 0 = contracts are dub-ready (findings, if any, are listed for teacher review), 1 = unusable contract.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / ".runtime-deps")]

from app.contract_precheck import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
