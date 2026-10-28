"""Renders the newest contracts from C:/Users/User/output one after another; status in outputs/queue5/status.json.

    python -u tools/run_queue5.py [count] [skip] [out_folder_name]

skip = how many of the newest contracts to leave out (the ones already rendered); output goes to outputs/<out_folder_name>.
"""
from __future__ import annotations

import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path("C:/Users/User/output")
OUT = ROOT / "outputs" / "queue5"
PY = "D:/AI/ComfyUI_windows_portable/python_embeded/python.exe"
CONFIG = "config.realistic.12gb.yaml"


def record(name, **info):
    OUT.mkdir(parents=True, exist_ok=True)
    f = OUT / "status.json"
    data = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    data[name] = {**data.get(name, {}), **info, "at": time.strftime("%H:%M:%S")}
    f.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    global OUT
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    skip = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    OUT = ROOT / "outputs" / (sys.argv[3] if len(sys.argv) > 3 else "queue5")
    contracts = sorted(SRC.glob("*_contract.json"), key=lambda p: p.stat().st_mtime, reverse=True)[skip:skip + count]
    for c in contracts:
        name = c.name[: -len("_contract.json")]
        slug = "".join(ch if ch.isalnum() else "_" for ch in name)
        out = OUT / slug
        earlier = [d for d in (ROOT / "outputs").glob("queue*/*") if d.parent != OUT and d.name in (name, slug) and (d / "final_qa.json").exists()
                   and json.loads((d / "final_qa.json").read_text(encoding="utf-8")).get("passed")]
        if earlier:
            record(name, state="done", note=f"already rendered in {earlier[0].parent.name}"); continue
        if (out / "final_qa.json").exists() and json.loads((out / "final_qa.json").read_text(encoding="utf-8")).get("passed"):
            record(name, state="done", note="already rendered"); continue
        record(name, state="running", output=str(out))
        with open(OUT / f"{slug}.log", "a", encoding="utf-8") as log:
            code = subprocess.call([PY, "-u", "scripts/local_entry.py", "--config", CONFIG, "--input", str(c), "--output", str(out)],
                                   cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        qa = out / "final_qa.json"
        res = json.loads(qa.read_text(encoding="utf-8")) if qa.exists() else {}
        record(name, state="done" if code == 0 and res.get("passed") else "FAILED", exit=code, critical=res.get("critical"), duration=res.get("duration"))


if __name__ == "__main__":
    main()
