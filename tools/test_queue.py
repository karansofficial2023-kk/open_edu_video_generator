"""Runs the pending verification renders one after another and records the result of each in outputs/queue_status.json.

    python -u tools/test_queue.py

Steps never abort the queue: a failing step is recorded and the next one starts. Only one heavy job runs at a time.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = Path("C:/Users/User/AppData/Local/Temp/claude/D--Python-claude-version/54acc0e5-1a19-414c-b74e-1a43614d55b9/scratchpad/golden")
STATUS = ROOT / "outputs" / "queue_status.json"
CONFIG = "config.realistic.12gb.yaml"


def record(name: str, **info) -> None:
    data = json.loads(STATUS.read_text(encoding="utf-8")) if STATUS.exists() else {}
    data[name] = {**data.get(name, {}), **info, "at": time.strftime("%H:%M:%S")}
    STATUS.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def render(name: str, contract: Path, output: str) -> None:
    record(name, state="running")
    log = ROOT / "outputs" / f"{output}.log"
    with open(log, "w", encoding="utf-8") as handle:
        code = subprocess.call([sys.executable, "-u", "-m", "app.main", "--config", CONFIG, "--input", str(contract),
                                "--output", f"outputs/{output}"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT)
    qa = ROOT / "outputs" / output / "final_qa.json"
    result = json.loads(qa.read_text(encoding="utf-8")) if qa.exists() else {}
    warnings = [f["message"] for f in result.get("findings", []) if f["level"] == "warning"]
    record(name, state="done" if code == 0 and result.get("passed") else "FAILED", exit=code, critical=result.get("critical"),
           duration=result.get("duration"), warnings=len(warnings), sample_warnings=warnings[:3])


def english_lesson() -> None:
    """Full Java pipeline on an English lesson (video -> storyboard -> contract) with isolated folders, then the render."""
    name = "english_ohm_pipeline"
    record(name, state="running")
    base = SCRATCH / "en_ohm"
    for sub in ("in", "tmp", "mat", "out"):
        shutil.rmtree(base / sub, ignore_errors=True)
        (base / sub).mkdir(parents=True, exist_ok=True)
    shutil.copy(SCRATCH / "in" / "Ohms_Law_and_Resistor_Circuits.mp4", base / "in")
    args = (f"-Dvideo.input.folder={base / 'in'} -Dvideo.output.dir={base / 'out'} -Dtemp.dir={base / 'tmp'} "
            f"-Dstoryboard.materials.dir={base / 'mat'}").replace("\\", "/")
    java_log = ROOT / "outputs" / "english_ohm_java.log"
    with open(java_log, "w", encoding="utf-8") as handle:
        launcher = subprocess.Popen([str(ROOT.parent / "transcribe" / "mvnw.cmd"), "-o", "-q", "spring-boot:run", f"-Dspring-boot.run.jvmArguments={args}"],
                                    cwd=ROOT.parent / "transcribe", stdout=handle, stderr=subprocess.STDOUT)
        deadline = time.time() + 120 * 60
        contract = None
        while time.time() < deadline and launcher.poll() is None:
            found = list((base / "out").glob("*_contract.json"))
            if found and list((base / "out").glob("*_manifest.json")):
                contract = found[0]
                time.sleep(20)                           # let the app finish writing and release the GPU lease
                break
            time.sleep(30)
        if contract is None:                              # the launcher can exit between two polls, after writing the contract
            found = list((base / "out").glob("*_contract.json"))
            contract = found[0] if found and list((base / "out").glob("*_manifest.json")) else None
        subprocess.call(["taskkill", "/T", "/F", "/PID", str(launcher.pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if contract is None:
        record(name, state="FAILED", reason="no contract produced (timeout or Java exited early); see outputs/english_ohm_java.log")
        return
    record(name, state="done", contract=str(contract))
    render("english_ohm_render", contract, "real_english_ohm")


def main() -> None:
    STATUS.parent.mkdir(exist_ok=True)
    STATUS.write_text("{}", encoding="utf-8")
    render("telugu_rerender", SCRATCH / "out_telugu2" / "Photosynthesis_Telugu_contract.json", "real_telugu")
    render("tamil_rerender", SCRATCH / "real_ta2" / "Photosynthesis_Tamil_contract.json", "real_tamil")
    render("kohlrausch_render", SCRATCH / "rerun_k" / "Applications of Kohlrausch Law_contract.json", "real_kohlrausch")
    english_lesson()
    record("queue", state="finished")


if __name__ == "__main__":
    main()
