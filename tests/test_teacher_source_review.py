"""A Hindi correction must not silently clear uncertainty in the English transcript."""

import json
from pathlib import Path

from windows_queue import video_dubbing_queue as queue


def test_source_asr_requires_verified_english(tmp_path, monkeypatch):
    input_root = tmp_path / "input"
    output_root = tmp_path / "output"
    (input_root / "teacher_corrections").mkdir(parents=True)
    (output_root / "transcribe").mkdir(parents=True)
    monkeypatch.setattr(queue, "INPUT_ROOT", input_root)
    monkeypatch.setattr(queue, "OUTPUT_ROOT", output_root)
    video = Path("lesson.mp4")
    contract = output_root / "transcribe" / "lesson_hi_contract.json"
    review = output_root / "transcribe" / "lesson_hi_translation_review.json"
    correction = input_root / "teacher_corrections" / "V000001.json"
    contract.write_text(json.dumps({"scenes": [{"shots": [{"shot_id": "1.1", "source_narration": "old English", "narration": "old Hindi"}]}]}), encoding="utf-8")
    review.write_text(json.dumps({"review_items": [
        {"where": "scene 1.1 source_asr", "problem": "uncertain English"},
        {"where": "scene 1.1 semantic_audit", "problem": "changed fact"},
    ]}), encoding="utf-8")
    correction.write_text(json.dumps({"corrections": [{"shot_id": "1.1", "narration": "new Hindi"}]}), encoding="utf-8")
    queue.apply_teacher_corrections(video, "V000001")
    active = json.loads(review.read_text(encoding="utf-8"))["review_items"]
    assert [item["where"] for item in active] == ["scene 1.1 source_asr"]
    correction.write_text(json.dumps({"corrections": [{"shot_id": "1.1", "source_narration": "verified English", "narration": "verified Hindi"}]}), encoding="utf-8")
    queue.apply_teacher_corrections(video, "V000001")
    assert json.loads(review.read_text(encoding="utf-8"))["review_items"] == []
    shot = json.loads(contract.read_text(encoding="utf-8"))["scenes"][0]["shots"][0]
    assert (shot["source_narration"], shot["narration"]) == ("verified English", "verified Hindi")
