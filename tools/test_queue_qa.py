"""Re-runs the cached Telugu and Tamil renders so that only the final QA (revised prompt) is recomputed."""
from test_queue import SCRATCH, render, record

if __name__ == "__main__":
    render("telugu_qa_rerun", SCRATCH / "out_telugu2" / "Photosynthesis_Telugu_contract.json", "real_telugu")
    render("tamil_qa_rerun", SCRATCH / "real_ta2" / "Photosynthesis_Tamil_contract.json", "real_tamil")
    record("qa_rerun", state="finished")
