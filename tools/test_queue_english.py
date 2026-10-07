"""Runs only the English end-to-end step of test_queue."""
from test_queue import STATUS, english_lesson, record

if __name__ == "__main__":
    english_lesson()
    record("queue_english", state="finished")
