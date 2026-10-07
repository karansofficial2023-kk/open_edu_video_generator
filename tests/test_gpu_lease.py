import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from app import gpu_lease


class GpuLeaseTest(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.original = gpu_lease.LOCK
        gpu_lease.LOCK = Path(self.folder.name) / "gpu.lock"

    def tearDown(self):
        gpu_lease.LOCK = self.original
        self.folder.cleanup()

    def test_lease_is_taken_and_released(self):
        with gpu_lease.gpu_lease("test"):
            self.assertEqual(json.loads(gpu_lease.LOCK.read_text())["pid"], os.getpid())
        self.assertFalse(gpu_lease.LOCK.exists())

    def test_a_lock_of_a_dead_process_is_taken_over(self):
        gpu_lease.LOCK.write_text(json.dumps({"pid": 4000000, "owner": "crashed"}))
        with gpu_lease.gpu_lease("test", wait_seconds=5, poll=0.05):
            self.assertEqual(json.loads(gpu_lease.LOCK.read_text())["pid"], os.getpid())

    def test_a_half_written_lock_is_not_stolen_but_an_old_empty_one_is(self):
        gpu_lease.LOCK.write_text("")
        self.assertFalse(gpu_lease._stale())                        # being created right now
        old = time.time() - 120
        os.utime(gpu_lease.LOCK, (old, old))
        self.assertTrue(gpu_lease._stale())

    def test_a_live_owner_is_never_displaced_however_old_the_lock(self):
        gpu_lease.LOCK.write_text(json.dumps({"pid": os.getpid(), "owner": "long job"}))
        old = time.time() - 30 * 3600
        os.utime(gpu_lease.LOCK, (old, old))
        self.assertFalse(gpu_lease._stale())
        with self.assertRaises(TimeoutError):
            with gpu_lease.gpu_lease("other", wait_seconds=0.2, poll=0.05):
                pass
        self.assertTrue(gpu_lease.LOCK.exists())

    def test_release_leaves_a_lock_that_now_belongs_to_someone_else(self):
        with gpu_lease.gpu_lease("test"):
            gpu_lease.LOCK.write_text(json.dumps({"pid": 4000000, "owner": "thief"}))
        self.assertTrue(gpu_lease.LOCK.exists())


if __name__ == "__main__":
    unittest.main()
