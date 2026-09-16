"""Skipping the face detector on some frames -- and why that alone may not help.

``--yolo-every N`` runs the YOLO face detector only on every Nth frame and reuses the previous
box on the others, to buy back the detector's per-frame cost.

Measured on this machine (2026-09-16), the two input paths differ completely:

    video file  : read=1.5  yolo=9.5  mediapipe=11.0  total=22.8 ms  -> 38.0 fps
    webcam      : read=15.8 yolo=7.8  mediapipe=8.4   total=32.6 ms  -> 29.3 fps

On the webcam the frame **read** is the largest single cost, and its 15.8 ms is time spent waiting
for a camera that delivers roughly 30 frames per second. The pipeline's own compute is only
~17 ms, so skipping work cannot lift the rate above the camera's ceiling -- it only creates idle
time. Keep that in mind before expecting a big gain from this option.

The decision itself is a pure function so the "which frames run the detector" contract is testable
without a camera.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

# Direct import, no skipIf: a missing helper must FAIL this file, not skip it silently.
from vtube_studio_bridge.vtube_studio_bridge import should_run_detection  # noqa: E402


class TestShouldRunDetection(unittest.TestCase):
    def test_every_one_runs_on_every_frame(self):
        self.assertTrue(all(should_run_detection(i, 1) for i in range(6)))

    def test_zero_or_negative_means_every_frame(self):
        """A nonsensical value must fall back to 'always run', never to 'never run'."""
        self.assertTrue(all(should_run_detection(i, 0) for i in range(3)))
        self.assertTrue(all(should_run_detection(i, -5) for i in range(3)))

    def test_every_two_runs_on_even_frames(self):
        self.assertEqual([should_run_detection(i, 2) for i in range(6)],
                         [True, False, True, False, True, False])

    def test_every_three_runs_one_frame_in_three(self):
        self.assertEqual([should_run_detection(i, 3) for i in range(7)],
                         [True, False, False, True, False, False, True])

    def test_the_first_frame_always_runs(self):
        """Frame 0 has no previous box to reuse -- it must always detect."""
        for every in (1, 2, 3, 10):
            self.assertTrue(should_run_detection(0, every))


if __name__ == "__main__":
    unittest.main(verbosity=2)
