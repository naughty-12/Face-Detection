"""The end-of-run summary has to say where the time went.

The bridge already records per-stage durations for the (optional) debug HUD, but a ``--no-gui`` run
printed none of them, so a slow pipeline could only be guessed at. That is exactly what happened:
the same video measured **40.2 fps** when the number was first recorded and **23.0 fps** on
2026-09-16, and nothing in the output could say whether YOLO, MediaPipe, the frame read or the
sinks had changed -- while lowering ``--imgsz`` from 640 to 320 changed nothing at all.

``format_stage_timings`` renders the recorded means (``record_timing`` already stores milliseconds)
as one summary line, so the next slow run diagnoses itself.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

# Imported directly (no skipIf): a missing function must fail this file, not silently skip it.
# Skipping is only for "the environment cannot import the module at all".
from vtube_studio_bridge.vtube_studio_bridge import format_stage_timings  # noqa: E402


class TestStageTimingSummary(unittest.TestCase):
    def test_stage_means_are_rendered_in_milliseconds(self):
        line = format_stage_timings({"read": 5.0, "yolo": 12.34, "mediapipe": 45.6})

        self.assertIn("read=5.0", line)
        self.assertIn("yolo=12.3", line)
        self.assertIn("mediapipe=45.6", line)

    def test_the_total_stage_is_included_when_present(self):
        line = format_stage_timings({"total": 41.0, "mediapipe": 30.0})

        self.assertIn("total=41.0", line)

    def test_no_recorded_stages_produce_no_line(self):
        self.assertEqual(format_stage_timings({}), "")
        self.assertEqual(format_stage_timings(None), "")

    def test_stage_names_keep_their_order(self):
        line = format_stage_timings({"read": 1.0, "yolo": 2.0, "mediapipe": 3.0})

        self.assertLess(line.index("read=1.0"), line.index("yolo=2.0"))
        self.assertLess(line.index("yolo=2.0"), line.index("mediapipe=3.0"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
