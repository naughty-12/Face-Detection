"""Tests for the VTube Studio bridge's temporal filtering.

``FaceBoxFilter`` is the bridge's main robustness mechanism: EMA smoothing against jitter,
a median window against single-frame jumps, a hold window so brief detection dropouts do
not make the puppet flicker, and a reset rule so switching to a different face does not
slide the box across the screen. None of it was covered by tests, and every one of those
behaviours is easy to get subtly wrong.

Importing the bridge pulls in mediapipe/ultralytics/torch (all installed; mediapipe is a
hard requirement of the application itself), so the suite skips only if that import is
genuinely broken -- and says so rather than failing obscurely.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

_IMPORT_ERROR = None
try:
    from vtube_studio_bridge.vtube_studio_bridge import (  # noqa: E402
        FaceBoxFilter,
        ParameterFilter,
        face_to_vts_position,
        smooth_position,
    )
except Exception as exc:  # pragma: no cover - only on a broken environment
    _IMPORT_ERROR = exc
    FaceBoxFilter = ParameterFilter = None

W, H = 640, 480


def face(cx, cy, w=100.0, h=100.0, conf=0.9):
    return {
        "xyxy": (cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0),
        "center": (cx, cy),
        "confidence": conf,
    }


@unittest.skipIf(FaceBoxFilter is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestFaceBoxFilter(unittest.TestCase):
    def test_first_face_initialises_without_smoothing(self):
        f = FaceBoxFilter(alpha=0.1)
        out, ok = f.update(face(320, 240), W, H)
        self.assertTrue(ok)
        self.assertFalse(out["held"])
        self.assertAlmostEqual(out["center"][0], 320.0)
        self.assertAlmostEqual(out["center"][1], 240.0)

    def test_repeated_identical_measurements_stay_put(self):
        f = FaceBoxFilter(alpha=0.55, window_size=5)
        for _ in range(10):
            out, ok = f.update(face(320, 240), W, H)
        self.assertTrue(ok)
        self.assertAlmostEqual(out["center"][0], 320.0, places=3)
        self.assertAlmostEqual(out["center"][1], 240.0, places=3)

    def test_confidence_is_passed_through_on_a_real_detection(self):
        f = FaceBoxFilter()
        out, _ = f.update(face(320, 240, conf=0.77), W, H)
        self.assertAlmostEqual(out["confidence"], 0.77)
        self.assertFalse(out["held"])

    def test_missing_face_is_held_then_gives_up(self):
        f = FaceBoxFilter(alpha=0.55, hold_frames=3)
        f.update(face(320, 240), W, H)

        for i in range(3):
            out, ok = f.update(None, W, H)
            self.assertTrue(ok, f"miss {i + 1} should still be held")
            self.assertTrue(out["held"])
            self.assertAlmostEqual(out["confidence"], 0.0)

        out, ok = f.update(None, W, H)
        self.assertFalse(ok)
        self.assertIsNone(out)

    def test_hold_frames_zero_gives_up_immediately(self):
        f = FaceBoxFilter(hold_frames=0)
        f.update(face(320, 240), W, H)
        out, ok = f.update(None, W, H)
        self.assertFalse(ok)
        self.assertIsNone(out)

    def test_recovery_after_a_dropout(self):
        f = FaceBoxFilter(alpha=0.55, hold_frames=2)
        f.update(face(320, 240), W, H)
        f.update(None, W, H)
        out, ok = f.update(face(320, 240), W, H)
        self.assertTrue(ok)
        self.assertFalse(out["held"])
        self.assertAlmostEqual(out["center"][0], 320.0, places=3)

    def test_median_window_suppresses_a_single_outlier(self):
        """alpha=1 isolates the median filter: the state becomes the window median."""
        f = FaceBoxFilter(alpha=1.0, window_size=5)
        for _ in range(3):
            f.update(face(320, 240), W, H)

        # a 6 px nudge: close enough not to trigger a reset, far enough to matter
        out, _ = f.update(face(326, 240), W, H)
        self.assertAlmostEqual(out["center"][0], 320.0, places=3,
                               msg="median of [320,320,320,326] is 320, so the nudge is dropped")

    def test_far_jump_resets_instead_of_sliding(self):
        """Switching faces must not drag the box across the frame."""
        f = FaceBoxFilter(alpha=0.05, window_size=5)
        f.update(face(100, 100), W, H)
        f.update(face(100, 100), W, H)

        out, _ = f.update(face(500, 400), W, H)   # ~0.78 normalised away: far beyond 0.35
        self.assertAlmostEqual(out["center"][0], 500.0, places=3)
        self.assertAlmostEqual(out["center"][1], 400.0, places=3)

    def test_large_size_change_resets(self):
        f = FaceBoxFilter(alpha=0.05, window_size=5)
        f.update(face(320, 240, w=50, h=50), W, H)

        out, _ = f.update(face(320, 240, w=200, h=200), W, H)   # 4x larger, ratio > 2.5
        self.assertAlmostEqual(out["xyxy"][2] - out["xyxy"][0], 200.0, places=1)

    def test_output_is_clamped_inside_the_frame(self):
        f = FaceBoxFilter()
        out, ok = f.update(face(5, 5, w=200, h=200), W, H)
        self.assertTrue(ok)
        x1, y1, x2, y2 = out["xyxy"]
        self.assertGreaterEqual(x1, 0.0)
        self.assertGreaterEqual(y1, 0.0)
        self.assertLessEqual(x2, W - 1.0)
        self.assertLessEqual(y2, H - 1.0)

    def test_window_size_one_still_works(self):
        f = FaceBoxFilter(alpha=1.0, window_size=1)
        out, ok = f.update(face(320, 240), W, H)
        self.assertTrue(ok)
        self.assertAlmostEqual(out["center"][0], 320.0, places=3)

    def test_reset_clears_state(self):
        f = FaceBoxFilter()
        f.update(face(320, 240), W, H)
        f.reset()
        self.assertIsNone(f.state)
        self.assertEqual(f.missed_frames, 0)


@unittest.skipIf(ParameterFilter is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestParameterFilter(unittest.TestCase):
    def test_first_update_is_adopted_verbatim(self):
        f = ParameterFilter(alpha=0.45)
        self.assertEqual(f.update({"a": 1.0, "b": 2.0}), {"a": 1.0, "b": 2.0})

    def test_empty_input_clears_state(self):
        f = ParameterFilter()
        f.update({"a": 1.0})
        self.assertEqual(f.update({}), {})
        self.assertIsNone(f.state)

    def test_values_converge_toward_the_target(self):
        f = ParameterFilter(alpha=0.5)
        f.update({"a": 0.0})
        for _ in range(20):
            out = f.update({"a": 1.0})
        self.assertAlmostEqual(out["a"], 1.0, places=4)

    def test_channels_are_independent(self):
        f = ParameterFilter(alpha=0.5)
        f.update({"a": 0.0, "b": 10.0})
        out = f.update({"a": 0.0, "b": 0.0})
        self.assertAlmostEqual(out["a"], 0.0, places=6)
        self.assertAlmostEqual(out["b"], 5.0, places=6)

    def test_a_key_absent_from_an_update_keeps_its_smoothed_value(self):
        f = ParameterFilter(alpha=0.5)
        f.update({"a": 0.0, "b": 0.0})
        out = f.update({"a": 1.0})
        self.assertAlmostEqual(out["a"], 0.5, places=6)
        self.assertAlmostEqual(out["b"], 0.0, places=6)

    def test_reset_clears_state(self):
        f = ParameterFilter()
        f.update({"a": 1.0})
        f.reset()
        self.assertIsNone(f.state)


@unittest.skipIf(FaceBoxFilter is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestPositionMapping(unittest.TestCase):
    def test_centred_face_maps_to_origin(self):
        pos = face_to_vts_position(face(320, 240), W, H)
        self.assertAlmostEqual(pos["x"], 0.0, places=6)
        self.assertAlmostEqual(pos["y"], 0.0, places=6)

    def test_x_increases_to_the_right_and_y_upwards(self):
        right = face_to_vts_position(face(600, 240), W, H)
        left = face_to_vts_position(face(40, 240), W, H)
        self.assertGreater(right["x"], 0.0)
        self.assertLess(left["x"], 0.0)

        high = face_to_vts_position(face(320, 40), W, H)
        low = face_to_vts_position(face(320, 440), W, H)
        self.assertGreater(high["y"], 0.0)
        self.assertLess(low["y"], 0.0)

    def test_values_are_bounded(self):
        for cx, cy in ((0, 0), (W, H), (-500, -500), (5000, 5000)):
            pos = face_to_vts_position(face(cx, cy, w=1, h=1), W, H)
            for axis in ("x", "y", "z"):
                self.assertGreaterEqual(pos[axis], -10.0)
                self.assertLessEqual(pos[axis], 10.0)

    def test_smooth_position_without_history_returns_current(self):
        current = {"x": 1.0, "y": 2.0, "z": 3.0}
        self.assertEqual(smooth_position(None, current, 0.5), current)

    def test_smooth_position_interpolates(self):
        out = smooth_position({"x": 0.0, "y": 0.0, "z": 0.0},
                              {"x": 10.0, "y": 10.0, "z": 10.0}, 0.25)
        for axis in ("x", "y", "z"):
            self.assertAlmostEqual(out[axis], 2.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
