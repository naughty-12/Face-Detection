"""Camera capture settings -- the only lever left for the webcam path's frame rate.

Measured (2026-09-16) on the webcam path:

    read=15.8  yolo=7.8  mediapipe=8.4  total=32.6 ms  ->  29.3 fps

Nearly half of every frame is spent **waiting for a ~30 fps camera**, while the pipeline's own
compute needs only ~17 ms (≈59 fps of headroom). `--imgsz` does nothing (measured 320/480/640:
no difference) and `--yolo-every` only creates idle time, so the capture settings are what is left.
`apps/vtube_bridge/probe_camera.py` measures the candidates; these helpers wire the winner in.
"""
import os
import sys
import unittest
from unittest import mock

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

import cv2  # noqa: E402

# Direct imports, no skipIf: missing helpers must FAIL this file, not skip it silently.
from vtube_studio_bridge.vtube_studio_bridge import (  # noqa: E402
    camera_backend_code,
    open_capture,
    parse_fourcc,
)


class FakeCapture:
    def __init__(self, opened=True):
        self.settings = []
        self.released = False
        self._opened = opened

    def isOpened(self):
        return self._opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True

    def release(self):
        self.released = True


class FakeCv2:
    """Stands in for the cv2 module so ``open_capture`` can be tested without a camera."""

    CAP_MSMF = cv2.CAP_MSMF
    CAP_DSHOW = cv2.CAP_DSHOW
    CAP_PROP_FOURCC = cv2.CAP_PROP_FOURCC
    CAP_PROP_FRAME_WIDTH = cv2.CAP_PROP_FRAME_WIDTH
    CAP_PROP_FRAME_HEIGHT = cv2.CAP_PROP_FRAME_HEIGHT
    CAP_PROP_FPS = cv2.CAP_PROP_FPS
    VideoWriter_fourcc = staticmethod(cv2.VideoWriter_fourcc)

    def __init__(self, opened=True):
        self.capture = FakeCapture(opened)
        self.calls = []

    def VideoCapture(self, *args):
        self.calls.append(args)
        return self.capture


class TestFourcc(unittest.TestCase):
    def test_a_four_letter_code_becomes_the_opencv_integer(self):
        self.assertEqual(parse_fourcc("MJPG"), cv2.VideoWriter_fourcc(*"MJPG"))

    def test_nothing_requested_stays_none(self):
        self.assertIsNone(parse_fourcc(None))
        self.assertIsNone(parse_fourcc(""))
        self.assertIsNone(parse_fourcc("   "))

    def test_a_wrong_length_is_rejected_loudly(self):
        """Silently passing a 3-letter code would make the driver ignore the request."""
        with self.assertRaises(ValueError):
            parse_fourcc("MPG")


class TestCameraBackend(unittest.TestCase):
    def test_known_names_map_to_opencv_backends(self):
        self.assertEqual(camera_backend_code("msmf"), cv2.CAP_MSMF)
        self.assertEqual(camera_backend_code("MSMF"), cv2.CAP_MSMF)
        self.assertEqual(camera_backend_code(" dshow "), cv2.CAP_DSHOW)

    def test_nothing_requested_means_the_system_default(self):
        self.assertIsNone(camera_backend_code(None))
        self.assertIsNone(camera_backend_code(""))

    def test_an_unknown_name_is_rejected_loudly(self):
        with self.assertRaises(ValueError):
            camera_backend_code("obs-virtual")


class TestOpenCapture(unittest.TestCase):
    def open_with(self, **kwargs):
        fake = FakeCv2()
        with mock.patch("vtube_studio_bridge.vtube_studio_bridge.cv2", fake):
            capture = open_capture(0, **kwargs)
        return fake, capture

    def test_no_settings_opens_the_source_without_a_backend_preference(self):
        fake, _ = self.open_with()

        self.assertEqual(fake.calls, [(0,)])

    def test_the_chosen_backend_is_passed_to_opencv(self):
        fake, _ = self.open_with(backend="dshow")

        self.assertEqual(fake.calls, [(0, cv2.CAP_DSHOW)])

    def test_fourcc_is_applied_before_the_resolution(self):
        """Drivers commonly reset the format when the resolution changes, so MJPG goes first."""
        fake, _ = self.open_with(backend="msmf", width=640, height=480, fps=60, fourcc="MJPG")

        props = [prop for prop, _ in fake.capture.settings]
        self.assertEqual(props, [cv2.CAP_PROP_FOURCC, cv2.CAP_PROP_FRAME_WIDTH,
                                 cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])

    def test_the_requested_values_are_the_ones_sent(self):
        fake, _ = self.open_with(width=640, height=480, fps=60)

        self.assertEqual(fake.capture.settings,
                         [(cv2.CAP_PROP_FRAME_WIDTH, 640.0),
                          (cv2.CAP_PROP_FRAME_HEIGHT, 480.0),
                          (cv2.CAP_PROP_FPS, 60.0)])

    def test_a_capture_that_failed_to_open_is_returned_for_the_caller_to_report(self):
        fake = FakeCv2(opened=False)
        with mock.patch("vtube_studio_bridge.vtube_studio_bridge.cv2", fake):
            capture = open_capture(0, backend="msmf", width=640, height=480)

        self.assertIs(capture, fake.capture)
        self.assertEqual(fake.capture.settings, [],
                         "a closed capture must not be configured (it would just be ignored)")

    def test_camera_settings_are_ignored_for_a_file_source(self):
        """`--camera-backend msmf` with a video file: MSMF cannot open files at all, so the
        backend and the capture settings must not be applied to a path."""
        fake = FakeCv2()
        with mock.patch("vtube_studio_bridge.vtube_studio_bridge.cv2", fake):
            open_capture("clip.mp4", backend="msmf", width=640, height=480, fps=60, fourcc="MJPG")

        self.assertEqual(fake.calls, [("clip.mp4",)],
                         "a file source must be opened without a camera backend preference")
        self.assertEqual(fake.capture.settings, [],
                         "resolution/fps/FOURCC are camera properties and must be skipped for files")


if __name__ == "__main__":
    unittest.main(verbosity=2)
