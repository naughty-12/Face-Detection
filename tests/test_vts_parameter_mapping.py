"""The source -> VTS parameter mapping must not collapse one-directional parameters.

``TrackingCalibrationManager._map_value`` remaps a pipeline value into the range VTube Studio
reports for that parameter. It is a *two-segment* map around the source "center": the segment
below the center goes to ``[target_min, target_default]`` and the one above it to
``[target_default, target_max]``.

That only works when the center sits strictly inside the source range. For one-directional
parameters the center IS an endpoint -- ``EyeOpenLeft`` is 0..1 with a default of 1.0 (eye open)
-- so the "below center" segment covers the entire source range and is mapped onto
``[target_min, target_default]``. VTube Studio reports ``defaultValue = 0.0`` for the eye
parameters (0 = closed), so ``target_default == target_min`` and that segment is a single point.

Measured on this machine (2026-09-16), reproducing it in isolation:

    EyeOpenLeft  state: min=0.0 max=1.0 default=1.0 center=1.0
    EyeOpenLeft  0.97 -> 0.000     <- every input became 0
    EyeOpenRight 0.96 -> 0.000
    MouthOpen    0.50 -> 0.500     ok (its center is at the minimum, taking the other branch)
    FaceAngleX  22.50 -> 15.000    ok (intended remap: source +-45 deg -> VTS +-30 deg)

Live consequence: the avatar's eyes stayed shut for the whole session while head, mouth and gaze
worked (``ParamEyeLOpen = ParamEyeROpen = 0.0`` on screen), and the bridge kept sending
``EyeOpenLeft = 0`` no matter what MediaPipe reported (~0.97, which the camera image did produce).
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
        TRACKING_PARAMETER_SPECS,
        TrackingCalibrationManager,
    )
except Exception as exc:  # pragma: no cover - only on a broken environment
    _IMPORT_ERROR = exc
    TrackingCalibrationManager = None

# What VTube Studio actually reports for these parameters (InputParameterListRequest, hiyori).
VTS_PARAMETERS = {
    "EyeOpenLeft": {"min": 0.0, "max": 1.0, "defaultValue": 0.0},
    "EyeOpenRight": {"min": 0.0, "max": 1.0, "defaultValue": 0.0},
    "MouthOpen": {"min": 0.0, "max": 1.0, "defaultValue": 0.0},
    "MouthSmile": {"min": 0.0, "max": 1.0, "defaultValue": 0.0},
    "FaceAngleX": {"min": -30.0, "max": 30.0, "defaultValue": 0.0},
    "FaceAngleY": {"min": -30.0, "max": 30.0, "defaultValue": 0.0},
    "EyeLeftX": {"min": -1.0, "max": 1.0, "defaultValue": 0.0},
}


def mapped(raw):
    manager = TrackingCalibrationManager(TRACKING_PARAMETER_SPECS)
    return manager.map_values(raw, VTS_PARAMETERS)


@unittest.skipIf(TrackingCalibrationManager is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestOneDirectionalParameters(unittest.TestCase):
    def test_open_eyes_stay_open(self):
        """The regression that made the avatar blink for a whole session."""
        out = mapped({"EyeOpenLeft": 0.97, "EyeOpenRight": 0.96})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.97, places=2,
                               msg="an open eye (0.97) must not be mapped to 0 = closed")
        self.assertAlmostEqual(out["EyeOpenRight"], 0.96, places=2)

    def test_closed_eyes_are_still_closed(self):
        out = mapped({"EyeOpenLeft": 0.0})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.0, places=6)

    def test_a_blink_partway_through_is_not_snapped_to_an_endpoint(self):
        out = mapped({"EyeOpenLeft": 0.5})

        self.assertAlmostEqual(out["EyeOpenLeft"], 0.5, places=2,
                               msg="half-open eyes must stay half-open")

    def test_mouth_open_still_passes_through(self):
        out = mapped({"MouthOpen": 0.5, "MouthSmile": 0.25})

        self.assertAlmostEqual(out["MouthOpen"], 0.5, places=6)
        self.assertAlmostEqual(out["MouthSmile"], 0.25, places=6)


@unittest.skipIf(TrackingCalibrationManager is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestBidirectionalParametersKeepTheirRemap(unittest.TestCase):
    """Parameters whose center sits inside the range must keep the two-segment behaviour."""

    def test_head_angle_is_remapped_into_the_vts_range(self):
        out = mapped({"FaceAngleX": 22.5, "FaceAngleY": -22.5})

        # source +-45 deg -> VTS +-30 deg, so 22.5 becomes 15.0
        self.assertAlmostEqual(out["FaceAngleX"], 15.0, places=3)
        self.assertAlmostEqual(out["FaceAngleY"], -15.0, places=3)

    def test_head_angle_at_the_neutral_point_stays_neutral(self):
        out = mapped({"FaceAngleX": 0.0})

        self.assertAlmostEqual(out["FaceAngleX"], 0.0, places=6)

    def test_eye_gaze_is_remapped_with_its_sign_preserved(self):
        out = mapped({"EyeLeftX": 0.6, "EyeLeftY": -0.25})

        self.assertAlmostEqual(out["EyeLeftX"], 0.6, places=6)
        self.assertAlmostEqual(out["EyeLeftY"], -0.25, places=6)


@unittest.skipIf(TrackingCalibrationManager is None,
                 f"bridge module could not be imported: {_IMPORT_ERROR}")
class TestCalibratedRanges(unittest.TestCase):
    """A calibrated (measured) source range must survive any center the user picks.

    Calibration is how an under-responsive channel gets fixed: measured on this machine
    (2026-09-16), MediaPipe's blink score tops out at ~0.75, so ``EyeOpenLeft`` only ever
    fell to 0.251 -- a squint, never a closed eye. Narrowing the source range to the measured
    0.25..0.99 restores a full blink. But if the user then sets a center *inside* that range,
    the lower segment would be mapped onto ``[target_min, target_default]`` again -- and
    VTube Studio reports ``defaultValue == min`` for the eye parameters, so that segment is a
    single point and the mapping collapses once more.
    """

    def test_a_center_inside_the_source_range_cannot_collapse_the_mapping(self):
        manager = TrackingCalibrationManager(TRACKING_PARAMETER_SPECS)
        manager.set_limits("EyeOpenLeft", 0.25, 0.99)
        manager.set_center("EyeOpenLeft", 0.6)

        out = manager.map_values({"EyeOpenLeft": 0.5},
                                 {"EyeOpenLeft": {"min": 0.0, "max": 1.0, "defaultValue": 0.0}})

        # linear over the calibrated range: (0.5 - 0.25) / (0.99 - 0.25)
        self.assertAlmostEqual(out["EyeOpenLeft"], 0.3378, places=3,
                               msg="a middle center must not flatten the lower segment to the target default")

    def test_calibrating_the_eye_range_restores_a_full_blink(self):
        manager = TrackingCalibrationManager(TRACKING_PARAMETER_SPECS)
        manager.set_limits("EyeOpenLeft", 0.25, 0.99)   # measured: 0.251 .. 0.988
        manager.set_center("EyeOpenLeft", 0.99)

        eyes_shut = manager.map_values({"EyeOpenLeft": 0.251},
                                       {"EyeOpenLeft": {"min": 0.0, "max": 1.0, "defaultValue": 0.0}})
        eyes_open = manager.map_values({"EyeOpenLeft": 0.988},
                                       {"EyeOpenLeft": {"min": 0.0, "max": 1.0, "defaultValue": 0.0}})

        self.assertLess(eyes_shut["EyeOpenLeft"], 0.02, "a full blink must reach closed")
        self.assertGreater(eyes_open["EyeOpenLeft"], 0.98, "wide open eyes must reach open")


if __name__ == "__main__":
    unittest.main(verbosity=2)
