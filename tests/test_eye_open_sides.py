"""Which eye the avatar closes -- settled by winking at it, not by derivation.

``estimate_expressions`` maps MediaPipe's ``eyeBlinkLeft`` / ``eyeBlinkRight`` blendshapes onto the
VTS parameters ``EyeOpenLeft`` / ``EyeOpenRight``.

MediaPipe names those blendshapes from the **subject's** point of view (ARKit convention), so the
straight-through mapping "user's left eye -> ``EyeOpenLeft``" is the anatomically consistent one:
hiyori routes ``EyeOpenLeft`` to ``ParamEyeLOpen``, the model's own left eye.

It is not what the user wants to see. Reported on 2026-09-16, once the rest of the rig worked:

    "眨左眼反馈的是右眼眨"

i.e. winking their own left eye closed the eye on the *other* side of the screen. The expectation
is mirror-like -- an avatar facing you should close the eye that lines up with yours in your view --
so the two sides are swapped here. Only the swap itself is locked by these tests; the visual
promptness of a blink is a smoothing question, see ``--expression-alpha``.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

_IMPORT_ERROR = None
try:
    from thirdparty.MediaPipe.face_landmarker import estimate_expressions  # noqa: E402
except Exception as exc:  # pragma: no cover - only on a broken environment
    _IMPORT_ERROR = exc
    estimate_expressions = None


@unittest.skipIf(estimate_expressions is None,
                 f"pipeline module could not be imported: {_IMPORT_ERROR}")
class TestEyeOpenSides(unittest.TestCase):
    def test_both_eyes_open_report_both_open(self):
        expressions = estimate_expressions({"eyeBlinkLeft": 0.0, "eyeBlinkRight": 0.0})

        self.assertAlmostEqual(expressions["EyeOpenLeft"], 1.0, places=6)
        self.assertAlmostEqual(expressions["EyeOpenRight"], 1.0, places=6)

    def test_winking_the_users_left_eye_closes_the_models_other_eye(self):
        """The reported bug: the sides were the wrong way round."""
        expressions = estimate_expressions({"eyeBlinkLeft": 1.0, "eyeBlinkRight": 0.0})

        self.assertAlmostEqual(expressions["EyeOpenRight"], 0.0, places=6,
                               msg="the user's left wink must drive EyeOpenRight (mirror-like)")
        self.assertAlmostEqual(expressions["EyeOpenLeft"], 1.0, places=6)

    def test_winking_the_users_right_eye_closes_the_models_other_eye(self):
        expressions = estimate_expressions({"eyeBlinkLeft": 0.0, "eyeBlinkRight": 1.0})

        self.assertAlmostEqual(expressions["EyeOpenLeft"], 0.0, places=6)
        self.assertAlmostEqual(expressions["EyeOpenRight"], 1.0, places=6)

    def test_blinking_both_eyes_closes_both(self):
        expressions = estimate_expressions({"eyeBlinkLeft": 1.0, "eyeBlinkRight": 1.0})

        self.assertAlmostEqual(expressions["EyeOpenLeft"], 0.0, places=6)
        self.assertAlmostEqual(expressions["EyeOpenRight"], 0.0, places=6)

    def test_a_partial_wink_is_not_snapped_to_an_endpoint(self):
        expressions = estimate_expressions({"eyeBlinkLeft": 0.4, "eyeBlinkRight": 0.0})

        self.assertAlmostEqual(expressions["EyeOpenRight"], 0.6, places=6)
        self.assertAlmostEqual(expressions["EyeOpenLeft"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
