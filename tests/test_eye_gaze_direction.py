"""Eye-gaze signs -- settled by visual tests plus explicit geometry, **not** by guessing.

History on 2026-09-16, because it decides what this file locks in:

1. the original code negated all four values; the user reported the gaze mirrored;
2. the negation was removed; the user again reported the gaze mirrored;
3. the same user later reported that **the left/right head direction was mirrored too**.

(3) matters: when a person looks to one side their head turns as well, so a mirrored *head* makes
the whole gaze look mirrored. That is the most likely reading of (2) -- which means the eye sign
itself was probably already correct after step (2), and the head is what actually needs fixing
(see ``tests/test_head_rotation_direction.py``).

The geometry agrees with keeping the gaze un-negated, given the usual conventions:

* a webcam image is **not** mirrored, so the user's right side lands on the image's left;
* when the user looks to their own right, the iris shifts toward the image's left, so the raw
  offset is negative;
* VTS ``EyeRightX`` (the parameter hiyori drives ``ParamEyeBallX`` from) is positive toward the
  viewer's right -- so "looking to the user's right" must be sent as **negative**, un-negated.

Unverified: the vertical axis could not be affected by left/right mirroring, so "up = positive"
comes straight from the geometry (smaller image y = iris up) and carries no negation. It has not
been confirmed by eye yet.

These expectations are written in *image* coordinates, so they stay meaningful either way.
"""
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

import numpy as np  # noqa: E402

from thirdparty.MediaPipe.face_landmarker import estimate_eye_gaze  # noqa: E402

# MediaPipe indices 33/133/159/145 describe the eye on the image's left; 263/362/386/374 the other.
IMAGE_LEFT_EYE = {"outer": 33, "inner": 133, "top_lid": 159, "bottom_lid": 145, "iris": range(468, 473),
                  "centre_x": 120.0, "centre_y": 60.0, "x_radius": 20.0, "y_radius": 10.0}
IMAGE_RIGHT_EYE = {"outer": 263, "inner": 362, "top_lid": 386, "bottom_lid": 374, "iris": range(473, 478),
                   "centre_x": 220.0, "centre_y": 60.0, "x_radius": 20.0, "y_radius": 10.0}


def make_landmarks(iris_x_offset=0.0, iris_y_offset=0.0):
    """478 landmarks with both irises displaced by the given image-space offsets."""
    points = np.zeros((478, 3), dtype=np.float32)
    for eye in (IMAGE_LEFT_EYE, IMAGE_RIGHT_EYE):
        points[eye["outer"]] = (eye["centre_x"] - eye["x_radius"], eye["centre_y"], 0.0)
        points[eye["inner"]] = (eye["centre_x"] + eye["x_radius"], eye["centre_y"], 0.0)
        points[eye["top_lid"]] = (eye["centre_x"], eye["centre_y"] - eye["y_radius"], 0.0)
        points[eye["bottom_lid"]] = (eye["centre_x"], eye["centre_y"] + eye["y_radius"], 0.0)
        for index in eye["iris"]:
            points[index] = (eye["centre_x"] + iris_x_offset, eye["centre_y"] + iris_y_offset, 0.0)
    return points


class TestEyeGazeXSign(unittest.TestCase):
    def test_iris_at_the_centre_is_neutral(self):
        gaze = estimate_eye_gaze(make_landmarks())

        for name in ("EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY"):
            self.assertAlmostEqual(gaze[name], 0.0, places=6)

    def test_iris_toward_the_image_left_stays_negative_on_x(self):
        """Image-left iris = the user looking to their own right -> negative, un-negated."""
        gaze = estimate_eye_gaze(make_landmarks(iris_x_offset=-10.0))

        self.assertAlmostEqual(gaze["EyeRightX"], -0.5, places=6)
        self.assertAlmostEqual(gaze["EyeLeftX"], -0.5, places=6)

    def test_iris_toward_the_image_right_stays_positive_on_x(self):
        gaze = estimate_eye_gaze(make_landmarks(iris_x_offset=10.0))

        self.assertAlmostEqual(gaze["EyeRightX"], 0.5, places=6)
        self.assertAlmostEqual(gaze["EyeLeftX"], 0.5, places=6)

    def test_the_two_eyes_keep_their_own_lateral_offset(self):
        """Only the image-left eye moves: the two parameters must not become identical."""
        points = make_landmarks()
        for index in IMAGE_LEFT_EYE["iris"]:
            points[index] = (IMAGE_LEFT_EYE["centre_x"] - 10.0, IMAGE_LEFT_EYE["centre_y"], 0.0)

        gaze = estimate_eye_gaze(points)

        self.assertAlmostEqual(gaze["EyeRightX"], -0.5, places=6)
        self.assertAlmostEqual(gaze["EyeLeftX"], 0.0, places=6)


class TestEyeGazeYSign(unittest.TestCase):
    """Mirroring cannot affect the vertical axis, so this one comes from geometry alone."""

    def test_iris_above_the_lid_centre_is_positive_on_y(self):
        gaze = estimate_eye_gaze(make_landmarks(iris_y_offset=-5.0))

        self.assertAlmostEqual(gaze["EyeRightY"], 0.5, places=6)
        self.assertAlmostEqual(gaze["EyeLeftY"], 0.5, places=6)

    def test_iris_below_the_lid_centre_is_negative_on_y(self):
        gaze = estimate_eye_gaze(make_landmarks(iris_y_offset=5.0))

        self.assertAlmostEqual(gaze["EyeRightY"], -0.5, places=6)
        self.assertAlmostEqual(gaze["EyeLeftY"], -0.5, places=6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
