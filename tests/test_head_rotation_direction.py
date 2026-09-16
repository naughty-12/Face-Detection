"""Head-rotation signs. Left/right is verified by eye; pitch and roll are locked as-is.

``estimate_angles`` turns MediaPipe's facial transformation matrix into ``FaceAngleX/Y/Z``.

Verified (2026-09-16): the user reported the **left/right** head direction mirrored
("左右摇头方向...相反"), so ``FaceAngleX`` carries a negation. Pitch ("点头") and roll ("歪头")
were never reported wrong, so they keep the original sign -- these tests lock **both** facts so a
later change has to be deliberate.

The synthetic matrices below are pure rotations in the *matrix* frame, so the expectations say
exactly one thing: "what sign does a +X-degree matrix rotation produce". Which physical head
movement that corresponds to on screen can only be settled by watching the avatar -- which is why
the left/right sign here comes from a user report rather than from the maths.
"""
import math
import os
import sys
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "apps", "vtube_bridge"))

import numpy as np  # noqa: E402

from thirdparty.MediaPipe.face_landmarker import estimate_angles  # noqa: E402

DEGREES = 10.0


def as_matrix(rotation3x3):
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, :3] = rotation3x3
    return matrix


def yaw_matrix(degrees):
    """Rotation about Y -- what ``estimate_angles`` reads as FaceAngleX."""
    a = math.radians(degrees)
    return as_matrix([[math.cos(a), 0.0, math.sin(a)],
                      [0.0, 1.0, 0.0],
                      [-math.sin(a), 0.0, math.cos(a)]])


def pitch_matrix(degrees):
    """Rotation about X -- read as FaceAngleY."""
    a = math.radians(degrees)
    return as_matrix([[1.0, 0.0, 0.0],
                      [0.0, math.cos(a), -math.sin(a)],
                      [0.0, math.sin(a), math.cos(a)]])


def roll_matrix(degrees):
    """Rotation about Z -- read as FaceAngleZ."""
    a = math.radians(degrees)
    return as_matrix([[math.cos(a), -math.sin(a), 0.0],
                      [math.sin(a), math.cos(a), 0.0],
                      [0.0, 0.0, 1.0]])


class TestHeadRotationSigns(unittest.TestCase):
    def test_a_neutral_matrix_is_neutral(self):
        angles = estimate_angles(as_matrix(np.eye(3)))

        for name in ("FaceAngleX", "FaceAngleY", "FaceAngleZ"):
            self.assertAlmostEqual(angles[name], 0.0, places=4)

    def test_left_right_is_negated(self):
        """Verified against the avatar: without this the head shook the wrong way."""
        angles = estimate_angles(yaw_matrix(DEGREES))

        self.assertAlmostEqual(angles["FaceAngleX"], -DEGREES, places=4,
                               msg="FaceAngleX must negate the matrix yaw (user-verified direction)")

    def test_up_down_keeps_its_original_sign(self):
        angles = estimate_angles(pitch_matrix(DEGREES))

        self.assertAlmostEqual(angles["FaceAngleY"], -DEGREES, places=4)

    def test_head_tilt_keeps_its_original_sign(self):
        angles = estimate_angles(roll_matrix(DEGREES))

        self.assertAlmostEqual(angles["FaceAngleZ"], DEGREES, places=4)

    def test_the_three_axes_stay_independent(self):
        """A pure yaw must not leak into pitch or roll."""
        angles = estimate_angles(yaw_matrix(DEGREES))

        self.assertAlmostEqual(angles["FaceAngleY"], 0.0, places=4)
        self.assertAlmostEqual(angles["FaceAngleZ"], 0.0, places=4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
