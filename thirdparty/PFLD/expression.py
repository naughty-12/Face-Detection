"""Expression heuristics from WFLW/PFLD 98-point landmarks."""
import math

import numpy as np


WFLW_IMAGE_LEFT_EYE = list(range(60, 68))
WFLW_IMAGE_RIGHT_EYE = list(range(68, 76))
WFLW_IMAGE_LEFT_BROW = list(range(33, 42))
WFLW_IMAGE_RIGHT_BROW = list(range(42, 51))
WFLW_OUTER_MOUTH = list(range(76, 88))
WFLW_INNER_MOUTH = list(range(88, 96))
MOUTH_INNER_VERTICAL_PAIRS = ((90, 94), (89, 95), (91, 93))


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


def distance(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)))


def mean_point(points):
    return np.mean(np.asarray(points, dtype=np.float32), axis=0)


def eye_open_ratio(points):
    """Estimate eye openness from a closed 8-point eye contour."""
    points = np.asarray(points, dtype=np.float32)
    width = max(distance(points[0], points[4]), 1e-6)
    vertical = (
        distance(points[1], points[7])
        + distance(points[2], points[6])
        + distance(points[3], points[5])
    ) / 3.0
    return vertical / width


def normalize_eye_open(ratio):
    # Typical open-eye ratios land roughly around 0.20-0.35, closed eyes near 0.05-0.12.
    return clamp((ratio - 0.08) / 0.22)


def estimate_expressions(landmarks):
    """Return VTube Studio expression parameters derived from PFLD landmarks."""
    if landmarks is None or len(landmarks) < 96:
        return {}

    points = np.asarray(landmarks, dtype=np.float32)

    image_left_eye = points[WFLW_IMAGE_LEFT_EYE]
    image_right_eye = points[WFLW_IMAGE_RIGHT_EYE]
    image_left_brow = points[WFLW_IMAGE_LEFT_BROW]
    image_right_brow = points[WFLW_IMAGE_RIGHT_BROW]
    mouth = points[WFLW_OUTER_MOUTH]
    inner_mouth = points[WFLW_INNER_MOUTH]

    image_left_eye_open = normalize_eye_open(eye_open_ratio(image_left_eye))
    image_right_eye_open = normalize_eye_open(eye_open_ratio(image_right_eye))

    mouth_width = max(distance(mouth[0], mouth[6]), 1e-6)
    mouth_vertical = np.mean([
        distance(points[upper], points[lower])
        for upper, lower in MOUTH_INNER_VERTICAL_PAIRS
    ])
    mouth_open_ratio = mouth_vertical / mouth_width
    mouth_open = clamp((mouth_open_ratio - 0.02) / 0.34)

    mouth_center_y = float(mean_point([mouth[0], mouth[6]])[1])
    corner_lift = ((mouth_center_y - float(mouth[0][1])) + (mouth_center_y - float(mouth[6][1]))) / 2.0
    mouth_smile = clamp((corner_lift / mouth_width + 0.03) / 0.18)

    left_eye_center = mean_point(image_left_eye)
    right_eye_center = mean_point(image_right_eye)
    left_brow_center = mean_point(image_left_brow)
    right_brow_center = mean_point(image_right_brow)
    eye_gap = max(distance(left_eye_center, right_eye_center), 1e-6)
    image_left_brow_y = clamp(((left_eye_center[1] - left_brow_center[1]) / eye_gap - 0.12) / 0.18)
    image_right_brow_y = clamp(((right_eye_center[1] - right_brow_center[1]) / eye_gap - 0.12) / 0.18)

    # Webcam frames are not mirrored here: image-right corresponds to the user's left side.
    return {
        "EyeOpenLeft": float(image_right_eye_open),
        "EyeOpenRight": float(image_left_eye_open),
        "MouthOpen": float(mouth_open),
        "MouthSmile": float(mouth_smile),
        "BrowLeftY": float(image_right_brow_y),
        "BrowRightY": float(image_left_brow_y),
    }


class ExpressionFilter:
    """EMA smoothing for expression parameters."""

    def __init__(self, alpha=0.45):
        self.alpha = alpha
        self.state = None

    def update(self, values):
        if not values:
            self.state = None
            return {}
        if self.state is None:
            self.state = dict(values)
            return dict(self.state)
        for key, value in values.items():
            previous = self.state.get(key, value)
            self.state[key] = previous * (1.0 - self.alpha) + value * self.alpha
        return dict(self.state)

    def reset(self):
        self.state = None
