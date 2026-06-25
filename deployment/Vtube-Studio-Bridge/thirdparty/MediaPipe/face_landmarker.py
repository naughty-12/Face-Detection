"""MediaPipe Face Landmarker wrapper for VTube Studio expression parameters."""
import math
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "models" / "face_landmarker.task"


def clamp(value, min_value=0.0, max_value=1.0):
    return max(min_value, min(max_value, value))


class MediaPipeFaceLandmarker:
    """Run MediaPipe Face Landmarker and map blendshapes to VTS parameters."""

    def __init__(self, model_path=DEFAULT_MODEL, crop_scale=1.45):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"MediaPipe Face Landmarker model not found: {self.model_path}")
        self.crop_scale = float(crop_scale)

        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

    def close(self):
        detector = self.detector
        self.detector = None
        if detector is not None:
            detector.close()

    def detect(self, frame, face=None):
        if self.detector is None:
            return None

        input_frame = frame
        offset = (0.0, 0.0)
        scale = 1.0
        if face is not None:
            crop = self._crop_face(frame, face)
            if crop is None:
                return None
            input_frame, offset, scale = crop

        rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self.detector.detect(image)
        if not result.face_landmarks:
            return None

        h, w = input_frame.shape[:2]
        landmarks = np.array(
            [(lm.x * w, lm.y * h, lm.z * w) for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        landmarks[:, 0] = landmarks[:, 0] * scale + offset[0]
        landmarks[:, 1] = landmarks[:, 1] * scale + offset[1]
        landmarks[:, 2] *= scale
        blendshapes = {}
        if result.face_blendshapes:
            blendshapes = {
                category.category_name: float(category.score)
                for category in result.face_blendshapes[0]
            }
        return {
            "landmarks": landmarks,
            "blendshapes": blendshapes,
            "matrix": result.facial_transformation_matrixes[0]
            if result.facial_transformation_matrixes
            else None,
        }

    def _crop_face(self, frame, face):
        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = map(float, face["xyxy"])
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        size = max(x2 - x1, y2 - y1) * self.crop_scale
        if size <= 1:
            return None

        sx = max(0, int(round(cx - size / 2.0)))
        sy = max(0, int(round(cy - size / 2.0)))
        ex = min(frame_w, int(round(cx + size / 2.0)))
        ey = min(frame_h, int(round(cy + size / 2.0)))
        if ex <= sx or ey <= sy:
            return None

        crop = frame[sy:ey, sx:ex]
        return crop, (float(sx), float(sy)), 1.0


def blendshape_score(blendshapes, name, default=0.0):
    return float(blendshapes.get(name, default))


def estimate_expressions(blendshapes):
    """Map MediaPipe blendshape scores to VTube Studio expression parameters."""
    if not blendshapes:
        return {}

    eye_open_left = 1.0 - blendshape_score(blendshapes, "eyeBlinkLeft")
    eye_open_right = 1.0 - blendshape_score(blendshapes, "eyeBlinkRight")
    mouth_open = max(
        blendshape_score(blendshapes, "jawOpen"),
        blendshape_score(blendshapes, "mouthFunnel"),
        blendshape_score(blendshapes, "mouthPucker") * 0.6,
    )
    mouth_smile = (
        blendshape_score(blendshapes, "mouthSmileLeft")
        + blendshape_score(blendshapes, "mouthSmileRight")
    ) * 0.5
    brow_left = max(
        blendshape_score(blendshapes, "browOuterUpLeft"),
        blendshape_score(blendshapes, "browInnerUp"),
    )
    brow_right = max(
        blendshape_score(blendshapes, "browOuterUpRight"),
        blendshape_score(blendshapes, "browInnerUp"),
    )

    return {
        "EyeOpenLeft": clamp(eye_open_left),
        "EyeOpenRight": clamp(eye_open_right),
        "MouthOpen": clamp(mouth_open),
        "MouthSmile": clamp(mouth_smile),
        "BrowLeftY": clamp(brow_left),
        "BrowRightY": clamp(brow_right),
    }


def estimate_angles(matrix):
    """Convert MediaPipe facial transformation matrix to VTS FaceAngleX/Y/Z."""
    if matrix is None:
        return {}

    matrix = np.asarray(matrix, dtype=np.float32)
    if matrix.shape != (4, 4):
        matrix = matrix.reshape(4, 4)

    rotation = matrix[:3, :3]
    # The transform can contain tiny scale/shear components; keep only rotation.
    u, _, vt = np.linalg.svd(rotation)
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt

    sy = math.sqrt(rotation[0, 0] * rotation[0, 0] + rotation[1, 0] * rotation[1, 0])
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(rotation[2, 1], rotation[2, 2])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = math.atan2(rotation[1, 0], rotation[0, 0])
    else:
        pitch = math.atan2(-rotation[1, 2], rotation[1, 1])
        yaw = math.atan2(-rotation[2, 0], sy)
        roll = 0.0

    return {
        "FaceAngleX": float(math.degrees(yaw)),
        "FaceAngleY": float(-math.degrees(pitch)),
        "FaceAngleZ": float(math.degrees(roll)),
    }


def estimate_eye_gaze(landmarks):
    """Estimate VTS eye direction parameters from MediaPipe iris landmarks."""
    if landmarks is None or len(landmarks) < 478:
        return {}

    points = np.asarray(landmarks, dtype=np.float32)
    image_left = _estimate_single_eye_gaze(
        points,
        outer_corner=33,
        inner_corner=133,
        top_lid=159,
        bottom_lid=145,
        iris_indices=range(468, 473),
    )
    image_right = _estimate_single_eye_gaze(
        points,
        outer_corner=263,
        inner_corner=362,
        top_lid=386,
        bottom_lid=374,
        iris_indices=range(473, 478),
    )

    # Webcam frames are not mirrored here: image-right corresponds to the user's left eye.
    return {
        "EyeLeftX": -image_right["x"],
        "EyeLeftY": -image_right["y"],
        "EyeRightX": -image_left["x"],
        "EyeRightY": -image_left["y"],
    }


def _estimate_single_eye_gaze(points, outer_corner, inner_corner, top_lid, bottom_lid, iris_indices):
    outer = points[outer_corner]
    inner = points[inner_corner]
    top = points[top_lid]
    bottom = points[bottom_lid]
    iris = np.mean(points[list(iris_indices)], axis=0)

    x_min = min(float(outer[0]), float(inner[0]))
    x_max = max(float(outer[0]), float(inner[0]))
    y_min = min(float(top[1]), float(bottom[1]))
    y_max = max(float(top[1]), float(bottom[1]))
    x_center = (x_min + x_max) / 2.0
    y_center = (y_min + y_max) / 2.0
    x_radius = max((x_max - x_min) / 2.0, 1e-6)
    y_radius = max((y_max - y_min) / 2.0, 1e-6)

    x = clamp((float(iris[0]) - x_center) / x_radius, -1.0, 1.0)
    y = clamp((y_center - float(iris[1])) / y_radius, -1.0, 1.0)
    return {"x": float(x), "y": float(y)}


def draw_landmarks(frame, landmarks, color=(80, 255, 160), radius=1, show_indexes=False):
    if landmarks is None:
        return frame
    for index, point in enumerate(np.asarray(landmarks, dtype=np.int32)):
        x, y = int(point[0]), int(point[1])
        cv2.circle(frame, (x, y), radius, color, -1, cv2.LINE_AA)
        if show_indexes:
            cv2.putText(
                frame,
                str(index),
                (x + 2, y - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3,
                color,
                1,
                cv2.LINE_AA,
            )
    return frame
