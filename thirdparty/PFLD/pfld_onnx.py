"""ONNX Runtime wrapper for yakhyo/face-landmark-detection PFLD."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


PFLD_INPUT_SIZE = 112
_ORT_DLLS_PRELOADED = False


def preload_onnxruntime_gpu_dlls():
    """Load CUDA/cuDNN DLLs from PyTorch/NVIDIA packages when available."""
    global _ORT_DLLS_PRELOADED
    if _ORT_DLLS_PRELOADED:
        return
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            # CPU fallback still works if CUDA provider dependencies are unavailable.
            pass
    _ORT_DLLS_PRELOADED = True


class PFLDLandmarkDetector:
    """Run PFLD on a single face box and map 98 landmarks back to frame space."""

    def __init__(self, model_path, providers=None, crop_scale=1.3):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"PFLD model not found: {self.model_path}")

        preload_onnxruntime_gpu_dlls()

        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.crop_scale = crop_scale

    @property
    def providers(self):
        return self.session.get_providers()

    def detect(self, frame, face):
        crop, origin, size = self._crop_face(frame, face)
        if crop is None:
            return None

        input_img = cv2.resize(crop, (PFLD_INPUT_SIZE, PFLD_INPUT_SIZE))
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_img = np.expand_dims(input_img, axis=0)

        outputs = self.session.run(self.output_names, {self.input_name: input_img})
        landmark_output = outputs[1]
        landmarks = landmark_output[0].reshape(-1, 2) * [size, size]
        landmarks += np.array(origin, dtype=np.float32)

        height, width = frame.shape[:2]
        landmarks[:, 0] = np.clip(landmarks[:, 0], 0, width - 1)
        landmarks[:, 1] = np.clip(landmarks[:, 1], 0, height - 1)
        return landmarks.astype(np.float32)

    def _crop_face(self, frame, face):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = face["xyxy"]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        size = max(2, int(max(box_w, box_h) * self.crop_scale))

        crop_x1 = int(round(cx - size / 2.0))
        crop_y1 = int(round(cy - size / 2.0))
        crop_x2 = crop_x1 + size
        crop_y2 = crop_y1 + size

        src_x1 = max(0, crop_x1)
        src_y1 = max(0, crop_y1)
        src_x2 = min(width, crop_x2)
        src_y2 = min(height, crop_y2)
        if src_x2 <= src_x1 or src_y2 <= src_y1:
            return None, None, None

        crop = frame[src_y1:src_y2, src_x1:src_x2]
        pad_left = src_x1 - crop_x1
        pad_top = src_y1 - crop_y1
        pad_right = crop_x2 - src_x2
        pad_bottom = crop_y2 - src_y2
        if any(pad > 0 for pad in (pad_left, pad_top, pad_right, pad_bottom)):
            crop = cv2.copyMakeBorder(
                crop,
                pad_top,
                pad_bottom,
                pad_left,
                pad_right,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        return crop, (crop_x1, crop_y1), size


def draw_landmarks(frame, landmarks, color=(0, 255, 0), radius=1):
    if landmarks is None:
        return frame
    for x, y in landmarks.astype(np.int32):
        cv2.circle(frame, (int(x), int(y)), radius, color, -1)
    return frame
