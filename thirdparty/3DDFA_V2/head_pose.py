"""Lightweight 3DDFA_V2 ONNX head-pose wrapper."""
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import yaml

from utils.functions import crop_img, parse_roi_box_from_bbox
from utils.io import _load
from utils.pose import calc_pose


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configs" / "mb1_120x120.yml"
_ORT_DLLS_PRELOADED = False


def preload_onnxruntime_gpu_dlls():
    global _ORT_DLLS_PRELOADED
    if _ORT_DLLS_PRELOADED:
        return
    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls()
        except Exception:
            pass
    _ORT_DLLS_PRELOADED = True


class HeadPoseEstimator3DDFA:
    """Estimate yaw/pitch/roll from an image and a face bounding box."""

    def __init__(self, config_path=DEFAULT_CONFIG, providers=None):
        self.config_path = Path(config_path)
        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self.root = self.config_path.parents[1]
        self.size = int(cfg.get("size", 120))
        checkpoint = self.root / cfg["checkpoint_fp"]
        self.onnx_path = checkpoint.with_suffix(".onnx")
        if not self.onnx_path.exists():
            raise FileNotFoundError(f"3DDFA ONNX model not found: {self.onnx_path}")

        mean_std_path = self.root / f"configs/param_mean_std_62d_{self.size}x{self.size}.pkl"
        mean_std = _load(str(mean_std_path))
        self.param_mean = mean_std["mean"]
        self.param_std = mean_std["std"]

        preload_onnxruntime_gpu_dlls()
        available = ort.get_available_providers()
        if providers is None:
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(self.onnx_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    @property
    def providers(self):
        return self.session.get_providers()

    def estimate(self, frame, face):
        bbox = np.array(face["xyxy"], dtype=np.float32)
        roi_box = parse_roi_box_from_bbox(bbox)
        img = crop_img(frame, roi_box)
        if img.size == 0:
            return None

        img = cv2.resize(img, dsize=(self.size, self.size), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32).transpose(2, 0, 1)[np.newaxis, ...]
        img = (img - 127.5) / 128.0

        param = self.session.run(None, {self.input_name: img})[0]
        param = param.flatten().astype(np.float32)
        param = param * self.param_std + self.param_mean

        _, pose = calc_pose(param)
        yaw, pitch, roll = pose
        return {
            "FaceAngleX": float(yaw),
            # 3DDFA pitch direction is opposite to VTube Studio's FaceAngleY.
            "FaceAngleY": float(-pitch),
            "FaceAngleZ": float(roll),
        }


class HeadPoseFilter:
    """EMA smoothing for VTS face-angle parameters."""

    def __init__(self, alpha=0.35):
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
