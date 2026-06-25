"""Bridge YOLO face position to VTube Studio tracking parameters."""
import argparse
import json
import math
import os
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from functools import partial
from collections import deque
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1].parents[1]
BRIDGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from websocket import create_connection

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except Exception:  # pragma: no cover - optional GUI dependency
    QtCore = QtGui = QtWidgets = None

from thirdparty.MediaPipe.face_landmarker import MediaPipeFaceLandmarker
from thirdparty.MediaPipe.face_landmarker import draw_landmarks as draw_mediapipe_landmarks
from thirdparty.MediaPipe.face_landmarker import estimate_angles as estimate_mediapipe_angles
from thirdparty.MediaPipe.face_landmarker import estimate_eye_gaze as estimate_mediapipe_eye_gaze
from thirdparty.MediaPipe.face_landmarker import estimate_expressions as estimate_mediapipe_expressions


DEFAULT_MODEL = PROJECT_ROOT / "training" / "checkpoints" / "best_model_v2.onnx"
DEFAULT_MEDIAPIPE_MODEL = PROJECT_ROOT / "deployment" / "Vtube-Studio-Bridge" / "thirdparty" / "MediaPipe" / "models" / "face_landmarker.task"
PLUGIN_NAME = "Face Detection VTube Studio Bridge"
PLUGIN_DEVELOPER = "Face-Detection Project"
TOKEN_FILE = Path(os.getenv("APPDATA", Path.home())) / "FaceDetectionVTubeStudioBridge" / "vts_token.json"
VTS_RETRY_INTERVAL = 10.0
CONFIG_DIR = BRIDGE_ROOT / "config"
CALIBRATION_CONFIG_FILE = CONFIG_DIR / "tracking_calibration.json"

TRACKING_PARAMETER_SPECS = [
    {"name": "FacePositionX", "source_default": 0.0, "source_min": -10.0, "source_max": 10.0},
    {"name": "FacePositionY", "source_default": 0.0, "source_min": -10.0, "source_max": 10.0},
    {"name": "FacePositionZ", "source_default": 0.0, "source_min": -10.0, "source_max": 10.0},
    {"name": "EyeOpenLeft", "source_default": 1.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "EyeOpenRight", "source_default": 1.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "MouthOpen", "source_default": 0.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "MouthSmile", "source_default": 0.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "BrowLeftY", "source_default": 0.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "BrowRightY", "source_default": 0.0, "source_min": 0.0, "source_max": 1.0},
    {"name": "FaceAngleX", "source_default": 0.0, "source_min": -45.0, "source_max": 45.0},
    {"name": "FaceAngleY", "source_default": 0.0, "source_min": -45.0, "source_max": 45.0},
    {"name": "FaceAngleZ", "source_default": 0.0, "source_min": -45.0, "source_max": 45.0},
    {"name": "EyeLeftX", "source_default": 0.0, "source_min": -1.0, "source_max": 1.0},
    {"name": "EyeLeftY", "source_default": 0.0, "source_min": -1.0, "source_max": 1.0},
    {"name": "EyeRightX", "source_default": 0.0, "source_min": -1.0, "source_max": 1.0},
    {"name": "EyeRightY", "source_default": 0.0, "source_min": -1.0, "source_max": 1.0},
]
TRACKING_PARAMETER_NAMES = [spec["name"] for spec in TRACKING_PARAMETER_SPECS]
_ACTIVE_MEDIAPIPE_DETECTOR = None


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def build_tracking_values(position, expressions, angles, eye_gaze):
    return {
        "FacePositionX": float(position.get("x", 0.0)),
        "FacePositionY": float(position.get("y", 0.0)),
        "FacePositionZ": float(position.get("z", 0.0)),
        "EyeOpenLeft": float(expressions.get("EyeOpenLeft", 1.0)),
        "EyeOpenRight": float(expressions.get("EyeOpenRight", 1.0)),
        "MouthOpen": float(expressions.get("MouthOpen", 0.0)),
        "MouthSmile": float(expressions.get("MouthSmile", 0.0)),
        "BrowLeftY": float(expressions.get("BrowLeftY", 0.0)),
        "BrowRightY": float(expressions.get("BrowRightY", 0.0)),
        "FaceAngleX": float(angles.get("FaceAngleX", 0.0)),
        "FaceAngleY": float(angles.get("FaceAngleY", 0.0)),
        "FaceAngleZ": float(angles.get("FaceAngleZ", 0.0)),
        "EyeLeftX": float(eye_gaze.get("EyeLeftX", 0.0)),
        "EyeLeftY": float(eye_gaze.get("EyeLeftY", 0.0)),
        "EyeRightX": float(eye_gaze.get("EyeRightX", 0.0)),
        "EyeRightY": float(eye_gaze.get("EyeRightY", 0.0)),
    }


def build_tracking_valid_mask(face_found, mediapipe_ready):
    valid = {
        "FacePositionX": bool(face_found),
        "FacePositionY": bool(face_found),
        "FacePositionZ": bool(face_found),
        "EyeOpenLeft": bool(mediapipe_ready),
        "EyeOpenRight": bool(mediapipe_ready),
        "MouthOpen": bool(mediapipe_ready),
        "MouthSmile": bool(mediapipe_ready),
        "BrowLeftY": bool(mediapipe_ready),
        "BrowRightY": bool(mediapipe_ready),
        "FaceAngleX": bool(mediapipe_ready),
        "FaceAngleY": bool(mediapipe_ready),
        "FaceAngleZ": bool(mediapipe_ready),
        "EyeLeftX": bool(mediapipe_ready),
        "EyeLeftY": bool(mediapipe_ready),
        "EyeRightX": bool(mediapipe_ready),
        "EyeRightY": bool(mediapipe_ready),
    }
    return valid


def build_parameter_values(mapped_values):
    return [
        {"id": name, "value": float(mapped_values.get(name, 0.0)), "weight": 1.0}
        for name in TRACKING_PARAMETER_NAMES
    ]


@dataclass
class TrackingCalibrationState:
    name: str
    source_default: float
    source_min: float
    source_max: float
    current_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    calibrating: bool = False

    def __post_init__(self):
        self.current_value = float(self.source_default)
        self.min_value = float(self.source_min)
        self.max_value = float(self.source_max)

    def reset_to_default(self):
        default_value = float(self.source_default)
        self.min_value = default_value
        self.max_value = default_value

    def set_limits(self, min_value, max_value):
        min_value = float(min_value)
        max_value = float(max_value)
        if min_value <= max_value:
            self.min_value = min_value
            self.max_value = max_value
        else:
            self.min_value = max_value
            self.max_value = min_value

    def update_current(self, value, valid=True):
        self.current_value = float(value)
        if self.calibrating and valid:
            if self.current_value < self.min_value:
                self.min_value = self.current_value
            if self.current_value > self.max_value:
                self.max_value = self.current_value

    def toggle_calibration(self):
        self.calibrating = not self.calibrating
        if self.calibrating:
            self.reset_to_default()


class TrackingCalibrationManager:
    def __init__(self, specs):
        self.lock = threading.RLock()
        self.states = {
            spec["name"]: TrackingCalibrationState(
                name=spec["name"],
                source_default=spec["source_default"],
                source_min=spec["source_min"],
                source_max=spec["source_max"],
            )
            for spec in specs
        }

    def update_current_values(self, values, valid_mask=None):
        valid_mask = valid_mask or {}
        with self.lock:
            for name, state in self.states.items():
                if name in values:
                    state.update_current(values[name], valid_mask.get(name, True))

    def toggle(self, name):
        with self.lock:
            self.states[name].toggle_calibration()
            return self.states[name]

    def set_limits(self, name, min_value, max_value):
        with self.lock:
            self.states[name].set_limits(min_value, max_value)
            return self.states[name]

    def items(self):
        with self.lock:
            return list(self.states.items())

    def snapshot(self):
        with self.lock:
            return {
                name: {
                    "name": state.name,
                    "current_value": state.current_value,
                    "min_value": state.min_value,
                    "max_value": state.max_value,
                    "calibrating": state.calibrating,
                }
                for name, state in self.states.items()
            }

    def load(self, path=CALIBRATION_CONFIG_FILE):
        path = Path(path)
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"[WARN] Failed to load config from {path}: {exc}")
            return {}

        parameters = data.get("parameters", {})
        with self.lock:
            for name, values in parameters.items():
                if name not in self.states or not isinstance(values, dict):
                    continue
                if "min" in values and "max" in values:
                    self.states[name].set_limits(values["min"], values["max"])
        return data

    def save(self, path=CALIBRATION_CONFIG_FILE, camera_preview_enabled=None):
        path = Path(path)
        if camera_preview_enabled is None and path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    camera_preview_enabled = json.load(f).get("cameraPreviewEnabled", True)
            except (OSError, json.JSONDecodeError):
                camera_preview_enabled = True
        elif camera_preview_enabled is None:
            camera_preview_enabled = True

        with self.lock:
            data = {
                "version": 1,
                "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "cameraPreviewEnabled": bool(camera_preview_enabled),
                "parameters": {
                    name: {
                        "min": state.min_value,
                        "max": state.max_value,
                    }
                    for name, state in self.states.items()
                },
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return path

    @staticmethod
    def _map_value(state, raw_value, param_info):
        source_min = float(state.min_value)
        source_max = float(state.max_value)
        if source_min > source_max:
            source_min, source_max = source_max, source_min

        raw_value = clamp(float(raw_value), source_min, source_max)
        if abs(source_max - source_min) < 1e-9:
            if param_info is not None and "defaultValue" in param_info:
                return clamp(float(param_info["defaultValue"]), float(param_info["min"]), float(param_info["max"]))
            return raw_value

        normalized = (raw_value - source_min) / (source_max - source_min)
        if param_info is None:
            return raw_value
        target_min = float(param_info["min"])
        target_max = float(param_info["max"])
        mapped = target_min + normalized * (target_max - target_min)
        return clamp(mapped, min(target_min, target_max), max(target_min, target_max))

    def map_values(self, values, input_parameters):
        with self.lock:
            mapped = {}
            for name, state in self.states.items():
                mapped[name] = self._map_value(
                    state,
                    float(values.get(name, state.source_default)),
                    input_parameters.get(name),
                )
            return mapped


class VTubeStudioClient:
    """Tiny VTube Studio Public API client for parameter injection."""

    def __init__(self, host="127.0.0.1", port=8001, timeout=1.0):
        self.url = f"ws://{host}:{port}"
        self.timeout = timeout
        self.ws = None
        self.input_parameters = {}
        self.input_model_loaded = False
        self.input_model_name = ""

    @property
    def connected(self):
        return self.ws is not None

    def connect(self):
        self.close()
        self.ws = create_connection(self.url, timeout=self.timeout)

    def close(self, timeout=0.2):
        if self.ws is not None:
            ws = self.ws
            self.ws = None
            try:
                ws.close(timeout=timeout)
            except Exception:
                pass

    def request(self, message_type, data=None):
        payload = {
            "apiName": "VTubeStudioPublicAPI",
            "apiVersion": "1.0",
            "requestID": str(uuid.uuid4()),
            "messageType": message_type,
            "data": data or {},
        }
        if self.ws is None:
            raise RuntimeError("VTube Studio is not connected.")
        try:
            self.ws.send(json.dumps(payload))
            response = json.loads(self.ws.recv())
        except Exception:
            self.close()
            raise
        if response.get("messageType") == "APIError":
            error = response.get("data", {})
            raise RuntimeError(f"VTube Studio API error: {error}")
        return response.get("data", {})

    def authenticate(self):
        token = self._load_token()
        if token and self._authenticate_with_token(token):
            return

        print("Requesting VTube Studio auth token. Please allow the plugin in VTube Studio.")
        data = self.request(
            "AuthenticationTokenRequest",
            {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
            },
        )
        token = data["authenticationToken"]
        self._save_token(token)
        if not self._authenticate_with_token(token):
            raise RuntimeError("VTube Studio authentication failed after token approval.")

    def refresh_input_parameters(self):
        data = self.request("InputParameterListRequest")
        self.input_model_loaded = bool(data.get("modelLoaded"))
        self.input_model_name = data.get("modelName", "")

        parameters = {}
        for section in ("defaultParameters", "customParameters"):
            for param in data.get(section, []):
                name = param.get("name")
                if name:
                    parameters[name] = param
        self.input_parameters = parameters

        if self.input_model_loaded:
            print(
                f"VTS input parameters: {len(self.input_parameters)} "
                f"for model '{self.input_model_name}'"
            )
        else:
            print("VTS input parameters: no model loaded")
        return data

    def inject_face_position(self, face_found, parameter_values):
        self.request(
            "InjectParameterDataRequest",
            {
                "faceFound": bool(face_found),
                "mode": "set",
                "parameterValues": parameter_values,
            },
        )

    def _authenticate_with_token(self, token):
        data = self.request(
            "AuthenticationRequest",
            {
                "pluginName": PLUGIN_NAME,
                "pluginDeveloper": PLUGIN_DEVELOPER,
                "authenticationToken": token,
            },
        )
        return bool(data.get("authenticated"))

    @staticmethod
    def _load_token():
        if not TOKEN_FILE.exists():
            return None
        try:
            with open(TOKEN_FILE, "r", encoding="utf-8") as f:
                return json.load(f).get("authenticationToken")
        except (OSError, json.JSONDecodeError):
            return None

    @staticmethod
    def _save_token(token):
        TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_FILE, "w", encoding="utf-8") as f:
            json.dump({"authenticationToken": token}, f, indent=2)


class VTubeStudioWorker:
    """Background VTS sender so websocket latency does not block inference."""

    def __init__(self, host, port, send_fps):
        self.vts = VTubeStudioClient(host, port)
        self.send_interval = 1.0 / max(1.0, send_fps)
        self.latest = {"face_found": False, "parameter_values": []}
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="VTubeStudioWorker", daemon=True)
        self.last_send = 0.0
        self.next_retry_at = 0.0

    @property
    def connected(self):
        return self.vts.connected

    def start(self):
        self.thread.start()

    def update(self, face_found, parameter_values):
        with self.lock:
            self.latest = {
                "face_found": bool(face_found),
                "parameter_values": [dict(item) for item in parameter_values],
            }

    def stop(self):
        self.stop_event.set()
        # Closing the socket first interrupts any blocking send/recv/connect cleanup.
        self.vts.close(timeout=0.2)
        self.thread.join(timeout=0.5)
        if self.thread.is_alive():
            print("[WARN] VTube Studio worker did not stop within 0.5s; exiting anyway.")

    def _snapshot(self):
        with self.lock:
            return {
                "face_found": self.latest["face_found"],
                "parameter_values": [dict(item) for item in self.latest["parameter_values"]],
            }

    def _run(self):
        while not self.stop_event.is_set():
            now = time.perf_counter()
            if not self.vts.connected:
                if now >= self.next_retry_at:
                    self._connect_once(now)
                self.stop_event.wait(0.05)
                continue

            if now - self.last_send < self.send_interval:
                self.stop_event.wait(0.002)
                continue

            state = self._snapshot()
            try:
                self.vts.inject_face_position(
                    state["face_found"],
                    state["parameter_values"],
                )
            except Exception as exc:
                print(f"[WARN] Lost VTube Studio connection: {exc}. Retrying in {VTS_RETRY_INTERVAL:.0f}s.")
                self.vts.close()
                self.next_retry_at = time.perf_counter() + VTS_RETRY_INTERVAL
            self.last_send = time.perf_counter()

    def _connect_once(self, now):
        try:
            print(f"Connecting to VTube Studio at {self.vts.url} ...")
            self.vts.connect()
            self.vts.authenticate()
            print("VTube Studio connected.")
            self.vts.refresh_input_parameters()
        except Exception as exc:
            self.vts.close()
            print(f"[WARN] VTube Studio unavailable: {exc}. Retrying in {VTS_RETRY_INTERVAL:.0f}s.")
            self.next_retry_at = now + VTS_RETRY_INTERVAL
        else:
            self.next_retry_at = now + VTS_RETRY_INTERVAL


def select_center_face(boxes, frame_width, frame_height, target_center=None):
    """Pick the detected face whose center is closest to the selected preview point."""
    if boxes is None or len(boxes) == 0:
        return None

    xyxy = boxes.xyxy.detach().cpu().numpy()
    conf = boxes.conf.detach().cpu().numpy()
    if target_center is None:
        target_x = frame_width / 2.0
        target_y = frame_height / 2.0
    else:
        target_x = clamp(float(target_center[0]), 0.0, 1.0) * frame_width
        target_y = clamp(float(target_center[1]), 0.0, 1.0) * frame_height

    best = None
    best_distance = math.inf
    for box, score in zip(xyxy, conf):
        x1, y1, x2, y2 = map(float, box)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        distance = (cx - target_x) ** 2 + (cy - target_y) ** 2
        if distance < best_distance:
            best_distance = distance
            best = {
                "xyxy": (x1, y1, x2, y2),
                "center": (cx, cy),
                "confidence": float(score),
            }
    return best


class FaceBoxFilter:
    """Temporal filter for face box center and size."""

    def __init__(self, alpha=0.35, window_size=5, hold_frames=3, reset_distance=0.35):
        self.alpha = alpha
        self.window = deque(maxlen=max(1, window_size))
        self.hold_frames = max(0, hold_frames)
        self.reset_distance = reset_distance
        self.state = None
        self.missed_frames = 0

    def update(self, face, frame_width, frame_height):
        if face is None:
            self.missed_frames += 1
            if self.state is not None and self.missed_frames <= self.hold_frames:
                return self._state_to_face(frame_width, frame_height, 0.0, held=True), True
            self.reset()
            return None, False

        measurement = self._face_to_measurement(face, frame_width, frame_height)
        if self.state is None or self._should_reset(measurement):
            self.window.clear()
            self.window.append(measurement)
            self.state = measurement
        else:
            self.window.append(measurement)
            measurement = self._median_measurement()
            self.state = tuple(
                previous * (1.0 - self.alpha) + current * self.alpha
                for previous, current in zip(self.state, measurement)
            )

        self.missed_frames = 0
        return self._state_to_face(frame_width, frame_height, face["confidence"], held=False), True

    def reset(self):
        self.window.clear()
        self.state = None
        self.missed_frames = 0

    @staticmethod
    def _face_to_measurement(face, frame_width, frame_height):
        x1, y1, x2, y2 = face["xyxy"]
        cx, cy = face["center"]
        box_w = max(1.0, x2 - x1)
        box_h = max(1.0, y2 - y1)
        return (
            cx / frame_width,
            cy / frame_height,
            box_w / frame_width,
            box_h / frame_height,
        )

    def _should_reset(self, measurement):
        cx, cy, box_w, box_h = measurement
        prev_cx, prev_cy, prev_w, prev_h = self.state
        center_distance = math.hypot(cx - prev_cx, cy - prev_cy)
        size_ratio = max(box_w / max(prev_w, 1e-6), prev_w / max(box_w, 1e-6),
                         box_h / max(prev_h, 1e-6), prev_h / max(box_h, 1e-6))
        return center_distance > self.reset_distance or size_ratio > 2.5

    def _median_measurement(self):
        values = np.array(self.window, dtype=np.float32)
        return tuple(np.median(values, axis=0).tolist())

    def _state_to_face(self, frame_width, frame_height, confidence, held=False):
        cx_n, cy_n, box_w_n, box_h_n = self.state
        cx = cx_n * frame_width
        cy = cy_n * frame_height
        box_w = box_w_n * frame_width
        box_h = box_h_n * frame_height
        x1 = clamp(cx - box_w / 2.0, 0.0, frame_width - 1.0)
        y1 = clamp(cy - box_h / 2.0, 0.0, frame_height - 1.0)
        x2 = clamp(cx + box_w / 2.0, 0.0, frame_width - 1.0)
        y2 = clamp(cy + box_h / 2.0, 0.0, frame_height - 1.0)
        return {
            "xyxy": (x1, y1, x2, y2),
            "center": (cx, cy),
            "confidence": confidence,
            "held": held,
        }


def face_to_vts_position(face, frame_width, frame_height):
    """Map a face bounding box to VTube Studio FacePositionX/Y/Z."""
    x1, y1, x2, y2 = face["xyxy"]
    cx, cy = face["center"]
    box_w = max(1.0, x2 - x1)
    box_h = max(1.0, y2 - y1)

    # VTS default tracking position parameters are commonly mapped around -10..10.
    x = clamp((cx / frame_width - 0.5) * 20.0, -10.0, 10.0)
    y = clamp((0.5 - cy / frame_height) * 20.0, -10.0, 10.0)

    # Approximate depth from face size. This is not true 3D depth, just useful puppet motion.
    relative_size = max(box_w / frame_width, box_h / frame_height)
    z = clamp((relative_size - 0.25) * 40.0, -10.0, 10.0)
    return {"x": x, "y": y, "z": z}


def smooth_position(previous, current, alpha):
    if previous is None:
        return current
    return {
        key: previous[key] * (1.0 - alpha) + current[key] * alpha
        for key in ("x", "y", "z")
    }


class ParameterFilter:
    """EMA smoothing for MediaPipe expression and angle parameters."""

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


def record_timing(timings, name, start_time):
    timings[name].append((time.perf_counter() - start_time) * 1000.0)


def average_timings(timings):
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in timings.items()
    }


if QtWidgets is not None:
    class PreviewLabel(QtWidgets.QLabel):
        centerSelected = QtCore.pyqtSignal(float, float)

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._pixmap_size = None

        def set_preview_pixmap(self, pixmap):
            self._pixmap_size = pixmap.size()
            self.setPixmap(pixmap)

        def mousePressEvent(self, event):
            if event.button() != QtCore.Qt.LeftButton or self._pixmap_size is None:
                super().mousePressEvent(event)
                return

            pixmap_w = self._pixmap_size.width()
            pixmap_h = self._pixmap_size.height()
            if pixmap_w <= 0 or pixmap_h <= 0:
                super().mousePressEvent(event)
                return

            left = (self.width() - pixmap_w) / 2.0
            top = (self.height() - pixmap_h) / 2.0
            x = event.pos().x()
            y = event.pos().y()
            if x < left or y < top or x > left + pixmap_w or y > top + pixmap_h:
                super().mousePressEvent(event)
                return

            self.centerSelected.emit(
                clamp((x - left) / pixmap_w, 0.0, 1.0),
                clamp((y - top) / pixmap_h, 0.0, 1.0),
            )
            event.accept()


    class QtDebugWindow(QtWidgets.QWidget):
        """Qt-based debug preview window for future calibration tools."""

        def __init__(self, calibration_manager, camera_preview_enabled=True, title="VTube Studio Bridge"):
            super().__init__()
            self.calibration_manager = calibration_manager
            self._rows = {}
            self._closed = False
            self._target_center = (0.5, 0.5)
            self._camera_preview_enabled = bool(camera_preview_enabled)
            self.setWindowTitle(title)
            self.setMinimumSize(1280, 720)
            self.setFocusPolicy(QtCore.Qt.StrongFocus)

            self._camera_preview_button = QtWidgets.QPushButton()
            self._sync_camera_preview_button()
            self._camera_preview_button.clicked.connect(self._toggle_camera_preview)

            self._label = PreviewLabel(alignment=QtCore.Qt.AlignCenter)
            self._label.setMinimumSize(640, 360)
            self._label.setStyleSheet("background-color: #111; color: #eee;")
            self._label.setText("Waiting for frames...")
            self._label.centerSelected.connect(self._on_preview_center_selected)

            self._table = QtWidgets.QTableWidget(len(TRACKING_PARAMETER_SPECS), 5)
            self._table.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Calibration"])
            self._table.verticalHeader().setVisible(False)
            self._table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
            self._table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
            self._table.setFocusPolicy(QtCore.Qt.NoFocus)
            header = self._table.horizontalHeader()
            header.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
            header.setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(3, QtWidgets.QHeaderView.ResizeToContents)
            header.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)

            self._build_parameter_table()

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            preview_layout.addWidget(self._camera_preview_button)
            preview_layout.addWidget(self._label)

            table_panel = QtWidgets.QWidget()
            table_layout = QtWidgets.QVBoxLayout(table_panel)
            table_layout.setContentsMargins(0, 0, 0, 0)
            table_layout.addWidget(self._table)

            splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            splitter.addWidget(preview_panel)
            splitter.addWidget(table_panel)
            splitter.setStretchFactor(0, 1)
            splitter.setStretchFactor(1, 1)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(8, 8, 8, 8)
            layout.addWidget(splitter)

        def _build_parameter_table(self):
            for row, spec in enumerate(TRACKING_PARAMETER_SPECS):
                name = spec["name"]
                name_item = QtWidgets.QTableWidgetItem(name)
                name_item.setFlags(QtCore.Qt.ItemIsSelectable | QtCore.Qt.ItemIsEnabled)
                self._table.setItem(row, 0, name_item)

                value_label = QtWidgets.QLabel("0.0000")
                value_label.setAlignment(QtCore.Qt.AlignCenter)
                self._table.setCellWidget(row, 1, value_label)

                min_box = QtWidgets.QDoubleSpinBox()
                max_box = QtWidgets.QDoubleSpinBox()
                for box in (min_box, max_box):
                    box.setDecimals(4)
                    box.setRange(-999999.0, 999999.0)
                    box.setSingleStep(0.01)
                    box.setKeyboardTracking(False)

                button = QtWidgets.QPushButton("校准")
                button.setMinimumWidth(72)

                self._table.setCellWidget(row, 2, min_box)
                self._table.setCellWidget(row, 3, max_box)
                self._table.setCellWidget(row, 4, button)

                self._rows[name] = {
                    "value_label": value_label,
                    "min_box": min_box,
                    "max_box": max_box,
                    "button": button,
                }

                min_box.valueChanged.connect(partial(self._on_limits_changed, name))
                max_box.valueChanged.connect(partial(self._on_limits_changed, name))
                button.clicked.connect(partial(self._toggle_calibration, name))

            self.refresh_calibration_table()

        def _on_limits_changed(self, name, *_):
            row = self._rows[name]
            state = self.calibration_manager.set_limits(name, row["min_box"].value(), row["max_box"].value())
            self._sync_row(name, state)

        def _toggle_calibration(self, name, *_):
            state = self.calibration_manager.toggle(name)
            self._sync_row(name, state)
            if not state.calibrating:
                path = self._save_config()
                print(f"Calibration saved: {path}")

        def refresh_calibration_table(self):
            for name, state in self.calibration_manager.items():
                self._sync_row(name, state)

        def update_tracking_snapshot(self, snapshot):
            for name, state in snapshot.items():
                if name not in self._rows:
                    continue
                self._sync_row(name, state)

        def _sync_row(self, name, state):
            row = self._rows[name]
            current_value = self._state_value(state, "current_value", 0.0)
            min_value = self._state_value(state, "min_value", 0.0)
            max_value = self._state_value(state, "max_value", 0.0)
            calibrating = self._state_value(state, "calibrating", False)

            value_label = row["value_label"]
            value_label.setText(f"{float(current_value):.4f}")
            if not row["min_box"].hasFocus():
                blocker = QtCore.QSignalBlocker(row["min_box"])
                row["min_box"].setValue(float(min_value))
                del blocker
            if not row["max_box"].hasFocus():
                blocker = QtCore.QSignalBlocker(row["max_box"])
                row["max_box"].setValue(float(max_value))
                del blocker
            row["button"].setText("结束" if calibrating else "校准")

        @staticmethod
        def _state_value(state, key, default):
            if isinstance(state, dict):
                return state.get(key, default)
            return getattr(state, key, default)

        def update_frame(self, frame):
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            image = QtGui.QImage(
                rgb.data,
                width,
                height,
                3 * width,
                QtGui.QImage.Format_RGB888,
            ).copy()
            pixmap = QtGui.QPixmap.fromImage(image)
            pixmap = pixmap.scaled(
                self._label.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation,
            )
            self._label.set_preview_pixmap(pixmap)

        @property
        def closed(self):
            return self._closed

        @property
        def target_center(self):
            return self._target_center

        @property
        def camera_preview_enabled(self):
            return self._camera_preview_enabled

        def _toggle_camera_preview(self, *_):
            self._camera_preview_enabled = not self._camera_preview_enabled
            self._sync_camera_preview_button()
            path = self._save_config()
            print(f"Config saved: {path}")

        def _sync_camera_preview_button(self):
            self._camera_preview_button.setText(
                "关闭摄像头预览" if self._camera_preview_enabled else "打开摄像头预览"
            )

        def _save_config(self):
            return self.calibration_manager.save(
                camera_preview_enabled=self._camera_preview_enabled,
            )

        def _on_preview_center_selected(self, x, y):
            self._target_center = (float(x), float(y))
            print(f"Face priority center: x={x:.3f}, y={y:.3f}")

        def closeEvent(self, event):
            self._closed = True
            event.accept()
else:
    QtDebugWindow = None


def draw_debug(frame, face, position, vts_connected, fps=0.0, landmarks=None,
               expressions=None, angles=None, eye_gaze=None,
               show_landmark_indexes=False, timings=None, target_center=None,
               show_camera_preview=True):
    if not show_camera_preview:
        frame = np.zeros_like(frame)

    if face is not None:
        x1, y1, x2, y2 = map(int, face["xyxy"])
        cx, cy = map(int, face["center"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.circle(frame, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(
            frame,
            f"conf={face['confidence']:.2f}{' held' if face.get('held') else ''}",
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    if landmarks is not None:
        draw_mediapipe_landmarks(
            frame,
            landmarks,
            color=(80, 255, 160),
            radius=1,
            show_indexes=show_landmark_indexes,
        )

    target_x = 0.5 if target_center is None else clamp(float(target_center[0]), 0.0, 1.0)
    target_y = 0.5 if target_center is None else clamp(float(target_center[1]), 0.0, 1.0)
    target_px = (
        int(round(target_x * (frame.shape[1] - 1))),
        int(round(target_y * (frame.shape[0] - 1))),
    )
    cv2.drawMarker(
        frame,
        target_px,
        (255, 0, 0),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=2,
    )
    cv2.circle(frame, target_px, 5, (255, 0, 0), -1)
    cv2.putText(
        frame,
        f"VTS Pos X={position['x']:.2f} Y={position['y']:.2f} Z={position['z']:.2f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    cv2.putText(
        frame,
        f"VTS: {'connected' if vts_connected else 'retrying'}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0) if vts_connected else (0, 180, 255),
        2,
    )
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2,
    )
    if expressions:
        cv2.putText(
            frame,
            "Expr "
            f"EyeL={expressions.get('EyeOpenLeft', 0.0):.2f} "
            f"EyeR={expressions.get('EyeOpenRight', 0.0):.2f} "
            f"Mouth={expressions.get('MouthOpen', 0.0):.2f} "
            f"Smile={expressions.get('MouthSmile', 0.0):.2f}",
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 180, 255),
            2,
        )
    if angles:
        cv2.putText(
            frame,
            "Angle "
            f"X={angles.get('FaceAngleX', 0.0):.1f} "
            f"Y={angles.get('FaceAngleY', 0.0):.1f} "
            f"Z={angles.get('FaceAngleZ', 0.0):.1f}",
            (10, 150),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 180, 0),
            2,
        )
    if eye_gaze:
        cv2.putText(
            frame,
            "Eye "
            f"LX={eye_gaze.get('EyeLeftX', 0.0):.2f} "
            f"LY={eye_gaze.get('EyeLeftY', 0.0):.2f} "
            f"RX={eye_gaze.get('EyeRightX', 0.0):.2f} "
            f"RY={eye_gaze.get('EyeRightY', 0.0):.2f}",
            (10, 180),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (80, 255, 160),
            2,
        )
    if timings:
        cv2.putText(
            frame,
            "ms "
            f"read={timings.get('read', 0.0):.1f} "
            f"yolo={timings.get('yolo', 0.0):.1f} "
            f"mp={timings.get('mediapipe', 0.0):.1f}",
            (10, 210),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 255),
            2,
        )
        cv2.putText(
            frame,
            "ms "
            f"vts={timings.get('vts', 0.0):.1f} "
            f"draw={timings.get('debug', 0.0):.1f} "
            f"total={timings.get('total', 0.0):.1f}",
            (10, 235),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (180, 220, 255),
            2,
        )
    return frame


def run_bridge(args):
    global _ACTIVE_MEDIAPIPE_DETECTOR

    device = 0 if torch.cuda.is_available() else "cpu"
    model_path = Path(args.model)
    is_onnx_model = model_path.suffix.lower() == ".onnx"
    use_half = (not args.nohalf) and device != "cpu" and not is_onnx_model
    model = YOLO(args.model, task="detect")
    try:
        mediapipe_detector = MediaPipeFaceLandmarker(
            args.mediapipe_model,
            crop_scale=args.mediapipe_crop_scale,
        )
        _ACTIVE_MEDIAPIPE_DETECTOR = mediapipe_detector
    except Exception as exc:
        raise RuntimeError(f"MediaPipe Face Landmarker unavailable: {exc}") from exc

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index: {args.input}")

    vts_worker = VTubeStudioWorker(args.vtshost, args.vtsport, args.send_fps)
    vts_worker.start()
    calibration_manager = TrackingCalibrationManager(TRACKING_PARAMETER_SPECS)
    config_data = calibration_manager.load()
    if QtWidgets is None:
        raise RuntimeError("PyQt5 is not available.")
    qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    qt_window = QtDebugWindow(
        calibration_manager,
        camera_preview_enabled=bool(config_data.get("cameraPreviewEnabled", True)),
    )
    qt_window.show()

    print(f"Camera: {args.input}")
    print(f"Model: {args.model}")
    print(f"YOLO backend: {'ONNX Runtime' if is_onnx_model else 'PyTorch'}")
    print(f"Device: {device}")
    print(f"Half precision: {use_half}")
    print(f"Box filter: alpha={args.bbox_alpha}, window={args.bbox_window}, hold={args.hold_frames}")
    print(f"MediaPipe Face Landmarker: {args.mediapipe_model}")
    print(f"MediaPipe crop scale: {args.mediapipe_crop_scale}")
    print(f"Expression smoothing: alpha={args.expression_alpha}")
    print(f"Head pose smoothing: alpha={args.head_pose_alpha}")
    print(f"Eye gaze smoothing: alpha={args.eye_gaze_alpha}")
    print(f"VTube Studio: ws://{args.vtshost}:{args.vtsport}")
    print(f"VTube Studio retry interval: {VTS_RETRY_INTERVAL:.0f}s")
    print("Close the Qt debug window or press Ctrl+C to stop.")

    smoothed_position = None
    fps_deque = deque(maxlen=30)
    timing_deques = {
        "read": deque(maxlen=30),
        "yolo": deque(maxlen=30),
        "mediapipe": deque(maxlen=30),
        "vts": deque(maxlen=30),
        "debug": deque(maxlen=30),
        "total": deque(maxlen=30),
    }
    last_frame_time = None
    face_filter = FaceBoxFilter(
        alpha=args.bbox_alpha,
        window_size=args.bbox_window,
        hold_frames=args.hold_frames,
    )
    expression_filter = ParameterFilter(alpha=args.expression_alpha)
    head_pose_filter = ParameterFilter(alpha=args.head_pose_alpha)
    eye_gaze_filter = ParameterFilter(alpha=args.eye_gaze_alpha)
    last_target_center = None

    try:
        while True:
            qt_app.processEvents()
            if qt_window.closed:
                print("\nQt debug window closed. Stopping bridge.")
                break

            loop_start = time.perf_counter()
            frame_time = loop_start
            if last_frame_time is not None:
                elapsed = frame_time - last_frame_time
                if elapsed > 0:
                    fps_deque.append(1.0 / elapsed)
            last_frame_time = frame_time

            stage_start = time.perf_counter()
            ok, frame = cap.read()
            record_timing(timing_deques, "read", stage_start)
            if not ok:
                break

            frame_h, frame_w = frame.shape[:2]
            target_center = qt_window.target_center
            if target_center != last_target_center:
                face_filter.reset()
                smoothed_position = None
                expression_filter.reset()
                head_pose_filter.reset()
                eye_gaze_filter.reset()
                last_target_center = target_center

            stage_start = time.perf_counter()
            results = model(
                frame,
                imgsz=args.imgsz,
                conf=args.conf,
                device=device,
                half=use_half,
                verbose=False,
            )
            record_timing(timing_deques, "yolo", stage_start)
            raw_face = select_center_face(results[0].boxes, frame_w, frame_h, target_center)
            face, face_found = face_filter.update(raw_face, frame_w, frame_h)

            mediapipe_ready = False
            if not face_found:
                position = {"x": 0.0, "y": 0.0, "z": 0.0}
                smoothed_position = None
                expression_filter.reset()
                head_pose_filter.reset()
                eye_gaze_filter.reset()
            else:
                position = face_to_vts_position(face, frame_w, frame_h)
                smoothed_position = smooth_position(smoothed_position, position, args.smoothing)
                position = smoothed_position

            landmarks = None
            expressions = {}
            angles = {}
            eye_gaze = {}
            mediapipe_result = None
            if face is not None and not face.get("held"):
                stage_start = time.perf_counter()
                mediapipe_result = mediapipe_detector.detect(frame, face)
                record_timing(timing_deques, "mediapipe", stage_start)
            if mediapipe_result is not None:
                landmarks = mediapipe_result["landmarks"]
                mediapipe_ready = True
                expressions = expression_filter.update(
                    estimate_mediapipe_expressions(mediapipe_result["blendshapes"])
                )
                angles = head_pose_filter.update(estimate_mediapipe_angles(mediapipe_result["matrix"]))
                eye_gaze = eye_gaze_filter.update(estimate_mediapipe_eye_gaze(landmarks))
                expressions.update(eye_gaze)
            elif face is None:
                expression_filter.reset()
                head_pose_filter.reset()
                eye_gaze_filter.reset()

            tracking_values = build_tracking_values(position, expressions, angles, eye_gaze)
            calibration_manager.update_current_values(
                tracking_values,
                build_tracking_valid_mask(face_found, mediapipe_ready),
            )
            mapped_values = calibration_manager.map_values(tracking_values, vts_worker.vts.input_parameters)
            parameter_values = build_parameter_values(mapped_values)

            stage_start = time.perf_counter()
            vts_worker.update(face_found, parameter_values)
            record_timing(timing_deques, "vts", stage_start)

            qt_window.update_tracking_snapshot(calibration_manager.snapshot())
            fps = float(np.mean(fps_deque)) if fps_deque else 0.0
            debug_landmarks = landmarks if args.landmarks else None
            timing_summary = average_timings(timing_deques)
            stage_start = time.perf_counter()
            debug_frame = draw_debug(
                frame, face, position, vts_worker.connected, fps,
                debug_landmarks, expressions, angles, eye_gaze, args.show_landmark_indexes,
                timing_summary, target_center, qt_window.camera_preview_enabled
            )
            qt_window.update_frame(debug_frame)
            qt_app.processEvents()
            if qt_window.closed:
                print("\nQt debug window closed. Stopping bridge.")
                break
            record_timing(timing_deques, "debug", stage_start)
            record_timing(timing_deques, "total", loop_start)
    except KeyboardInterrupt:
        print("\nStopping bridge.")
    finally:
        try:
            vts_worker.stop()
        finally:
            cap.release()
            qt_window.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge YOLO face position to VTube Studio.")
    parser.add_argument("--input", type=int, default=0, help="Camera index.")
    parser.add_argument("--vtshost", type=str, default="127.0.0.1", help="VTube Studio API host.")
    parser.add_argument("--vtsport", type=int, default=8001, help="VTube Studio API port.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to YOLO .pt or .onnx model.")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--landmarks", action="store_true", help="Draw MediaPipe landmarks in debug preview.")
    parser.add_argument("--show-landmark-indexes", action="store_true",
                        help="Draw landmark indexes in debug preview.")
    parser.add_argument("--mediapipe-model", type=str, default=str(DEFAULT_MEDIAPIPE_MODEL),
                        help="Path to MediaPipe Face Landmarker .task model.")
    parser.add_argument("--mediapipe-crop-scale", type=float, default=1.45,
                        help="Scale YOLO face box before feeding the cropped ROI to MediaPipe.")
    parser.add_argument("--send-fps", type=float, default=30.0, help="Max parameter updates per second.")
    parser.add_argument("--smoothing", type=float, default=0.55, help="Position smoothing alpha, 0..1.")
    parser.add_argument("--bbox-alpha", type=float, default=0.55, help="Face box EMA alpha, 0..1.")
    parser.add_argument("--bbox-window", type=int, default=5, help="Face box median filter window size.")
    parser.add_argument("--hold-frames", type=int, default=3, help="Keep the last face box for brief detection drops.")
    parser.add_argument("--expression-alpha", type=float, default=0.45, help="Expression EMA alpha, 0..1.")
    parser.add_argument("--head-pose-alpha", type=float, default=0.35, help="Head pose EMA alpha, 0..1.")
    parser.add_argument("--eye-gaze-alpha", type=float, default=0.35, help="Eye gaze EMA alpha, 0..1.")
    parser.add_argument("--nohalf", action="store_true", help="Disable FP16 inference on CUDA.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.smoothing = clamp(args.smoothing, 0.0, 1.0)
    args.bbox_alpha = clamp(args.bbox_alpha, 0.0, 1.0)
    args.expression_alpha = clamp(args.expression_alpha, 0.0, 1.0)
    args.head_pose_alpha = clamp(args.head_pose_alpha, 0.0, 1.0)
    args.eye_gaze_alpha = clamp(args.eye_gaze_alpha, 0.0, 1.0)
    args.mediapipe_crop_scale = max(1.0, args.mediapipe_crop_scale)
    args.bbox_window = max(1, args.bbox_window)
    args.hold_frames = max(0, args.hold_frames)
    args.send_fps = max(1.0, args.send_fps)
    run_bridge(args)
    # MediaPipe 0.10.35 can block interpreter shutdown while its native Clearcut
    # uploader times out. run_bridge() has already released our resources here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
