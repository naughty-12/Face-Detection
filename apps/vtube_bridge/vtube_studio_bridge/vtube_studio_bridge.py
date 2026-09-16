"""Bridge YOLO face position to VTube Studio tracking parameters."""
import argparse
import json
import math
import os
import socket
import struct
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
# `thirdparty/` lives inside this bridge, so it is BRIDGE_ROOT (not PROJECT_ROOT)
# that must be importable -- previously this relied on the current working
# directory happening to be the bridge folder.
sys.path.insert(0, str(BRIDGE_ROOT))

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


# PyTorch weights are the default. On this machine onnxruntime exposes no CUDA provider,
# so an .onnx model would silently run on the CPU (~2.4-3.3x slower, measured) while the
# GPU sat idle. A .pt model takes the CUDA path and also enables FP16 inference below.
# Pass --model <...>.onnx to use ONNX Runtime explicitly (useful where PyTorch is absent).
DEFAULT_MODEL = PROJECT_ROOT / "artifacts" / "checkpoints" / "best_model_v2.pt"
DEFAULT_MEDIAPIPE_MODEL = BRIDGE_ROOT / "thirdparty" / "MediaPipe" / "models" / "face_landmarker.task"
PLUGIN_NAME = "Face Detection VTube Studio Bridge"
PLUGIN_DEVELOPER = "Face-Detection Project"
TOKEN_FILE = Path(os.getenv("APPDATA", Path.home())) / "FaceDetectionVTubeStudioBridge" / "vts_token.json"
VTS_RETRY_INTERVAL = 10.0
# VTS 对 AuthenticationTokenRequest 的回应方式是**弹窗等人点"允许"**，而其余请求都是毫秒级。
# 用统一的 1s 超时会让这个请求必然超时：实测（2026-09-16）桥接在发出请求 1 秒后断开，
# 用户 5 秒后才点下"允许"，token 被返回到一个已关闭的 socket —— 于是 token 永远缓存不下来，
# 每次运行都要再问一次。所以这一个请求单独给足等待时间。
AUTH_TOKEN_TIMEOUT = 120.0
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


# ---------------------------------------------------------------------------
# VMC Protocol (OSC/UDP) —— 速通线出口
#
# 本工具在 VMC 里扮演的角色是 **Assistant**：只把"部分骨骼 + 面部表情"发给
# Performer（如 VSeeFace），由 Performer 负责渲染皮套。协议原文：
#   Assistant - Send some bones, facial expressions, etc to Performer. (optional)
#               It works client to Performer, commonly send to 39540.
# 规范：https://protocol.vmc.info/english （MIT）
#
# 只发两类消息，其余可选消息一概不发：
#   /VMC/Ext/Blend/Val (string){name} (float){value}   ← 每个表情一条
#   /VMC/Ext/Blend/Apply                                ← 全部发完后统一应用
# 骨骼消息默认也发（头部朝向），可用 --vmc-no-head 关掉。
# ---------------------------------------------------------------------------

# VRM0 预设表情名。协议明确要求：使用 VRM1 的发送端也必须**按 VRM0 格式发送**，
# 以兼容既有 VMC 应用；所以默认用 VRM0 名（--vmc-vrm1-names 可切换）。
#   参考对照：Joy→happy、A→aa、Blink_L→blinkLeft
VMC_BLENDSHAPE_MAP = {
    "EyeOpenLeft": ("Blink_L", True),    # 我们的参数是"睁眼 0..1"，VMC 是"闭眼"，故取反
    "EyeOpenRight": ("Blink_R", True),
    "MouthOpen": ("A", False),
    "MouthSmile": ("Joy", False),
}
VMC_VRM1_NAMES = {"Blink_L": "blinkLeft", "Blink_R": "blinkRight", "A": "aa", "Joy": "happy"}


def osc_string(value):
    """OSC-string：UTF-8 + NUL 终止，并补齐到 4 字节边界。"""
    data = value.encode("utf-8") + b"\x00"
    return data + b"\x00" * ((4 - len(data) % 4) % 4)


def osc_message(address, *args):
    """单条 OSC 消息（不打包成 bundle）。

    规范允许 bundle，但"packets may be bundled"是可选行为，接收端本来就必须能处理
    未打包的消息；因此这里发独立消息，少一层编码风险。
    """
    typetags = ","
    payload = b""
    for value in args:
        if isinstance(value, bool):
            raise TypeError("OSC 的 bool 需由调用方显式转成 int")
        if isinstance(value, int):
            typetags += "i"
            payload += struct.pack(">i", value)
        elif isinstance(value, float):
            typetags += "f"
            payload += struct.pack(">f", value)
        elif isinstance(value, str):
            typetags += "s"
            payload += osc_string(value)
        else:
            raise TypeError(f"不支持的 OSC 参数类型: {type(value)!r}")
    return osc_string(address) + osc_string(typetags) + payload


def euler_to_vmc_quaternion(pitch_x, yaw_y, roll_z):
    """角度(度) → 四元数 (x, y, z, w)，按 Unity `Quaternion.Euler` 的 ZXY 顺序。

    VMC 的骨骼名取自 UnityEngine.HumanBodyBones，四元数也按 Unity 约定，
    因此这里刻意复刻 Unity 的欧拉角顺序（先 Z 再 X 再 Y），而不是常见的 XYZ。
    ⚠️ 轴向符号未经真机验证：需在目标应用里确认左右/俯仰是否需要取反。
    """
    hx = math.radians(pitch_x) * 0.5
    hy = math.radians(yaw_y) * 0.5
    hz = math.radians(roll_z) * 0.5
    qx = (math.sin(hx), 0.0, 0.0, math.cos(hx))
    qy = (0.0, math.sin(hy), 0.0, math.cos(hy))
    qz = (0.0, 0.0, math.sin(hz), math.cos(hz))

    def multiply(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    return multiply(multiply(qy, qx), qz)


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
    center_value: float = 0.0
    current_value: float = 0.0
    min_value: float = 0.0
    max_value: float = 0.0
    calibrating: bool = False

    def __post_init__(self):
        self.center_value = float(self.source_default)
        self.current_value = float(self.source_default)
        self.min_value = float(self.source_min)
        self.max_value = float(self.source_max)

    def reset_to_default(self):
        default_value = float(self.center_value)
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

    def set_center(self, value):
        self.center_value = float(value)

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

    def set_center(self, name, center_value):
        with self.lock:
            self.states[name].set_center(center_value)
            return self.states[name]

    def set_center_to_current(self, name):
        with self.lock:
            state = self.states[name]
            state.set_center(state.current_value)
            return state

    def set_all_centers_to_current(self):
        with self.lock:
            for state in self.states.values():
                state.set_center(state.current_value)
            return self.snapshot()

    def items(self):
        with self.lock:
            return list(self.states.items())

    def snapshot(self):
        with self.lock:
            return {
                name: {
                    "name": state.name,
                    "center_value": state.center_value,
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
                if "center" in values:
                    self.states[name].set_center(values["center"])
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
                        "center": state.center_value,
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
        source_center = clamp(float(getattr(state, "center_value", state.source_default)), source_min, source_max)

        raw_value = clamp(float(raw_value), source_min, source_max)
        if abs(source_max - source_min) < 1e-9:
            if param_info is not None and "defaultValue" in param_info:
                return clamp(float(param_info["defaultValue"]), float(param_info["min"]), float(param_info["max"]))
            return raw_value

        if param_info is None:
            return raw_value
        target_min = float(param_info["min"])
        target_max = float(param_info["max"])
        target_default = float(param_info.get("defaultValue", (target_min + target_max) / 2.0))

        # 两段式映射要求"源中心"与"目标默认值"都**严格落在各自区间内部**：
        #   ① 源中心落在端点（如 EyeOpenLeft：源 0..1、源默认 1.0 = 睁眼）时，整个源范围
        #      都被塞进下段；
        #   ② 而 VTS 对这类参数报的 defaultValue 等于 min（0 = 闭眼），下段的目标区间
        #      [target_min, target_default] 退化成**单点**。
        # 两者任一成立，任何输入都会被压成同一个值。实测（2026-09-16）：0.97 -> 0.000，
        # 皮套整场闭着眼，而感知层（相机图）其实正常给出 0.97。
        # 因此只要任一侧不可分段，就用全范围线性映射。
        source_splittable = source_min + 1e-9 < source_center < source_max - 1e-9
        target_splittable = target_min + 1e-9 < target_default < target_max - 1e-9
        if not (source_splittable and target_splittable):
            ratio = (raw_value - source_min) / (source_max - source_min)
            return clamp(
                target_min + ratio * (target_max - target_min),
                min(target_min, target_max),
                max(target_min, target_max),
            )

        if abs(source_center - source_min) < 1e-9 and raw_value <= source_center:
            return clamp(target_default, min(target_min, target_max), max(target_min, target_max))
        if abs(source_max - source_center) < 1e-9 and raw_value >= source_center:
            return clamp(target_default, min(target_min, target_max), max(target_min, target_max))

        if raw_value <= source_center:
            if abs(source_center - source_min) < 1e-9:
                mapped = target_default
            else:
                ratio = (raw_value - source_min) / (source_center - source_min)
                mapped = target_min + ratio * (target_default - target_min)
        else:
            if abs(source_max - source_center) < 1e-9:
                mapped = target_default
            else:
                ratio = (raw_value - source_center) / (source_max - source_center)
                mapped = target_default + ratio * (target_max - target_default)
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
        # 这一个请求的回应要等真人点弹窗，所以临时把 socket 超时放宽（见 AUTH_TOKEN_TIMEOUT 的注释）。
        if self.ws is not None:
            self.ws.settimeout(AUTH_TOKEN_TIMEOUT)
        try:
            data = self.request(
                "AuthenticationTokenRequest",
                {
                    "pluginName": PLUGIN_NAME,
                    "pluginDeveloper": PLUGIN_DEVELOPER,
                },
            )
        finally:
            # request() 失败时会 close() 并把 self.ws 置空，此时不必也不能再设超时。
            if self.ws is not None:
                self.ws.settimeout(self.timeout)
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
        # VTS 出口保留原有"线程 + last_send"实现（它是被放弃的路线，不改造）；
        # 只有 VMC / Unity 两个 UDP 出口改用内联发送 + next_send_at 信用累积。
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


class HeadlessDebugWindow:
    """`--no-gui` 时替代 QtDebugWindow 的空实现。

    捕捉循环只用到窗口的一小块接口：停止标志、可选的"点击选脸"目标点、
    以及每帧两个更新入口。把这几个放在这里，循环内部就不需要为无窗口模式
    再加分支 —— 也让桥接在**没装 PyQt5** 的机器上可以只做参数发送。

    注意：无窗口模式下没有"关窗口即停止"的退出方式，只能用 Ctrl+C。
    """

    def __init__(self):
        self.closed = False
        self.target_center = None
        self.camera_preview_enabled = False

    def show(self):
        pass

    def close(self):
        self.closed = True

    def update_tracking_snapshot(self, snapshot):
        pass

    def update_frame(self, frame):
        pass


class _HeadlessApp:
    """无窗口模式下的 QApplication 替身（循环里只调用 processEvents）。"""

    def processEvents(self):
        pass


class VmcOscSink:
    """把追踪参数按 VMC Protocol（OSC/UDP）发给 Performer（如 VSeeFace）。

    与 UnityUdpSink 同构：主循环只发布最新快照，daemon 线程持 socket 并限频发送，
    UDP 无连接语义 —— 目标应用没开也不会影响捕捉循环。

    每帧发送顺序（顺序有意义：Apply 必须在所有 Val 之后）：
        /VMC/Ext/T          (float) 相对时间
        /VMC/Ext/Blend/Val  (string)name (float)value   × N
        [可选] /VMC/Ext/Bone/Pos (string)"Head" (float)p.xyz (float)q.xyzw
        /VMC/Ext/Blend/Apply
    """

    def __init__(self, host, port, send_head=True, vrm1_names=False):
        self.address = (host, int(port))
        self.send_head = bool(send_head)
        self.vrm1_names = bool(vrm1_names)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packets_sent = 0
        self.last_sent_at = 0.0
        self.started_at = time.time()
        self.frames_sent = 0

    @property
    def connected(self):
        # 与 UnityUdpSink 同一口径：UDP 无握手，只报告"最近还在发"。
        if self.packets_sent == 0:
            return False
        return (time.perf_counter() - self.last_sent_at) < 2.0

    def start(self):
        # VMC 的"已加载/已校准"状态：Assistant 只发一次即可（非周期消息）。
        self._send(osc_message("/VMC/Ext/OK", 1, 3, 0))

    def update(self, face_found, tracking_values):
        """入参是**原始追踪值**（不做目标映射），映射到 VMC 表情名在本类内完成。

        **逐帧发送，不限频**。这里刻意不做速率上限，原因有实测依据：
        UDP 是 fire-and-forget、没有背压，可曾尝试过的"距上次发送够不够久"限频
        在管线帧间隔小于限频间隔时会**整帧整帧地丢** —— 实测 41 帧/秒的管线配
        30 帧/秒的上限，只剩 20 帧/秒发出（240 帧只发了 120 帧），皮套会明显卡顿。
        `--send-fps` 现在只对 VTS（WebSocket）出口生效。

        顺带：发送不放在独立线程里。UDP `sendto` 无握手、耗时几十微秒，
        内联即可；独立线程与主循环争 GIL 只会让节奏更不可控。
        """
        self.frames_sent += 1
        for message in self._messages(dict(tracking_values), time.time() - self.started_at):
            self._send(message)

    def stop(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def summary(self):
        return f"VMC: 发送 {self.frames_sent} 帧 / {self.packets_sent} 个 OSC 包"

    def _messages(self, tracking, elapsed):
        """构造一帧的全部 OSC 消息（纯函数，便于自检）。"""
        out = [osc_message("/VMC/Ext/T", float(elapsed))]
        for source, (name, invert) in VMC_BLENDSHAPE_MAP.items():
            value = clamp(float(tracking.get(source, 0.0)), 0.0, 1.0)
            if invert:
                value = 1.0 - value
            if self.vrm1_names:
                name = VMC_VRM1_NAMES.get(name, name)
            out.append(osc_message("/VMC/Ext/Blend/Val", name, float(value)))

        if self.send_head and any(
            abs(float(tracking.get(k, 0.0))) > 1e-6
            for k in ("FaceAngleX", "FaceAngleY", "FaceAngleZ")
        ):
            qx, qy, qz, qw = euler_to_vmc_quaternion(
                float(tracking.get("FaceAngleY", 0.0)),   # 图像俯仰
                float(tracking.get("FaceAngleX", 0.0)),   # 图像偏航
                float(tracking.get("FaceAngleZ", 0.0)),   # 图像翻滚
            )
            out.append(osc_message(
                "/VMC/Ext/Bone/Pos", "Head",
                0.0, 0.0, 0.0, qx, qy, qz, qw,
            ))

        out.append(osc_message("/VMC/Ext/Blend/Apply"))
        return out

    def _send(self, message):
        try:
            self.sock.sendto(message, self.address)
        except OSError as exc:
            # 出口不可用绝不致命：捕捉循环必须继续跑。
            print(f"[WARN] VMC 出口不可用 ({exc})；继续重试。")
            return False
        self.packets_sent += 1
        self.last_sent_at = time.perf_counter()
        return True


class UnityUdpSink:
    """UDP parameter sink for a Unity receiver (see apps/unity_link/).

    UDP on purpose: it is fire-and-forget, so a missing or restarting Unity
    process can never block or slow the capture loop -- the same reason the VTS
    sender lives on its own thread. The structure mirrors VTubeStudioWorker:
    the capture loop only publishes the latest snapshot, while a daemon thread
    owns the socket and enforces the send rate.

    Payload (UTF-8 JSON), field names match the C# side verbatim:

        {"seq": 42, "t": 1770000000.123, "face_found": true,
         "parameter_values": [{"id": "MouthOpen", "value": 0.31, "weight": 1.0}, ...]}
    """

    def __init__(self, host, port):
        self.address = (host, int(port))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.packets_sent = 0
        self.last_sent_at = 0.0
        self.seq = 0

    @property
    def connected(self):
        # UDP has no handshake, so do not fake a connection state: report
        # "recently sending" instead. Before the first packet this is False,
        # which is exactly what the debug overlay should show.
        if self.packets_sent == 0:
            return False
        return (time.perf_counter() - self.last_sent_at) < 2.0

    def start(self):
        # 内联发送，无需启动线程（见下方 update 的说明）。
        pass

    def update(self, face_found, parameter_values):
        """**逐帧发送，不限频**（理由见 VmcOscSink.update 的说明）。

        `--send-fps` 只对 VTS（WebSocket）出口生效：那条路是有状态连接，
        接收端有自己的处理节奏；而两个 UDP 出口是无背压的数据报，
        能发多少就发多少，发送率自然等于管线帧率。
        """
        self.seq += 1
        payload = {
            "seq": self.seq,
            "t": time.time(),
            "face_found": bool(face_found),
            "parameter_values": [dict(item) for item in parameter_values],
        }
        try:
            self.sock.sendto(json.dumps(payload).encode("utf-8"), self.address)
        except OSError as exc:
            # Never fatal: the capture loop must keep running without Unity.
            print(f"[WARN] Unity UDP sink unavailable ({exc}); retrying.")
            return
        self.packets_sent += 1
        self.last_sent_at = time.perf_counter()

    def stop(self):
        try:
            self.sock.close()
        except OSError:
            pass

    def summary(self):
        return f"Unity: 发送 {self.packets_sent} 包 / seq 到 {self.seq}"


CAMERA_BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}


def parse_fourcc(text):
    """`"MJPG"` → OpenCV 的 fourcc 整数；空 / None → None（不设置）。

    值得单独测的原因：**长度不对的编码会被驱动静默忽略**，现象就是"设了没用"。
    """
    if text is None:
        return None
    text = str(text).strip()
    if not text:
        return None
    if len(text) != 4:
        raise ValueError(f"FOURCC 必须是 4 个字符，收到 {text!r}")
    return int(cv2.VideoWriter_fourcc(*text))


def camera_backend_code(name):
    """`"msmf"` / `"dshow"` → OpenCV 常量；空 / None → None（用系统默认后端）。"""
    if name is None:
        return None
    key = str(name).strip().lower()
    if not key:
        return None
    if key not in CAMERA_BACKENDS:
        raise ValueError(f"未知的摄像头后端 {name!r}（可选：{', '.join(sorted(CAMERA_BACKENDS))}）")
    return CAMERA_BACKENDS[key]


def open_capture(source, backend=None, width=None, height=None, fps=None, fourcc=None):
    """按可选的后端 / 分辨率 / 帧率 / FOURCC 打开输入源。

    为什么需要（实测 2026-09-16，**摄像头路径**）：
        read=15.8  yolo=7.8  mediapipe=8.4  total=32.6 ms → 29.3 帧/秒
    **read 占近一半帧时间**，而管线自身只需约 17 ms —— 是在**等一个约 30 帧/秒的摄像头**。
    `--imgsz`（320/480/640 实测无差别）与 `--yolo-every`（只多出空闲）都动不了这一段，
    只有换采集方式才行。先用 `apps/vtube_bridge/probe_camera.py` 量出最省时间的组合。

    设置顺序：**先 FOURCC，再分辨率** —— 不少驱动在改分辨率时会把像素格式重置回默认。
    打不开时直接返回（不做任何 set），把报错留给调用方，保持原有错误信息不变。

    ⚠️ 摄像头参数**只对摄像头序号生效**：`--camera-backend msmf` 配视频文件会直接失败
    （MSMF 根本打不开文件），分辨率/帧率/FOURCC 对文件也没有意义，故文件源一律跳过这些设置。
    """
    is_camera = isinstance(source, int)
    code = camera_backend_code(backend) if is_camera else None
    cap = cv2.VideoCapture(source, code) if code is not None else cv2.VideoCapture(source)
    if not (is_camera and cap.isOpened()):
        return cap
    fourcc_code = parse_fourcc(fourcc)
    if fourcc_code is not None:
        cap.set(cv2.CAP_PROP_FOURCC, float(fourcc_code))
    if width and height:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    if fps:
        cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def should_run_detection(frame_index, every):
    """这一帧要不要跑 YOLO（`--yolo-every N`：每 N 帧跑一次，其余帧复用上一次的框）。

    实测（2026-09-16）两条输入路径的瓶颈完全不同：
        视频文件：read=1.5 ms  yolo=9.5 ms  mediapipe=11.0 ms  → 38.0 帧/秒
        摄像头  ：read=15.8 ms yolo=7.8 ms  mediapipe=8.4 ms   → 29.3 帧/秒
    摄像头那条的 **read 才是大头**（15.8 ms 是在等一个约 30 帧/秒的摄像头），管线自身只需 ~17 ms，
    因此跳帧省算力**无法突破摄像头的上限** —— 只能让管线多出空闲时间。
    `every <= 1` 视为"每帧都跑"，绝不退化成"从不跑"。
    """
    every = int(every)
    if every <= 1:
        return True
    return frame_index % every == 0


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


class BlinkStabiliser:
    """非对称平滑 + 带迟滞的"闭眼吸附" + **按持续时间区分眨眼与眯眼**。

    实测（2026-09-16）同一个人的反馈，把设计一步步逼出来：
      · 对称 EMA `--expression-alpha 0.65` → "**更抖**"；
      · 默认 0.45 → 快眨"**闭不上，有稍微一点点的睁开**"；
      · 加非对称平滑 → "**不抖但不实**"；
      · 加纯阈值吸附 → "**眯眼很容易闭上**"（吸附过头）。
    150 秒采样（从 VTS 回读、按当时校准反推）给出关键数字：**自然快眨的谷底落在 raw 0.35–0.49**，
    而**眯眼也在 0.40–0.50 一带** —— 两者**在数值上重叠，靠深度根本分不开**。

    **能分开它们的维度是持续时间**：真眨眼约 0.15 s（30 帧/秒 下 4–5 帧），眯眼是持续的。于是：
      · `target < deep_below`（0.38）→ **深闭合**：判为闭眼，**不限时长**（真闭眼可以保持很久）；
      · `snap_below`（0.5）以下的**浅而持久**值（超过 `squint_frames` 帧）→ 判为**眯眼**：
        放开吸附，按平滑值输出半闭；
      · 浅且**短暂**（几帧内就回来）→ 判为**眨眼**：吸合到 0。
    另外：**不缩放映射**（缩放会放大睁眼抖动，见第一条反馈），只做"替换"；`unsnap_above`（0.7）
    提供迟滞，避免阈值附近反复闪。
    """

    KEYS = ("EyeOpenLeft", "EyeOpenRight")

    # 默认阈值来自**实测**（2026-09-16，用 `--blink-snap-below 0 --blink-deep-below 0` 关掉吸附，
     # 从 VTS 回读值反推 raw，240 秒采样）：
    #   · 眨眼谷底 / 真闭眼：raw 0.300–0.365
    #   · 眯眼水平：        raw 0.410–0.550
    #   · 睁眼：            raw 0.63–0.97
    # 眨眼与眯眼之间有条约 0.04 的空隙，阈值就设在空隙里 —— 所以**不需要靠时长去赌**，
    # 时长规则（squint_frames）只作为"眯眼过程中偶发下探"的兜底。
    def __init__(self, close_alpha=0.8, open_alpha=0.45, snap_threshold=0.25,
                 snap_below=0.40, unsnap_above=0.7, deep_below=0.33, squint_frames=6):
        self.close_alpha = float(close_alpha)
        self.open_alpha = float(open_alpha)
        self.snap_threshold = float(snap_threshold)
        self.snap_below = float(snap_below)
        self.unsnap_above = float(unsnap_above)
        self.deep_below = float(deep_below)
        self.squint_frames = int(squint_frames)
        self.state = {}
        self.closed = {}
        self.low_frames = {}

    def update(self, values):
        out = dict(values)
        for name in self.KEYS:
            if name not in values:
                continue
            target = float(values[name])
            current = self.state.get(name)
            if current is None:
                self.state[name] = target
                self.low_frames[name] = 0
                continue

            # 统计"浅而低"持续了多少帧 —— 这是区分眨眼与眯眼的唯一依据。
            if self.deep_below <= target < self.snap_below:
                self.low_frames[name] = self.low_frames.get(name, 0) + 1
            else:
                self.low_frames[name] = 0

            if target < self.deep_below:
                # 深闭合：真闭眼 / 深眨眼，不限时长
                self.closed[name] = True
                self.low_frames[name] = 0
                self.state[name] = 0.0
                out[name] = 0.0
                continue

            if self.closed.get(name):
                if target > self.unsnap_above:
                    self.closed[name] = False
                    self.low_frames[name] = 0
                elif self.low_frames[name] >= self.squint_frames:
                    # 浅而持久 = 眯眼，不是眨眼：放开吸附，交给平滑值输出半闭
                    self.closed[name] = False
                else:
                    self.state[name] = 0.0
                    out[name] = 0.0
                    continue

            closing = (current - target) > self.snap_threshold
            alpha = self.close_alpha if closing else self.open_alpha
            current = current + alpha * (target - current)
            # 只在该值**正在往下走**时吸附：否则"从 0 重新睁开"会被自己又吸回去。
            if target < current and 0.0 < self.snap_below and current < self.snap_below:
                self.closed[name] = True
                self.low_frames[name] = 0
                current = 0.0
            self.state[name] = current
            out[name] = current
        return out

    def reset(self):
        self.state.clear()
        self.closed.clear()
        self.low_frames.clear()


def record_timing(timings, name, start_time):
    timings[name].append((time.perf_counter() - start_time) * 1000.0)

def average_timings(timings):
    return {
        name: float(np.mean(values)) if values else 0.0
        for name, values in timings.items()
    }


def format_stage_timings(timings):
    """把各段平均耗时（**毫秒**，由 record_timing 记录）渲染成一行 summary。

    没有它时，`--no-gui` 的跑到最后只打印总帧率，"慢在哪一段"只能靠猜 ——
    实测（2026-09-16）同一段视频从 40.2 帧/秒变成 23.0 帧/秒，而输出里没有任何分段信息，
    把 `--imgsz` 从 640 降到 320 也完全没有变化。
    """
    if not timings:
        return ""
    return "分段耗时（均值，ms）：" + "  ".join(
        f"{name}={value:.1f}" for name, value in timings.items()
    )


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
            self._all_center_button = QtWidgets.QPushButton("一键回中")
            self._all_center_button.clicked.connect(self._set_all_centers)

            self._label = PreviewLabel(alignment=QtCore.Qt.AlignCenter)
            self._label.setMinimumSize(640, 360)
            self._label.setStyleSheet("background-color: #111; color: #eee;")
            self._label.setText("Waiting for frames...")
            self._label.centerSelected.connect(self._on_preview_center_selected)

            self._table = QtWidgets.QTableWidget(len(TRACKING_PARAMETER_SPECS), 6)
            self._table.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Center", "Calibration"])
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
            header.setSectionResizeMode(5, QtWidgets.QHeaderView.ResizeToContents)

            self._build_parameter_table()

            preview_panel = QtWidgets.QWidget()
            preview_layout = QtWidgets.QVBoxLayout(preview_panel)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            preview_layout.addWidget(self._label)

            table_panel = QtWidgets.QWidget()
            table_layout = QtWidgets.QVBoxLayout(table_panel)
            table_layout.setContentsMargins(0, 0, 0, 0)
            table_layout.addWidget(self._table)
            control_layout = QtWidgets.QHBoxLayout()
            control_layout.addWidget(self._camera_preview_button)
            control_layout.addWidget(self._all_center_button)
            control_layout.addStretch(1)
            table_layout.addLayout(control_layout)

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
                center_button = QtWidgets.QPushButton("回中")
                center_button.setMinimumWidth(72)

                self._table.setCellWidget(row, 2, min_box)
                self._table.setCellWidget(row, 3, max_box)
                self._table.setCellWidget(row, 4, center_button)
                self._table.setCellWidget(row, 5, button)

                self._rows[name] = {
                    "value_label": value_label,
                    "min_box": min_box,
                    "max_box": max_box,
                    "center_button": center_button,
                    "button": button,
                }

                min_box.valueChanged.connect(partial(self._on_limits_changed, name))
                max_box.valueChanged.connect(partial(self._on_limits_changed, name))
                center_button.clicked.connect(partial(self._set_center, name))
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

        def _set_center(self, name, *_):
            state = self.calibration_manager.set_center_to_current(name)
            self._sync_row(name, state)
            path = self._save_config()
            print(f"Center saved: {path}")

        def _set_all_centers(self, *_):
            self.calibration_manager.set_all_centers_to_current()
            self.refresh_calibration_table()
            path = self._save_config()
            print(f"All centers saved: {path}")

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
            center_value = self._state_value(state, "center_value", 0.0)
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
            row["center_button"].setToolTip(f"Center: {float(center_value):.4f}")
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


SINK_ALIASES = {
    "both": ("vts", "unity"),          # 保留旧写法
    "all": ("vts", "unity", "vmc"),
}
SINK_NAMES = ("vmc", "unity", "vts")


def parse_sink_names(value):
    """把 --sink 的字符串解析成出口名列表（支持逗号分隔与别名，去重且保持顺序）。"""
    names = []
    for part in str(value).split(","):
        part = part.strip().lower()
        if not part:
            continue
        names.extend(SINK_ALIASES.get(part, (part,)))
    unknown = [name for name in names if name not in SINK_NAMES]
    if unknown:
        raise SystemExit(
            f"--sink 不认识这些出口: {', '.join(unknown)}；可选: "
            f"{', '.join(SINK_NAMES)}，或别名 {', '.join(SINK_ALIASES)}"
        )
    if not names:
        raise SystemExit("--sink 不能为空")
    return list(dict.fromkeys(names))


def parse_input_source(value):
    """`--input` 既接受摄像头序号（"0"），也接受视频文件路径。

    支持文件路径是为了**在没有摄像头时也能把全链路跑一遍**：视频放完 cap.read() 失败，
    主循环自然退出，因此非常适合做无人值守的验证与回归。
    """
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    path = Path(text)
    if not path.exists():
        raise SystemExit(f"--input 指向的文件不存在: {path}（也可以传摄像头序号，如 0）")
    return str(path)


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

    cap = open_capture(
        args.input,
        args.camera_backend,
        args.camera_width,
        args.camera_height,
        args.camera_fps,
        args.camera_fourcc,
    )
    if not cap.isOpened():
        raise RuntimeError(
            f"无法打开输入源 {args.input!r}：摄像头序号打不开设备，或视频文件不存在/编码不支持。"
        )

    # --- parameter sinks -------------------------------------------------
    # 捕捉管线共用，只有"出口"不同。三条线路对应三个出口：
    #   vmc   → 速通线：VMC/OSC 发给 VSeeFace 等 Performer（不需要 Unity）
    #   unity → 目标线：自定义 UDP JSON 发给本项目的 Unity 接收端
    #   vts   → 已放弃的 VTS 路线，保留仅作历史与架构对照
    # --sink 支持逗号分隔多个，便于同屏对照（见 parse_sink_names）。
    sinks = []
    vts_worker = None
    unity_sink = None
    vmc_sink = None
    if "vts" in args.sinks:
        vts_worker = VTubeStudioWorker(args.vtshost, args.vtsport, args.send_fps)
        vts_worker.start()
        sinks.append(vts_worker)
    if "unity" in args.sinks:
        unity_sink = UnityUdpSink(args.unity_host, args.unity_port)
        unity_sink.start()
        sinks.append(unity_sink)
    if "vmc" in args.sinks:
        vmc_sink = VmcOscSink(
            args.vmc_host, args.vmc_port,
            send_head=not args.vmc_no_head, vrm1_names=args.vmc_vrm1_names,
        )
        vmc_sink.start()
        sinks.append(vmc_sink)

    calibration_manager = TrackingCalibrationManager(TRACKING_PARAMETER_SPECS)
    config_data = calibration_manager.load()
    headless = bool(args.no_gui)
    if headless:
        qt_app = _HeadlessApp()
        qt_window = HeadlessDebugWindow()
    else:
        if QtWidgets is None:
            raise RuntimeError(
                "PyQt5 is not available. Pass --no-gui to run without the debug window "
                "(no window is created, so no PyQt5 is needed)."
            )
        qt_app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
        qt_window = QtDebugWindow(
            calibration_manager,
            camera_preview_enabled=bool(config_data.get("cameraPreviewEnabled", True)),
        )
    qt_window.show()

    print(f"Input: {args.input}" + ("" if isinstance(args.input, int) else " (video file: ends at EOF)"))
    print(f"Capture size: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
          f" @{cap.get(cv2.CAP_PROP_FPS):.0f} fps")
    if args.camera_backend or args.camera_width or args.camera_fps or args.camera_fourcc:
        print(f"Camera request: backend={args.camera_backend or 'default'} "
              f"{args.camera_width or '-'}x{args.camera_height or '-'} "
              f"@{args.camera_fps or '-'} {args.camera_fourcc or ''}")
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
    print(f"Sink(s): {', '.join(args.sinks)}")
    print(f"Debug window: {'disabled (--no-gui)' if headless else 'Qt'}")
    if vmc_sink is not None:
        print(f"VMC (OSC/UDP) target: {args.vmc_host}:{args.vmc_port} "
              f"(Assistant → Performer; head bone {'on' if not args.vmc_no_head else 'off'}, "
              f"{'VRM1' if args.vmc_vrm1_names else 'VRM0'} blendshape names)")
    if vts_worker is not None:
        print(f"VTube Studio: ws://{args.vtshost}:{args.vtsport}")
        print(f"VTube Studio retry interval: {VTS_RETRY_INTERVAL:.0f}s")
    if unity_sink is not None:
        print(f"Unity UDP target: {args.unity_host}:{args.unity_port} (fire-and-forget)")
        print("  Unity receiver lives in apps/unity_link/; mock receiver: "
              "python apps/unity_link/mock_receiver.py")
    print("Close the Qt debug window or press Ctrl+C to stop." if not headless
          else "Press Ctrl+C to stop (headless: no debug window, nothing to close).")

    smoothed_position = None
    fps_deque = deque(maxlen=30)
    frames_read = 0
    cached_raw_face = None
    loop_t0 = time.perf_counter()
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
    blink_stabiliser = BlinkStabiliser(
        snap_below=args.blink_snap_below,
        deep_below=args.blink_deep_below,
        squint_frames=args.blink_squint_frames,
    )
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
                blink_stabiliser.reset()
                head_pose_filter.reset()
                eye_gaze_filter.reset()
                last_target_center = target_center

            # --yolo-every：只在第 N 帧跑检测，其余帧**复用上一次的框**。
            # ⚠️ 必须复用框，不能传 None —— 传 None 会被 FaceBoxFilter 当成"丢脸"，
            # 跳帧一多就进入 held 状态，而 held 会跳过 MediaPipe（见下面的条件），
            # 于是表情每帧回落到默认值，**眼睛会一跳一跳地弹开**。
            if should_run_detection(frames_read, args.yolo_every):
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
                cached_raw_face = select_center_face(results[0].boxes, frame_w, frame_h, target_center)
            raw_face = cached_raw_face
            face, face_found = face_filter.update(raw_face, frame_w, frame_h)

            mediapipe_ready = False
            if not face_found:
                position = {"x": 0.0, "y": 0.0, "z": 0.0}
                smoothed_position = None
                expression_filter.reset()
                blink_stabiliser.reset()
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
                raw_expressions = estimate_mediapipe_expressions(mediapipe_result["blendshapes"])
                blink_values = {
                    name: raw_expressions.pop(name)
                    for name in BlinkStabiliser.KEYS
                    if name in raw_expressions
                }
                # 眼睛走**非对称平滑**（快闭、稳睁），其余表情走原来的 EMA。
                expressions = expression_filter.update(raw_expressions)
                expressions.update(blink_stabiliser.update(blink_values))
                angles = head_pose_filter.update(estimate_mediapipe_angles(mediapipe_result["matrix"]))
                eye_gaze = eye_gaze_filter.update(estimate_mediapipe_eye_gaze(landmarks))
                expressions.update(eye_gaze)
            elif face is None:
                expression_filter.reset()
                blink_stabiliser.reset()
                head_pose_filter.reset()
                eye_gaze_filter.reset()

            tracking_values = build_tracking_values(position, expressions, angles, eye_gaze)
            calibration_manager.update_current_values(
                tracking_values,
                build_tracking_valid_mask(face_found, mediapipe_ready),
            )
            # 三个出口各取所需，但都来自同一份 tracking_values：
            #   VTS   → 按 VTS 自己返回的参数范围映射后的值（校准在这里生效）
            #   Unity → 只按源范围裁剪、不做目标映射（input_parameters 传空即此语义），
            #           由 Unity 侧映射到模型参数
            #   VMC   → 同上（源范围裁剪后的原始值），由 VmcOscSink 映射到 VMC 表情名
            # 此前出口共用一份"已按 VTS 范围映射"的值，于是 Unity 收到的是被映射过一次的数，
            # 再映射一次就成了双重映射；现按出口分别计算。
            raw_values = calibration_manager.map_values(tracking_values, {})
            stage_start = time.perf_counter()
            if vts_worker is not None:
                vts_worker.update(
                    face_found,
                    build_parameter_values(
                        calibration_manager.map_values(tracking_values, vts_worker.vts.input_parameters)
                    ),
                )
            if unity_sink is not None:
                unity_sink.update(face_found, build_parameter_values(raw_values))
            if vmc_sink is not None:
                vmc_sink.update(face_found, raw_values)
            # 计时标签沿用 "vts"（调试叠加层按此键读取），实际已覆盖所有出口。
            record_timing(timing_deques, "vts", stage_start)
            frames_read += 1

            qt_window.update_tracking_snapshot(calibration_manager.snapshot())
            fps = float(np.mean(fps_deque)) if fps_deque else 0.0
            debug_landmarks = landmarks if args.landmarks else None
            timing_summary = average_timings(timing_deques)
            stage_start = time.perf_counter()
            # 无窗口模式跳过整帧叠加绘制：没有窗口可显示，画了纯属浪费主循环时间。
            # 因此 headless 下 "debug" 计时约为 0 —— 这是真实值，不是省掉了测量。
            if not headless:
                debug_frame = draw_debug(
                    frame, face, position, any(sink.connected for sink in sinks), fps,
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
        elapsed = time.perf_counter() - loop_t0 if frames_read else 0.0
        print(f"\n[summary] 处理帧数 {frames_read}" +
              (f"，耗时 {elapsed:.1f} s → 管线 {frames_read / elapsed:.1f} 帧/秒" if elapsed > 0 else ""))
        stage_line = format_stage_timings(average_timings(timing_deques))
        if stage_line:
            print(f"[summary] {stage_line}")
        for sink in sinks:
            if hasattr(sink, "summary"):
                print(f"[summary] {sink.summary()}")
        try:
            for sink in sinks:
                sink.stop()
        finally:
            cap.release()
            qt_window.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Bridge YOLO face position to VTube Studio.")
    parser.add_argument("--input", type=str, default="0",
                        help="摄像头序号（如 0），或视频文件路径 —— "
                             "后者用于**没有摄像头时也能跑全链路**（放完自动结束）。")
    parser.add_argument("--vtshost", type=str, default="127.0.0.1", help="VTube Studio API host.")
    parser.add_argument("--vtsport", type=int, default=8001, help="VTube Studio API port.")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL),
                        help="Path to a YOLO .pt model (CUDA, default) or .onnx (CPU-only here).")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold.")
    parser.add_argument("--camera-backend", type=str, default=None, choices=sorted(CAMERA_BACKENDS),
                        help="摄像头后端：msmf（Windows 默认）或 dshow。两者 read 耗时可能差很多 —— "
                             "先用 apps/vtube_bridge/probe_camera.py 量，再把结论写在这里。")
    parser.add_argument("--camera-width", type=int, default=None, help="请求的采集宽度（如 640）")
    parser.add_argument("--camera-height", type=int, default=None, help="请求的采集高度（如 480）")
    parser.add_argument("--camera-fps", type=int, default=None,
                        help="请求的采集帧率（如 60）。实测本机摄像头路径受限于采集帧率，"
                             "这是唯一能显著提高帧率的参数。")
    parser.add_argument("--camera-fourcc", type=str, default=None,
                        help="请求的像素格式，四个字符（如 MJPG）。**先设它再设分辨率**，"
                             "否则部分驱动会把它重置回默认。")
    parser.add_argument("--blink-snap-below", type=float, default=0.40,
                        help="浅闭合吸附阈值（0 = **关闭吸附**，用于测量真实眨眼幅度）。平滑值一路下降"
                             "到它以下就直接输出 0（= 全闭）。**实测**：眨眼谷底 raw 0.300–0.365、"
                             "眯眼水平 raw 0.410–0.550 → 阈值设在两者之间的空隙里，眯眼就不会被吸附。")
    parser.add_argument("--blink-deep-below", type=float, default=0.33,
                        help="深闭合阈值：低于它判为**真闭眼**，保持多久都算闭合（不会被当成眯眼放开）。"
                             "默认 0.33 来自实测：真闭眼能到 raw 0.300–0.33。")
    parser.add_argument("--blink-squint-frames", type=int, default=6,
                        help="兜底：浅而低持续超过这么多帧即判为**眯眼**并放开吸附（约 0.2 s @30 帧/秒）。"
                             "阈值已经能分开眨眼与眯眼，这一条只防「眯眼过程中偶发下探」。")
    parser.add_argument("--yolo-every", type=int, default=1,
                        help="每隔 N 帧才跑一次 YOLO 人脸检测，其余帧复用上一次的框（默认 1 = 每帧都跑）。"
                             "YOLO 约占单帧 8–10 ms。⚠️ 实测本机**摄像头路径的瓶颈是帧读取**"
                             "（read=15.8 ms，约 30 帧/秒的摄像头上限），跳帧只能让管线多出空闲，"
                             "**不能突破摄像头帧率**；视频路径上才有明显收益。")
    parser.add_argument("--landmarks", action="store_true", help="Draw MediaPipe landmarks in debug preview.")
    parser.add_argument("--show-landmark-indexes", action="store_true",
                        help="Draw landmark indexes in debug preview.")
    parser.add_argument("--mediapipe-model", type=str, default=str(DEFAULT_MEDIAPIPE_MODEL),
                        help="Path to MediaPipe Face Landmarker .task model.")
    parser.add_argument("--mediapipe-crop-scale", type=float, default=1.45,
                        help="Scale YOLO face box before feeding the cropped ROI to MediaPipe.")
    parser.add_argument("--send-fps", type=float, default=30.0,
                        help="VTS（WebSocket）出口的发送上限。"
                             "两个 UDP 出口（vmc / unity）**逐帧发送、不受此限** —— "
                             "限频曾把 41 帧/秒的管线砍到 20 帧/秒，见 CHANGELOG。")
    parser.add_argument("--sink", type=str, default="vts",
                        help="出口，支持逗号分隔多个：vmc / unity / vts；"
                             "别名 both=vts+unity，all=全部。vmc 不需要 Unity 或 VTS。")
    parser.add_argument("--no-gui", action="store_true",
                        help="Run without the Qt debug window (no PyQt5 needed). Stop with Ctrl+C.")
    parser.add_argument("--unity-host", type=str, default="127.0.0.1",
                        help="Unity UDP receiver host (used when --sink includes unity).")
    parser.add_argument("--unity-port", type=int, default=39540,
                        help="Unity UDP receiver port (must match FaceParamReceiver in Unity).")
    parser.add_argument("--vmc-host", type=str, default="127.0.0.1",
                        help="VMC (OSC/UDP) target host, e.g. the machine running VSeeFace.")
    parser.add_argument("--vmc-port", type=int, default=39540,
                        help="VMC port. 39540 = Assistant→Performer（规范约定）；39539 = Marionette。"
                             "注意与 --unity-port 默认值相同，两条线不要同时用同一端口。")
    parser.add_argument("--vmc-no-head", action="store_true",
                        help="不发送 /VMC/Ext/Bone/Pos 头部骨骼（只发表情 blendshape）。")
    parser.add_argument("--vmc-vrm1-names", action="store_true",
                        help="表情名用 VRM1 预设（aa/happy/blinkLeft）而非默认的 VRM0（A/Joy/Blink_L）。")
    parser.add_argument("--smoothing", type=float, default=0.55, help="Position smoothing alpha, 0..1.")
    parser.add_argument("--bbox-alpha", type=float, default=0.55, help="Face box EMA alpha, 0..1.")
    parser.add_argument("--bbox-window", type=int, default=5, help="Face box median filter window size.")
    parser.add_argument("--hold-frames", type=int, default=3, help="Keep the last face box for brief detection drops.")
    parser.add_argument("--expression-alpha", type=float, default=0.45, help="Expression EMA alpha, 0..1.")
    parser.add_argument("--head-pose-alpha", type=float, default=0.35, help="Head pose EMA alpha, 0..1.")
    parser.add_argument("--eye-gaze-alpha", type=float, default=0.35, help="Eye gaze EMA alpha, 0..1.")
    parser.add_argument("--nohalf", action="store_true",
                        help="Disable FP16 inference on CUDA (applies to .pt models only).")
    args = parser.parse_args()
    # 提前校验 FOURCC：否则非法值要等 YOLO/MediaPipe 加载完、真的去开摄像头时才报错（约 20 秒后）。
    parse_fourcc(args.camera_fourcc)
    return args


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
    args.sinks = parse_sink_names(args.sink)
    args.input = parse_input_source(args.input)
    run_bridge(args)
    # MediaPipe 0.10.35 can block interpreter shutdown while its native Clearcut
    # uploader times out. run_bridge() has already released our resources here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
