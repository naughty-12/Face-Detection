"""自检：验证 UnityUdpSink 的 UDP 发送端能否被正确接收与解析。

不依赖摄像头、Unity 或 VTube Studio，只用本机回环：
  1. 起一个本地 UDP 监听；
  2. 实例化桥接里的 UnityUdpSink（真实代码路径，不是复制实现）；
  3. 按 30 FPS 推送若干帧合成参数；
  4. 校验收到的包数、字段完整性与序号单调性。

用法（项目根目录）：
    python apps/unity_link/selfcheck.py
"""
import json
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "vtube_bridge"))

from vtube_studio_bridge.vtube_studio_bridge import (  # noqa: E402
    TRACKING_PARAMETER_NAMES,
    UnityUdpSink,
    build_parameter_values,
)


SEND_FPS = 30.0
FRAMES = 40
PORT = 39541


def main():
    received = []
    stop_event = threading.Event()

    import socket

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", PORT))
    listener.settimeout(0.5)

    def listen():
        while not stop_event.is_set():
            try:
                data, _ = listener.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            received.append(json.loads(data.decode("utf-8")))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    sink = UnityUdpSink("127.0.0.1", PORT)
    sink.start()

    sent_frames = 0
    for i in range(FRAMES):
        raw = {
            name: (0.0 if name not in ("EyeOpenLeft", "EyeOpenRight") else 1.0)
            for name in TRACKING_PARAMETER_NAMES
        }
        raw["MouthOpen"] = min(1.0, i / 10.0)
        raw["FaceAngleX"] = -20.0 + i
        sink.update(True, build_parameter_values(raw))
        sent_frames += 1
        time.sleep(1.0 / SEND_FPS)

    deadline = time.perf_counter() + 1.0
    while time.perf_counter() < deadline and len(received) < sent_frames - 2:
        time.sleep(0.05)

    connected_before_stop = sink.connected
    sink.stop()
    stop_event.set()
    try:
        listener.close()
    except OSError:
        pass
    thread.join(timeout=1.0)

    print(f"推送帧数: {sent_frames}   收到包数: {len(received)}   丢弃(正常，UDP 允许丢包): {sent_frames - len(received)}")
    print(f"sink.connected (发送中): {connected_before_stop}")
    if not received:
        print("[FAIL] 一个包都没收到 —— 检查端口是否被占用/防火墙是否拦截回环 UDP。")
        return 1

    failures = []
    first = received[0]
    for key in ("seq", "t", "face_found", "parameter_values"):
        if key not in first:
            failures.append(f"缺字段 {key}")
    if len(first.get("parameter_values", [])) != len(TRACKING_PARAMETER_NAMES):
        failures.append("parameter_values 数量与 TRACKING_PARAMETER_NAMES 不一致")
    sample = {item["id"]: item["value"] for item in first.get("parameter_values", [])}
    if set(sample) != set(TRACKING_PARAMETER_NAMES):
        failures.append("parameter_values 的参数名集合不一致")
    if not all(isinstance(v, float) for v in sample.values()):
        failures.append("参数值不是 float（Unity 侧 JsonUtility 需要数值）")

    seqs = [p.get("seq") for p in received]
    if seqs != sorted(seqs):
        failures.append(f"seq 非单调递增: {seqs[:10]}...")
    if first.get("face_found") is not True:
        failures.append("face_found 未正确编码为 true")

    print(f"首个包: seq={first.get('seq')} face_found={first.get('face_found')} "
          f"MouthOpen={sample.get('MouthOpen')} FaceAngleX={sample.get('FaceAngleX')}")
    print(f"参数名({len(sample)}): {', '.join(sorted(sample))}")

    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1

    print("[PASS] 发送端可用：字段完整、数值为 float、序号单调、face_found 正确。")
    print("下一步：Unity 侧 FaceParamReceiver 绑定同一端口即可收到同样的包。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
