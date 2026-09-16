"""速通线自检：验证 VmcOscSink 发出的消息符合 VMC Protocol。

不依赖摄像头、VSeeFace 或 Unity：直接实例化桥接里的真实发送类（不是复制实现），
在本机回环上收包、按规范解码，逐项核对地址、类型、表情名、取值方向与发送顺序。

用法（项目根目录）：
    python apps/vmc_link/selfcheck.py
"""
import socket
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "apps" / "vtube_bridge"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osc_reader import parse_packet  # noqa: E402
from vtube_studio_bridge.vtube_studio_bridge import (  # noqa: E402
    VMC_BLENDSHAPE_MAP,
    VmcOscSink,
)

SEND_FPS = 30.0
TRACKING_ZERO_ANGLES = {
    "EyeOpenLeft": 0.2, "EyeOpenRight": 1.0, "MouthOpen": 0.6, "MouthSmile": 0.3,
    "FaceAngleX": 0.0, "FaceAngleY": 0.0, "FaceAngleZ": 0.0,
}


def capture(port, tracking, frames=12, **sink_kwargs):
    """在本地监听该端口，跑一段真实发送，返回 (消息列表, 包数)。"""
    messages = []
    packets = [0]
    stop_event = threading.Event()

    listener = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    listener.bind(("127.0.0.1", port))
    listener.settimeout(0.5)

    def listen():
        while not stop_event.is_set():
            try:
                data, _ = listener.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            packets[0] += 1
            try:
                messages.extend(parse_packet(data))
            except Exception as exc:
                messages.append(("__parse_error__", [str(exc)]))

    thread = threading.Thread(target=listen, daemon=True)
    thread.start()

    sink = VmcOscSink("127.0.0.1", port, **sink_kwargs)
    sink.start()
    for _ in range(frames):
        sink.update(True, tracking)
        time.sleep(1.0 / SEND_FPS)
    time.sleep(0.3)
    sink.stop()
    stop_event.set()
    try:
        listener.close()
    except OSError:
        pass
    thread.join(timeout=1.0)
    return messages, packets[0]


def decode(sink, tracking):
    """直接调用消息构造函数（纯函数），返回 [(address, args), ...]。"""
    return [parse_packet(raw)[0] for raw in sink._messages(tracking, 0.0)]


def main():
    failures = []

    # --- 第一轮：默认（VRM0 名、角度为 0 故不发骨骼）-----------------------
    messages, packets = capture(39542, TRACKING_ZERO_ANGLES)
    print(f"第一轮：收到 {packets} 个包、{len(messages)} 条 OSC 消息")
    if not messages:
        print("[FAIL] 一条都没收到 —— 检查回环 UDP 是否被拦截。")
        return 1

    order = [address for address, _ in messages]
    for required in ("/VMC/Ext/OK", "/VMC/Ext/T", "/VMC/Ext/Blend/Val", "/VMC/Ext/Blend/Apply"):
        if required not in order:
            failures.append(f"缺少必需消息 {required}")

    blends = {}
    bad_types = False
    for address, values in messages:
        if address != "/VMC/Ext/Blend/Val":
            continue
        if not (isinstance(values[0], str) and isinstance(values[1], float)):
            bad_types = True
        else:
            blends[values[0]] = values[1]
    if bad_types:
        failures.append("Blend/Val 的类型不是 (string, float)")

    expected_names = {name for name, _invert in VMC_BLENDSHAPE_MAP.values()}
    if set(blends) != expected_names:
        failures.append(f"表情名不符：收到 {sorted(blends)}，期望 {sorted(expected_names)}")

    # 取值方向：睁眼 0.2 → 闭眼 Blink_L 应为 0.8；张嘴直传 0.6；微笑直传 0.3
    for name, want in (("Blink_L", 0.8), ("Blink_R", 0.0), ("A", 0.6), ("Joy", 0.3)):
        got = blends.get(name)
        if got is None or abs(got - want) > 1e-3:
            failures.append(f"{name} = {got}，期望 {want}（取反/直传方向不对？）")

    # Apply 必须在同一帧内排在该帧所有 Blend/Val 之后。
    # 注意：不能跨帧比较 —— 第 1 帧的 Apply 当然排在第 2 帧的 Val 之前。
    saw_val_since_apply = False
    apply_count = 0
    for address in order:
        if address == "/VMC/Ext/Blend/Val":
            saw_val_since_apply = True
        elif address == "/VMC/Ext/Blend/Apply":
            apply_count += 1
            if not saw_val_since_apply:
                failures.append("存在一帧 Apply 之前没有任何 Blend/Val（接收端会忽略该帧）")
            saw_val_since_apply = False
    if apply_count == 0:
        failures.append("没有任何 Blend/Apply —— 接收端不会应用表情值")

    # OK 消息按 V2.5 形式：(int)loaded (int)calibration state (int)mode
    ok_args = [values for address, values in messages if address == "/VMC/Ext/OK"]
    if ok_args and not (len(ok_args[0]) == 3 and ok_args[0][0] == 1):
        failures.append(f"/VMC/Ext/OK 参数不符规范：{ok_args[0]}")

    # 角度为 0 时不应发骨骼（省带宽的设计）
    if "/VMC/Ext/Bone/Pos" in order:
        failures.append("角度为 0 时不应发送骨骼消息")

    print(f"表情名({len(blends)}): " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(blends.items())))

    # --- 第二轮：VRM1 表情名 ---------------------------------------------
    messages2, packets2 = capture(39543, TRACKING_ZERO_ANGLES, vrm1_names=True)
    names2 = {values[0] for address, values in messages2 if address == "/VMC/Ext/Blend/Val"}
    expected2 = {"blinkLeft", "blinkRight", "aa", "happy"}
    if names2 != expected2:
        failures.append(f"VRM1 名不符：收到 {sorted(names2)}，期望 {sorted(expected2)}")
    else:
        print(f"第二轮：VRM1 名正确 —— {', '.join(sorted(names2))}")

    # --- 第三轮：非零角度 → 应发归一化四元数 ------------------------------
    sink3 = VmcOscSink("127.0.0.1", 39544, send_head=True)
    decoded = decode(sink3, {"FaceAngleX": 20.0, "FaceAngleY": -10.0, "FaceAngleZ": 5.0,
                             "MouthOpen": 0.5})
    bones = [values for address, values in decoded if address == "/VMC/Ext/Bone/Pos"]
    if not bones:
        failures.append("有非零头部角度时未发送 /VMC/Ext/Bone/Pos")
    else:
        args = bones[0]
        if args[0] != "Head":
            failures.append(f"骨骼名应为 Head，实际 {args[0]!r}")
        if len(args) != 8:
            failures.append(f"Bone/Pos 应有 8 个参数（name + p.xyz + q.xyzw），实际 {len(args)}")
        else:
            norm = sum(v * v for v in args[4:]) ** 0.5
            if abs(norm - 1.0) > 1e-4:
                failures.append(f"四元数未归一化，模长 {norm:.6f}")
            else:
                print(f"第三轮：头部骨骼四元数正常（模长 {norm:.6f}）")

    # --- 第四轮：关闭骨骼开关后不应再发 ----------------------------------
    sink4 = VmcOscSink("127.0.0.1", 39545, send_head=False)
    decoded4 = decode(sink4, {"FaceAngleX": 20.0, "FaceAngleY": -10.0, "FaceAngleZ": 5.0})
    if any(address == "/VMC/Ext/Bone/Pos" for address, _ in decoded4):
        failures.append("--vmc-no-head 未生效：仍然发送了骨骼消息")

    print()
    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1

    print("[PASS] VMC 编码符合规范：消息齐全、类型正确、取值方向正确、Apply 顺序正确、")
    print("       VRM0/VRM1 两套表情名可用、头部四元数归一化、骨骼开关生效。")
    print("下一步：先跑 apps/vmc_link/monitor.py 确认端口通，再让 VSeeFace 监听同一端口。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
