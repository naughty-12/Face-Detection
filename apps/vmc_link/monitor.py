"""VMC 监听器：在本机接收桥接发出的 VMC/OSC 消息并翻译成人话。

用途：**在装 VSeeFace 之前**先确认发送端到底发了什么 —— 这是速通线第一个要过的关。
只依赖标准库。默认监听 39540（VMC 约定：Assistant → Performer）。

用法：
    python apps/vmc_link/monitor.py                 # 监听 39540
    python apps/vmc_link/monitor.py --port 39539    # 监听 Marionette 端口
"""
import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from osc_reader import parse_packet  # noqa: E402


BLEND_PREFIX = "/VMC/Ext/Blend/Val"
APPLY = "/VMC/Ext/Blend/Apply"


def main():
    parser = argparse.ArgumentParser(description="监听并翻译 VMC/OSC 消息。")
    parser.add_argument("--port", type=int, default=39540,
                        help="监听端口（39540 = Assistant→Performer，VMC 约定）。")
    parser.add_argument("--interval", type=float, default=1.0, help="状态行间隔（秒）。")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("0.0.0.0", args.port))
    except OSError as exc:
        print(f"[FAIL] 无法绑定 UDP {args.port}：{exc}（端口被占用？）")
        return 1
    sock.settimeout(0.5)

    print(f"监听 UDP {args.port}。启动桥接：")
    print(f"  python apps/vtube_bridge/main.py --input 0 --sink vmc --no-gui --vmc-port {args.port}")
    print("Ctrl+C 停止。\n")

    pending = {}
    frames = 0
    packets = 0
    window_start = time.perf_counter()
    window_frames = 0
    last_seen = None
    first_frame_printed = False
    other_addresses = set()
    warned_silent = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                if last_seen is not None and not warned_silent and time.perf_counter() - last_seen > 2.0:
                    print("[WARN] 超过 2 秒没有收到消息 —— 桥接还在运行吗？")
                    warned_silent = True
                continue

            packets += 1
            last_seen = time.perf_counter()
            warned_silent = False

            try:
                messages = parse_packet(data)
            except Exception as exc:
                print(f"[WARN] 无法解析来自 {addr} 的包：{exc}")
                continue

            for address, values in messages:
                if address == BLEND_PREFIX and len(values) >= 2:
                    pending[str(values[0])] = float(values[1])
                elif address == APPLY:
                    frames += 1
                    window_frames += 1
                    if not first_frame_printed:
                        first_frame_printed = True
                        print("首帧表情（VRM0 预设名 → 值）：")
                        for name, value in sorted(pending.items()):
                            print(f"    {name:<12} {value:.3f}")
                        print("    （Blink_* 为 1 表示闭眼；A 为张嘴；Joy 为微笑）\n")
                    pending = {}
                else:
                    other_addresses.add(address)

            now = time.perf_counter()
            if now - window_start >= args.interval:
                rate = window_frames / (now - window_start)
                print(f"{rate:5.1f} 帧/秒  {packets:>6} 包  {frames:>6} 帧  "
                      f"非表情消息: {', '.join(sorted(other_addresses)) or '无'}")
                window_start = now
                window_frames = 0
                packets = 0
                other_addresses = set()
    except KeyboardInterrupt:
        print(f"\n共收到 {frames} 帧表情数据。")
        if frames == 0:
            print("一帧都没收到：确认桥接用的是 --sink vmc 且端口一致。")
        else:
            print("发送端正常。接下来让接收端（VSeeFace 等）监听同一端口即可。")
    finally:
        sock.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
