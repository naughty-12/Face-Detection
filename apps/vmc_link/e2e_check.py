"""端到端冒烟测试：一条命令验证「感知层 → VMC 出口」整条链路。

不需要摄像头、不需要 VSeeFace、不需要 Unity：
  1. 本进程内起一个 UDP 监听（VMC 约定端口）；
  2. 以子进程方式跑真实的桥接 —— 输入是**仓库里的演示视频**，放完自动退出；
  3. 按 VMC 规范解码收到的包，检查消息齐全、取值合法、帧数合理。

用途：在你装 VSeeFace 之前，先用一条命令确认"我们发出去的东西是对的"。
这样如果皮套不动，问题就一定在接收端，不必来回怀疑。

用法（项目根目录）：
    python apps/vmc_link/e2e_check.py
    python apps/vmc_link/e2e_check.py --video output_result.mp4 --min-frames 50
"""
import argparse
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from osc_reader import parse_packet  # noqa: E402

DEFAULT_VIDEO = "7f62f96bca5cffdfe2e0167bf3de3169.mp4"
DEFAULT_PORT = 39560
BLEND_VAL = "/VMC/Ext/Blend/Val"
APPLY = "/VMC/Ext/Blend/Apply"
BONE = "/VMC/Ext/Bone/Pos"


class Collector(threading.Thread):
    def __init__(self, port):
        super().__init__(daemon=True)
        self.messages = []
        self.packets = 0
        self.frames = 0
        self.values = {}
        self.apply_without_val = 0
        self.bad_types = 0
        self.parse_errors = 0
        self.stop_event = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.settimeout(0.3)

    def run(self):
        pending_seen = False
        while not self.stop_event.is_set():
            try:
                data, _ = self.sock.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self.packets += 1
            try:
                parsed = parse_packet(data)
            except Exception:
                self.parse_errors += 1
                continue
            for address, values in parsed:
                self.messages.append(address)
                if address == BLEND_VAL:
                    if len(values) >= 2 and isinstance(values[0], str) and isinstance(values[1], float):
                        self.values[values[0]] = values[1]
                        pending_seen = True
                    else:
                        self.bad_types += 1
                elif address == APPLY:
                    self.frames += 1
                    if not pending_seen:
                        self.apply_without_val += 1
                    pending_seen = False

    def stop(self):
        self.stop_event.set()
        try:
            self.sock.close()
        except OSError:
            pass


def main():
    parser = argparse.ArgumentParser(description="VMC 出口端到端冒烟测试（无需摄像头/VSeeFace）。")
    parser.add_argument("--video", default=DEFAULT_VIDEO, help="输入视频（默认仓库自带演示视频）。")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="测试用端口（避开 39540/39541）。")
    parser.add_argument("--min-frames", type=int, default=100, help="至少应收到的表情帧数。")
    parser.add_argument("--timeout", type=float, default=300.0, help="桥接最长运行秒数。")
    args = parser.parse_args()

    video = PROJECT_ROOT / args.video
    if not video.exists():
        print(f"[FAIL] 找不到视频 {video}")
        return 1

    collector = Collector(args.port)
    collector.start()
    time.sleep(0.3)

    cmd = [
        sys.executable, str(PROJECT_ROOT / "apps" / "vtube_bridge" / "main.py"),
        "--input", str(video),
        "--sink", "vmc",
        "--vmc-port", str(args.port),
        "--no-gui",
    ]
    print("运行桥接（视频放完自动退出）…")
    print("  " + " ".join(cmd[1:]))
    started = time.perf_counter()
    try:
        # 不捕获子进程输出：沙箱下管道可能不可用，而这里也不需要它的 stdout。
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                timeout=args.timeout)
        code = result.returncode
    except subprocess.TimeoutExpired:
        print(f"[FAIL] 桥接超过 {args.timeout:.0f} 秒未结束")
        collector.stop()
        return 1
    elapsed = time.perf_counter() - started

    time.sleep(0.5)
    collector.stop()
    collector.join(timeout=1.0)

    failures = []
    if code != 0:
        failures.append(f"桥接退出码 {code}（应为 0）")
    if collector.packets == 0:
        failures.append("一个 UDP 包都没收到 —— 端口或回环被拦截？")
    if collector.frames < args.min_frames:
        failures.append(f"只收到 {collector.frames} 帧表情，少于要求的 {args.min_frames}")
    if collector.parse_errors:
        failures.append(f"{collector.parse_errors} 个包无法按 OSC 解析")
    if collector.bad_types:
        failures.append(f"{collector.bad_types} 条 Blend/Val 的类型不是 (string, float)")
    if collector.apply_without_val:
        failures.append(f"{collector.apply_without_val} 次 Apply 之前没有任何 Val")
    for name in ("A", "Blink_L", "Blink_R", "Joy"):
        if name not in collector.values:
            failures.append(f"没有收到表情 {name}")
    for name, value in collector.values.items():
        if not (0.0 <= value <= 1.0):
            failures.append(f"{name} = {value} 超出 0..1")
    if BONE not in collector.messages:
        failures.append("没有收到 /VMC/Ext/Bone/Pos（视频里人脸角度非零，本应发送）")

    print()
    print(f"桥接耗时 {elapsed:.1f} s，退出码 {code}")
    print(f"收到 {collector.packets} 个 UDP 包 / {collector.frames} 帧表情 "
          f"（{collector.frames / max(elapsed, 0.1):.1f} 帧/秒）")
    print(f"消息类型: {', '.join(sorted(set(collector.messages)))}")
    print("最后一批表情值: " + ", ".join(f"{k}={v:.3f}" for k, v in sorted(collector.values.items())))

    if failures:
        for item in failures:
            print(f"[FAIL] {item}")
        return 1

    print()
    print("[PASS] 端到端链路正常：桥接把视频里的人脸解成了 VMC 表情与头部姿态，")
    print("       消息齐全、类型正确、取值合法、Apply 顺序正确。")
    print("       → 发送端不用再怀疑了。若 VSeeFace 里皮套不动，问题在接收端配置。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
