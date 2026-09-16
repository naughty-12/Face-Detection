"""测摄像头读取（read）耗时：找出最省时间的后端 / 分辨率 / 帧率 / FOURCC 组合。

为什么需要它
------------
实测（2026-09-16）摄像头路径的 summary 是：

    read=15.8  yolo=7.8  mediapipe=8.4  total=32.6 ms  →  29.3 帧/秒

**read 占了近一半帧时间**，而管线自身只需约 17 ms（≈59 帧/秒 的算力）—— 也就是说管线在
**等摄像头**，不是算得慢。因此 `--imgsz`（实测 320/480/640 无差别）与 `--yolo-every`
（跳帧只是多出空闲）都动不了这一段，只有换**采集方式**才行。

工具会做两件事：
1. 逐个打开配置，读回驱动**实际接受**的分辨率/帧率/FOURCC（很多摄像头会静默忽略请求）；
2. 丢掉前 10 帧预热后，连读 60 帧，报告 read 的均值 / p95 / 实际帧率。

用法（必须在**你自己的终端**里跑；先 Ctrl+C 停掉桥接，避免两个进程抢同一个设备）
------------------------------------------------------------------------
    python apps/vtube_bridge/probe_camera.py              # 设备 0
    python apps/vtube_bridge/probe_camera.py --device 1
    python apps/vtube_bridge/probe_camera.py --frames 120

输出：控制台（纯 ASCII，避免 Windows 控制台把中文弄乱）+ `artifacts/diag/camera_read_report.txt`

判读
----
* 某配置 read 明显更低、且实际帧率更高 → 就是它，回来把它写进桥接参数；
* 全部配置 read 都 ≈ 1000/摄像头帧率 → 说明瓶颈就是摄像头的输出帧率，只能靠换帧率档位解决；
* 若 `@60` 的实际帧率仍只有 30 → 该摄像头在这个分辨率下不支持 60 帧/秒，换更低分辨率或 MJPG 再试。
"""
import argparse
import os
import statistics
import sys
import time

import cv2

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(PROJECT_ROOT, "artifacts", "diag")

BACKENDS = {"msmf": cv2.CAP_MSMF, "dshow": cv2.CAP_DSHOW}

# (标签, 后端, 宽, 高, 帧率, FOURCC) —— None 表示不设置，用驱动默认
CONFIGS = [
    ("msmf  default (不设置任何参数)",   "msmf",  None, None, None, None),
    ("dshow default (不设置任何参数)",   "dshow", None, None, None, None),
    ("msmf  640x480",                    "msmf",  640,  480,  None, None),
    ("dshow 640x480",                    "dshow", 640,  480,  None, None),
    ("msmf  640x480 @30",                "msmf",  640,  480,  30,   None),
    ("msmf  640x480 @30 MJPG",           "msmf",  640,  480,  30,   "MJPG"),
    ("dshow 640x480 @30 MJPG",           "dshow", 640,  480,  30,   "MJPG"),
    ("msmf  640x480 @60",                "msmf",  640,  480,  60,   None),
    ("msmf  640x480 @60 MJPG",           "msmf",  640,  480,  60,   "MJPG"),
    ("dshow 640x480 @60 MJPG",           "dshow", 640,  480,  60,   "MJPG"),
    ("msmf  1280x720 @30 MJPG",          "msmf",  1280, 720,  30,   "MJPG"),
    ("msmf  1280x720 @60 MJPG",          "msmf",  1280, 720,  60,   "MJPG"),
]

WARMUP_FRAMES = 10
PER_CONFIG_BUDGET_S = 12.0          # 某配置太慢就提前收工，避免整体卡住


def fourcc_text(value):
    """把 cv2.CAP_PROP_FOURCC 读回的整数还原成四个字符（不可打印则给十六进制）。"""
    try:
        value = int(value)
    except (TypeError, ValueError):
        return "?"
    if value <= 0:
        return "?"
    chars = [chr((value >> (8 * i)) & 0xFF) for i in range(4)]
    if all(32 <= ord(c) < 127 for c in chars):
        return "".join(chars)
    return f"0x{value:08X}"


def probe(label, backend_name, width, height, fps, fourcc, device, frames, log):
    backend = BACKENDS[backend_name]
    cap = cv2.VideoCapture(device, backend)
    try:
        if not cap.isOpened():
            log(f"  {label:34s} -> 打不开（该后端/配置不可用）")
            return None
        if fourcc:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps:
            cap.set(cv2.CAP_PROP_FPS, float(fps))

        for _ in range(WARMUP_FRAMES):          # 预热：丢掉自动曝光/缓冲里那几帧
            cap.read()

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        actual_fourcc = fourcc_text(cap.get(cv2.CAP_PROP_FOURCC))

        read_ms = []
        started = time.perf_counter()
        while len(read_ms) < frames and (time.perf_counter() - started) < PER_CONFIG_BUDGET_S:
            t0 = time.perf_counter()
            ok, _ = cap.read()
            if not ok:
                break
            read_ms.append((time.perf_counter() - t0) * 1000.0)
        elapsed = time.perf_counter() - started
    finally:
        cap.release()

    if not read_ms:
        log(f"  {label:34s} -> 读不到帧")
        return None

    mean_ms = statistics.fmean(read_ms)
    p95_ms = sorted(read_ms)[int(len(read_ms) * 0.95) - 1] if len(read_ms) > 1 else read_ms[0]
    effective_fps = len(read_ms) / elapsed if elapsed > 0 else 0.0
    log(f"  {label:34s} -> 实际 {actual_w}x{actual_h} @{actual_fps:.0f} [{actual_fourcc}]  "
        f"read 均值 {mean_ms:6.2f} ms  p95 {p95_ms:6.2f} ms  实测 {effective_fps:5.1f} 帧/秒")
    return {"label": label, "w": actual_w, "h": actual_h, "fps": actual_fps,
            "fourcc": actual_fourcc, "read_mean_ms": mean_ms, "read_p95_ms": p95_ms,
            "effective_fps": effective_fps, "frames": len(read_ms)}


def main():
    parser = argparse.ArgumentParser(description="Measure webcam read() cost across capture settings.")
    parser.add_argument("--device", type=int, default=0, help="摄像头序号（默认 0）")
    parser.add_argument("--frames", type=int, default=60, help="每个配置连读多少帧（默认 60）")
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    lines = []

    def log(message):
        print(message, flush=True)
        lines.append(message)

    log(f"[probe] device={args.device}  frames_per_config={args.frames}  opencv={cv2.__version__}")
    log(f"[probe] 先确认没有别的进程占着摄像头（跑之前请 Ctrl+C 停掉桥接）")
    log("")

    results = []
    for label, backend_name, width, height, fps, fourcc in CONFIGS:
        result = probe(label, backend_name, width, height, fps, fourcc, args.device, args.frames, log)
        if result:
            results.append(result)
        time.sleep(0.6)                          # 给驱动一点时间真正释放设备

    log("")
    if results:
        best = min(results, key=lambda r: r["read_mean_ms"])
        log(f"[best] read 最省：{best['label']}")
        log(f"       {best['w']}x{best['h']} @{best['fps']:.0f} [{best['fourcc']}]  "
            f"read 均值 {best['read_mean_ms']:.2f} ms → 理论上限 {1000.0 / best['read_mean_ms']:.1f} 帧/秒")
        log("")
        log("[how-to-read]")
        log("  * read 均值 ≈ 1000/摄像头帧率 → 瓶颈是摄像头帧率本身（换档位才能改善）")
        log("  * 某档实际帧率低于请求值 → 驱动没接受该请求（对比「实际」列）")
        log("  * 把最优那行的参数告诉助手，写进桥接命令行参数即可")
    else:
        log("[!] 没有任何配置读到帧 —— 先停掉桥接，或换 --device")

    report_path = os.path.join(OUT_DIR, "camera_read_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    print(f"\n[probe] 报告已写入 {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
