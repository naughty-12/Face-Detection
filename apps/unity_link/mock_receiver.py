"""假 Unity：在本机接收桥接的 UDP 参数包并打印，用于在没有 Unity 时验证发送端。

只依赖标准库，因此在装 Unity 之前就能跑。
"""
import argparse
import json
import socket
import time


PARAM_PREVIEW = ["FaceAngleX", "FaceAngleY", "MouthOpen", "EyeOpenLeft", "EyeOpenRight"]


def main():
    parser = argparse.ArgumentParser(description="Mock Unity UDP receiver for the face bridge.")
    parser.add_argument("--port", type=int, default=39540, help="Must match --unity-port of the bridge.")
    parser.add_argument("--interval", type=float, default=1.0, help="Seconds between status lines.")
    args = parser.parse_args()

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", args.port))
    sock.settimeout(0.5)
    print(f"Listening on UDP {args.port}. Start the bridge with:")
    print(f"  python apps/vtube_bridge/main.py --input 0 --sink unity --unity-port {args.port}")
    print("Ctrl+C to stop.\n")

    received = 0
    window_start = time.perf_counter()
    last_packet_at = None
    last_seq = None
    out_of_order = 0
    last = None
    warned_silent = False

    try:
        while True:
            try:
                data, addr = sock.recvfrom(65535)
            except socket.timeout:
                if last_packet_at is not None and not warned_silent:
                    if time.perf_counter() - last_packet_at > 2.0:
                        print("[WARN] 超过 2 秒没有收到参数包 —— 桥接是否还在运行？")
                        warned_silent = True
                continue

            payload = json.loads(data.decode("utf-8"))
            received += 1
            seq = payload.get("seq")
            if last_seq is not None and seq is not None and seq < last_seq:
                out_of_order += 1
            last_seq = seq
            last_packet_at = time.perf_counter()
            warned_silent = False
            last = (payload, addr)

            now = time.perf_counter()
            if now - window_start >= args.interval:
                rate = received / (now - window_start)
                values = {item["id"]: item["value"] for item in payload.get("parameter_values", [])}
                preview = "  ".join(f"{name}={values.get(name, 0.0):+.3f}" for name in PARAM_PREVIEW)
                print(
                    f"seq={seq:<7} {rate:5.1f} pkt/s  face_found={payload.get('face_found')!s:<5} "
                    f"乱序={out_of_order}\n    {preview}"
                )
                received = 0
                window_start = now
    except KeyboardInterrupt:
        if last is not None:
            print("\n最后收到的包（原始 JSON）：")
            print(json.dumps(last[0], ensure_ascii=False, indent=2)[:800])
        else:
            print("\n没有收到任何参数包。")
        print("停止。")


if __name__ == "__main__":
    main()
