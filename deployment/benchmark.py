"""Performance benchmark: FPS, latency, model size"""
import os
import sys
import time
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
IMGSZ = 640


def resolve_existing_path(label, candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    joined = "\n  ".join(candidates)
    raise FileNotFoundError(f"{label} not found. Checked:\n  {joined}")


def benchmark_pytorch(model_path, num_warmup=50, num_test=200):
    from ultralytics import YOLO
    model = YOLO(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.model.to(device)
    model.model.eval()
    dummy = torch.randn(1, 3, IMGSZ, IMGSZ, device=device)

    with torch.inference_mode():
        for _ in range(num_warmup):
            _ = model.model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    times = []
    with torch.inference_mode():
        for _ in range(num_test):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model.model(dummy)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)

    avg_latency = np.mean(times)
    fps = 1000.0 / avg_latency
    print(f"PyTorch:  avg latency = {avg_latency:.2f} ms, FPS = {fps:.1f}")
    return fps, avg_latency


def benchmark_onnx(onnx_path, num_warmup=50, num_test=200):
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    providers = []
    if "CUDAExecutionProvider" in available_providers:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(onnx_path, providers=providers)
    print(f"ONNX provider: {session.get_providers()[0]}")
    input_name = session.get_inputs()[0].name
    dummy = np.random.randn(1, 3, IMGSZ, IMGSZ).astype(np.float32)

    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy})

    times = []
    for _ in range(num_test):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy})
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)

    avg_latency = np.mean(times)
    fps = 1000.0 / avg_latency
    print(f"ONNX:     avg latency = {avg_latency:.2f} ms, FPS = {fps:.1f}")
    return fps, avg_latency


def main():
    print("=" * 60)
    print("Performance Benchmark")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    v2_pt = resolve_existing_path(
        "PyTorch model",
        [
            os.path.join(CHECKPOINT_DIR, "best_model_v2.pt"),
            os.path.join(DEPLOY_DIR, "best_model_v2.pt"),
        ],
    )
    v2_onnx = resolve_existing_path(
        "ONNX model",
        [
            os.path.join(CHECKPOINT_DIR, "best_model_v2.onnx"),
            os.path.join(DEPLOY_DIR, "best_model_v2.onnx"),
        ],
    )

    print("\n[1] PyTorch Inference Speed")
    pt_fps, pt_latency = benchmark_pytorch(v2_pt)

    print("\n[2] ONNX Runtime Inference Speed")
    onnx_fps, onnx_latency = benchmark_onnx(v2_onnx)

    print("\n[3] Model Size")
    pt_size = os.path.getsize(v2_pt) / (1024*1024)
    onnx_size = os.path.getsize(v2_onnx) / (1024*1024)
    print(f"PyTorch: {pt_size:.2f} MB | ONNX: {onnx_size:.2f} MB")

    print("\n" + "=" * 60)
    print("Benchmark Summary")
    print("=" * 60)
    print(f"  PyTorch FPS:        {pt_fps:.1f}")
    print(f"  ONNX Runtime FPS:   {onnx_fps:.1f}")
    print(f"  PyTorch Latency:    {pt_latency:.2f} ms")
    print(f"  ONNX Latency:       {onnx_latency:.2f} ms")
    print(f"  PyTorch Model Size: {pt_size:.2f} MB")
    print(f"  ONNX Model Size:    {onnx_size:.2f} MB")
    fps_ok = onnx_fps >= 30
    size_ok = onnx_size <= 10
    print(f"\n  FPS Check:  {'PASS' if fps_ok else 'FAIL'} (target >=30)")
    print(f"  Size Check: {'PASS' if size_ok else 'FAIL'} (target <=10MB)")


if __name__ == "__main__":
    main()
