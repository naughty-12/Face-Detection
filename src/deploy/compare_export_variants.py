"""Decide the ONNX export precision by measuring DETECTION QUALITY, not tensor MAE.

Background: export precision used to be validated with an element-wise MAE between the
PyTorch and ONNX output tensors, at a 1e-4 tolerance. That criterion is unusable for
FP16 -- the tensor holds 8400 candidate boxes whose coordinates round to ~3 significant
digits, so a single coordinate drifting 320.12345 -> 320.1 already contributes ~2e-2,
two hundred times the tolerance, while the detections themselves are unchanged. Any
FP16 export fails it, which is presumably why the script was set to half=False, doubling
the file to 11.7 MiB and breaking the project's own <=10 MB target.

The right question is whether FP16 changes what the model DETECTS. So: run mAP on the
same 3,226-image validation set for PyTorch, ONNX FP32 and ONNX FP16, and compare.

Also reports end-to-end-ish throughput via model() (letterbox + forward + decode + NMS),
which is what the applications actually pay.
"""
import os
import shutil
import sys
import time

import cv2
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.getcwd())
from src.paths import BEST_MODEL_V2_ONNX, BEST_MODEL_V2_PT, CHECKPOINT_DIR, WIDER_YAML  # noqa: E402

SAMPLE_IMAGE = os.path.join("artifacts", "reports", "test_detect_0.jpg")


def evaluate(path, label):
    """mAP on the full val split plus a throughput sample."""
    model = YOLO(path)
    # workers=0 keeps validation single-process. The default worker pool spins up a
    # multiprocessing DataLoader, which needs named pipes -- forbidden in this
    # sandbox (WinError 5). Worker count does not affect mAP, only load speed.
    res = model.val(data=WIDER_YAML, split="val", batch=8, imgsz=640,
                    verbose=False, plots=False, workers=0)
    map50, map5095 = float(res.box.map50), float(res.box.map)

    frame = cv2.imdecode(np.fromfile(SAMPLE_IMAGE, dtype=np.uint8), cv2.IMREAD_COLOR)
    for _ in range(5):
        model(frame, imgsz=640, conf=0.25, verbose=False)
    times = []
    for _ in range(30):
        t0 = time.perf_counter()
        r = model(frame, imgsz=640, conf=0.25, verbose=False)
        times.append((time.perf_counter() - t0) * 1000)
    med = float(np.median(times))
    faces = 0 if r[0].boxes is None else len(r[0].boxes)
    size_mib = os.path.getsize(path) / (1024 * 1024)

    print(f"  {label:<20} mAP50={map50:.5f}  mAP50-95={map5095:.5f}  "
          f"P={float(res.box.mp):.4f}  R={float(res.box.mr):.4f}  "
          f"{med:6.1f} ms  {1000 / med:6.1f} FPS  {size_mib:5.2f} MiB  faces={faces}")
    return {"label": label, "map50": map50, "map5095": map5095,
            "p": float(res.box.mp), "r": float(res.box.mr),
            "ms": med, "fps": 1000 / med, "mib": size_mib}


def main():
    print("=" * 100)
    print("Export-variant comparison: detection quality (the deciding metric) + throughput")
    print("=" * 100)

    rows = []
    print("\n[1/3] PyTorch .pt (FP32, CUDA)")
    rows.append(evaluate(BEST_MODEL_V2_PT, "PyTorch .pt"))

    print("\n[2/3] ONNX FP32 (current build)")
    rows.append(evaluate(BEST_MODEL_V2_ONNX, "ONNX FP32"))

    tmp_pt = os.path.join(CHECKPOINT_DIR, "_cmp_model.pt")
    shutil.copy(BEST_MODEL_V2_PT, tmp_pt)
    tmp_onnx = tmp_pt.replace(".pt", ".onnx")
    try:
        print("\n[3/3] ONNX FP16 (exported from a copy)")
        YOLO(tmp_pt).export(format="onnx", imgsz=640, dynamic=False,
                            simplify=True, opset=12, half=True)
        rows.append(evaluate(tmp_onnx, "ONNX FP16"))
    finally:
        for f in (tmp_pt, tmp_onnx):
            if os.path.exists(f):
                os.remove(f)

    print("\n" + "=" * 100)
    print("Summary")
    print("=" * 100)
    print(f"  {'variant':<20}{'mAP50':>10}{'mAP50-95':>11}{'P':>9}{'R':>9}{'ms':>9}{'FPS':>8}{'MiB':>8}")
    for r in rows:
        print(f"  {r['label']:<20}{r['map50']:>10.5f}{r['map5095']:>11.5f}"
              f"{r['p']:>9.4f}{r['r']:>9.4f}{r['ms']:>9.1f}{r['fps']:>8.1f}{r['mib']:>8.2f}")

    base = next((r for r in rows if r["label"] == "PyTorch .pt"), None)
    if base:
        print("\n  Delta vs PyTorch .pt:")
        for r in rows[1:]:
            print(f"    {r['label']:<18} mAP50 {r['map50'] - base['map50']:+.5f}   "
                  f"mAP50-95 {r['map5095'] - base['map5095']:+.5f}   "
                  f"size {r['mib'] / base['mib']:.2f}x")


if __name__ == "__main__":
    main()
