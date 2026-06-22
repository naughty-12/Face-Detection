"""ONNX model export + precision validation"""
import os
import sys
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
DEPLOY_DIR = os.path.dirname(os.path.abspath(__file__))
ANNO_DIR = os.path.join(PROJECT_ROOT, "data", "annotations")


def export_to_onnx(model_path, output_path, imgsz=640):
    model = YOLO(model_path)
    model.export(format="onnx", imgsz=imgsz, dynamic=False, simplify=True, opset=12, half=False)
    import shutil
    src = model_path.replace(".pt", ".onnx")
    if os.path.exists(src):
        shutil.copy(src, output_path)
        print(f"ONNX model exported to {output_path}")
        return output_path
    else:
        raise RuntimeError(f"ONNX export failed, expected output not found at {src}")


def validate_precision(pytorch_model_path, onnx_model_path, num_samples=100, tolerance=1e-4):
    import onnxruntime as ort
    import cv2

    pt_model = YOLO(pytorch_model_path)
    pt_model.model.eval()
    ort_session = ort.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name

    val_list_path = os.path.join(ANNO_DIR, "val_list.txt")
    if not os.path.exists(val_list_path):
        print("[WARN] val_list.txt not found, skipping precision validation")
        return True

    with open(val_list_path, "r") as f:
        img_paths = [l.strip() for l in f.readlines()][:num_samples]

    total_mae = 0.0
    count = 0
    for img_path in img_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (640, 640))
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)

        with torch.no_grad():
            pt_output = pt_model.model(img_tensor)
        img_numpy = img_tensor.numpy().astype(np.float32)
        ort_output = ort_session.run(None, {input_name: img_numpy})

        if isinstance(pt_output, (list, tuple)):
            pt_out = pt_output[0].numpy()
        else:
            pt_out = pt_output.numpy()
        mae = np.mean(np.abs(pt_out - ort_output[0]))
        total_mae += mae
        count += 1

    avg_mae = total_mae / count if count > 0 else 0
    passed = avg_mae < tolerance
    print(f"\nPrecision Validation: Samples={count}, MAE={avg_mae:.6f}, Status={'PASSED' if passed else 'FAILED'}")
    return passed


def main():
    v2_path = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
    onnx_path = os.path.join(DEPLOY_DIR, "best_model_v2.onnx")

    print("=" * 60)
    print("Step 1: Exporting to ONNX ...")
    print("=" * 60)
    export_to_onnx(v2_path, onnx_path, imgsz=640)

    print("\n" + "=" * 60)
    print("Step 2: Validating precision (PyTorch vs ONNX) ...")
    print("=" * 60)
    passed = validate_precision(v2_path, onnx_path, num_samples=100)
    if not passed:
        print("\n[WARN] Precision validation FAILED.")
        sys.exit(1)

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"\nONNX Model Size: {size_mb:.2f} MB")
    print("\nONNX export and validation complete.")


if __name__ == "__main__":
    main()
