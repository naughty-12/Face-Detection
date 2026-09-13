"""Model diagnostics: forward-pass output shapes and TorchScript export readiness.

This is a diagnostic script, not a unit test -- it prints observations instead of
asserting anything. The pretrained weight used to be referenced as
"yolov8n-face.pt", a file that never existed in this project; it is now
"yolov8n.pt", matching `model_name` in configs/model.yaml.
"""
import torch
from ultralytics import YOLO


def test_model_output_shapes():
    model = YOLO("yolov8n.pt")
    model.model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    with torch.no_grad():
        output = model.model(dummy_input)

    if isinstance(output, (list, tuple)):
        print(f"Output is list/tuple with {len(output)} elements")
        for i, o in enumerate(output):
            if isinstance(o, torch.Tensor):
                print(f"  [{i}] shape: {o.shape}")
            elif isinstance(o, (list, tuple)):
                for j, oo in enumerate(o):
                    if isinstance(oo, torch.Tensor):
                        print(f"  [{i}][{j}] shape: {oo.shape}")
    elif isinstance(output, torch.Tensor):
        print(f"Output tensor shape: {output.shape}")
    print("Model forward pass: OK")
    params = sum(p.numel() for p in model.model.parameters()) / 1e6
    print(f"Model parameters: {params:.2f}M")


def test_model_export_readiness():
    model = YOLO("yolov8n.pt")
    model.model.eval()
    dummy_input = torch.randn(1, 3, 640, 640)
    try:
        traced = torch.jit.trace(model.model, dummy_input)
        print("TorchScript trace: OK (model is export-ready)")
    except Exception as e:
        print(f"TorchScript trace failed: {e}")
        print("Usually OK - ultralytics has its own export path.")


if __name__ == "__main__":
    print("Running model diagnostics ...")
    print("=" * 60)
    test_model_output_shapes()
    print("=" * 60)
    test_model_export_readiness()
    print("=" * 60)
    print("Diagnostics finished (no assertions -- inspect the output above).")
