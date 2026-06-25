"""Resume v1 training from last checkpoint, then run v2 fine-tune"""
import os
import sys
import shutil

# ── cv2.imread Unicode patch for Windows ──────────────────────────────────
import cv2
import numpy as np

_ORIG_IMREAD = cv2.imread

def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Unicode-safe cv2.imread — uses imdecode fallback for non-ASCII paths."""
    result = _ORIG_IMREAD(path, flags)
    if result is not None:
        return result
    try:
        with open(path, "rb") as f:
            data = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(data, flags)
    except Exception:
        return None

cv2.imread = _imread_unicode
# ──────────────────────────────────────────────────────────────────────────

from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")


def prepare_data_yaml():
    """Regenerate data.yaml (ensures val path is correct)"""
    data_yaml_path = os.path.join(ANNO_DIR, "widerface.yaml")
    rel_path = os.path.relpath(os.path.join(DATA_DIR, "raw"), PROJECT_ROOT)

    yaml_content = f"""# WIDER Face dataset config for YOLOv8
path: {rel_path}
train: WIDER_train
val: WIDER_val

names:
  0: face

nc: 1
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return data_yaml_path


def main():
    data_yaml_path = prepare_data_yaml()
    print(f"Data YAML: {data_yaml_path}")

    # ── Step 1: Resume v1 training from last checkpoint ──────────────────
    last_pt = os.path.join(CHECKPOINT_DIR, "v1_baseline", "weights", "last.pt")
    if not os.path.exists(last_pt):
        print(f"ERROR: last.pt not found at {last_pt}")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Resuming v1 baseline training from epoch 63/100")
    print(f"Checkpoint: {last_pt}")
    print("=" * 60)

    model = YOLO(last_pt)
    results = model.train(
        data=data_yaml_path,
        epochs=100,                # complete 100 epochs
        batch=8,
        imgsz=640,
        optimizer="AdamW",
        lr0=0.001,
        lrf=1e-5,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        amp=True,
        workers=2,
        mosaic=1.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        project=CHECKPOINT_DIR,
        name="v1_baseline",
        exist_ok=True,
        pretrained=True,
        resume=True,               # KEY: resume from last.pt
        val=True,                  # NOW ENABLED: WIDER_val has been re-downloaded
        verbose=True,
    )

    # Copy v1 best model
    v1_best = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
    src = os.path.join(CHECKPOINT_DIR, "v1_baseline", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, v1_best)
        print(f"v1 best model saved to {v1_best}")

    # ── Step 2: v2 fine-tune (+30 epochs from v1 best) ──────────────────
    print("\n" + "=" * 60)
    print("Phase 3: Fine-tuning v2 closed-loop (+30 epochs)")
    print("=" * 60)

    v1_weights = v1_best if os.path.exists(v1_best) else src
    if not os.path.exists(v1_weights):
        print(f"ERROR: v1 weights not found, skipping v2")
        return

    model_v2 = YOLO(v1_weights)
    results_v2 = model_v2.train(
        data=data_yaml_path,
        epochs=30,
        batch=8,
        imgsz=640,
        optimizer="AdamW",
        lr0=1e-4,                  # lower LR for fine-tuning
        lrf=1e-5,
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=1,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        cos_lr=True,
        amp=True,
        workers=2,
        mosaic=0.5,                # reduced augmentation for fine-tuning
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        translate=0.1,
        scale=0.5,
        project=CHECKPOINT_DIR,
        name="v2_finetune",
        exist_ok=True,
        pretrained=True,
        resume=False,
        val=True,
        verbose=True,
    )

    v2_best = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
    src_v2 = os.path.join(CHECKPOINT_DIR, "v2_finetune", "weights", "best.pt")
    if os.path.exists(src_v2):
        shutil.copy(src_v2, v2_best)
        print(f"v2 best model saved to {v2_best}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  v1 weights: {v1_weights}")
    print(f"  v2 weights: {v2_best}")
    print("=" * 60)


if __name__ == "__main__":
    main()
