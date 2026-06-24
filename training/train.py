"""YOLOv8n-face training script — v1 baseline (100 epochs) + v2 fine-tune (+30 epochs)"""
import os
import sys
import shutil

# ── cv2.imread Unicode patch for Windows ──────────────────────────────────
# cv2.imread fails on paths containing non-ASCII characters (e.g. Chinese).
# Replace it with a wrapper that falls back to imdecode when imread fails.
import cv2
import numpy as np

_ORIG_IMREAD = cv2.imread

def _imread_unicode(path, flags=cv2.IMREAD_COLOR):
    """Unicode-safe cv2.imread — uses imdecode fallback for non-ASCII paths."""
    result = _ORIG_IMREAD(path, flags)
    if result is not None:
        return result
    # Fallback: read bytes and decode
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

from training.config import load_config, print_training_summary, WIDER_TRAIN_SIZE

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")


def prepare_data_yaml():
    """Generate ultralytics data.yaml with UTF-8 encoding and relative path"""
    data_yaml_path = os.path.join(ANNO_DIR, "widerface.yaml")

    # Use path relative to PROJECT_ROOT (CWD when running training),
    # NOT relative to the YAML file — ultralytics resolves from CWD
    rel_path = os.path.relpath(os.path.join(DATA_DIR, "raw"), PROJECT_ROOT)

    # TODO: change val back to WIDER_val when WIDER_val.zip is re-downloaded
    yaml_content = f"""# WIDER Face dataset config for YOLOv8
path: {rel_path}
train: WIDER_train
val: WIDER_train

names:
  0: face

nc: 1
"""
    with open(data_yaml_path, "w", encoding="utf-8") as f:
        f.write(yaml_content)
    return data_yaml_path


def train_v1(config, data_yaml_path):
    """Phase 2: Train v1 baseline — 100 epochs"""
    model = YOLO(config["model_name"])
    cfg = config["training"]

    results = model.train(
        data=data_yaml_path,
        epochs=cfg["epochs"],
        batch=cfg["batch_size"],
        imgsz=cfg["imgsz"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"],
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        warmup_momentum=cfg["warmup_momentum"],
        warmup_bias_lr=cfg["warmup_bias_lr"],
        cos_lr=cfg["cos_lr"],
        amp=cfg["amp"],
        workers=cfg["workers"],
        mosaic=cfg["mosaic"],
        mixup=cfg["mixup"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=cfg["fliplr"],
        project=CHECKPOINT_DIR,
        name="v1_baseline",
        exist_ok=True,
        pretrained=True,
        verbose=True,
        val=False,  # validation disabled until WIDER_val is re-downloaded
    )

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_path = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
    src = os.path.join(CHECKPOINT_DIR, "v1_baseline", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, best_path)
        print(f"v1 best model copied to {best_path}")

    return best_path


def train_v2(config, data_yaml_path, v1_weights_path):
    """Phase 3: Fine-tune v2 — +30 epochs from v1 checkpoint"""
    model = YOLO(v1_weights_path)
    cfg = config["training"]

    results = model.train(
        data=data_yaml_path,
        epochs=30,
        batch=cfg["batch_size"],
        imgsz=cfg["imgsz"],
        optimizer=cfg["optimizer"],
        lr0=cfg["lr0"] * 0.1,
        lrf=cfg["lrf"],
        momentum=cfg["momentum"],
        weight_decay=cfg["weight_decay"],
        warmup_epochs=cfg["warmup_epochs"],
        warmup_momentum=cfg["warmup_momentum"],
        warmup_bias_lr=cfg["warmup_bias_lr"],
        cos_lr=True,
        amp=cfg["amp"],
        workers=cfg["workers"],
        mosaic=0.5,
        mixup=cfg["mixup"],
        hsv_h=cfg["hsv_h"],
        hsv_s=cfg["hsv_s"],
        hsv_v=cfg["hsv_v"],
        degrees=cfg["degrees"],
        translate=cfg["translate"],
        scale=cfg["scale"],
        fliplr=0.5,
        project=CHECKPOINT_DIR,
        name="v2_finetune",
        exist_ok=True,
        pretrained=True,
        resume=False,
        verbose=True,
    )

    best_path = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
    src = os.path.join(CHECKPOINT_DIR, "v2_finetune", "weights", "best.pt")
    if os.path.exists(src):
        shutil.copy(src, best_path)
        print(f"v2 best model copied to {best_path}")

    return best_path


def main():
    config = load_config()
    print_training_summary(config, WIDER_TRAIN_SIZE)

    data_yaml_path = prepare_data_yaml()
    print(f"\nData YAML prepared at: {data_yaml_path}")

    print("\n" + "=" * 60)
    print("Phase 2: Training v1 baseline (100 epochs)")
    print("=" * 60)
    v1_path = train_v1(config, data_yaml_path)

    print("\n" + "=" * 60)
    print("Phase 3: Fine-tuning v2 closed-loop (+30 epochs)")
    print("=" * 60)
    v2_path = train_v2(config, data_yaml_path, v1_path)

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"  v1 weights: {v1_path}")
    print(f"  v2 weights: {v2_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
