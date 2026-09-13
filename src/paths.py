"""Central path resolution — the single source of truth for where things live.

Run scripts as modules from the project root, e.g.:

    python -m src.train.config
    python -m src.deploy.detect --input 0

Keeping every path here means a script's own file depth never matters, which is
what the previous per-file ``os.path.dirname(os.path.dirname(__file__))`` idiom
could not guarantee once scripts moved into sub-packages.
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ── Configuration ─────────────────────────────────────────────────────────
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model.yaml")

# ── Dataset ───────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_IMAGES_DIR = os.path.join(RAW_DIR, "WIDER_train", "images")
VAL_IMAGES_DIR = os.path.join(RAW_DIR, "WIDER_val", "images")
WIDER_YAML = os.path.join(ANNO_DIR, "widerface.yaml")
TRAIN_LIST = os.path.join(ANNO_DIR, "train_list.txt")
VAL_LIST = os.path.join(ANNO_DIR, "val_list.txt")

# ── Outputs ───────────────────────────────────────────────────────────────
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")

# ── Deployed weights ──────────────────────────────────────────────────────
BEST_MODEL_V1_PT = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
BEST_MODEL_V2_PT = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
BEST_MODEL_V2_ONNX = os.path.join(CHECKPOINT_DIR, "best_model_v2.onnx")
