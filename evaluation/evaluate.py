"""WIDER Face multi-dimensional mAP evaluation + P-R curves"""
import os
import sys
import json
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")


def evaluate_widerface(model_path):
    """Evaluate model on WIDER Face val set. Falls back to ultralytics built-in val."""
    model = YOLO(model_path)
    data_yaml = os.path.join(ANNO_DIR, "widerface.yaml")

    results = model.val(data=data_yaml, split="val", batch=8, imgsz=640, verbose=True)

    metrics = {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
    }

    try:
        from widerface import evaluate as wf_eval
        pred_dir = os.path.join(REPORTS_DIR, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        generate_predictions(model_path, pred_dir)
        wf_results = wf_eval.evaluate(pred_dir, os.path.join(DATA_DIR, "annotations"))
        metrics["easy_mAP"] = wf_results.get("Easy", 0.0)
        metrics["medium_mAP"] = wf_results.get("Medium", 0.0)
        metrics["hard_mAP"] = wf_results.get("Hard", 0.0)
    except ImportError:
        print("[WARN] widerface-evaluate not installed. Using ultralytics mAP as reference.")
        metrics["easy_mAP"] = metrics["mAP50"]
        metrics["medium_mAP"] = metrics["mAP50"]
        metrics["hard_mAP"] = metrics["mAP50"]

    return metrics


def generate_predictions(model_path, output_dir):
    """Generate WIDER Face format prediction files"""
    model = YOLO(model_path)
    val_list_path = os.path.join(ANNO_DIR, "val_list.txt")
    if not os.path.exists(val_list_path):
        print("[WARN] val_list.txt not found, skipping prediction generation")
        return
    with open(val_list_path, "r") as f:
        img_paths = [l.strip() for l in f.readlines()]
    for img_path in img_paths:
        results = model(img_path, verbose=False)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        pred_file = os.path.join(output_dir, f"{img_name}.txt")
        with open(pred_file, "w") as f:
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    f.write(f"{img_name}\n{score:.6f}\n{x1:.1f} {y1:.1f} {x2-x1:.1f} {y2-y1:.1f}\n")


def save_metrics(metrics, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {output_path}")


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)

    v1_path = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
    v2_path = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")

    print("=" * 60)
    print("Evaluating v1 baseline model ...")
    print("=" * 60)
    v1_metrics = evaluate_widerface(v1_path)
    save_metrics(v1_metrics, os.path.join(REPORTS_DIR, "metrics_v1.json"))

    print("\n" + "=" * 60)
    print("Evaluating v2 fine-tuned model ...")
    print("=" * 60)
    v2_metrics = evaluate_widerface(v2_path)
    save_metrics(v2_metrics, os.path.join(REPORTS_DIR, "metrics_v2.json"))

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"{'Metric':<20} {'v1 Baseline':>12} {'v2 Fine-tuned':>12} {'Improvement':>12}")
    print("-" * 60)
    for key in ["easy_mAP", "medium_mAP", "hard_mAP", "mAP50"]:
        v1_val = v1_metrics.get(key, 0.0)
        v2_val = v2_metrics.get(key, 0.0)
        print(f"{key:<20} {v1_val:>12.4f} {v2_val:>12.4f} {v2_val-v1_val:>+12.4f}")

    comparison = {"v1": v1_metrics, "v2": v2_metrics}
    save_metrics(comparison, os.path.join(REPORTS_DIR, "comparison.json"))


if __name__ == "__main__":
    main()
