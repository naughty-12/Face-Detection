"""WIDER Face multi-dimensional mAP evaluation + P-R curves"""
import os
import json
import numpy as np
from ultralytics import YOLO

from src.paths import ANNO_DIR, CHECKPOINT_DIR, PROJECT_ROOT, REPORTS_DIR, VAL_LIST, WIDER_YAML


def evaluate_widerface(model_path):
    """Evaluate model on WIDER Face val set. Falls back to ultralytics built-in val."""
    model = YOLO(model_path)
    data_yaml = WIDER_YAML

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
        wf_results = wf_eval.evaluate(pred_dir, ANNO_DIR)
        metrics["easy_mAP"] = wf_results.get("Easy", 0.0)
        metrics["medium_mAP"] = wf_results.get("Medium", 0.0)
        metrics["hard_mAP"] = wf_results.get("Hard", 0.0)
    except ImportError:
        print("[UNAVAILABLE] widerface-evaluate is not installed, so the official")
        print("              Easy/Medium/Hard split metrics CANNOT be computed.")
        print("              They are reported as None on purpose -- filling them with")
        print("              mAP50 would look like a real per-subset result but is not.")
        print("              Install with:  pip install widerface-evaluate")
        metrics["easy_mAP"] = None
        metrics["medium_mAP"] = None
        metrics["hard_mAP"] = None

    return metrics


def generate_predictions(model_path, output_dir):
    """Generate WIDER Face format prediction files"""
    model = YOLO(model_path)
    val_list_path = VAL_LIST
    if not os.path.exists(val_list_path):
        print("[WARN] val_list.txt not found, skipping prediction generation")
        return
    with open(val_list_path, "r", encoding="utf-8") as f:
        img_paths = [l.strip() for l in f.readlines()]
    for img_path in img_paths:
        results = model(img_path, verbose=False)
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        pred_file = os.path.join(output_dir, f"{img_name}.txt")
        with open(pred_file, "w") as f:
            if results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                scores = results[0].boxes.conf.cpu().numpy()
                f.write(f"{img_name}\n")
                f.write(f"{len(boxes)}\n")
                for box, score in zip(boxes, scores):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    f.write(f"{x1:.1f} {y1:.1f} {w:.1f} {h:.1f} {score:.6f}\n")
            else:
                f.write(f"{img_name}\n0\n")


def plot_pr_curve(model_path, save_path=None):
    """Generate P-R curve using ultralytics built-in plotting"""
    model = YOLO(model_path)
    data_yaml = WIDER_YAML

    results = model.val(data=data_yaml, split="val", batch=8, imgsz=640, plots=True)

    if save_path:
        import shutil
        import glob
        run_dirs = sorted(glob.glob(os.path.join(PROJECT_ROOT, "runs", "detect", "val*")),
                          key=os.path.getmtime, reverse=True)
        if run_dirs:
            pr_src = os.path.join(run_dirs[0], "PR_curve.png")
            if os.path.exists(pr_src):
                shutil.copy(pr_src, save_path)
                print(f"P-R curve saved to {save_path}")


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

    print("\nGenerating P-R curves ...")
    plot_pr_curve(v1_path, save_path=os.path.join(REPORTS_DIR, "pr_curve_v1.png"))
    plot_pr_curve(v2_path, save_path=os.path.join(REPORTS_DIR, "pr_curve_v2.png"))

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"{'Metric':<20} {'v1 Baseline':>12} {'v2 Fine-tuned':>12} {'Improvement':>12}")
    print("-" * 60)
    def _fmt(v):
        return "n/a" if v is None else f"{v:.4f}"

    for key in ["easy_mAP", "medium_mAP", "hard_mAP", "mAP50"]:
        v1_val = v1_metrics.get(key)
        v2_val = v2_metrics.get(key)
        if v1_val is None or v2_val is None:
            delta = "n/a"
        else:
            delta = f"{v2_val - v1_val:+.4f}"
        print(f"{key:<20} {_fmt(v1_val):>12} {_fmt(v2_val):>12} {delta:>12}")

    comparison = {"v1": v1_metrics, "v2": v2_metrics}
    save_metrics(comparison, os.path.join(REPORTS_DIR, "comparison.json"))

    print("\nGenerating improvement chart ...")
    plot_improvement_chart(v1_metrics, v2_metrics, REPORTS_DIR)


def plot_improvement_chart(v1_metrics, v2_metrics, output_dir):
    """Generate v1→v2 improvement bar chart"""
    import matplotlib.pyplot as plt

    metrics_keys = ["easy_mAP", "medium_mAP", "hard_mAP"]
    v1_vals = [v1_metrics.get(k) for k in metrics_keys]
    v2_vals = [v2_metrics.get(k) for k in metrics_keys]

    if any(v is None for v in v1_vals + v2_vals):
        print("[SKIP] Improvement chart needs Easy/Medium/Hard values, which are unavailable.")
        print("       Run:  pip install widerface-evaluate")
        return

    x = np.arange(len(metrics_keys))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, v1_vals, width, label='v1 Baseline', color='steelblue')
    bars2 = ax.bar(x + width/2, v2_vals, width, label='v2 Fine-tuned', color='darkorange')

    ax.set_ylabel('mAP')
    ax.set_title('v1 vs v2 Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_keys)
    ax.legend()

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "improvement_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Improvement chart saved to {chart_path}")


if __name__ == "__main__":
    main()
