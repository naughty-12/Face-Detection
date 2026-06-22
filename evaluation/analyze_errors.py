"""Hard sample analysis: TOP20 false negatives/positives + error distribution charts"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cv2
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, PROJECT_ROOT)

REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "training", "checkpoints")
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
RAW_DIR = os.path.join(DATA_DIR, "raw")


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def analyze_errors(model_path, split="val", iou_threshold=0.5, top_k=20):
    model = YOLO(model_path)
    model.model.eval()

    anno_file = os.path.join(ANNO_DIR, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
    image_root = os.path.join(RAW_DIR, f"WIDER_{split}", "images")

    from dataset.dataloader import parse_wider_annotation
    samples = parse_wider_annotation(anno_file, image_root)

    false_negatives = []
    false_positives = []

    print(f"Analyzing {len(samples)} images ...")
    for idx, (img_path, gt_boxes_xywh) in enumerate(samples):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(samples)}")

        results = model(img_path, verbose=False)
        pred_boxes = []
        pred_scores = []
        if results[0].boxes is not None:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()

        gt_boxes_xyxy = [xywh_to_xyxy(*box) for box in gt_boxes_xywh]
        matched_gt = set()
        matched_pred = set()

        for pi, pbox in enumerate(pred_boxes):
            best_iou = 0
            best_gi = -1
            for gi, gbox in enumerate(gt_boxes_xyxy):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                matched_pred.add(pi)

        for gi, gbox in enumerate(gt_boxes_xyxy):
            if gi not in matched_gt:
                false_negatives.append({
                    "img_path": img_path,
                    "gt_box_xywh": gt_boxes_xywh[gi],
                    "gt_box_xyxy": gbox,
                })

        for pi, pbox in enumerate(pred_boxes):
            if pi not in matched_pred:
                false_positives.append({
                    "img_path": img_path,
                    "pred_box": pbox.tolist(),
                    "score": float(pred_scores[pi]),
                })

    false_positives.sort(key=lambda x: x["score"], reverse=True)
    false_negatives.sort(key=lambda x: (x["gt_box_xywh"][2]-x["gt_box_xywh"][0]) *
                                       (x["gt_box_xywh"][3]-x["gt_box_xywh"][1]))

    return false_negatives[:top_k], false_positives[:top_k]


def save_hard_samples_report(fn_list, fp_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    fn_report = os.path.join(output_dir, "false_negatives_top20.txt")
    with open(fn_report, "w") as f:
        f.write(f"TOP {len(fn_list)} False Negatives (Missed Detections)\n{'='*60}\n")
        for i, item in enumerate(fn_list):
            box = item["gt_box_xywh"]
            f.write(f"\n[{i+1}] {item['img_path']}\n")
            f.write(f"    GT box (xywh): {box[0]} {box[1]} {box[2]} {box[3]}\n")
    fp_report = os.path.join(output_dir, "false_positives_top20.txt")
    with open(fp_report, "w") as f:
        f.write(f"TOP {len(fp_list)} False Positives (Wrong Detections)\n{'='*60}\n")
        for i, item in enumerate(fp_list):
            f.write(f"\n[{i+1}] {item['img_path']} (score={item['score']:.4f})\n")
            box = item["pred_box"]
            f.write(f"    Pred box (xyxy): {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n")
    print(f"FN report: {fn_report}")
    print(f"FP report: {fp_report}")


def plot_error_visualization(fn_list, fp_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def create_grid(items, title, output_path, color, max_items=20):
        n = min(len(items), max_items)
        cols = 4
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4*rows))
        fig.suptitle(title, fontsize=14)
        if rows == 1 and cols == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        for i in range(n):
            item = items[i]
            img = cv2.imread(item["img_path"])
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if "gt_box_xyxy" in item:
                box = item["gt_box_xyxy"]
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                axes[i].set_title(f"FN: {os.path.basename(item['img_path'])[:20]}", fontsize=8)
            else:
                box = item["pred_box"]
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                axes[i].set_title(f"FP (s={item['score']:.2f})", fontsize=8)
            axes[i].imshow(img)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Visualization saved to {output_path}")

    create_grid(fn_list, "False Negatives (Missed Faces)",
                os.path.join(output_dir, "fn_visualization.png"), color=(255, 0, 0))
    create_grid(fp_list, "False Positives (Wrong Detections)",
                os.path.join(output_dir, "fp_visualization.png"), color=(0, 0, 255))


def plot_error_distribution(fn_list, fp_list, output_dir):
    def face_size(box_xyxy):
        w = box_xyxy[2] - box_xyxy[0]
        h = box_xyxy[3] - box_xyxy[1]
        return w * h

    fn_sizes = [face_size(item["gt_box_xyxy"]) for item in fn_list]
    fp_sizes = [face_size(item["pred_box"]) for item in fp_list]
    bins = [0, 32**2, 96**2, 256**2, 1e10]
    labels = ["Tiny\n(<32px)", "Small\n(32-96px)", "Medium\n(96-256px)", "Large\n(>256px)"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fn_hist, _ = np.histogram(fn_sizes, bins=bins)
    axes[0].bar(labels, fn_hist, color="red", alpha=0.7)
    axes[0].set_title("False Negatives by Face Size")
    axes[0].set_ylabel("Count")
    fp_hist, _ = np.histogram(fp_sizes, bins=bins)
    axes[1].bar(labels, fp_hist, color="blue", alpha=0.7)
    axes[1].set_title("False Positives by Face Size")
    axes[1].set_ylabel("Count")
    plt.tight_layout()
    dist_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"Error distribution saved to {dist_path}")


def main():
    v1_path = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
    fn_list, fp_list = analyze_errors(v1_path, top_k=20)
    print(f"\nFound {len(fn_list)} false negatives, {len(fp_list)} false positives (TOP shown)")
    save_hard_samples_report(fn_list, fp_list, REPORTS_DIR)
    plot_error_visualization(fn_list, fp_list, REPORTS_DIR)
    plot_error_distribution(fn_list, fp_list, REPORTS_DIR)
    print("\nError analysis complete.")


if __name__ == "__main__":
    main()
