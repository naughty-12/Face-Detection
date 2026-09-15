"""Hard-sample analysis on the WIDER Face val set: size-stratified recall and precision.

Numbers first, pictures second. The previous version had a methodological bug: it
truncated the error lists to 20 items *inside* ``analyze_errors`` and then built the
size histogram from those 20 -- and because false negatives were sorted by face size
ascending before slicing, ``error_distribution.png`` was really a histogram of the 20
*smallest* missed faces, not an error distribution at all. Recall by face size cannot
be derived from a list that has been pre-filtered to small faces.

This version keeps every error, computes per-size-bin GT/prediction counters, prints
them as text, and only truncates for the human-facing TOP-N report and image grid.
"""
import argparse
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless: no window is ever opened
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.paths import ANNO_DIR, CHECKPOINT_DIR, RAW_DIR, REPORTS_DIR  # noqa: E402

# Bins are in equivalent face side length, sqrt(w * h), in pixels.
SIZE_BINS = [
    ("Tiny   (<32px)", 0, 32),
    ("Small  (32-96px)", 32, 96),
    ("Medium (96-256px)", 96, 256),
    ("Large  (>256px)", 256, float("inf")),
]


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def xywh_to_xyxy(x, y, w, h):
    return [x, y, x + w, y + h]


def _side(box_xyxy):
    """Equivalent face side length in pixels."""
    w = max(0.0, box_xyxy[2] - box_xyxy[0])
    h = max(0.0, box_xyxy[3] - box_xyxy[1])
    return float(np.sqrt(w * h))


def _bin_of(side):
    for i, (_, lo, hi) in enumerate(SIZE_BINS):
        if lo <= side < hi:
            return i
    return len(SIZE_BINS) - 1


def _imread_unicode(path):
    """cv2.imread returns None on non-ASCII paths on Windows; fall back to imdecode."""
    img = cv2.imread(path)
    if img is not None:
        return img
    try:
        return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    except Exception:
        return None


def analyze_errors(model_path, split="val", iou_threshold=0.5):
    """Match predictions against ground truth over the entire split.

    Returns ``(false_negatives, false_positives, stats)`` where the error lists are
    complete -- not truncated -- and ``stats`` holds the per-size-bin counters.
    """
    model = YOLO(model_path)

    anno_file = os.path.join(ANNO_DIR, "wider_face_split", f"wider_face_{split}_bbx_gt.txt")
    image_root = os.path.join(RAW_DIR, f"WIDER_{split}", "images")

    from src.data.loader import parse_wider_annotation  # local import: heavy deps

    samples = parse_wider_annotation(anno_file, image_root)

    false_negatives = []
    false_positives = []

    gt_total = [0] * len(SIZE_BINS)
    gt_matched = [0] * len(SIZE_BINS)
    pred_total = [0] * len(SIZE_BINS)
    pred_matched = [0] * len(SIZE_BINS)
    n_gt = n_pred = n_tp = 0

    print(f"Analyzing {len(samples)} images ...")
    for idx, (img_path, gt_boxes_xywh) in enumerate(samples):
        if idx % 500 == 0:
            print(f"  Progress: {idx}/{len(samples)}")

        results = model(img_path, verbose=False)
        pred_boxes, pred_scores = [], []
        if results[0].boxes is not None:
            pred_boxes = results[0].boxes.xyxy.cpu().numpy()
            pred_scores = results[0].boxes.conf.cpu().numpy()

        gt_boxes_xyxy = [xywh_to_xyxy(*box) for box in gt_boxes_xywh]
        n_gt += len(gt_boxes_xyxy)
        n_pred += len(pred_boxes)

        for gbox in gt_boxes_xyxy:
            gt_total[_bin_of(_side(gbox))] += 1
        for pbox in pred_boxes:
            pred_total[_bin_of(_side(pbox))] += 1

        matched_gt, matched_pred = set(), set()
        for pi, pbox in enumerate(pred_boxes):
            best_iou, best_gi = 0.0, -1
            for gi, gbox in enumerate(gt_boxes_xyxy):
                if gi in matched_gt:
                    continue
                iou = compute_iou(pbox, gbox)
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            if best_iou >= iou_threshold:
                matched_gt.add(best_gi)
                matched_pred.add(pi)
                n_tp += 1
                gt_matched[_bin_of(_side(gt_boxes_xyxy[best_gi]))] += 1
                pred_matched[_bin_of(_side(pbox))] += 1

        for gi, gbox in enumerate(gt_boxes_xyxy):
            if gi not in matched_gt:
                false_negatives.append({
                    "img_path": img_path,
                    "gt_box_xywh": gt_boxes_xywh[gi],
                    "gt_box_xyxy": gbox,
                    "side": _side(gbox),
                })

        for pi, pbox in enumerate(pred_boxes):
            if pi not in matched_pred:
                false_positives.append({
                    "img_path": img_path,
                    "pred_box": pbox.tolist(),
                    "score": float(pred_scores[pi]),
                    "side": _side(pbox),
                })

    stats = {
        "n_gt": n_gt,
        "n_pred": n_pred,
        "n_tp": n_tp,
        "n_fn": len(false_negatives),
        "n_fp": len(false_positives),
        "gt_total": gt_total,
        "gt_matched": gt_matched,
        "pred_total": pred_total,
        "pred_matched": pred_matched,
        "iou_threshold": iou_threshold,
        "model_path": model_path,
        "n_images": len(samples),
    }
    return false_negatives, false_positives, stats


def print_report(stats):
    n_gt, n_tp, n_fn = stats["n_gt"], stats["n_tp"], stats["n_fn"]
    n_pred, n_fp = stats["n_pred"], stats["n_fp"]

    print("\n" + "=" * 74)
    print("Size-stratified detection analysis")
    print("=" * 74)
    print(f"  model:   {os.path.basename(stats['model_path'])}")
    print(f"  images:  {stats['n_images']}")
    print(f"  IoU threshold for a match: {stats['iou_threshold']}")
    print(f"\n  ground-truth faces: {n_gt}")
    print(f"  matched (TP):       {n_tp}")
    print(f"  missed    (FN):     {n_fn}   -> recall    {n_tp / n_gt:.3f}" if n_gt else "")
    print(f"  predictions:        {n_pred}")
    print(f"  spurious  (FP):     {n_fp}   -> precision {n_tp / n_pred:.3f}" if n_pred else "")

    print("\n  " + "-" * 70)
    print(f"  {'size bin':<20} {'GT':>7} {'matched':>8} {'recall':>8} "
          f"{'pred':>7} {'matched':>8}")
    print("  " + "-" * 70)
    for i, (label, _, _) in enumerate(SIZE_BINS):
        g, gm = stats["gt_total"][i], stats["gt_matched"][i]
        p, pm = stats["pred_total"][i], stats["pred_matched"][i]
        recall = f"{gm / g:.3f}" if g else "  n/a"
        print(f"  {label:<20} {g:>7} {gm:>8} {recall:>8} {p:>7} {pm:>8}")
    print("  " + "-" * 70)

    tiny = stats["gt_matched"][0] / stats["gt_total"][0] if stats["gt_total"][0] else 0
    large = stats["gt_matched"][3] / stats["gt_total"][3] if stats["gt_total"][3] else 0
    if stats["gt_total"][0] and stats["gt_total"][3]:
        print(f"\n  Recall on Tiny faces is {tiny:.3f} vs {large:.3f} on Large faces "
              f"({large / tiny:.1f}x better on large)" if tiny > 0 else "")
    return stats


def save_reports(fn_list, fp_list, stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    summary = os.path.join(output_dir, "error_analysis_summary.txt")
    with open(summary, "w", encoding="utf-8") as f:
        f.write("Size-stratified detection analysis\n")
        f.write("=" * 74 + "\n")
        f.write(f"model:    {stats['model_path']}\n")
        f.write(f"images:   {stats['n_images']}\n")
        f.write(f"GT faces: {stats['n_gt']}   matched: {stats['n_tp']}   missed: {stats['n_fn']}\n")
        f.write(f"preds:    {stats['n_pred']}   spurious: {stats['n_fp']}\n")
        if stats["n_gt"]:
            f.write(f"recall:   {stats['n_tp'] / stats['n_gt']:.4f}\n")
        if stats["n_pred"]:
            f.write(f"precision:{stats['n_tp'] / stats['n_pred']:.4f}\n")
        f.write("\n")
        f.write(f"{'size bin':<20}{'GT':>8}{'matched':>9}{'recall':>9}{'pred':>8}{'matched':>9}\n")
        for i, (label, _, _) in enumerate(SIZE_BINS):
            g, gm = stats["gt_total"][i], stats["gt_matched"][i]
            p, pm = stats["pred_total"][i], stats["pred_matched"][i]
            r = f"{gm / g:.4f}" if g else "n/a"
            f.write(f"{label:<20}{g:>8}{gm:>9}{r:>9}{p:>8}{pm:>9}\n")

    top_fn = sorted(fn_list, key=lambda x: x["side"])[:20]          # smallest missed faces
    top_fp = sorted(fp_list, key=lambda x: x["score"], reverse=True)[:20]  # most confident mistakes

    fn_report = os.path.join(output_dir, "false_negatives_top20.txt")
    with open(fn_report, "w", encoding="utf-8") as f:
        f.write(f"20 SMALLEST missed faces (of {len(fn_list)} total false negatives)\n{'='*60}\n")
        for i, item in enumerate(top_fn):
            box = item["gt_box_xywh"]
            f.write(f"\n[{i+1}] {item['img_path']}\n")
            f.write(f"    GT box (xywh): {box[0]} {box[1]} {box[2]} {box[3]}   side={item['side']:.1f}px\n")

    fp_report = os.path.join(output_dir, "false_positives_top20.txt")
    with open(fp_report, "w", encoding="utf-8") as f:
        f.write(f"20 HIGHEST-CONFIDENCE wrong detections (of {len(fp_list)} total)\n{'='*60}\n")
        for i, item in enumerate(top_fp):
            box = item["pred_box"]
            f.write(f"\n[{i+1}] {item['img_path']} (score={item['score']:.4f})\n")
            f.write(f"    Pred box (xyxy): {box[0]:.1f} {box[1]:.1f} {box[2]:.1f} {box[3]:.1f}\n")

    print(f"\n  summary    -> {summary}")
    print(f"  FN report  -> {fn_report}")
    print(f"  FP report  -> {fp_report}")
    return top_fn, top_fp


def plot_error_distribution(stats, output_dir):
    """Full-set counts (not a 20-item sample) of GT vs matched vs missed, by size."""
    labels = [b[0].split("(")[0].strip() for b in SIZE_BINS]
    gt = stats["gt_total"]
    matched = stats["gt_matched"]
    missed = [g - m for g, m in zip(gt, matched)]
    pred = stats["pred_total"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    x = np.arange(len(labels))
    axes[0].bar(x - 0.2, gt, 0.4, label="ground truth", color="steelblue")
    axes[0].bar(x + 0.2, missed, 0.4, label="missed (FN)", color="crimson")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_title(f"Faces by size (all {stats['n_gt']} GT faces)")
    axes[0].set_ylabel("count")
    axes[0].legend()

    recall = [(m / g if g else 0.0) for g, m in zip(gt, matched)]
    bars = axes[1].bar(labels, recall, color="darkorange")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("Recall by face size")
    axes[1].set_ylabel("recall")
    for bar, r in zip(bars, recall):
        axes[1].annotate(f"{r:.3f}", xy=(bar.get_x() + bar.get_width() / 2, r),
                         xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9)

    plt.tight_layout()
    dist_path = os.path.join(output_dir, "error_distribution.png")
    plt.savefig(dist_path, dpi=150)
    plt.close()
    print(f"  distribution chart -> {dist_path}  (pred total: {pred})")


def plot_error_visualization(top_fn, top_fp, output_dir):
    def create_grid(items, title, output_path, color):
        n = len(items)
        cols = 4
        rows = max(1, (n + cols - 1) // cols)
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle(title, fontsize=14)
        for i in range(n):
            item = items[i]
            img = _imread_unicode(item["img_path"])
            if img is None:
                axes[i].axis("off")
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if "gt_box_xyxy" in item:
                box = item["gt_box_xyxy"]
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                axes[i].set_title(f"FN side={item['side']:.0f}px", fontsize=8)
            else:
                box = item["pred_box"]
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
                axes[i].set_title(f"FP s={item['score']:.2f}", fontsize=8)
            axes[i].imshow(img)
            axes[i].axis("off")
        for i in range(n, len(axes)):
            axes[i].axis("off")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  grid -> {output_path}")

    create_grid(top_fn, "20 smallest missed faces (false negatives)",
                os.path.join(output_dir, "fn_visualization.png"), color=(255, 0, 0))
    create_grid(top_fp, "20 highest-confidence wrong detections (false positives)",
                os.path.join(output_dir, "fp_visualization.png"), color=(0, 0, 255))


def main():
    parser = argparse.ArgumentParser(description="Size-stratified hard-sample analysis")
    parser.add_argument("--model", default=os.path.join(CHECKPOINT_DIR, "best_model_v2.pt"),
                        help="model to analyse (default: the deployed v2 weights)")
    parser.add_argument("--split", default="val")
    parser.add_argument("--iou", type=float, default=0.5)
    args = parser.parse_args()

    fn_list, fp_list, stats = analyze_errors(args.model, split=args.split, iou_threshold=args.iou)
    print_report(stats)
    top_fn, top_fp = save_reports(fn_list, fp_list, stats, REPORTS_DIR)
    plot_error_distribution(stats, REPORTS_DIR)
    plot_error_visualization(top_fn, top_fp, REPORTS_DIR)
    print("\nError analysis complete.")


if __name__ == "__main__":
    main()
