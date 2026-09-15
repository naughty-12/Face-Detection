"""Precision / recall as a function of the confidence threshold.

Answers the open question from the size-stratified analysis: is the default
``conf=0.25`` costing recall, and what would a lower threshold actually buy?

The sweep does not re-run inference per threshold. ``model.val()`` already evaluates
the full precision-recall curve because detection AP is threshold-independent; it then
picks the max-F1 operating point to report P/R. The curve data (P, R, F1 sampled along
a confidence axis) is exposed on the returned metrics object, so every threshold can be
read off a single validation pass.

Usage:
    python -m src.eval.conf_sweep
    python -m src.eval.conf_sweep --model artifacts/checkpoints/best_model_v1.pt
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from src.paths import CHECKPOINT_DIR, REPORTS_DIR, WIDER_YAML  # noqa: E402

DEFAULT_CONFS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]


def _flatten(a):
    return np.asarray(a, dtype=float).ravel()


def extract_curves(box):
    """Return (conf_axis, precision, recall, f1) as 1-D arrays.

    ultralytics has exposed these under slightly different names across releases, so
    several spellings are tried rather than relying on one of them.
    """
    x = p = r = f1 = None

    for attr in ("x", "px", "conf"):
        if hasattr(box, attr):
            cand = getattr(box, attr)
            if cand is not None:
                x = _flatten(cand)
                break

    for name, target in (("p_curve", "p"), ("r_curve", "r"), ("f1_curve", "f1")):
        if hasattr(box, name):
            cand = getattr(box, name)
            if cand is not None:
                arr = _flatten(cand)
            else:
                arr = None
        else:
            arr = None
        if target == "p":
            p = arr
        elif target == "r":
            r = arr
        else:
            f1 = arr

    # Fallback: curves_results is [p_curve, r_curve, f1_curve, px, x]
    if p is None or r is None or f1 is None or x is None:
        for attr in ("curves_results", "curves"):
            holder = getattr(box, attr, None)
            if not holder:
                continue
            item = holder[0] if isinstance(holder[0], (list, tuple, np.ndarray)) else holder
            if isinstance(item, (list, tuple, np.ndarray)) and len(item) == 5:
                p, r, f1, _px, x = (_flatten(v) for v in item)
                break

    n = min(len(v) for v in (x, p, r, f1) if v is not None)
    return x[:n], p[:n], r[:n], f1[:n]


def at(conf, x, p, r, f1):
    i = int(np.argmin(np.abs(x - conf)))
    return x[i], p[i], r[i], f1[i]


def main():
    parser = argparse.ArgumentParser(description="Confidence-threshold sweep")
    parser.add_argument("--model", default=os.path.join(CHECKPOINT_DIR, "best_model_v2.pt"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--workers", type=int, default=0,
                        help="0 keeps validation single-process (needed in sandboxes that "
                             "forbid the multiprocessing pipes a worker pool opens)")
    args = parser.parse_args()

    print("=" * 78)
    print(f"Confidence sweep — {os.path.basename(args.model)} on WIDER Face {args.split}")
    print("=" * 78)

    model = YOLO(args.model)
    metrics = model.val(data=WIDER_YAML, split=args.split, batch=8, imgsz=640,
                        verbose=False, plots=True, workers=args.workers)

    box = metrics.box
    x, p, r, f1 = extract_curves(box)

    if x is None or p is None or r is None or len(x) == 0:
        print("\n[ERROR] could not extract curve data from the validation metrics.")
        print("        available attributes containing 'curve'/'x':",
              [a for a in dir(box) if "curve" in a.lower() or a in ("x", "px")])
        return 1

    print(f"\n  AP (confidence-independent): mAP50={float(box.map50):.5f}  "
          f"mAP50-95={float(box.map):.5f}")

    best_i = int(np.argmax(f1))
    print(f"  Max-F1 operating point:      conf={x[best_i]:.3f}  "
          f"P={p[best_i]:.4f}  R={r[best_i]:.4f}  F1={f1[best_i]:.4f}")
    print(f"  ultralytics reports this point as P={float(box.mp):.4f} R={float(box.mr):.4f}")

    print("\n  " + "-" * 74)
    print(f"  {'conf':>6} {'precision':>11} {'recall':>9} {'F1':>9}   {'vs conf=0.25':>18}")
    print("  " + "-" * 74)

    ref_p, ref_r = None, None
    rows = []
    for c in DEFAULT_CONFS:
        cx, cp, cr, cf = at(c, x, p, r, f1)
        if abs(c - 0.25) < 1e-9:
            ref_p, ref_r = cp, cr
        rows.append((cx, cp, cr, cf))

    for cx, cp, cr, cf in rows:
        if ref_r is None:
            delta = ""
        else:
            delta = f"R {cr - ref_r:+.4f}"
        mark = "  <-- default" if abs(cx - 0.25) < 0.03 else ""
        print(f"  {cx:>6.2f} {cp:>11.4f} {cr:>9.4f} {cf:>9.4f}   {delta:>18}{mark}")
    print("  " + "-" * 74)

    if ref_r is not None:
        lo = at(0.10, x, p, r, f1)
        hi = at(0.50, x, p, r, f1)
        print(f"\n  Lowering conf 0.25 -> 0.10 changes recall {ref_r:.4f} -> {lo[2]:.4f} "
              f"({lo[2] - ref_r:+.4f}) and precision {ref_p:.4f} -> {lo[1]:.4f} "
              f"({lo[1] - ref_p:+.4f}).")
        print(f"  Raising  conf 0.25 -> 0.50 changes recall {ref_r:.4f} -> {hi[2]:.4f} "
              f"({hi[2] - ref_r:+.4f}) and precision {ref_p:.4f} -> {hi[1]:.4f} "
              f"({hi[1] - ref_p:+.4f}).")
        print("  Interpretation: AP50 is unaffected by this choice, so the threshold only")
        print("  trades recall against precision at a fixed operating point.")

    os.makedirs(REPORTS_DIR, exist_ok=True)

    report = os.path.join(REPORTS_DIR, "conf_sweep.txt")
    with open(report, "w", encoding="utf-8") as fh:
        fh.write(f"Confidence sweep — {args.model} on {args.split}\n")
        fh.write("=" * 70 + "\n")
        fh.write(f"mAP50={float(box.map50):.5f}  mAP50-95={float(box.map):.5f}\n")
        fh.write(f"max-F1 point: conf={x[best_i]:.3f} P={p[best_i]:.4f} "
                 f"R={r[best_i]:.4f} F1={f1[best_i]:.4f}\n\n")
        fh.write(f"{'conf':>6}{'precision':>12}{'recall':>10}{'F1':>10}\n")
        for cx, cp, cr, cf in rows:
            fh.write(f"{cx:>6.2f}{cp:>12.4f}{cr:>10.4f}{cf:>10.4f}\n")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    ax[0].plot(x, p, label="precision")
    ax[0].plot(x, r, label="recall")
    ax[0].plot(x, f1, label="F1")
    ax[0].axvline(0.25, color="grey", linestyle="--", linewidth=1, label="default 0.25")
    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1.02)
    ax[0].set_xlabel("confidence threshold")
    ax[0].set_ylabel("value")
    ax[0].set_title("P / R / F1 vs confidence")
    ax[0].legend()

    ax[1].plot(r, p)
    ax[1].set_xlabel("recall")
    ax[1].set_ylabel("precision")
    ax[1].set_title("Precision-recall curve")
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1.02)

    plt.tight_layout()
    chart = os.path.join(REPORTS_DIR, "conf_sweep.png")
    plt.savefig(chart, dpi=150)
    plt.close()

    print(f"\n  report -> {report}")
    print(f"  chart  -> {chart}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
