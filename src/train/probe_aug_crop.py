"""A/B experiment: does crop-and-zoom raise small-face recall?

Hypothesis, from the size-stratified analysis: 72.5% of WIDER val faces are smaller than
32 px and recall on that bin is only 0.493. Cropping a sub-region and rescaling it to 640
makes small faces larger during training, which should raise recall on that bin.

Design notes
------------
* **Both arms fine-tune the shipped ``best_model_v2.pt``**, not a fresh ImageNet model.
  The question is "does this augmentation change what the model already knows", so starting
  from a competent face detector with a small LR isolates the augmentation. It is also far
  cheaper than relearning from scratch, which matters because of the next point.
* ``fraction`` is deliberately left at 1.0. Two reasons:
    1. ``ultralytics/data/base.py`` truncates ``im_files`` when ``fraction < 1`` **before**
       ``get_labels()`` runs, so the cached label hash no longer matches and the cache is
       rebuilt on every run -- which needs a multiprocessing worker pool that this sandbox
       forbids.
    2. ``fraction`` takes a **prefix** of the sorted file list, not a random sample. WIDER's
       images are grouped by event class, so a fraction is not representative of the whole
       distribution -- a bad basis for a recall comparison.
* Two arms, identical in every respect except the ``augmentations`` argument: same starting
  weights, epochs, batch, imgsz, LR, seed and workers.
* Transforms are mounted through ultralytics' ``augmentations`` hyperparameter rather than
  through a custom Dataset. ``v8_transforms()`` inserts them between Mosaic/affine/MixUp and
  RandomHSV/RandomFlip, so Mosaic is preserved and nothing is applied twice.
* Bounding boxes are only transformed if the transform's class name is in ultralytics'
  hardcoded spatial whitelist. ``BBoxSafeRandomCrop`` and ``Resize`` are both verified to be
  present, and box handling is exercised before training starts (see --check-only).
* The baseline deliberately omits ``augmentations`` entirely, which leaves the hook on its
  built-in defaults -- exactly what the original training runs had. The treatment replaces
  those defaults with the crop pipeline, so the only difference between arms is the crop.

Evaluate the results with the size-stratified analysis, which is a separate step:

    python -m src.eval.analyze_errors --model artifacts/logs/aug_ab/base/weights/best.pt
    python -m src.eval.analyze_errors --model artifacts/logs/aug_ab/crop/weights/best.pt
"""
import argparse
import os
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.getcwd())
from src.paths import BEST_MODEL_V2_PT, LOGS_DIR, MODEL_CONFIG_PATH, WIDER_YAML  # noqa: E402

OUT_ROOT = os.path.join(LOGS_DIR, "aug_ab")


def crop_pipeline():
    """Crop a box-safe sub-region, then scale it back to a square 640.

    Resize (not LongestMaxSize) is used so every image in a batch ends up exactly 640x640 --
    the trainer stacks batches into one tensor and variable sizes would break that. The
    aspect distortion that comes with a square resize is accepted for this prototype.
    """
    import albumentations as A

    return [
        A.BBoxSafeRandomCrop(erosion_rate=0.0, p=1.0),
        A.Resize(640, 640, p=1.0),
    ]


def check_hook():
    """Prove the hook treats the pipeline as spatial and actually moves the boxes."""
    from ultralytics.data.augment import Albumentations

    hook = Albumentations(p=1.0, transforms=crop_pipeline())
    print(f"  contains_spatial = {hook.contains_spatial}")
    if not hook.contains_spatial:
        raise SystemExit("ABORT: pipeline not detected as spatial -> boxes would not be "
                         "transformed and labels would be corrupted")

    class _Inst:
        def __init__(self, b):
            self.bboxes = np.asarray(b, dtype=np.float32)

        def convert_bbox(self, fmt=None):
            return None

        def normalize(self, w, h):
            return None

        def update(self, bboxes=None):
            self.bboxes = np.asarray(bboxes, dtype=np.float32)

    img = np.zeros((640, 640, 3), dtype=np.uint8)
    before = np.array([[0.40, 0.40, 0.10, 0.10]], dtype=np.float32)  # small face, xywh
    labels = {"img": img, "cls": np.array([[0]]), "instances": _Inst(before)}
    out = hook(labels)
    after = np.asarray(out["instances"].bboxes)
    print(f"  bbox {before.ravel()} -> {after.ravel()}")
    print(f"  box side before={before[0, 2]:.4f} after={after[0, 2]:.4f} "
          f"({'enlarged' if after[0, 2] > before[0, 2] else 'NOT enlarged'})")
    if np.allclose(before, after):
        raise SystemExit("ABORT: boxes unchanged -> the crop is not reaching the labels")
    return True


def train_arm(name, cfg, args, use_augmentations):
    from ultralytics import YOLO

    tcfg = cfg["training"]          # model_name is top level; hyperparameters are nested
    weights = args.from_weights if args.from_weights else cfg["model_name"]
    model = YOLO(weights)
    print(f"  starting weights: {weights}")
    kwargs = dict(
        data=WIDER_YAML,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        fraction=args.fraction,
        optimizer=tcfg["optimizer"],
        lr0=args.lr0,
        lrf=tcfg["lrf"],
        cos_lr=tcfg["cos_lr"],
        amp=tcfg["amp"],
        seed=0,
        deterministic=True,
        val=args.val,
        plots=False,
        verbose=False,
        project=OUT_ROOT,
        name=name,
        exist_ok=True,
    )
    if use_augmentations:
        kwargs["augmentations"] = crop_pipeline()

    print(f"\n{'=' * 70}\nArm: {name}   augmentations={'crop+resize' if use_augmentations else 'built-in default'}\n{'=' * 70}")
    t0 = time.perf_counter()
    model.train(**kwargs)
    elapsed = time.perf_counter() - t0

    # last.pt, not best.pt: with val disabled there is no mAP to select 'best' on, and using
    # the final weights makes both arms an exact comparison at an identical training budget.
    last = os.path.join(OUT_ROOT, name, "weights", "last.pt")
    print(f"  arm {name} finished in {elapsed / 60:.1f} min -> {last}")
    return last


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="keep at 1.0: <1 forces a label-cache rebuild and is not representative")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--from-weights", type=str, default=BEST_MODEL_V2_PT,
                        help="starting weights for both arms (default: the shipped v2 model)")
    parser.add_argument("--lr0", type=float, default=1e-4,
                        help="fine-tune LR; matches the project's own v2 fine-tune")
    parser.add_argument("--batch", type=int, default=6,
                        help="6 is the measured maximum that fits 6 GiB physical VRAM")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workers", type=int, default=0,
                        help="0 keeps the dataloader single-process")
    parser.add_argument("--val", action="store_true",
                        help="validate every epoch (slower; unnecessary for an A/B on final weights)")
    parser.add_argument("--check-only", action="store_true",
                        help="verify the hook and exit without training")
    args = parser.parse_args()

    print("=" * 70)
    print("Small-face crop-and-zoom A/B")
    print("=" * 70)

    print("\n[pre-flight] does the hook treat the crop as spatial?")
    check_hook()
    if args.check_only:
        return

    cfg = yaml.safe_load(open(MODEL_CONFIG_PATH, encoding="utf-8"))
    tcfg = cfg["training"]

    base_best = train_arm("base", cfg, args, use_augmentations=False)
    crop_best = train_arm("crop", cfg, args, use_augmentations=True)

    print("\n" + "=" * 70)
    print("Both arms finished. Weights to evaluate:")
    print(f"  baseline  : {base_best}")
    print(f"  crop-zoom : {crop_best}")
    print("\nNext: size-stratified recall for each --")
    print(f"  python -m src.eval.analyze_errors --model {base_best}")
    print(f"  python -m src.eval.analyze_errors --model {crop_best}")


if __name__ == "__main__":
    main()
