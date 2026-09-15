"""Check WIDER Face annotation quality"""
import os

import cv2

from src.data.wider_annotations import parse_samples_with_image_root
from src.paths import ANNO_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR

REPORT_PATH = os.path.join(ANNO_DIR, "quality_report.txt")

MIN_BOX_SIZE = 5
MAX_AREA_RATIO = 0.8
MIN_FACE_RATIO = 0.001


def parse_annotation(anno_file, image_root):
    """Return [{"img_path", "w", "h", "boxes"}] for entries whose image is readable."""
    samples = []
    for img_path, boxes in parse_samples_with_image_root(anno_file, image_root):
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        samples.append({"img_path": img_path, "w": w, "h": h, "boxes": boxes})
    return samples


def check_sample(sample):
    issues = []
    w, h = sample["w"], sample["h"]
    img_area = w * h
    for idx, box in enumerate(sample["boxes"]):
        x, y, bw, bh = box
        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            issues.append(f"  Box {idx}: out of bounds ({x},{y},{bw},{bh}) img=({w},{h})")
        if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
            issues.append(f"  Box {idx}: too small ({bw}x{bh})")
        box_area = bw * bh
        if box_area > img_area * MAX_AREA_RATIO:
            issues.append(f"  Box {idx}: area too large ({box_area/img_area:.1%})")
        if box_area < img_area * MIN_FACE_RATIO:
            issues.append(f"  Box {idx}: area too small ({box_area/img_area:.4%})")
    return len(issues) == 0, issues


def main():
    report_lines = []
    total = 0
    invalid_total = 0
    for split, anno_file, img_root in [
        ("train", os.path.join(ANNO_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt"),
         TRAIN_IMAGES_DIR),
        ("val", os.path.join(ANNO_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt"),
         VAL_IMAGES_DIR),
    ]:
        print(f"\nChecking {split} set...")
        samples = parse_annotation(anno_file, img_root)
        invalid_count = 0
        for s in samples:
            total += 1
            valid, issues = check_sample(s)
            if not valid:
                invalid_count += 1
                report_lines.append(f"\n[INVALID] {s['img_path']}")
                report_lines.extend(issues)
        invalid_total += invalid_count
        print(f"  Total: {len(samples)} | Invalid: {invalid_count}")

    with open(REPORT_PATH, "w") as f:
        f.write(f"WIDER Face Annotation Quality Report\n{'='*60}\n")
        f.write(f"Total images checked: {total}\n")
        f.write(f"Images with issues: {invalid_total}\n")
        f.write(f"Valid images: {total - invalid_total}\n")
        f.write("\nDetails:\n" + "\n".join(report_lines))
    print(f"\nReport saved to {REPORT_PATH}")


if __name__ == "__main__":
    main()
