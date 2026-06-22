"""Check WIDER Face annotation quality"""
import os
import cv2

ANNO_DIR = os.path.join(os.path.dirname(__file__), "annotations")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
REPORT_PATH = os.path.join(ANNO_DIR, "quality_report.txt")

MIN_BOX_SIZE = 5
MAX_AREA_RATIO = 0.8
MIN_FACE_RATIO = 0.001


def parse_annotation(anno_file, image_root):
    samples = []
    with open(anno_file, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    i = 0
    while i < len(lines):
        img_name = lines[i]
        i += 1
        if i >= len(lines):
            break
        num_faces = int(lines[i])
        i += 1
        img_path = os.path.join(image_root, img_name)
        if not os.path.exists(img_path):
            for _ in range(num_faces):
                if i < len(lines):
                    i += 1
            continue
        h, w = cv2.imread(img_path).shape[:2]
        boxes = []
        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = list(map(int, lines[i].split()[:4]))
            x, y, bw, bh = parts
            boxes.append([x, y, bw, bh])
            i += 1
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
         os.path.join(RAW_DIR, "WIDER_train", "images")),
        ("val", os.path.join(ANNO_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt"),
         os.path.join(RAW_DIR, "WIDER_val", "images")),
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
