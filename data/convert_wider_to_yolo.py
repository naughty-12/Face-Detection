"""Convert WIDER Face annotations to YOLO-format label files required by ultralytics"""
import os

ANNO_DIR = os.path.join(os.path.dirname(__file__), "annotations")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def parse_wider_annotation(anno_file):
    """Parse WIDER Face annotation file, return [(img_name, [[x,y,w,h],...]), ...]"""
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
        boxes = []
        # WIDER Face quirk: some entries have num_faces=0 but still
        # include a dummy all-zero bbox line — skip it
        if num_faces == 0 and i < len(lines):
            parts = lines[i].split()
            if all(int(v) == 0 for v in parts):
                i += 1
        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = lines[i].split()
            x, y, w, h = map(int, parts[:4])
            boxes.append([x, y, w, h])
            i += 1
        samples.append((img_name, boxes))
    return samples


def boxes_to_yolo(boxes, img_w, img_h):
    """Convert [x,y,w,h] absolute to YOLO format [class, cx, cy, w, h] normalized"""
    yolo_lines = []
    for box in boxes:
        x, y, bw, bh = box
        cx = (x + bw / 2.0) / img_w
        cy = (y + bh / 2.0) / img_h
        nw = bw / img_w
        nh = bh / img_h
        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
    return yolo_lines


def convert_split(split_name, anno_file, image_root):
    """Convert one split (train/val) to YOLO format"""
    import cv2

    label_root = os.path.join(os.path.dirname(image_root), "labels")
    os.makedirs(label_root, exist_ok=True)

    samples = parse_wider_annotation(anno_file)
    converted = 0
    skipped = 0

    for img_name, boxes in samples:
        img_path = os.path.normpath(os.path.join(image_root, img_name))
        if not os.path.exists(img_path):
            skipped += 1
            continue

        # Get image dimensions (use imdecode to avoid cv2 Unicode path bug on Windows)
        import numpy as np
        with open(img_path, "rb") as f_img:
            img_data = f_img.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            skipped += 1
            continue
        h, w = img.shape[:2]

        # Convert to YOLO
        yolo_lines = boxes_to_yolo(boxes, w, h)

        # Write label file (same name as image, .txt extension)
        base_name = os.path.splitext(img_name)[0]
        label_path = os.path.join(label_root, base_name + ".txt")

        # Ensure subdirectories exist
        os.makedirs(os.path.dirname(label_path), exist_ok=True)

        with open(label_path, "w") as f:
            f.write("\n".join(yolo_lines) + "\n" if yolo_lines else "")

        converted += 1
        if converted % 2000 == 0:
            print(f"  Converted {converted} images...")

    print(f"  {split_name}: {converted} images converted, {skipped} skipped")


def main():
    print("Converting WIDER Face annotations to YOLO format...")

    # Train split
    train_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt")
    train_images = os.path.join(RAW_DIR, "WIDER_train", "images")
    print("\n[1/2] Training set")
    convert_split("train", train_anno, train_images)

    # Val split
    val_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt")
    val_images = os.path.join(RAW_DIR, "WIDER_val", "images")
    print("\n[2/2] Validation set")
    convert_split("val", val_anno, val_images)

    print("\nYOLO label conversion complete!")
    print(f"Labels saved to: {os.path.join(RAW_DIR, 'WIDER_train', 'labels')}")
    print(f"Labels saved to: {os.path.join(RAW_DIR, 'WIDER_val', 'labels')}")


if __name__ == "__main__":
    main()
