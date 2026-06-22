"""Convert WIDER Face annotations to train_list.txt / val_list.txt"""
import os

ANNO_DIR = os.path.join(os.path.dirname(__file__), "annotations")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")


def parse_wider_annotation(anno_file, image_root):
    """Parse WIDER Face annotation, returns list of (img_path, boxes) tuples"""
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
        for _ in range(num_faces):
            if i >= len(lines):
                break
            parts = lines[i].split()
            x, y, w, h = map(int, parts[:4])
            boxes.append([x, y, w, h])
            i += 1
        img_path = os.path.join(image_root, img_name)
        if os.path.exists(img_path):
            samples.append((img_path, boxes))
    return samples


def main():
    train_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt")
    train_images = os.path.join(RAW_DIR, "WIDER_train", "images")
    val_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt")
    val_images = os.path.join(RAW_DIR, "WIDER_val", "images")

    train_samples = parse_wider_annotation(train_anno, train_images)
    print(f"Training samples: {len(train_samples)}")
    val_samples = parse_wider_annotation(val_anno, val_images)
    print(f"Validation samples: {len(val_samples)}")

    with open(os.path.join(ANNO_DIR, "train_list.txt"), "w") as f:
        for img_path, _ in train_samples:
            f.write(f"{img_path}\n")
    with open(os.path.join(ANNO_DIR, "val_list.txt"), "w") as f:
        for img_path, _ in val_samples:
            f.write(f"{img_path}\n")

    print("train_list.txt and val_list.txt generated.")


if __name__ == "__main__":
    main()
