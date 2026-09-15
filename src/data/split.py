"""Convert WIDER Face annotations to train_list.txt / val_list.txt"""
import os

from src.data.wider_annotations import parse_samples_with_image_root
from src.paths import ANNO_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LIST, VAL_LIST


def parse_wider_annotation(anno_file, image_root):
    """Parse WIDER Face annotation, returns list of (img_path, boxes) tuples."""
    return list(parse_samples_with_image_root(anno_file, image_root))


def main():
    train_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_train_bbx_gt.txt")
    train_images = TRAIN_IMAGES_DIR
    val_anno = os.path.join(ANNO_DIR, "wider_face_split", "wider_face_val_bbx_gt.txt")
    val_images = VAL_IMAGES_DIR

    train_samples = parse_wider_annotation(train_anno, train_images)
    print(f"Training samples: {len(train_samples)}")
    val_samples = parse_wider_annotation(val_anno, val_images)
    print(f"Validation samples: {len(val_samples)}")

    # Always write UTF-8 explicitly: the default on Windows is the ANSI code page,
    # which mangles the Chinese characters in these absolute paths for any reader
    # that assumes UTF-8.
    with open(TRAIN_LIST, "w", encoding="utf-8") as f:
        for img_path, _ in train_samples:
            f.write(f"{img_path}\n")
    with open(VAL_LIST, "w", encoding="utf-8") as f:
        for img_path, _ in val_samples:
            f.write(f"{img_path}\n")

    print("train_list.txt and val_list.txt generated.")


if __name__ == "__main__":
    main()
