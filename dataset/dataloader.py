"""Unified DataLoader for WIDER Face dataset"""
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader

ANNO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annotations")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")


def parse_wider_annotation(anno_file, image_root):
    """Parse WIDER Face annotations, return [(img_path, [[x,y,w,h],...]), ...]"""
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


class WiderFaceDataset(Dataset):
    """WIDER Face dataset loader"""

    def __init__(self, split="train", transform=None):
        anno_file = os.path.join(ANNO_DIR, "wider_face_split",
                                 f"wider_face_{split}_bbx_gt.txt")
        image_root = os.path.join(RAW_DIR, f"WIDER_{split}", "images")
        self.samples = parse_wider_annotation(anno_file, image_root)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, boxes_xywh = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        bboxes = []
        class_labels = []
        for box in boxes_xywh:
            x, y, bw, bh = box
            cx = (x + bw / 2) / w
            cy = (y + bh / 2) / h
            nw = bw / w
            nh = bh / h
            bboxes.append([cx, cy, nw, nh])
            class_labels.append(0)

        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed["image"]
            bboxes = transformed["bboxes"]
            class_labels = transformed["class_labels"]

        if len(bboxes) > 0:
            boxes_tensor = torch.tensor(bboxes, dtype=torch.float32)
            labels_tensor = torch.tensor(class_labels, dtype=torch.int64)
            targets = {"boxes": boxes_tensor, "labels": labels_tensor}
        else:
            targets = {"boxes": torch.zeros((0, 4), dtype=torch.float32),
                       "labels": torch.zeros((0,), dtype=torch.int64)}

        return image, targets


def collate_fn(batch):
    images = []
    targets = []
    for img, tgt in batch:
        images.append(img)
        targets.append(tgt)
    return torch.stack(images, dim=0), targets


def create_dataloader(split="train", batch_size=8, num_workers=2, phase="early"):
    from dataset.augmentation import get_train_augmentation, get_val_augmentation

    if split == "val" or phase == "val":
        transform = get_val_augmentation()
    else:
        transform = get_train_augmentation(phase=phase)

    dataset = WiderFaceDataset(split=split, transform=transform)
    shuffle = (split == "train")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    loader = create_dataloader(split="train", batch_size=4, num_workers=0, phase="early")
    images, targets = next(iter(loader))
    print(f"Batch images: {images.shape}")
    print(f"Batch targets: {len(targets)} items")
    print(f"First target boxes: {targets[0]['boxes'].shape}")
