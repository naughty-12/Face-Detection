"""Save augmented batch images with bboxes for manual verification"""
import os
import cv2
import numpy as np

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "evaluation", "reports")


def draw_boxes(image_tensor, targets, save_path):
    img = image_tensor.permute(1, 2, 0).cpu().numpy()
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    boxes = targets["boxes"].cpu().numpy()
    for box in boxes:
        cx, cy, nw, nh = box
        x1 = int((cx - nw / 2) * w)
        y1 = int((cy - nh / 2) * h)
        x2 = int((cx + nw / 2) * w)
        y2 = int((cy + nh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(save_path, img)
    print(f"Visualization saved to {save_path}")


def visualize_batch(dataloader, num_samples=4):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    images, targets = next(iter(dataloader))
    for i in range(min(num_samples, len(images))):
        save_path = os.path.join(OUTPUT_DIR, f"aug_vis_{i}.jpg")
        draw_boxes(images[i], targets[i], save_path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dataset.dataloader import create_dataloader
    loader = create_dataloader(split="train", batch_size=4, num_workers=0, phase="late")
    visualize_batch(loader)
    print("Done. Check evaluation/reports/aug_vis_*.jpg")
