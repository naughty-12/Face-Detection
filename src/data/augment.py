"""Albumentations-based online data augmentation pipeline"""
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_augmentation(phase="early"):
    """Get training augmentation pipeline.
    Args:
        phase: "early" (first 50 epochs) or "late" (last 50 epochs)
    """
    if phase == "early":
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=20, p=0.3),
                A.Blur(blur_limit=3, p=0.1),
                A.Resize(640, 640),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )
    else:
        return A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=25, val_shift_limit=25, p=0.3),
                A.Blur(blur_limit=5, p=0.2),
                A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
                A.Resize(640, 640),
                ToTensorV2(),
            ],
            bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
        )


def get_val_augmentation():
    """Validation augmentation: resize only, no data perturbation"""
    return A.Compose(
        [A.Resize(640, 640), ToTensorV2()],
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
