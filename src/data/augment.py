"""Albumentations augmentation pipelines — **VISUALISATION ONLY, not used in training.**

⚠️ 定位说明（对应差距 G12，完整论证见 docs/项目现状与差距.md 第八节）：

本模块**不参与训练，也不打算参与**。训练使用的是 ultralytics 内置增强，而本模块定义的
每一个变换都已被其覆盖：

    本模块                                ultralytics 对应
    HorizontalFlip(p=0.5)            →   fliplr=0.5
    RandomBrightnessContrast         →   hsv_v
    HueSaturationValue               →   hsv_h / hsv_s
    Blur(blur_limit=3 / 5)           →   auto_augment='randaugment'（含模糊）
    CoarseDropout（即 Cutout）        →   erasing=0.4
    Resize(640, 640)                 →   letterbox 到 imgsz

而且本模块**严格更弱**：它没有 Mosaic，而 Mosaic 是 ultralytics 在检测任务上最有价值的
增强（mosaic=1.0）。因此：

  - **替换** ultralytics 的增强 → 会丢掉 Mosaic；
  - **叠加** → 翻转与 HSV 被施加两次、概率失准。

它当前的唯一用途，是让 src/data/vis_aug.py 渲染增强后的批次供人工检查——这恰好也是将来
设计"定向增强"（属性感知增强、小脸放大裁剪）时的验证手段。
"""
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
                # Albumentations 2.x API. The 1.x names (max_holes / max_height / max_width)
                # are **silently ignored** by 2.0: passing them only emits a UserWarning and
                # the transform then runs at its defaults (1-2 holes of 10-20%), not at the
                # intended up-to-8 holes of <=32 px. Verified against albumentations 2.0.8.
                A.CoarseDropout(
                    num_holes_range=(1, 8),
                    hole_height_range=(1, 32),
                    hole_width_range=(1, 32),
                    p=0.3,
                ),
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
