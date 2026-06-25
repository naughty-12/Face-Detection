# 高精度人脸检测算法及实践

基于 YOLOv8n 的高精度实时人脸检测系统，在 WIDER Face 验证集上达到 mAP50 = 0.664，推理速度 160+ FPS (RTX 3060)。

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt
```

### 实时摄像头检测

```bash
python deployment/realtime_detect.py --input 0
# 按 q 退出，按 s 截图
```

### 视频文件分析

```bash
python deployment/realtime_detect.py --input video.mp4
python deployment/realtime_detect.py --input video.mp4 --save output.mp4   # 保存结果视频
```

### 单张图片分析

```bash
python deployment/realtime_detect.py --input photo.jpg
python deployment/realtime_detect.py --input photo.jpg --save results/     # 保存到文件夹
```

### 批量文件夹分析

```bash
python deployment/realtime_detect.py --input my_photos/
python deployment/realtime_detect.py --input my_photos/ --save results/    # 全部保存
```

## 项目结构

| 模块 | 负责内容 | 核心文件 |
|:---|:---|:---|
| A — 数据采集与标注 | WIDER Face 数据集、标注质检 | `data/download_widerface.py`, `data/check_annotation.py` |
| B — 数据增强与加载 | Albumentations 增强、DataLoader | `dataset/dataloader.py`, `dataset/augmentation.py` |
| C — 模型结构设计 | YOLOv8n 配置 | `model/model_config.yaml`, `model/model_test.py` |
| D — 训练与调优 | 100 epoch 训练 + 30 epoch 微调 | `training/train.py`, `training/config.py` |
| E — 评估与错误分析 | mAP 评估、困难样本分析 | `evaluation/evaluate.py`, `evaluation/analyze_errors.py` |
| F — 部署与集成 | ONNX 导出、实时检测 | `deployment/realtime_detect.py`, `deployment/benchmark.py` |

## 模型性能

| 指标 | v1 基线 | v2 微调 |
|:---|:---|:---|
| mAP50 | 0.663 | 0.662 |
| Precision | 0.849 | 0.846 |
| Recall | 0.595 | 0.595 |
| 参数量 | 3.0M | 3.0M |
| 模型大小 | 6.0 MB | 6.0 MB |

| 部署指标 | 数值 | 目标 |
|:---|:---|:---|
| PyTorch GPU FPS (RTX 3060) | 162.8 | ≥ 30 ✅ |
| ONNX CPU FPS | 54.9 | ≥ 30 ✅ |
| ONNX FP16 模型大小 | 6.17 MB | ≤ 10 MB ✅ |

## 环境要求

- Python 3.8+ | PyTorch ≥ 1.10 (CUDA 12.1)
- RTX 3060 6GB+ 显存 | ultralytics ≥ 8.0

## 使用指南

### 训练

```bash
python training/train.py              # 从头训练（v1 100 epoch + v2 30 epoch）
python training/resume_train.py       # 断点续训
tensorboard --logdir training/checkpoints --port 6006  # 监控训练
```

### 评估

```bash
python evaluation/evaluate.py         # mAP 评估 + P-R 曲线
python evaluation/analyze_errors.py   # 困难样本可视化分析
```

### 导出 ONNX

```bash
python deployment/export_onnx.py      # ONNX FP16 导出 + 精度验证
```

### 性能测试

```bash
python deployment/benchmark.py        # FPS + 模型大小基准
```

## 技术栈

Python 3.8+ | PyTorch 2.5 | ultralytics 8.4 | OpenCV 4.13 | Albumentations 2.0 | ONNX Runtime 1.27 | scikit-learn 1.9 | Matplotlib 3.10 | TensorBoard 2.5
