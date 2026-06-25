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

### VTS面部捕捉
### 默认host与port，无调试窗口
```bash
python deployment/vtube_studio_bridge.py --input 0
```

### 指定VTube Studio API地址，并显示调试窗口
```bash
python deployment/vtube_studio_bridge.py --input 0 --vtshost 127.0.0.1 --vtsport 8001 --show
```

### Vtube Studio桥接默认启用FP16推理，如需关闭：
```bash
python deployment/vtube_studio_bridge.py --input 0 --nohalf
```

## 项目结构

| 模块 | 负责内容 | 核心文件 |
|:---|:---|:---|
| A — 数据采集与标注 | WIDER Face 数据集、标注质检 | `data/download_widerface.py`, `data/check_annotation.py` |
| B — 数据增强与加载 | Albumentations 增强、DataLoader | `dataset/dataloader.py`, `dataset/augmentation.py` |
| C — 模型结构设计 | YOLOv8n 配置 | `model/model_config.yaml`, `model/model_test.py` |
| D — 训练与调优 | 100 epoch 训练 + 30 epoch 微调 | `training/train.py`, `training/config.py` |
| E — 评估与错误分析 | mAP 评估、困难样本分析 | `evaluation/evaluate.py`, `evaluation/analyze_errors.py` |
| F — 部署与集成 | ONNX 导出、实时检测、VTube Studio 桥接 | `deployment/realtime_detect.py`, `deployment/vtube_studio_bridge.py`, `deployment/benchmark.py` |

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

### 实时检测

```bash
# 摄像头检测，默认使用FP32推理
python deployment/realtime_detect.py --input 0

# 视频文件检测
python deployment/realtime_detect.py --input path/to/video.mp4

# CUDA环境下启用FP16推理，便于利用Tensor Core
python deployment/realtime_detect.py --input 0 --half
```

### VTube Studio 人脸追踪

运行前请在 VTube Studio 中开启插件 API。首次运行时，VTube Studio 会弹出插件授权请求，同意后脚本会缓存 token。VTube Studio 未启动或连接失败时，脚本不会退出，会保持摄像头/调试窗口运行，并每 10 秒自动重试连接。连接成功后会读取 `InputParameterListRequest`，并按 VTS 返回的同名输入参数 `min/max` 裁剪注入值。

```bash
# 默认连接 ws://127.0.0.1:8001，使用0号摄像头，默认启用FP16推理
python deployment/vtube_studio_bridge.py --input 0

# 指定VTube Studio API地址
python deployment/vtube_studio_bridge.py --input 1 --vtshost 127.0.0.1 --vtsport 8001

# 显示调试窗口，方便确认选中的人脸框和映射位置
python deployment/vtube_studio_bridge.py --input 0 --show

# 关闭FP16推理
python deployment/vtube_studio_bridge.py --input 0 --nohalf
```

常用参数：

| 参数 | 默认值 | 说明 |
|:---|:---|:---|
| `--input` | `0` | 摄像头编号 |
| `--model` | `training/checkpoints/best_model_v2.onnx` | YOLO 人脸框检测模型，默认使用 ONNX Runtime GPU |
| `--vtshost` | `127.0.0.1` | VTube Studio API host |
| `--vtsport` | `8001` | VTube Studio API port |
| `--landmark-model` | `thirdparty/PFLD/weights/pfld_landmark.onnx` | PFLD 98 点关键点 ONNX 模型路径 |
| `--no-landmarks` | 关闭 | 关闭调试窗口中的 PFLD 关键点绘制 |
| `--show-landmark-indexes` | 关闭 | 在调试窗口绘制 PFLD 关键点编号，用于校准表情映射 |
| `--head-pose-config` | `thirdparty/3DDFA_V2/configs/mb1_120x120.yml` | 3DDFA_V2 头部姿态配置路径 |
| `--no-head-pose` | 关闭 | 关闭 `FaceAngleX/Y/Z` 估计 |
| `--show` | 关闭 | 显示调试预览窗口 |
| `--nohalf` | 关闭 | 关闭 bridge 默认启用的 FP16 推理 |
| `--send-fps` | `30` | 向 VTube Studio 发送参数的最高频率 |
| `--smoothing` | `0.55` | 位置平滑系数，越大响应越快 |
| `--bbox-alpha` | `0.55` | 人脸框 EMA 滤波系数，越大响应越快 |
| `--bbox-window` | `5` | 人脸框中值滤波窗口大小 |
| `--hold-frames` | `3` | 检测短暂丢失时保持上一人脸框的帧数 |
| `--expression-alpha` | `0.45` | 表情参数 EMA 平滑系数，越大响应越快 |
| `--head-pose-alpha` | `0.35` | 头部角度 EMA 平滑系数，越大响应越快 |

### 性能测试

```bash
python deployment/benchmark.py        # FPS + 模型大小基准
```

## 技术栈

Python 3.8+ | PyTorch 2.5 | ultralytics 8.4 | OpenCV 4.13 | Albumentations 2.0 | ONNX Runtime 1.27 | scikit-learn 1.9 | Matplotlib 3.10 | TensorBoard 2.5
