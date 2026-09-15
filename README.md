# 高精度人脸检测与实时表情捕捉

基于 YOLOv8n-face 的实时人脸检测系统，覆盖从数据准备、模型训练、评估到部署推理与
VTube Studio 虚拟形象驱动的完整链路。**由一人独立完成的工程实践项目。**

> 所有指标均标注了**测量口径**。本项目坚持一个原则：不把口径不同的数字放在一起比较，
> 也不引用未经复现的数值。未完成的部分如实列在[已知限制](#已知限制与未完成项)中。

---

## 实测指标

### 检测精度（WIDER Face 验证集）

| 指标 | v1 基线（100 epoch） | v2 微调（+30 epoch） |
|:---|---:|---:|
| mAP50 | **0.663** | 0.662 |
| mAP50-95 | 0.354 | 0.353 |
| Precision | 0.849 | 0.846 |
| Recall | 0.595 | 0.595 |

> **口径说明（重要）**：上表是**全部 3,226 张验证图**上按 COCO 方式计算的 AP50，
> 分母包含大量极小尺寸人脸。设计文档中的目标 `0.76` 指的是 WIDER Face **Hard 子集**
> 的 VOC AP，**两者口径不同，不可直接比较**。Easy/Medium/Hard 分项评估**尚未完成**，
> 因此原目标目前无法对账。
>
> **v2 是负收益**：三项指标全部略降。它只是在已收敛点附近做了 30 epoch 低学习率微调，
> 训练数据分布没有变化。真正的闭环优化需要困难样本回流，这一步没有做。

### 推理速度与体积

| 项 | 实测 | 说明 |
|:---|---:|:---|
| PyTorch 网络前向 | **145.0 – 146.9 FPS**（6.9 ms） | RTX 3060 Laptop |
| ONNX Runtime 前向 | 43.9 – 44.8 FPS（22.3 ms） | 见下方警告 |
| 模型体积（`.pt`） | 5.95 MB | 3.0M 参数 |
| 模型体积（`.onnx`） | 5.88 MB | FP32 + simplify |

> ⚠️ **速度口径**：以上数字由 `src/deploy/benchmark.py` 测得，调用的是 `model.model(dummy)`
> **裸网络前向**，**不含** letterbox 预处理、解码与 NMS 后处理，输入为随机张量。
> 因此它**不是端到端管线帧率**。
>
> ⚠️ **ONNX 未跑在 GPU 上**：ONNX 前向比 PyTorch 慢约 3 倍，且该脚本同时注册 CUDA 与 CPU
> 两个 ExecutionProvider 却**从不打印实际生效者**。所以"ONNX 44.8 FPS"目前无法确认其运行设备，
> 这一项需要先固定 provider 再重新测量。

---

## 环境要求

| 项 | 实测值 |
|:---|:---|
| 操作系统 | Windows |
| Python | 3.12.8 |
| PyTorch | 2.5.1+cu121 |
| GPU | NVIDIA GeForce RTX 3060 Laptop，**6144 MiB（6 GB）** |
| 驱动 / CUDA | 581.95 / CUDA 12.1 |

```bash
pip install -r requirements.txt
```

> **显存注意**：6 GB 卡上 `batch=8 + imgsz=640` 会出现超额分配 —— 训练日志记录的峰值
> 显存达 **11.1 GB**（Windows WDDM 溢出到共享系统内存），伴随吞吐从 5.8 it/s 掉到 1.7 it/s。
> 建议降到 `batch=4` 或 `imgsz=512`。

---

## 快速开始

所有脚本都从**项目根目录**以模块方式运行。

### 实时摄像头检测

```bash
python -m src.deploy.detect --input 0
# 按 q 退出，按 s 截图
```

### 视频 / 单图 / 文件夹

```bash
python -m src.deploy.detect --input video.mp4 --save output.mp4
python -m src.deploy.detect --input photo.jpg --save results/
python -m src.deploy.detect --input my_photos/ --save results/
python -m src.deploy.detect --input photo.jpg --no-show      # 不弹结果窗口（脚本/批处理用）
```

### VTube Studio 表情驱动

```bash
# 前置：VTube Studio 中开启"启动插件 API"，首次运行会弹授权请求
python apps/vtube_bridge/main.py --input 0 --landmarks
```

---

## 目录结构

```
configs/model.yaml          训练与模型配置
src/                        库代码（被导入，不直接运行）
├── paths.py                统一路径解析 —— 全项目唯一的路径来源
├── data/                   数据管线
│   ├── download.py         数据集准备指引（不执行下载）
│   ├── qc.py               标注质检
│   ├── convert.py          WIDER 标注 → YOLO 格式
│   ├── split.py            生成 train/val 图片清单
│   ├── loader.py           数据集封装
│   ├── augment.py          Albumentations 增强管线（当前未接入训练）
│   └── vis_aug.py          增强效果可视化
├── train/                  训练
│   ├── config.py           超参管理
│   ├── train.py            v1 基线 + v2 微调
│   ├── resume.py           断点续训
│   └── check_model.py      模型诊断（输出 shape / 导出可行性）
├── eval/                   评估
│   ├── evaluate.py         mAP 与 P-R 曲线
│   └── analyze_errors.py   困难样本分析
└── deploy/                 部署
    ├── detect.py           实时检测统一入口（摄像头/视频/图片/文件夹）
    ├── export_onnx.py      ONNX 导出
    └── benchmark.py        性能基准

apps/vtube_bridge/          VTube Studio 表情驱动（独立应用）
data/                       数据集（raw/ 与 annotations/）
artifacts/                  产出（checkpoints/ 权重与曲线，reports/ 图表，logs/ 日志）
docs/                       现状文档；archive/ 历史文档；plans/ 工作计划
tests/                      测试（待填充）
```

---

## 数据管线

```bash
python -m src.data.download     # 打印 WIDER Face 手动下载指引（约 3 GB，需自行下载）
python -m src.data.qc           # 标注质检 → data/annotations/quality_report.txt
python -m src.data.convert      # 标注 → YOLO labels/（训练真正依赖的产物）
python -m src.data.split        # 生成 train_list.txt / val_list.txt
```

数据集规模：训练 **12,880** 张、验证 **3,226** 张，沿用 WIDER Face 官方划分。

> 设计文档中承诺的"自采 500 张侧脸/遮挡/暗光数据"**未采集**，最终只使用公开数据集。

---

## 训练

```bash
python -m src.train.train        # v1 基线 100 epoch + v2 微调 30 epoch
python -m src.train.resume       # 断点续训（从 artifacts/checkpoints/*/weights/last.pt 接续）
tensorboard --logdir artifacts/checkpoints --port 6006
```

实际使用的超参（以 `artifacts/checkpoints/*/args.yaml` 为准）：

| 参数 | 值 |
|:---|:---|
| batch / imgsz | 8 / 640 |
| optimizer | AdamW，lr 1e-3 → 1e-5 余弦退火 |
| 混合精度 | `amp=true` |
| 增强 | `mosaic=1.0`（v2 降至 0.5）、`hsv_*`、`fliplr=0.5`、`translate=0.1`、`scale=0.5`、`close_mosaic=10` |
| 其他 | `nbs=64`、`workers=2`、`seed=0`、`deterministic=true` |

> 训练用的是 **ultralytics 内置增强**。`src/data/augment.py` 里的 Albumentations 管线
> 目前**没有接入训练流程**，只用于增强效果的可视化验证。

---

## 评估

```bash
python -m src.data.split          # 前置：生成 val_list.txt
python -m src.eval.evaluate       # mAP + Easy/Medium/Hard 分项 + P-R 曲线
python -m src.eval.analyze_errors # TOP20 漏检/误检 + 可视化
```

> ⚠️ 两个脚本都依赖 `data/annotations/val_list.txt`。该文件**不在仓库中**，缺失时会打印
> `[WARN]` 并跳过相应步骤。`export_onnx.py` 的精度验证同样依赖它，缺失时**会直接判定通过**，
> 因此"ONNX 精度损失 < 1e-4"这一说法**目前没有实测支撑**。

---

## 导出与基准

```bash
python -m src.deploy.export_onnx   # ONNX 导出（FP32 + simplify，opset 12）+ 精度验证
python -m src.deploy.benchmark     # 网络前向延迟与体积
```

---

## VTube Studio 实时表情驱动

用自训练的检测模型精确定位人脸，再送 MediaPipe 提取关键点与表情系数，通过 WebSocket
实时注入 VTube Studio，驱动 Live2D 虚拟形象。

```
摄像头帧
   │
   ▼ ① YOLOv8n-face 检测人脸框                        ~6 ms
   ▼ ② FaceBoxFilter：EMA(0.55) + 中值(5) + hold 3 帧 + 新人脸重置
   ▼ ③ 裁剪 ROI（外扩 1.45 倍）← 核心设计：先检测再裁剪
   ▼ ④ MediaPipe FaceLandmarker                       ~8 ms
        478 关键点 + 52 ARKit blendshape
        + 变换矩阵 SVD 分解 → 头部欧拉角
        + 虹膜关键点归一化 → 视线方向
   ▼ ⑤ ParameterFilter：表情/头部/眼球 三通道独立 EMA
   ▼ ⑥ 校准映射：按 VTS 返回的 min/max 裁剪 + 源范围采集 + 一键回中
   ▼ ⑦ VTubeStudioWorker：daemon 后台线程，30 FPS 注入，断线 10s 重连
VTube Studio → Live2D 形象跟随表情
```

**"先检测再裁剪"是这套管线与 VTS 自带跟踪的关键差别**：全图直接送关键点模型时人脸往往只占
画面一小部分，裁剪后人脸占比大幅提升，关键点定位与表情系数都更准；同时外扩比例 1.45 是
权衡值——裁太紧会切掉下巴/额头导致姿态估计不稳，裁太松则背景干扰回归。

常用参数见 `python apps/vtube_bridge/main.py --help`（16 个可调参数：平滑系数、滤波窗口、
保持帧数、发送频率、MediaPipe 裁剪比例等）。

---

## 已知限制与未完成项

| 项 | 说明 |
|:---|:---|
| WIDER 分项评估 | Easy/Medium/Hard 尚未评估，原始 0.76 目标无法对账 |
| ONNX 精度验证 | 因 `val_list.txt` 缺失而静默跳过，MAE 未经实测 |
| ONNX 运行设备 | benchmark 不打印实际生效的 ExecutionProvider，且 ONNX 比 PyTorch 慢 3 倍 |
| Recall 偏低 | 0.595，漏检主要来自小脸与遮挡；尚无 conf 阈值扫描数据 |
| v2 无增益 | 微调为负收益，缺困难样本回流 |
| 增强模块未接入 | `src/data/augment.py` 为独立模块，训练只用 ultralytics 内置增强 |
| 自采数据缺失 | 设计中的 500 张侧脸/遮挡/暗光数据未采集 |
| 训练吞吐 | 6 GB 显存下超额分配，需降 batch 或 imgsz |
| 标注解析重复 | WIDER 解析在 `convert/qc/split/loader` 中有 4 份实现，边界处理不一致 |
| 无单元测试 | `tests/` 为空；`check_model.py` 是诊断脚本，不含断言 |

---

## 技术栈

Python 3.12 | PyTorch 2.5 | ultralytics 8.4 | OpenCV 4.13 | MediaPipe 0.10 |
Albumentations 2.0 | ONNX Runtime 1.27 | PyQt5 5.15 | websocket-client 1.8 |
scikit-learn | Matplotlib | TensorBoard

---

## 文档

| 文档 | 内容 |
|:---|:---|
| [`docs/架构说明.md`](docs/架构说明.md) | 模块划分、数据流、依赖关系与设计取舍 |
| [`docs/项目现状与差距.md`](docs/项目现状与差距.md) | 实现与设计承诺的逐条差距（含证据） |
| [`docs/使用指南.md`](docs/使用指南.md) | 面向使用者的完整操作说明与 FAQ |
| [`docs/面试讲解提纲.md`](docs/面试讲解提纲.md) | 讲解流程与预设问答 |
| [`CHANGELOG.md`](CHANGELOG.md) | 变更记录 |
| [`docs/archive/`](docs/archive/README.md) | 历史设计文档（**描述的是当初的目标，不是现状**） |
