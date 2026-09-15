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
| mAP50 | **0.66501** | 0.66404 |
| mAP50-95 | 0.35545 | **0.35618** |
| Precision | 0.849 | 0.846 |
| Recall | 0.595 | 0.595 |

> **口径说明（重要）**：上表是**全部 3,226 张验证图**上按 COCO 方式计算的 AP50，
> 分母包含大量极小尺寸人脸。设计文档中的目标 `0.76` 指的是 WIDER Face **Hard 子集**
> 的 VOC AP，**两者口径不同，不可直接比较**。官方 Easy/Medium/Hard 分项指标需要
> `widerface-evaluate` 包，而**本机网络不可用、装不上**，所以该目标无法对账
> （脚本在该包缺失时返回 `null`，不再用 mAP50 冒充分项结果）。
>
> **v2 与 v1 基本持平**：mAP50 微降 0.001，mAP50-95 微升 0.0007，属噪声量级。它只是在已收敛点
> 附近做了 30 epoch 低学习率微调，数据分布没变，所以**没有增益也不意外**。真正的闭环优化
> 需要困难样本回流，这一步没有做。
>
> 注：训练日志 `results.csv` 末行数字略低（0.66349 / 0.66171），那是**最后一个 epoch**的值，
> 与保存下来的 best 权重不是同一组参数；上表由 `src/eval/evaluate.py` 在 best 权重上重新测得。

### 按人脸尺寸分层（同一组权重，39,708 张标注人脸）

| 尺寸桶 | GT 人脸数 | 命中 | 召回 | 占全部 GT |
|:---|---:|---:|---:|---:|
| **Tiny（<32px）** | **28,775** | 14,186 | **0.493** | **72.5%** |
| Small（32–96px） | 8,606 | 7,780 | 0.904 | 21.7% |
| Medium（96–256px） | 1,890 | 1,805 | 0.955 | 4.8% |
| Large（>256px） | 437 | 432 | **0.989** | 1.1% |
| 合计 | 39,708 | 24,203 | **0.610** | 100% |

> **这是本项目最重要的一条结论**：汇总召回 0.610 看起来偏低，但 **72.5% 的标注人脸小于 32 像素**
> （人群远景里的脸），模型在这些极小脸上召回仅 0.493。**排除极小脸后，≥32px 人脸的召回是 0.916。**
>
> 也就是说：分数低主要是**基准集本身极端**，而非模型在实际场景不可用。虚拟形象驱动场景中
> 人脸占据画面主要位置（Medium/Large 桶），召回 0.955–0.989。这也解释了原始 `0.76` 目标
> 为何无法达成 —— Hard 子集几乎全是极小脸，**目标设定与实际应用场景错配**。
>
> 注：此处召回 0.610 与上表 ultralytics 报的 0.595 **口径不同**（前者为 IoU≥0.5 贪心匹配 +
> `conf=0.25`，后者为 ultralytics 自身的置信度扫描），两个数都真实，不可互相替换引用。

### 置信度阈值的影响（`python -m src.eval.conf_sweep`）

| conf | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.4387 | 0.6996 | 0.5392 |
| 0.10 | 0.5812 | 0.6746 | 0.6244 |
| 0.15 | 0.6817 | 0.6515 | 0.6663 |
| 0.20 | 0.7560 | 0.6308 | 0.6877 |
| **0.25（默认）** | 0.8139 | 0.6097 | **0.6971** |
| 0.30 | 0.8581 | 0.5887 | 0.6983 |
| 0.40 | 0.9187 | 0.5438 | 0.6832 |
| 0.50 | 0.9537 | 0.4909 | 0.6482 |

> **结论：默认的 `conf=0.25` 几乎正好在最优 F1 点上**（max-F1 点 `conf=0.297`，F1 0.6985
> 对比 0.25 处的 0.6971，仅差 **0.0014**）。此前文档里"`conf=0.25` 对召回不友好"的说法
> **经实测被推翻**。
>
> - **AP50 与阈值无关**（始终 0.66404），阈值只是在同一曲线上选取工作点。
> - 降到 0.10：召回 **+0.065**，但精确率 **−0.233** —— 代价很大。
> - 对"宁可多检也不漏检"的场景（如人脸跟踪），0.15–0.20 是合理选择。
> - 本项目**未改动任何默认阈值**，因为默认值本身就是最优 F1 点。

### 运行方式与体积（含 letterbox + 解码 + NMS 的实际吞吐）

| 变体 | mAP50 | mAP50-95 | 吞吐（中位） | 运行设备 | 体积 |
|:---|---:|---:|---:|:---|---:|
| **PyTorch `.pt`** | 0.66404 | 0.35618 | **10.31 ms / 97 FPS** | **CUDA（GPU）** | 5.95 MiB |
| ONNX **FP16** | 0.66382 | 0.35627 | 24.80 ms / 40 FPS | CPU | **5.88 MiB** |
| ONNX FP32 | 0.66379 | 0.35670 | ~25 ms / ~40 FPS | CPU | 11.70 MiB |

> **ONNX 精度的取舍已用实测解决**：FP16 相对 PyTorch 的 mAP50 差 **−0.0002**、mAP50-95 差
> **+0.0001**，即**检测质量无差别**；相对 FP32 体积减半、CPU 吞吐相同。
> 因此导出采用 **FP16**（`src/deploy/export_onnx.py` 的 `EXPORT_HALF = True`），
> 项目原定 ≤10 MB 目标随之重新满足（FP32 为 11.70 MiB，一度超标）。
>
> ⚠️ **ONNX 在本机只能跑 CPU**：`onnxruntime` 1.27 实际只提供 `CPUExecutionProvider` 与
> `AzureExecutionProvider`，**没有 CUDA**；安装 `onnxruntime-gpu` 需要网络，本机不可用。
> 所以 ONNX 比 PyTorch 慢约 **2.4–3.3 倍** —— 它在本机的价值是**可移植性**，不是速度。
> `src/deploy/benchmark.py` 与 ultralytics 现在都会明确打印这一降级，不再静默。
>
> ⚠️ **速度口径（重要）**：本机单次测量噪声极大 —— 同一调用在不同轮次测到 **4.8 – 18.8 ms**（跨度约 3 倍）。
> 本会话中所有"数字对不上"的现象都源于此，而非代码变化。因此 `src/deploy/benchmark.py` 已改为报告
> **中位数 + 最小值 / p95 / 最大值 + 样本数**，不再只报一个平均值。
> 最新中位结果：**PyTorch 10.31 ms（97.0 FPS）** / **ONNX 24.80 ms（40.3 FPS，CPU）**。
> 引用任何速度数字时请注明测量次数与统计量，**单次读数不可作为结论**。

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

> **显存注意（已实测）**：6 GB 卡上 `batch=8 + imgsz=640` 的峰值**保留**显存达 **6.18 GiB**，
> 已超过 6.00 GiB 物理上限，从第一轮起就在向共享内存溢出；100 epoch 长跑会因内存碎片继续增高
> （历史日志达 11.1 GiB），吞吐从 5.8 it/s 崩到 1.7 it/s。
> **配置已下调为 `batch=6`**（峰值保留 4.41 GiB，留 27% 余量）。实测脚本：`src/train/probe_vram.py`。

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
| batch / imgsz | 历史训练 **8 / 640**（见 `artifacts/checkpoints/*/args.yaml`）；**配置已改为 6 / 640**（显存实测见上） |
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
python -m src.deploy.export_onnx   # ONNX 导出（FP16 + simplify，opset 12）+ 精度验证
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
| WIDER 分项评估 | 官方 Easy/Medium/Hard 需要 `widerface-evaluate` 包，**本机网络不可用、装不上**，原始 0.76 目标无法对账。已有替代：按人脸尺寸分层的召回表（见上方） |
| 小脸召回 | 汇总召回 0.610，其中 **72.5% 的 GT 人脸 <32px、召回仅 0.493**；≥32px 人脸召回 0.916。**conf 扫描已完成**：默认 0.25 已接近最优 F1，阈值不是瓶颈（见上方阈值表） |
| v2 无增益 | 与 v1 基本持平（mAP50 −0.001，mAP50-95 +0.0007），缺困难样本回流 |
| 增强模块未接入 | `src/data/augment.py` 为独立模块，训练只用 ultralytics 内置增强 |
| 自采数据缺失 | 设计中的 500 张侧脸/遮挡/暗光数据未采集 |
| 训练吞吐 | 已实测并修复：6 GB 显存下 `batch=8` 峰值保留 **6.18 GiB** 超上限，**配置已改为 `batch=6`**（4.41 GiB）。**尚未用新 batch 重训验证** |
| **测量噪声** | 同一调用实测跨度 **4.8–18.8 ms（约 3 倍）**，单次平均值无意义。`src/deploy/benchmark.py` 已改为报告中位数与分布 |
| 测试覆盖有限 | `tests/` 有 **56 个测试**覆盖 WIDER 解析、坐标转换、匹配几何，以及桥接的时序滤波（EMA/中值/hold/reset）与位置映射（`unittest`，零依赖）；**模型 IO、训练全流程仍无覆盖** |
| 死配置键 | `configs/model.yaml` 的 `accumulation_steps`、`multi_scale`、`architecture` 从未被代码读取 |

### 运行方式建议（本机实测）

| 场景 | 建议 | 原因 |
|:---|:---|:---|
| **日常使用、追求速度** | **PyTorch `.pt` + CUDA** | 中位 10.31 ms / 97 FPS，比 ONNX（CPU）快约 2.4 倍 |
| 免装 PyTorch 的部署 | ONNX **FP16** | 5.88 MiB，mAP 与 PyTorch 无差别，但只能跑 CPU |
| **VTube Studio 桥接** | **默认已用 `.pt`（CUDA + FP16）** | 曾默认 `.onnx` → 落在 CPU、GPU 闲置；已修正，FP16 另获 7.4% 提速 |

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
