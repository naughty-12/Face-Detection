# 高精度人脸检测与实时表情捕捉

用**普通摄像头**驱动 Live2D 虚拟形象的实时客户端工具：自训练的 **YOLOv8n** 人脸检测器定位人脸，
裁剪后交给 **MediaPipe** 提取关键点与表情系数，经多级时序滤波与按模型自适应的参数校准，
通过 **VTube Studio 插件 API（WebSocket）** 实时注入虚拟形象 —— 张嘴、眨眼（左右可分辨）、
三轴转头、眼球视线均已实测跟随，**连续 599.7 秒稳定性实测通过**。

上半段是完整的算法链路（数据处理 → 训练 → 评估 → ONNX 导出 → 部署推理），下半段是真正的目的：
把它做成一个能长时间稳定运行的工具。**感知层与出口层解耦**，另挂两条出口（VMC/OSC、Unity UDP）
复用同一条捕捉管线。

> **本 README 是仓库唯一的说明文档，也是唯一的文档来源。**
> 每个数字都标注**测量口径**并给出可复现的命令；不同口径的数字不并列比较；没测过的一律写"未实测"。
> 未完成的部分如实列在[§14 已知限制](#14-已知限制与未完成项)。
>
> **角色说明（如实交底）**：本项目**由我主导，实现环节由 AI 辅助完成**。
> 我负责技术选型、架构约束、验收判据与方案否决；仓库里的脚本、测试与证据是用来**验收**这些东西的
> （详见[§1.2](#12-角色与我的职责)）。协作与验收方法见[§12](#12-工程规范测试与判据纪律)。

---

## 目录

| 节 | 内容 |
|:---|:---|
| [1](#1-项目概览与角色) | 项目概览与角色（状态矩阵：什么真跑过、什么没跑过） |
| [2](#2-实测指标与口径基准) | **实测指标与口径基准**（精度 / 分层 / 阈值 / 四个速度口径 / 导出变体） |
| [3](#3-架构与代码组织) | 架构与代码组织（设计原则、模块、依赖方向、运行时线程模型、决策与代价） |
| [4](#4-环境与安装) | 环境与安装（含 OpenCV 三包共存、显存、中文路径三个真实坑） |
| [5](#5-快速开始) | 快速开始（四种检测输入、皮套桥接、三条出口自检、常见问题） |
| [6](#6-数据管线) | 数据管线 |
| [7](#7-训练) | 训练（超参、时长、收敛判断、踩坑与修复） |
| [8](#8-评估与误差分析) | 评估与误差分析 |
| [9](#9-导出与基准) | 导出与基准 |
| [10](#10-实时表情捕捉管线) | 实时表情捕捉管线（主线：先检测再裁剪、滤波、校准、稳定性） |
| [11](#11-另外两条出口vmcosc-与-unity-udp) | 另外两条出口：VMC/OSC 与 Unity UDP |
| [12](#12-工程规范测试与判据纪律) | 工程规范、测试与判据纪律 |
| [13](#13-被否决的结论不值得再试一遍) | 被否决的结论（不值得再试一遍） |
| [14](#14-已知限制与未完成项) | 已知限制与未完成项 |
| [15](#15-复现命令与证据索引) | 复现命令与证据索引 |
| [16](#16-文档导航与许可) | 文档导航与许可 |

---

## 1. 项目概览与角色

### 1.1 一句话与状态矩阵

| 层 | 内容 | 状态 |
|:---|:---|:---|
| **检测模型** | YOLOv8n 在 WIDER Face 上单类微调（v1 基线 100 epoch + v2 微调 30 epoch），3.011 M 参数 / 5.95 MiB | ✅ 已训练、已评估、已导出 |
| **部署推理** | `src/deploy/detect.py`：摄像头 / 视频 / 单图 / 文件夹四种输入；PyTorch `.pt`（CUDA + FP16）与 ONNX FP16 两个运行时 | ✅ 可运行 |
| **表情捕捉** | 自训练检测器定位人脸 → 裁剪 ROI → MediaPipe Face Landmarker（478 关键点 + 52 ARKit blendshape）→ 时序滤波 → 参数映射与校准 | ✅ 已实现并实测 |
| **出口 ①：VTube Studio（主线）** | WebSocket Plugin API 注入 Live2D 参数，16 个参数 | ★ **端到端已跑通**，连续 **599.7 s** 稳定性实测通过 |
| **出口 ②：VMC/OSC（3D 备选）** | OSC over UDP → VSeeFace + VRM | ⚠️ 发送端已实测（回环 + 编码自检 + 一键冒烟），**未与真机 VSeeFace 联调** |
| **出口 ③：Unity UDP（目标线）** | UDP JSON（16 参数 + `seq`）→ Unity C# 接收端 → 占位物体 | ⚠️ 发送端已实测；**C# 侧只过"桩编译"静态检查，未在 Unity 中编译或运行** |
| **工程** | `src/` 五层 + 3 个独立应用；**155 个单元测试**（`unittest`，零额外依赖） | ✅ 本轮实跑全绿 |

**明确没做 / 没测**：端到端延迟（未实测，因此**不给毫秒数**）· 真机 VSeeFace 联通 · Unity 内编译与运行 ·
WIDER 官方 Easy/Medium/Hard 分项评估（依赖包在本机装不上）· 自采 500 张侧脸/遮挡/暗光数据 · 演示视频。

三条出口复用**同一条**捕捉管线，只是挂不同的 sink（`--sink vts` / `vmc` / `unity`，可逗号组合），
**没有复制任何感知代码** —— 这也是"可插拔出口"从主张变成证据的地方（见[§11](#11-另外两条出口vmcosc-与-unity-udp)）。

### 1.2 角色与我的职责

| 归属 | 我负责什么 | 落地物 |
|:---|:---|:---|
| **技术选型** | 检测方案（用成熟实现微调而非自建）、运行时主路径（PyTorch/CUDA）与可移植路径（ONNX FP16）、三条出口的协议选择 | `configs/model.yaml`、`src/deploy/export_onnx.py`、`apps/` |
| **架构设计** | `src/` 五层与严格单向依赖、`src/paths.py` 作为唯一路径来源、`apps/` 作为可独立交付的应用、感知与出口解耦 | `src/paths.py`、[§3](#3-架构与代码组织) |
| **验收判据** | 指标必须标口径；精度验收用 **mAP** 而非张量 MAE；"未测"与"通过"在代码层分开；把判据固化成带退出码的工具 | `compare_export_variants.py`、`sample_vts_params.py`、`tests/` |
| **取舍与否决** | 小脸裁剪增强经 A/B 否决、batch 按显存实测下调、重复的标注解析收敛为一份、移除对 UDP 无意义的限频 | [§13](#13-被否决的结论不值得再试一遍) |

实现环节由 AI 辅助完成；**我修改、修复、否决的东西几乎都不是"能不能跑"，而是"结论是不是真的"** ——
判据有没有在伪造通过、指标口径可不可比、参数是不是真的生效、数字有没有被测量污染（见[§12.3](#123-三次同类失误都是取数姿势的问题)）。

---

## 2. 实测指标与口径基准

> **本节是全仓库的口径基准。** 任何数字引用前先看这里属于哪个口径；**不同口径不可交叉引用**。

### 2.1 检测精度（WIDER Face 验证集，3,226 张）

| 指标 | v1 基线（100 epoch） | v2 微调（+30 epoch） |
|:---|---:|---:|
| mAP50 | **0.66501** | 0.66404 |
| mAP50-95 | 0.35545 | **0.35618** |
| Precision | 0.849 | 0.846 |
| Recall | 0.595 | 0.595 |

来源：`artifacts/reports/metrics_v1.json` / `metrics_v2.json`，`src/eval/evaluate.py` 在 **best 权重**上重测。

> **口径（重要）**：上表是**全部 3,226 张验证图**按 COCO 方式算的 AP50，分母包含大量极小尺寸人脸。
> 历史设计文档里的目标 `0.76` 指的是 WIDER **Hard 子集**的 VOC AP —— **两者口径不同，不能对账**。
> 官方 Easy/Medium/Hard 分项需要 `widerface-evaluate`，**本机网络不可用装不上**；
> `evaluate.py` 在该包缺失时把三个字段报 `null`（**不再用 mAP50 冒充分项结果**），替代分析见 2.2。
>
> **v2 与 v1 基本持平**：mAP50 −0.001、mAP50-95 +0.0007，属噪声量级。v2 只是在已收敛点附近做 30 epoch
> 低学习率微调，数据分布没变，**没有增益也不意外**；真正的闭环需要困难样本回流，这一步没有做。
>
> 注：训练日志 `results.csv` 末行是 0.66349 / 0.66171 —— 那是**最后一个 epoch** 的值，
> 与保存下来的 best 权重不是同一组参数。

### 2.2 按人脸尺寸分层（同一组权重，39,708 张标注人脸）

| 尺寸桶 | GT 人脸数 | 命中 | 召回 | 占全部 GT |
|:---|---:|---:|---:|---:|
| **Tiny（<32px）** | **28,775** | 14,186 | **0.493** | **72.5%** |
| Small（32–96px） | 8,606 | 7,780 | 0.904 | 21.7% |
| Medium（96–256px） | 1,890 | 1,805 | 0.955 | 4.8% |
| Large（>256px） | 437 | 432 | **0.989** | 1.1% |
| **合计** | **39,708** | 24,203 | **0.610** | 100% |

（预测 29,785 个，其中 5,582 个为误检 → 精确率 0.813；来源 `artifacts/reports/error_analysis_summary.txt`）

> **这是本项目最重要的一条结论**：汇总召回 0.610 看着偏低，但 **72.5% 的标注人脸小于 32 像素**
> （人群远景里的脸），模型在这些极小脸上召回仅 0.493。**排除极小脸后，≥32px 人脸的召回是 0.916。**
>
> 也就是说：分数低主要是**基准集本身极端**，而非模型在实际场景不可用。虚拟形象驱动场景中人脸占据
> 画面主要位置（Medium/Large 桶，召回 0.955–0.989）。这也解释了原始 `0.76` 目标为何达不成 ——
> Hard 子集几乎全是极小脸，**目标设定与实际应用场景错配**。
>
> 口径提醒：此处 **0.610** 与 2.1 表中 ultralytics 报的 **0.595 口径不同**
> （前者 = IoU≥0.5 贪心匹配 + `conf=0.25`；后者 = ultralytics 自身置信度扫描）。
> 两个数都真实，**不可互相替换引用**。

### 2.3 置信度阈值的影响（`python -m src.eval.conf_sweep`）

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

> **结论：默认 `conf=0.25` 几乎正好在最优 F1 点上**（max-F1 点为 `conf=0.297`，F1 0.6985 vs 0.6971，
> 仅差 **0.0014**）→ **阈值不是瓶颈**。此前文档里"`conf=0.25` 对召回不友好"的说法**经实测被推翻**。
>
> - **AP50 与阈值无关**（始终 0.66404）—— 阈值只是在同一根 P-R 曲线上选工作点。
> - 降到 0.10：召回 **+0.065**，精确率 **−0.233**（代价很大）。对"宁可多检也不漏检"的场景，
>   0.15–0.20 是合理工作点；本项目**未改默认阈值**，因为默认值本身已接近最优 F1。

### 2.4 速度：**四个口径，必须说清是哪一个**

| # | 口径 | 含义 | 实测 |
|:--:|:---|:---|:---|
| ① | **含前后处理的单帧吞吐** | `model(frame)` = letterbox + 前向 + 解码 + NMS，即应用真正付出的代价 | **PyTorch/CUDA 11.4 ms ≈ 87.8 FPS**；ONNX FP32 27.7 ms ≈ 36.1 FPS；ONNX FP16 27.9 ms ≈ 35.8 FPS |
| ② | **裸网络前向** | `model.model(dummy)`，**不含**前后处理 | PyTorch/CUDA **10.31 ms ≈ 97.0 FPS**；ONNX/CPU 24.80 ms ≈ 40.3 FPS（`src/deploy/benchmark.py`） |
| ③ | **管线循环帧率** | 桥接主循环（取帧 + 绘制 + 网络 I/O） | 摄像头路径 **29.3 FPS**；视频路径 **38.0 FPS**；VMC 发送率 40.2 FPS |
| ④ | **端到端延迟** | 采集 → 形象响应 | **未实测**（缺同步外部刺激，故不给毫秒数） |

- 口径 ① 来源：`artifacts/logs/export_variant_comparison.log`（同一次运行的汇总表，`rc=0`）；
  口径 ② 来源：`src/deploy/benchmark.py` 的 50 次预热 + 200 次计时。
- 设备：**RTX 3060 Laptop 6 GB**，Windows 11，PyTorch 2.5.1+cu121。
- ⚠️ **测量噪声很大，单次读数不可作为结论**：本机同一调用在不同轮次实测 **4.8–18.8 ms（跨度约 3 倍）**。
  因此 `benchmark.py` 报的是**中位数 + 最小值 + p95 + 最大值 + 样本数**；引用速度时必须带统计量与样本数。
- 跳帧实测（视频路径，基线 38.0 FPS）：每 2 帧跑一次 → **49.2 FPS**；每 3 帧 → **52.0 FPS**；
  **摄像头路径无提升**（瓶颈是采集侧 30 FPS 硬上限，见 10.4）。
- 桥接单帧分段（摄像头路径，单次分段读数）：`read 15.8 + yolo 7.8 + mediapipe 8.4 = 32.6 ms → 29.3 FPS`。

### 2.5 导出变体对比（模型体积 vs 检测质量）

`python src/deploy/compare_export_variants.py`：三个变体各跑**同一套 3,226 张验证集**的 mAP，
吞吐按口径 ① 计（含 letterbox + 解码 + NMS）：

| 变体 | mAP50 | mAP50-95 | 吞吐 | 运行设备 | 体积 |
|:---|---:|---:|---:|:---|---:|
| **PyTorch `.pt`** | 0.66404 | 0.35618 | **11.4 ms / 87.8 FPS** | **CUDA（GPU）** | 5.95 MiB |
| ONNX **FP16** | 0.66382 | 0.35627 | 27.9 ms / 35.8 FPS | CPU | **5.88 MiB** |
| ONNX FP32 | 0.66379 | 0.35670 | 27.7 ms / 36.1 FPS | CPU | 11.70 MiB |

> **FP16 的取舍用实测解决**：相对 PyTorch 的 mAP50 差 **−0.0002**、mAP50-95 差 **+0.0001**，
> 即**检测质量无差别**；相对 FP32 体积减半、CPU 吞吐相同。因此导出用 **FP16**
> （`export_onnx.py` 的 `EXPORT_HALF = True`），项目"≤10 MB"目标随之重新满足。
>
> ⚠️ **ONNX 在本机只能跑 CPU**：`onnxruntime` 1.27 实际只提供 `CPUExecutionProvider` 与
> `AzureExecutionProvider`，**没有 CUDA**；装 `onnxruntime-gpu` 需要网络，本机不可用。
> 所以 ONNX 比 PyTorch 慢约 **2.4 倍** —— 它在本机的价值是**可移植性**，不是速度。
> `benchmark.py` 与 ultralytics 现在都会**打印实际生效的 provider**，不再静默降级。
>
> ⚠️ **精度验证判据已修正**：张量 MAE 门限按精度区分（FP32 `1e-4` / FP16 `1e-1`，见 `MAE_TOLERANCE`），
> 实测 FP32 MAE = **0.000015**、FP16 = **0.050635**。但**张量 MAE 比的是浮点表示而非模型行为** ——
> FP16 下 8400 个候选框坐标只保留约 3 位有效数字，单个坐标从 `320.12345` 漂到 `320.1` 就贡献约 2e-2。
> **真正的验收判据是 mAP**（上表），MAE 只作参考。原判据"逐元素 MAE < 1e-4"对 FP16 **不可能通过**，
> 曾迫使脚本用 `half=False` 规避，结果文件翻倍到 11.70 MiB、还破了体积目标。

### 2.6 口径规则（写死在项目约定里）

1. 任何指标必须标注**测量口径**：数据集范围、计算方式、是否含前后处理、运行设备、统计量。
2. **禁止把口径不同的数字并列比较**（例：全验证集 COCO 式 AP50 与 Hard 子集 VOC AP；0.610 与 0.595）。
3. 未实测的数字不得写入文档；**跳过验证时不得输出"通过"**。
4. 测量对象的状态要先确认再取数 —— **空结果不是否定证据**（见[§12.3](#123-三次同类失误都是取数姿势的问题)）。

---

## 3. 架构与代码组织

### 3.1 三条设计原则

**1. 代码 / 数据 / 产出三者分离** —— `src/`、`data/`、`artifacts/` 互不混放：
代码进版本库、数据体积大可重新获取、产出可随时重建，三者生命周期不同。

**2. 路径只有一处定义** —— 所有路径必须从 `src/paths.py` 导入，**禁止**任何模块用 `__file__` 自行推算项目根。

> **这是踩过并解决的真实问题**：项目早期每个脚本写
> `os.path.dirname(os.path.dirname(__file__))` 求项目根，在扁平目录下成立；一旦脚本进入子包
> （如 `src/deploy/detect.py`），同一表达式会算出 `src/` 而不是项目根，**所有数据与权重路径同时失效**。
> 把根目录解析收敛到唯一一处之后，**脚本可以自由移动**。

**3. 依赖严格单向**：

```
deploy  →  eval  →  train  →  data  →  paths
```

反向依赖一律不允许：数据管线不知道训练代码的存在，训练不依赖部署代码，`paths.py` 不依赖任何模块。
好处是可以单独测试或重跑任一环节，而不必把整条链路拖起来。

### 3.2 模块划分

| 模块 | 目录 | 职责 | 关键文件 |
|:---|:---|:---|:---|
| **路径基础** | `src/paths.py` | 全项目唯一的路径来源 | — |
| **数据管线** | `src/data/` | 下载指引、标注质检、WIDER 标注解析（**唯一实现**）、YOLO 格式转换、train/val 划分、数据集封装、增强可视化 | `wider_annotations.py`、`convert.py` |
| **训练** | `src/train/` | 超参管理、v1 基线 + v2 微调、断点续训、模型诊断、显存与增强探针 | `train.py`、`resume.py`、`probe_vram.py` |
| **评估** | `src/eval/` | mAP + P-R 曲线 + 分项（缺包报 `null`）、尺寸分层误差分析、阈值扫描 | `evaluate.py`、`analyze_errors.py`、`conf_sweep.py` |
| **部署** | `src/deploy/` | 实时检测（四种输入）、ONNX 导出与精度验证、裸前向基准、变体对比 | `detect.py`、`export_onnx.py` |
| **应用** | `apps/vtube_bridge/`、`apps/vmc_link/`、`apps/unity_link/` | 三条出口（可独立交付） | `vtube_studio_bridge.py` |
| **配置 / 产出** | `configs/model.yaml`、`artifacts/` | 训练超参；权重、报告、诊断证据、日志 | — |

**`src/` 与 `apps/` 的区别**：`src/` 是可被导入的库代码，脚本以 `python -m src.<pkg>.<module>` 运行
（导入关系显式、工作目录固定为项目根）；`apps/` 是可独立运行的完整应用 —— 它自带第三方模型文件与可选 GUI，
对外只需要一个入口。

**已知耦合（保留但登记）**：`apps/vtube_bridge` 不依赖 `src/`（模型路径解析自成一套，好处是可单独交付）；
`src/data/loader.py` 为孤立模块（供将来接入训练，当前无人引用）。

### 3.3 数据流（离线）

```
WIDER Face 原始数据（data/raw/、data/annotations/）        ← 不在仓库内，自行下载（约 3 GB）
        │
        ▼  src/data/wider_annotations.py   标注解析唯一实现
        ▼  src/data/convert.py             标注 → YOLO 归一化 labels/（与 images/ 平行）
        ▼  src/data/split.py               生成 train_list.txt / val_list.txt
        ▼  train.py 动态生成 data/annotations/widerface.yaml（path: data/raw，相对项目根）
        ▼  src/train/train.py              ultralytics 直接读 images/ ↔ labels/
        ▼  artifacts/checkpoints/best_model_v1.pt →（再 30 epoch）→ best_model_v2.pt
        │
        ├─▶ src/eval/*                      mAP、P-R、尺寸分层、阈值扫描 → artifacts/reports/
        └─▶ src/deploy/export_onnx.py       → best_model_v2.onnx（FP16 + simplify）
                    │
                    ▼  src/deploy/detect.py      实时检测
                    ▼  apps/vtube_bridge/         表情驱动 Live2D
```

> **一个容易混淆的点**：`train_list.txt` / `val_list.txt` 与训练的**数据加载**无关 ——
> ultralytics 依据 `images/` 与 `labels/` 的同构目录直接扫描。这两个清单只被**评估**与
> **ONNX 精度验证**使用；缺了它们训练照常，但评估环节会降级（现在都**明确打印** `[WARN]` / `[SKIPPED]`，不静默）。

### 3.4 运行时线程模型

```
主线程：取帧 → 检测 → 关键点 → 滤波 → 校准 → 产出参数          只做这些，保证帧预算稳定
   │
   ├─ daemon 线程：VTS WebSocket 发送（30 FPS 注入、断线 10 s 自动重连）  ← 网络 I/O 不阻塞主循环
   └─ 旁路：Qt 调试面板（16 参数实时显示 / 摄像头预览 / 点击选脸）        ← 可选依赖，本机未装
```

### 3.5 关键设计决策与代价

| 决策 | 理由 | 代价 |
|:---|:---|:---|
| 路径统一由 `src/paths.py` 提供 | 脚本可自由移动，不再依赖自身目录深度 | 少量间接层，需记住常量名 |
| 脚本以 `python -m src.<pkg>.<module>` 运行 | 导入关系显式，工作目录固定为项目根 | 不能再直接 `python path/to/script.py` |
| 桥接保持为独立应用（`apps/`），不并入 `src/` | 自带 MediaPipe 模型与可选 GUI，耦合低、可单独交付 | 路径解析方式与主项目不统一 |
| **先 YOLO 检测、再裁剪 ROI** 送关键点模型 | 人脸在输入中占比大幅提升，关键点与表情系数更准 | 多一次裁剪开销，外扩比例需调（1.45 为经验值） |
| 人脸框三级滤波（EMA + 中值 + 保持帧） | 抗抖动、抗单帧跳变、短暂遮挡不闪烁 | 引入可控延迟，需与响应速度权衡 |
| VTS 发送放在 daemon 线程 | 网络 I/O 不阻塞推理主循环 | 需处理线程间状态同步与断线重连 |
| ONNX 静态 shape 导出（`dynamic=False`） | 部署固定 640×640，静态图可获得更充分的图优化 | 无法变分辨率推理 |
| 检测框架选 YOLOv8n | 3.011 M 参数 / 5.95 MiB，6 GB 卡可训，实时性有富余；单类密集小目标任务下更大模型的收益远小于提高分辨率 | 极小脸召回弱（0.493，见 2.2） |
| `nc=1` 单类 | 只检人脸，head 分类分支从 80 类降到 1 类 | — |

### 3.6 扩展点

- **换检测模型**：`configs/model.yaml` 的 `model_name` 与 `src/deploy/detect.py` 的 `DEFAULT_MODEL` 两处。
- **换推理后端**：`detect.py` 通过 ultralytics 加载，换成 `.onnx` 只需改 `--model`。
- **加表情维度**：MediaPipe 已提供全部 52 个 ARKit blendshape，在 `estimate_expressions` 里补映射即可（当前精选 6 个维度）。
- **加数据源**：在 `src/data/` 加转换脚本产出 YOLO 格式 labels 即可，训练侧无需改动。
- **加出口**：实现一个新的 sink 挂到同一条管线（已有 vts / vmc / unity 三种范式可参照）。

---

## 4. 环境与安装

### 4.1 实测环境

| 项 | 实测值 |
|:---|:---|
| 操作系统 | Windows 11（26200），设备 Dell G15 5520 |
| Python | 3.12.8 |
| PyTorch | 2.5.1+cu121 |
| GPU | NVIDIA GeForce RTX 3060 Laptop，**6144 MiB（6 GB）** |
| 驱动 / CUDA | 581.95 / CUDA 12.1 |
| 摄像头 | 内置 Integrated Webcam，**硬上限 30 帧/秒**（见 10.4） |

```bash
pip install -r requirements.txt
```

### 4.2 三个必须先知道的坑

**① OpenCV 三个分发包共用同一个 `cv2/`，最后安装者生效** ——
`opencv-python` / `-headless` / `-contrib` 都往 `site-packages/cv2/` 里写。
本机实际生效的是 **opencv-contrib-python 5.0.0.93**（`cv2.__version__` = 5.0.0，而 `pip show opencv-python` 仍报 4.13.0.92），
因此 `requirements.txt` 声明 `opencv-contrib-python`。
⚠️ **绝不能让 `opencv-python-headless` 成为最后安装的那个** —— 它没有 `imshow` / `waitKey` / `destroyAllWindows`，
而 `src/deploy/detect.py` 用到它们，实时预览会直接崩，且报错只指向 `cv2.imshow`、完全不提安装顺序。
护栏：`tests/test_opencv_environment.py`（被 headless 覆盖时**变红，不会 skip**）。

**② 6 GB 显存下的 batch 选择** —— `batch=8 + imgsz=640` 的峰值**保留**显存达 **6.18 GiB**，
已超过 6.00 GiB 物理上限，从第一轮起向共享内存溢出；100 epoch 长跑会因内存碎片继续增高
（历史日志达 **11.1 GiB**），吞吐从 **5.8 it/s 崩到 1.7 it/s**。**配置已下调为 `batch=6`**（峰值保留 4.41 GiB，留 27% 余量）。
实测脚本：`python src/train/probe_vram.py`。

**③ 中文路径** —— Windows 下 `cv2.imread` 遇到非 ASCII 路径会**静默返回 `None`**，
项目统一用 `imdecode` 兜底（`src/train/train.py`、`src/data/convert.py`）；
MediaPipe 的 C++ 层按系统区域设置打开文件，非 ASCII 路径同样失效，`face_landmarker.py` 的 `_safe_path()`
会把模型复制到纯 ASCII 临时目录。

### 4.3 需要自己准备的三份大文件（**均不在仓库内**）

| 资源 | 大小 | 用途 | 获取方式 |
|:---|:---|:---|:---|
| WIDER Face 数据集 | ≈3 GB | 训练与评估 | `python -m src.data.download` **只打印手动下载指引**（脚本不执行下载），解压到 `data/raw/` |
| `face_landmarker.task` | 3.7 MB | MediaPipe 关键点与表情系数 | 见 `apps/vtube_bridge/thirdparty/MediaPipe/README.md` 里的一条 `Invoke-WebRequest` 命令，存到 `thirdparty/MediaPipe/models/`（该路径已被 `.gitignore` 的 `models/*.task` 排除） |
| `yolov8n.pt` | 6.2 MB | 训练起点（COCO 预训练权重） | ultralytics 首次训练时自动下载，或手动放入项目根目录 |

> 缺少 MediaPipe 模型时，桥接以明确的 `FileNotFoundError` 失败（**不会静默降级**）。

### 4.4 摄像头相关的实测约束

- 摄像头设备**只在交互式用户会话里可用**。受限执行环境（自动化助手 shell）中 `Get-PnpDevice` 返回 0 个设备、
  `Get-CimInstance` 拒绝访问 —— **空结果不是否定证据**。
  → **凡是要打开摄像头的命令，必须由使用者本人在自己的终端运行。**
- 没有摄像头也能跑全链路：`--input` 同时接受**视频文件路径**，仓库根目录随附演示视频
  `7f62f96bca5cffdfe2e0167bf3de3169.mp4`（586 KiB，已入库），放完自动退出、退出码 0，
  因此可作无人值守的回归冒烟。

---

## 5. 快速开始

所有脚本都从**项目根目录**以模块方式运行；路径一律经 `src/paths.py` 解析。

### 5.1 实时摄像头检测

```bash
python -m src.deploy.detect --input 0        # 按 q 退出，按 s 截图
```

### 5.2 视频 / 单图 / 文件夹

```bash
python -m src.deploy.detect --input video.mp4 --save output.mp4
python -m src.deploy.detect --input photo.jpg --save results/
python -m src.deploy.detect --input my_photos/ --save results/
python -m src.deploy.detect --input photo.jpg --no-show      # 不弹结果窗口（脚本/批处理/CI 用）
```

参数：`--input`（摄像头序号 / 视频 / 图片 / 文件夹）｜`--model`｜`--imgsz`（默认 640）｜
`--conf`（默认 0.25）｜`--save`｜`--no-show`。

> ⚠️ 单图模式默认**弹窗并等待关闭**；脚本场景务必加 `--no-show`，否则命令不返回。

### 5.3 表情驱动：VTube Studio 主线

```bash
# 前置：VTS 中开启「启动插件 API」并加载模型（如自带 hiyori）
#       首次运行会弹授权窗，有 120 秒时间点"允许"
python apps/vtube_bridge/main.py --input 0 --sink vts --no-gui

# 没有摄像头：用随仓库的演示视频，放完自动结束
python apps/vtube_bridge/main.py --input 7f62f96bca5cffdfe2e0167bf3de3169.mp4 --sink vts --no-gui
```

> `--no-gui` 必须带：**本机未安装 PyQt5**，不加会因缺 Qt 直接报错退出。
> PyQt5 **不在依赖清单内** —— 它只服务于内部调试面板，且该面板在本机从未运行过。

### 5.4 三条出口各自的"不装任何东西"自检

```bash
python apps/vmc_link/selfcheck.py                # VMC/OSC 编码逐字节是否符合规范
python apps/unity_link/selfcheck.py              # Unity UDP 字段完整性 + 序号单调
python apps/vmc_link/e2e_check.py                # 一条命令跑完整链路：真桥接 + 真视频 + UDP 收包
python apps/unity_link/mock_receiver.py          # 假 Unity：不装 Unity 也能验证发送端
python apps/vtube_bridge/sample_vts_params.py    # VTS 连续稳定性采样（默认 600 s，退出码 0/2）
```

### 5.5 常见问题

| 症状 | 处理 |
|:---|:---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| 单图检测跑完不返回 | 加 `--no-show` |
| 摄像头打不开 | 换 `--input 1` / `--input 2`；在"相机"应用确认设备可用 |
| 检测速度很慢 | `python -c "import torch; print(torch.cuda.is_available())"` —— 若为 `False`，装了 CPU 版 PyTorch |
| ONNX 推理报错 | `pip uninstall onnxruntime-gpu && pip install onnxruntime` |
| `python src/deploy/detect.py` 报导入错误 | 必须以模块方式运行：`python -m src.deploy.detect`，且在项目根目录 |
| 训练显存不足 / 速度骤降 | 6 GB 卡上 `batch=8` 已超额分配；把 `configs/model.yaml` 的 batch 降到 4，或 `imgsz` 降到 512 |
| 实时预览崩溃、报 `cv2.imshow` 不存在 | `cv2` 被 headless 版覆盖了 —— 见 4.2 ①，重装 `opencv-contrib-python` |
| 桥接报缺 `face_landmarker.task` | 见 4.3，按 README 里的命令下载到 `thirdparty/MediaPipe/models/` |
| 首次连接 VTS 一直失败 | 授权弹窗需要在 **120 秒内**点"允许"（旧版此处有缺陷，见 10.5） |

---

## 6. 数据管线

```bash
python -m src.data.download     # 打印 WIDER Face 手动下载指引（约 3 GB，需自行下载）
python -m src.data.qc           # 标注质检 → data/annotations/quality_report.txt
python -m src.data.convert      # 标注 → YOLO labels/（训练真正依赖的产物）
python -m src.data.split        # 生成 train_list.txt / val_list.txt
```

数据集规模（实测清单行数）：训练 **12,880** 张、验证 **3,226** 张，沿用 WIDER Face 官方划分。

> **`*_list.txt` 不在仓库里**（`.gitignore` 排除了 `data/annotations/*_list.txt`）—— 它们存的是
> **本机绝对路径**，换机器必然失效。**新环境第一步就是 `python -m src.data.split` 重新生成**，
> 否则依赖它的三个脚本会走各自的降级分支（见[§8](#8-评估与误差分析)）。
>
> **WIDER 标注解析已收敛为唯一实现** `src/data/wider_annotations.py`（原 `convert.py` / `qc.py` /
> `split.py` / `loader.py` 各写了一遍）。收敛的原因是一个真实缺陷：边界情况
> **`num_faces=0` 却带全零假框**只在 `convert.py` 里被处理，其余三份都漏 —— `split.py` 因此在假框行上崩，
> **清单从未成功生成**，连锁让 ONNX 精度验证与分项评估静默失效。
>
> WIDER 标注里每张脸还带 `blur / expression / illumination / occlusion / pose` 五个困难属性，
> 当前解析时**被跳过、未用于训练** —— 这是"属性感知增强"方向的前提（见[§14](#14-已知限制与未完成项)）。
> 设计文档中承诺的"自采 500 张侧脸/遮挡/暗光数据"**未采集**，最终只使用公开数据集。

---

## 7. 训练

```bash
python -m src.train.train        # v1 基线 100 epoch + v2 微调 30 epoch
python -m src.train.resume       # 断点续训（从 artifacts/checkpoints/*/weights/last.pt 接续）
tensorboard --logdir artifacts/checkpoints --port 6006
```

### 7.1 流程与超参

```
① 生成 data/annotations/widerface.yaml（UTF-8，path 相对项目根；nc=1，names: {0: face}）
② v1：YOLO("yolov8n.pt") → train(100 epoch, batch 8, imgsz 640)  历史配置
③ 拷贝 v1_baseline/weights/best.pt → best_model_v1.pt
④ v2：YOLO("best_model_v1.pt") → train(30 epoch, lr0=1e-4, mosaic 0.5)
⑤ 拷贝 v2_finetune/weights/best.pt → best_model_v2.pt
```

| 超参 | 值 | 依据 |
|:---|:---|:---|
| batch / imgsz | 历史训练 **8 / 640**；配置已改 **6 / 640** | 显存探针实测：4 → 3.58 GiB、6 → 4.41 GiB、**8 → 6.18 GiB 超 6.00 GiB 上限**（见 4.2 ②） |
| optimizer | AdamW（`momentum` 0.937、`weight_decay` 0.0005） | ultralytics 对检测任务的稳定默认 |
| lr0 / lrf | v1 **1e-3**；v2 **1e-4**；余弦退火到 `lr0×1e-5` | v2 在已收敛点附近微调，学习率降到 1/10 |
| warmup | v1 3 epoch；v2 1 epoch | v2 无需长 warmup |
| 增强 | `mosaic=1.0`（v2 → 0.5）、`hsv_*`、`fliplr=0.5`、`translate=0.1`、`scale=0.5`、`close_mosaic=10` | **ultralytics 内置增强**；`src/data/augment.py` 的 Albumentations 管线**不参与训练**（见 §13） |
| 其他 | `amp=true`、`workers=2`、`seed=0`、`deterministic=true`、`nbs=64` | 混合精度省显存；固定种子可复现 |
| 微调方式 | **全参数微调（`freeze: null`）**，仅重建检测头（80 类 → 1 类） | WIDER 与 COCO 域差异大，冻结底层不利于适应人脸纹理；12,880 张足以支撑全网络微调 |

**loss（ultralytics 默认，未改权重）**：分类 BCE（`cls=0.5`）、回归 CIoU（`box=7.5`）+ **DFL**（`dfl=1.5`），
标签分配用 **TaskAlignedAssigner**（topk=10）。`nbs=64` 在 batch<64 时按比例缩放 loss 并**隐式做梯度累积**
（batch=8 等效累积 8 步）—— 配置里的 `accumulation_steps` **从未被读取**（见 §14 第 14 条）。

**时长（实测）**：v1 累计 **6,544.6 s ≈ 1.82 h**；v2 累计 **7,281–7,282 s ≈ 2.02 h**。
（v2 的 30 epoch 比 v1 的 100 epoch 还慢，因为 v2 全程开验证，而 v1 前 60 epoch 没验。）

### 7.2 怎么判断收敛、有没有过拟合

**收敛看 mAP 曲线的平台位置**：`results.csv` 里**第一个有验证指标的 epoch（64）mAP50 已达 0.66259，
到 epoch 100 只有 0.66349 —— 最后 36 个 epoch 合计 +0.0009**，模型在早期就已收敛。
这同时解释了 v2 为什么没有增益（在已饱和点上再微调，数据分布没变）。
早停在该配置下不会触发（`patience=100` 等于 epoch 数）。

**过拟合：没有观察到**（依据是两侧 loss 的贴合）：末期 `train/box 1.376` vs `val/box 1.414`、
`train/cls 0.636` vs `val/cls 0.627`（验证侧反而更低）、`dfl 0.951` vs `0.975`，
没有出现"训练 loss 持续下降、验证 loss 抬头"的形态。300 万参数的小模型 + 12,880 张 + 一整套在线增强，
过拟合风险本就低。**真正的短板是极小脸上的欠拟合**（召回 0.493，见 2.2）。

### 7.3 训练中踩的坑（三段式）

> ① 早期 val 指标全为 0（`results.csv` 首行 `val/box_loss=0`）—— 先让训练跑起来，用了 `val=False`；
> ② 定位到 val split 数据缺失，重下 3,226 张；
> ③ 用 `resume=True` 从 `last.pt` **断点续训**补到 100 epoch。
>
> 代价要说清：`val=False` 让 epoch 1–60 没有验证指标（`results.csv` 里全为 0），
> **best 权重的挑选依据因此不可靠** —— 这是"如果重做第一个要改的地方"（见 §14 建议）。

---

## 8. 评估与误差分析

```bash
python -m src.data.split          # 前置：生成 val_list.txt（缺失时下面几步会降级）
python -m src.eval.evaluate       # mAP + Easy/Medium/Hard 分项 + P-R 曲线
python -m src.eval.analyze_errors # 按尺寸分层的召回/精确率 + TOP20 漏检/误检 + 可视化
python -m src.eval.conf_sweep     # 置信度阈值扫描
```

依赖 `val_list.txt` 的脚本，其降级行为（**都是明确打印，不静默**）：

| 脚本 | 前置缺失时的行为 |
|:---|:---|
| `evaluate.py` | 打印 `[WARN] val_list.txt not found, skipping prediction generation`；官方分项字段报 `null`；缺 P-R 曲线时打印 `[WARN]` 而非静默无操作 |
| `analyze_errors.py` | 依赖同一清单；缺失时无法产出尺寸分层表 |
| `export_onnx.py` | 打印 `[SKIPPED] ... No MAE was measured -- do NOT report a precision-loss figure.`，返回 `None`，并**明确说明"这不是通过"** |

> **判据设计原则：把"没测"和"通过"分开。** 早期版本在这两处的表现分别是
> "用 mAP50 冒充分项指标"和"缺前置文件直接 `return True`"，两处都已修掉。
> 另一处修掉的同类问题：`evaluate.py` 查找 `PR_curve.png`，而 ultralytics 8.4 实际写的是
> `BoxPR_curve.png`，且 `if os.path.exists` 没有 else 分支 → **静默无操作**。

分析结果（尺寸分层表、阈值扫描）见[§2.2](#22-按人脸尺寸分层同一组权重39708-张标注人脸)与
[§2.3](#23-置信度阈值的影响python--msrcevalconf_sweep)。

---

## 9. 导出与基准

```bash
python -m src.deploy.export_onnx                  # ONNX 导出（FP16 + simplify + opset 12 + 静态 shape）+ 精度验证
python -m src.deploy.benchmark                    # 裸网络前向延迟与体积（口径 ②）
python src/deploy/compare_export_variants.py      # 三变体 mAP + 端到端吞吐 + 体积（口径 ①，决策依据）
```

导出产物落在 `artifacts/checkpoints/best_model_v2.onnx`（5.88 MiB），与 `.pt` 同级，已是桥接的可选运行时。

**导出的接缝（实测）**：

| 项 | 实测值 |
|:---|:---|
| 输入 | `images`，`FLOAT [1, 3, 640, 640]` |
| 输出 | `output0`，`FLOAT [1, 5, 8400]`（5 = `cx, cy, w, h` + 1 个类别分数） |
| 权重精度 | 图内 132 个 initializer 为 FLOAT16，1 个 FLOAT，8 个 INT64 |
| 配置 | `opset 12`、`dynamic=False`、`simplify=True`、`half=True` |

> **裸 head 是 65 通道，导出的图是 5 通道** —— 差别在于导出把 **DFL 的解码折进了图里**，
> 而 **NMS 留在图外**由 Python 侧执行（8400 个候选 → NMS → 最终框 + 置信度）。
> 模型**输出不含关键点**：检测要的是"脸在哪"，关键点要的是"脸怎么动"，后者交给 MediaPipe，
> 且 ARKit blendshape 是行业标准格式，自训多任务模型还要重做映射、对皮套驱动没有额外收益。

---

## 10. 实时表情捕捉管线

### 10.1 数据流

```
摄像头帧
   │
   ▼ ① YOLOv8n-face 检测人脸框                                   实测 7.8 ms（GPU + FP16）
   ▼ ② FaceBoxFilter：EMA(α=0.55) + 中值(window=5) + hold 3 帧 + 换人重置(reset_distance=0.35)
   ▼ ③ 裁剪 ROI（外扩 1.45 倍）                     ★ 核心设计：先检测再裁剪
   ▼ ④ MediaPipe Face Landmarker                                 实测 8.4 ms（CPU）
        478 关键点 + 52 ARKit blendshape
        面部变换矩阵 → SVD 分解取纯旋转 → 头部欧拉角（yaw/pitch/roll）
        虹膜关键点（468–477）归一化 → 视线方向 (−1, 1)
   ▼ ⑤ 三通道独立平滑：表情 α=0.45 / 头部 0.35 / 眼球 0.35
        眨眼另走 BlinkStabiliser：非对称平滑（闭 0.8 / 开 0.45）+ 迟滞吸附 + 深度分档
   ▼ ⑥ TrackingCalibrationManager：按 VTS 返回的同名参数 min/max 裁剪 + 源范围采集 + 一键回中 + JSON 持久化
   ▼ ⑦ Sink（可并存）：
        VTubeStudioWorker  WebSocket → VTube Studio（30 FPS 注入，daemon 线程，断线 10 s 重连）
        VmcOscSink         OSC/UDP   → VSeeFace（逐帧发送）
        UnityUdpSink       UDP JSON  → Unity（逐帧发送，带 seq 去乱序）
   ▼
虚拟形象跟随表情、眨眼、转头与视线
  旁路：Qt 调试面板（16 个参数实时显示 / 摄像头预览 / 点击选脸）—— PyQt5 为可选依赖，本机未装
```

### 10.2 "先检测再裁剪"为什么是关键

VTS 自带的跟踪是把整张 640×480 画面直接送给内部模型，人脸可能只占画面六分之一。
本项目的做法是：**先用自训练检测器精确定位人脸框，外扩 1.45 倍裁剪后再送 MediaPipe** ——
人脸在输入中的占比大幅提升，478 个关键点的定位与表情系数都更准，同时排除了背景干扰。

外扩比例 1.45 是权衡值：裁太紧会切掉下巴/额头导致姿态估计不稳，裁太松背景干扰就回来了。
**如实说明：这个比例没有消融实验支撑，是经验值。**

### 10.3 手感：三类问题三套策略

| 问题 | 策略 | 参数 |
|:---|:---|:---|
| 抖动 | 人脸框 EMA + 中值滤波 | `--bbox-alpha 0.55`、`--bbox-window 5` |
| 单帧跳变 / 短暂丢失闪烁 | 中值 + 保持帧 | `--hold-frames 3` |
| 切换目标时框"滑过去" | `reset_distance` 判定新目标（超过即直接跳，不插值） | 0.35 |
| 表情/头部/眼球反应速度不同 | **三个独立**平滑通道 | 表情 0.45 / 头部 0.35 / 眼球 0.35 |
| 快眨被平滑削浅 | 非对称平滑 + 迟滞吸附 + 深度分档 | 闭 0.8 / 开 0.45；`--blink-snap-below 0.40`、`--blink-deep-below 0.33` |

**眨眼问题经过四轮定位，根因分布在三层**（这条最能说明"分通道调参不是拍脑袋"）：

| 轮次 | 现象 | 根因 | 处置 |
|:---|:---|:---|:---|
| 1 | 双眼整场闭着（模型侧 `ParamEyeLOpen = 0.0`） | **映射退化**：单向参数的源范围中心正好落在区间端点，整个源范围被塞进下半段，而 VTS 对该参数报的默认值恰等于 `min`，区间退化成单点（复现输出 `0.97 → 0.000`） | 两段式映射的前提是**源中心与目标默认值都严格落在区间内部**，任一侧不成立即改用全范围线性映射（9 个回归测试） |
| 2 | 能睁开了，但只能眯眼 | 用实测排除法定位：尽力闭眼**保持 2 秒**（约 50 帧，EMA 早该收敛）仍停在 0.251 → 排除平滑、排除 VTS 侧映射 → 根因是 **MediaPipe 的 `eyeBlink` 上限只有约 0.75**，属感知层幅度不足 | 用项目自带的校准机制按实测范围 `0.25/0.26..0.99` 校准；复测 `ParamEyeLOpen min = 0.0172`（闭到底） |
| 3 | 快眨"还是差一点" | 按动作分开看同一轮采样：**慢速闭紧保持 2 秒谷底 0.004 / 0.007 / 0.015**（完全闭），**自然快眨（约 0.15 s）谷底只到 0.09–0.12**。按 α=0.45、管线约 25 帧/秒推算，一次快眨只有约 4 帧，还没到底目标就回来了 → **平滑削浅了短时眨眼** | 改用**带迟滞的闭眼吸附**；刻意**不**抬高校准下限，因为线性映射压窄跨度会把睁眼抖动放大 1.5 倍 |

同一轮还修了两处方向问题：`FaceAngleX` 取负（本机摄像头相对 VTS/hiyori 约定是左右镜像的，
"数值一致不能证明方向对"，最终由目视确定）；左右眼归属改为镜像式（面对你的形象应闭上与你同侧的那只眼）。
头部幅度按反馈校准为 1:1，歪头源范围两轮放宽（±30 → ±50 → ±65）。

### 10.4 帧率为什么停在 29.3

单帧耗时分解（实测）：`read=15.8  yolo=7.8  mediapipe=8.4  total=32.6 ms → 29.3 FPS`。
`probe_camera.py` 跑满 **12 组「后端 × 分辨率 × 帧率 × FOURCC」**组合，结论三条：

- **全部组合的 `read` 都 ≈ 33.3 ms = 1000/30**，即内置摄像头的帧间隔（实测 28.3–30.0 FPS）；
- 那 15.8 ms 的 `read` **不是可优化的开销**，而是"帧周期 33.3 ms − 算力 16.2 ms"的**等待时间**；
- 请求 60 FPS 与 MJPG 的配置被**驱动静默忽略**，驱动还会**谎报**帧率 —— 请求 @60 时 `CAP_PROP_FPS`
  读回 60、实测仍 29.8。所以探针必须同时打印**「请求值」与「实际值」**。

推论：`--imgsz`（320/480/640 实测无差别）、`--yolo-every`（跳帧只多出空闲）、`--camera-*`
**都提高不了这台机器的帧率**；管线自身算力约 16 ms ≈ 59 FPS **有余量**，瓶颈在采集侧；
**想更高帧率只能换摄像头**。保留 5 个 `--camera-*` 参数与探针（换设备时用），本机无需设置。

### 10.5 端到端实测状态与稳定性验收（2026-09-16）

| 项 | 实测结果 |
|:---|:---|
| 端到端 | ✅ `摄像头 → 桥接 → VTube Studio 插件 API（WebSocket）→ Live2D 形象` 已跑通；**127 个输入参数中所需的 16 个全部存在**（`InputParameterListRequest` 核对） |
| 注入链路 | ✅ 写入特征值后立即回读，**5 项全部 MATCH**（`MouthOpen` / `FaceAngleX` / `EyeOpenLeft` / `EyeLeftX` / `MouthSmile`） |
| 跟随项 | ✅ 张嘴 / 眨眼（左右可分辨）/ 三轴转头 / 眼球视线 —— 使用者目视复验确认 |
| **连续稳定性** | ✅ **连续 599.734 s：19/19 个 30 秒窗口都有参数在变化；掉线 / 采样失败 / 采样中断 / VTS 重启全为 0** |
| 管线帧率 | 29.3 FPS（≈ 摄像头 30 FPS 硬上限，见 10.4） |
| 未做 | **端到端延迟未实测**；**无演示视频**（使用者决定）；真机 VSeeFace 与 Unity 未联调 |

**稳定性判据与工具**：`python apps/vtube_bridge/sample_vts_params.py`，默认采 600 秒（要求 ≥300 秒，留 5 分钟余量），
原始 CSV 边采边写，输出 **CSV + JSON 判据结论 + 报告**三件套，**退出码 0 = 通过 / 2 = 未通过**
（可被脚本消费）。本次证据：`artifacts/diag/vts_stability_20260916-150813.{csv,json,txt}`（2,090 行采样）。

> **判据要求"每个 30 秒窗口至少有一个参数在变化"，而不只是"连接没掉线"** —— 这不是过度设计：
> 同一判据下**另一次同长度运行连接侧零掉线**，但人不在镜头前，t≈102 s 起参数回落到 VTS 默认值，
> 后 15 个窗口全无变化，被**判 FAIL（4/19）**。**只看连接状态会把那次误判成"稳定运行 10 分钟"。**
> 同一轮还实测到"离开画面再回来仍能跟随"。

**一个已修复且很隐蔽的缺陷（授权握手必然失败）**：客户端 socket 超时是 **1 秒**，而
`AuthenticationTokenRequest` 的响应方式是 VTS 弹窗等人点击"允许"。VTS 日志显示
`13:05:15 触发授权弹窗 → 13:05:16 插件断开 → 13:05:21 用户点允许并把 token 返回到已关闭的 socket`。
修法：该请求单独使用 **120 秒**超时（`AUTH_TOKEN_TIMEOUT`），其余请求维持 1 秒。修完实测：
弹窗出现 → 3 秒后点击 → `Token valid, authenticated`，token 正常落盘。配 4 个回归测试（假 websocket，无需真 VTS）。

### 10.6 驱动的 16 个参数

| 组 | 参数 | 来源 |
|:---|:---|:---|
| 位置（3） | `FacePositionX / Y / Z` | 人脸框在画面中的位置与大小 |
| 表情（6） | `EyeOpenLeft`、`EyeOpenRight`、`MouthOpen`、`MouthSmile`、`BrowLeftY`、`BrowRightY` | 52 个 blendshape 中精选约 9 个合成（如 `MouthOpen` 取 `jawOpen`、`mouthFunnel` 与 `0.6 × mouthPucker` 的最大值） |
| 头部姿态（3） | `FaceAngleX / Y / Z` | 面部变换矩阵 → SVD 取纯旋转 → 欧拉角 |
| 视线（4） | `EyeLeftX / Y`、`EyeRightX / Y` | 虹膜关键点相对眼睛角点归一化到 (−1, 1) |

**为什么只注入 6 个表情维度（而不是 52 个）**：① **映射质量** —— 多数通道需要按权重合成才有意义；
② **皮套适配** —— VTS 参数是 Live2D 模型的输入定义（本项目核对过，hiyori 有 127 个输入参数），
不是每个 ARKit 通道都有调校过的形变参数；③ **抖动成本** —— 每个通道都要独立平滑与校准，
通道越多越容易引入需要逐个排查的抖动源，而直播对稳定性的要求高于表情丰富度。
扩展路径清晰：`estimate_expressions` 里逐个补映射即可。

**换皮套模型能直接用吗**：能，且是设计进去的 —— 参数**按名**注入（16 个都是 VTS 标准输入参数名）、
范围按**模型返回的 min/max** 裁剪而非写死、校准可重做（CLI 与调试面板都支持源范围采集与一键回中）。
需要重做的是校准 JSON（依赖相机/距离/光线，见 §14）与没有对应形变参数的通道。

### 10.7 桥接参数（实测 `--help` 共 36 个开关）

| 组 | 参数 | 说明 |
|:---|:---|:---|
| 输入 | `--input` | 摄像头序号或视频路径（后者放完自动退出） |
| 出口 | `--sink` | `vts` / `vmc` / `unity`，可逗号组合；别名 `both` = vts+unity、`all` = 全部 |
| 模型 | `--model` | 默认 `.pt`（CUDA + FP16）；也可指向 `.onnx`（本机只能 CPU） |
| 平滑 | `--expression-alpha`、`--head-pose-alpha`、`--eye-gaze-alpha`、`--smoothing` | 三个通道各自独立的 EMA 系数 + 位置平滑 |
| 人脸框 | `--bbox-alpha`、`--bbox-window`、`--hold-frames` | EMA / 中值窗口 / 保持帧数 |
| 眨眼 | `--blink-snap-below`、`--blink-deep-below`、`--blink-squint-frames` | 阈值设在实测的"真闭眼 0.30–0.365"与"眯眼 0.41–0.55"之间 |
| 关键点 | `--mediapipe-model`、`--mediapipe-crop-scale`（默认 1.45） | ROI 外扩比例 |
| 采集 | `--camera-backend`、`--camera-width`、`--camera-height`、`--camera-fps`、`--camera-fourcc` | 本机无需设置（见 10.4） |
| 其他 | `--send-fps`（**只对 VTS WebSocket 出口生效**）、`--no-gui`、`--nohalf` | 两个 UDP 出口**逐帧发送、不受限频**（见 §11.3） |

---

## 11. 另外两条出口：VMC/OSC 与 Unity UDP

### 11.1 出口 ②：VMC/OSC → VSeeFace（3D 备选）

```
摄像头 → 同一条捕捉管线 → VmcOscSink → OSC over UDP（默认 39540）→ VSeeFace → VRM 模型
```

每帧按规范顺序发送：`/VMC/Ext/T` → `/VMC/Ext/Blend/Val` × N → `/VMC/Ext/Bone/Pos` → `/VMC/Ext/Blend/Apply`。
表情映射 4 项：`EyeOpenLeft/Right` → `Blink_L/R`（**取反** —— 本项目是"睁眼度"、协议是"闭眼度"）、
`MouthOpen` → `A`、`MouthSmile` → `Joy`（默认 VRM0 名，`--vmc-vrm1-names` 切 VRM1 名）。

```bash
python apps/vmc_link/monitor.py                                   # 终端 A：监听器（把 OSC 翻译成人话）
python apps/vtube_bridge/main.py --input 7f62f96bca5cffdfe2e0167bf3de3169.mp4 --sink vmc --no-gui
python apps/vmc_link/e2e_check.py                                 # 一条命令跑完整冒烟
```

**已实测**（用仓库内演示视频跑的真实数据）：管线 40.2 帧/秒、发送 **240 帧**（= 管线帧率，逐帧）、
**1,681 个 OSC 包**（≈7 包/帧）、`Blink_L` 取反方向正确、头部骨骼确实在发、退出码 0。

⚠️ **未与真机 VSeeFace 联调**：本机没装 VSeeFace，验证只到"OSC 编码符合规范 + 本机回环收发正确"。
**头部骨骼的轴向符号未验证**；只发 4 个表情（VRM0 预设无眉毛项）。

### 11.2 出口 ③：Unity 接收端（目标线）

```
python apps/vtube_bridge/main.py --sink unity --unity-port 39540
        │  UDP JSON，逐帧发送，16 个参数，带 seq
        ▼
Unity: FaceParamReceiver ──► FaceParamMapper ──► 占位物体（P0）/ 皮套（P1）
                                  ▲
                            FaceParamHud（运行时诊断面板：收包速率 / 静默检测 / 关键参数）
```

```bash
python apps/unity_link/mock_receiver.py                            # 不装 Unity 先验证发送端
python apps/unity_link/selfcheck.py                                # 字段完整性 + 序号单调
powershell -NoProfile -ExecutionPolicy Bypass -File apps/unity_link/CompileCheck/check.ps1
```

**已实测**：`seq` 1→240（= 视频帧数）、40.2 包/秒、`face_found` 全程 `true`、**乱序 0**、参数逐帧变化；
同一个 `MouthOpen=0.570` 在 VMC 监听器里显示为 `A=0.570` —— **两个出口是同一份感知结果，互相印证**。

⚠️ **脚本从未在 Unity 里编译过**：本机没有 Unity，改用 `CompileCheck/` 里手写的 UnityEngine 桩 +
Roslyn `csc.exe` 做静态检查（结论：语法/成员名/类型自洽且未超过 C# 9）。**"过了桩"不等于"在 Unity 里能编译"** ——
首次导入仍可能有报错，以 Unity Console 为准。P0 的 5 条验收**尚未在 Unity 中执行**。
⚠️ **端口冲突**：`--unity-port` 与 `--vmc-port` 默认都是 **39540**，两条线不要同时启用同一端口。

### 11.3 一个来自实现的协议取舍：限频按语义拆开

VTS 走 WebSocket 是**有状态连接**、需要限频；两个 UDP 出口**没有背压**，限频只有害处。
早期让它们共用同一个 `--send-fps`，结果是管线帧间隔 **24.9 ms 小于限频门限 33.3 ms** →
**240 帧只发出 120 帧**（每帧都差一点，于是隔帧发一次）。
修法是**按语义把参数拆开**：限频只留给需要状态管理的 WebSocket 出口，UDP 出口逐帧发送。

---

## 12. 工程规范、测试与判据纪律

### 12.1 测试

```bash
python -m unittest discover -s tests -t .        # Ran 155 tests ... OK
```

**155 个用例 / 15 个文件**（`unittest`，零额外依赖）。覆盖分布：

| 领域 | 文件（用例数） |
|:---|:---|
| 数据解析与坐标 | `test_wider_annotations.py`(14)、`test_coordinates.py`(6) |
| 桥接时序滤波与映射 | `test_facebox_filter.py`(23)、`test_blink_stabiliser.py`(18)、`test_vts_parameter_mapping.py`(9)、`test_eye_open_sides.py`(5)、`test_head_rotation_direction.py`(5)、`test_eye_gaze_direction.py`(6) |
| 采集与跳帧 | `test_camera_capture_settings.py`(12)、`test_frame_skipping.py`(5) |
| 评估与诊断口径 | `test_error_metrics.py`(13)、`test_stage_timing_summary.py`(4) |
| 稳定性采样判据 | `test_vts_stability_sampler.py`(28) |
| 授权与运行环境护栏 | `test_vts_auth_wait.py`(4)、`test_opencv_environment.py`(3) |

> **回归测试是按"真实踩过的 bug"写的，不是为覆盖率而写**。几条刻意设计成"失败而不 skip"：
> `test_opencv_environment.py` 在 `cv2` 被 headless 覆盖时**变红**；
> `test_vts_parameter_mapping.py` 锁住"单向参数映射退化成常量 0 → 皮套整场闭着眼"那个必现缺陷；
> `test_vts_stability_sampler.py` 钉住"桥接启动前被算成掉线"的判据缺陷；
> `test_stage_timing_summary.py` 刻意不用 `skipIf` —— 函数缺失必须失败而不是跳过。
>
> **覆盖不到的地方（同样重要）**：**模型 IO、训练全流程、Unity C# 运行时、VMC 真机接收端**都没有测试。
> 训练数据路径无覆盖，是"接入定向增强前必须先补测试"的原因。

### 12.2 判据与失败可见性（这几条比测试更能说明工程质量）

| 原则 | 落地 |
|:---|:---|
| **接口存在 ≠ 生效** | 推理**实际生效的 provider 必须打印**（正因如此才发现 ONNX 一直跑在 CPU）；改完参数用 `-W error::UserWarning` 验证构造期不再抛警告 |
| **没测 ≠ 通过** | 缺前置文件时 `export_onnx.py` 返回 `None` 并打印 `[SKIPPED] ... do NOT report a precision-loss figure`，而不是 `return True` |
| **不伪造指标** | `evaluate.py` 缺包时把 Easy/Medium/Hard 报 `null`，而不是把 mAP50 填进三个字段冒充分项结果 |
| **判据要固化成工具** | 稳定性判据 → 入库采样工具（原始 CSV + 结论 + 退出码 0/2）；显存决策 → 探针脚本；增强假设 → A/B 脚本 |
| **非法参数早失败** | 在 `parse_args` 立即报错，不必等模型加载约 20 秒后才失败 |
| **可脚本化** | 交互式脚本带无头开关（`--no-show`），可在脚本与 CI 中运行 |
| **口径规范** | 任何指标必须标注测量口径；禁止口径不同的数字并列比较；未实测不得写入文档 |

### 12.3 三次同类失误都是"取数姿势"的问题

| 失误 | 现象 | 根因 |
|:---|:---|:---|
| 断言"本机没有摄像头" | 受限 shell 里 `Get-PnpDevice` 返回 0 个设备 | 没有先自证这次查询看得见已知为真的对象 —— **空结果不是否定证据** |
| 断言"端口没监听" | `Get-NetTCPConnection` 连确定在监听的 445 端口都报 0 条 | 同上 |
| 报"管线退化到 23.0 帧/秒" | 实际是**测量污染**：测的时候使用者自己的桥接实例也在跑，两个实例抢同一块 6 GB GPU；干净条件下同段视频 **38.0 帧/秒**（与记录的 40.2 同量级）→ **管线没有退化** | 没有先确认被测对象的状态就取数 |

规则因此从"记录教训"升级为**下一步动作**：**先确认被测对象状态再取数；空结果必须先做正对照。**
同类"看起来生效了"的另一个例子：增强代码用了 Albumentations **1.x 的参数名**（`max_holes` 等），
而环境装的是 **2.0.8** —— 这些参数**只产生一条 `UserWarning` 就被忽略，实际跑的是默认值**，
预期"最多 8 个 ≤32px 的洞"变成"1~2 个占 10~20% 的大洞"。它的症状是**不报错、代码在跑、图像确实被改了**，
是在检查构造期告警时抓到的；影响限于可视化与一次实验，**没有污染训练**，
修法是改 2.x API 并把依赖收紧到 `albumentations>=2.0.0`。

### 12.4 仓库约定

- **运行方式**：一律从项目根以 `python -m src.<pkg>.<module>` 运行；新脚本归入既有子包，不新增顶层目录。
- **路径**：必须从 `src/paths.py` 导入，禁止 `__file__` 自行推算。
- **依赖卫生**：`requirements.txt` 只列实际用到的东西（已删 PyQt5 / scikit-learn / onnx-simplifier 三项"写了但没用"的条目）；
  可选包注释掉，否则 `pip install -r` 在本机直接失败。
- **仓库卫生**：数据集与训练中间产物不入库，只提交 3 个部署权重（`best_model_v1.pt` / `best_model_v2.pt` / `best_model_v2.onnx`）。
- **过程留痕**：conventional commits，一个可独立验证的变更一次提交；决策与失误记入 `docs/`（本地，不随仓库发布，见 §16）。

**没有的东西也说清楚**：无 CI 配置（无 `.github/`）、无 lint / format / type check（无 `pyproject.toml`、无 ruff/mypy 配置）、
无代码评审记录（个人项目）。

---

## 13. 被否决的结论（不值得再试一遍）

| 方向 | 做法与结果 | 判定 |
|:---|:---|:---|
| **小脸放大裁剪**（数据侧提召回） | 只用一行增强配置做两臂 A/B，各从 `best_model_v2.pt` 微调 3 epoch（`lr0=1e-4`、`batch=6`、`imgsz=640`、`seed=0`），唯一差别是 `BBoxSafeRandomCrop` + `Resize(640)`。结果：Tiny 召回 0.494 → **0.501（+0.007）**、总体 +0.004，但精确率 0.808 → **0.765（−0.043）**、预测数 **+1,881**（约 **13:1** 的误检代价） | ❌ **否决**：收益在噪声边缘、代价明确，不足以支撑数天级的 Dataset 改造。局限四条（只跑 3 epoch、单次无种子重复、`Resize` 会拉伸长宽比、测的是随机裁剪而非定向小脸裁剪）已记录，避免重复评估 |
| **把 Albumentations 通用管线接进训练** | 该模块的全部变换已被 ultralytics 内置增强覆盖（翻转/HSV/模糊/Cutout 对应 `fliplr`/`hsv_*`/`randaugment`/`erasing`），且**没有 Mosaic**。替换会丢 Mosaic；叠加会让翻转与 HSV 施加两次 | ❌ **否决**（方案 B）：定位为可视化工具，不参与训练 |
| **靠调阈值提召回** | 3,226 张全量扫描：最优 F1 在 `conf=0.297`（0.6985），默认 0.25 是 0.6971，**只差 0.0014**；且 AP50 与阈值无关 | ❌ **推翻**旧判断："`conf=0.25` 对召回不友好"不成立；**未改默认阈值** |
| **降低 `imgsz` 换速度** | 检测只占 7.8 ms/帧，320/480/640 实测无差别；而 <32px 的人脸在 640 输入下已不足 10px，再降分辨率等于放弃那条线 | ❌ **不做**：精度换不到速度 |
| **让 UDP 出口共用限频** | 管线帧间隔 24.9 ms < 门限 33.3 ms → 240 帧只发出 120 帧 | ❌ **撤销**：UDP 无背压，逐帧发送 |
| **用张量 MAE 判精度** | "逐元素 MAE < 1e-4" 对 FP16 不可能通过（坐标舍入即贡献 2e-2），它衡量的是浮点表示而非模型行为 | ❌ **改判据**：以 mAP 为验收标准，MAE 按精度分门限、只作参考 |

> **下一步真正的高杠杆方向是提高分辨率**（`imgsz=960`）：尺寸分层已把根因定位到"<32px 的脸在 640 输入下不足 10px"，
> 提高分辨率直接作用于同一根因，且**用现成的 A/B 脚本只改一个参数即可验证**。

---

## 14. 已知限制与未完成项

| # | 项 | 说明 |
|:--|:---|:---|
| 1 | **WIDER 官方分项评估** | 需 `widerface-evaluate`，**本机网络不可用装不上**，原始 `0.76` 目标无法对账。`evaluate.py` 返回 `null` 而非用 mAP50 冒充。替代方案：2.2 的尺寸分层表 |
| 2 | **小脸召回** | 汇总召回 0.610，其中 **72.5% 的 GT 人脸 <32px、召回仅 0.493**；≥32px 为 0.916。conf 扫描已完成，**默认 0.25 已接近最优 F1，阈值不是瓶颈** |
| 3 | **v2 无增益** | 与 v1 基本持平（mAP50 −0.001，mAP50-95 +0.0007），缺困难样本回流 |
| 4 | **增强模块定位** | `src/data/augment.py` 仅用于可视化、不参与训练；数据侧提召回的方向是**属性感知增强**（需 Dataset 级改造，且前提是先补上被跳过的属性解析）与**分辨率提升**（`imgsz=960`）。小脸裁剪放大已 A/B 否决（[§13](#13-被否决的结论不值得再试一遍)） |
| 5 | **自采数据缺失** | 设计中的 500 张侧脸/遮挡/暗光数据未采集 |
| 6 | **暗光 / 遮挡 / 侧脸未系统评测** | WIDER 困难属性解析时被跳过 → 做不了分项统计（侧脸仅有一条目视观察：快速侧脸时检测框仍稳定、MediaPipe 继续出表情） |
| 7 | **训练吞吐修复未复训验证** | `batch` 已由 8 改为 6，**尚未用新 batch 重训验证效果** |
| 8 | **`val=False` 的历史影响** | v1 前 60 epoch 无验证指标（`results.csv` 里为 0），best 权重挑选依据不可靠；现已可用 `resume` 补验，但历史可比性无法恢复 |
| 9 | **授权弹窗窗口** | 首次连接需在 **120 秒**内点"允许"（旧版 1 秒超时是必现缺陷，已修，见 10.5） |
| 10 | **测量噪声** | 同一调用实测跨度 **4.8–18.8 ms（约 3 倍）**，单次平均值无意义。`benchmark.py` 已改为报告分布，**已缓解，未根治** |
| 11 | **测试覆盖有限** | 155 个用例集中在纯函数、滤波与判据逻辑；**模型 IO、训练全流程无覆盖** |
| 12 | **虚拟形象驱动两项缺口** | ① **端到端延迟未实测**（缺同步外部刺激）—— 故不给延迟毫秒数；② **无演示视频**（使用者决定），证据改为可复算的数字证据包 |
| 13 | **校准依赖使用条件** | `apps/vtube_bridge/config/tracking_calibration.json` 的源范围（`0.25/0.26..0.99`）是在当前相机/距离/光线下实测的，**换条件需重测**；`--no-gui` 下没有 GUI 校准入口，只能改 JSON |
| 14 | **VMC 线未联调** | 未与真机 VSeeFace 对接；头部骨骼轴向符号未验证；只发 4 个表情 |
| 15 | **Unity 线未编译未运行** | 3 个 C# 脚本只过"桩编译"静态检查；P0 的 5 条验收尚未执行 |
| 16 | **VTS 音频唇同步未关** | 本机 `Config_LipsyncType = ULipSync` 会自己驱动 `MouthOpen`，与注入**直接竞争**。实测注入后 +0.5 s 回读一致（该窗口内没被抢走），**长时间是否干扰未测**；注意要改的是**「对口型模式」**，不是麦克风开关 |
| 17 | **选择跟随对象是启发式** | "离画面中心最近" + 点击指定；**无身份跟踪（无 ReID、无跨帧 ID）**，有人频繁横穿会短暂跟错；更稳需引入轻量跟踪（如 ByteTrack） |
| 18 | **死配置键** | `configs/model.yaml` 的 `accumulation_steps`、`multi_scale`、`architecture` 从未被代码读取 |
| 19 | **仓库无 LICENSE** | 另有版权注意：使用 Live2D 官方样本模型发布内容**必须标注官方指定的版权声明**、部分角色不得改动设计；商用或长期直播应自制模型或购买授权 |
| 20 | **编辑器/工具链缺失** | 无 CI、无 lint/format/type check、无 `pyproject.toml`、无代码评审记录 |
| 21 | **工作区清理未做** | 仍留有若干测试临时目录（`facetest_*` / `.facetest_tmp_*` / `_wtest_*`）与本地实验日志，部分未被 `.gitignore` 覆盖 |

### 如果重做，按这个顺序改

| 优先 | 改什么 | 为什么排这个位置 |
|:--:|:---|:---|
| **1** | 修 `train_v1()` 里处于关闭状态的 `val=False` | 一行代码，代价是前 60 epoch 的评估盲区与不可靠的 best 权重 |
| **2** | 分辨率消融：`imgsz=960` | 尺寸分层已把根因定位到极小脸，这直接作用于根因；现成 A/B 脚本只改一个参数 |
| **3** | 端到端延迟实测 | 验收清单里唯一还空着的一条，也是直播体验的核心指标 |
| **4** | 补官方 Easy/Medium/Hard 分项评估 | 需 `widerface-evaluate`（有网络时一条 `pip install` 即可），补上后原始 0.76 目标才能对账 |
| **5** | 补训练数据路径的测试 | 现有 155 个用例不覆盖训练数据路径，是"做定向增强"的前提而非锦上添花 |
| **6** | 属性感知增强（需 Dataset 级改造） | WIDER 自带五个困难属性，正好对应未评测的困难场景；前提是先补第 5 条并证明收益 |
| **7** | 稳定性判据的判据扩展 + 清死配置键 + 补 LICENSE | 低风险收尾 |

---

## 15. 复现命令与证据索引

| 想复核什么 | 运行 | 产物 |
|:---|:---|:---|
| 155 个测试全绿 | `python -m unittest discover -s tests -t .` | 控制台（本轮实跑 OK） |
| 数据集清单 | `python -m src.data.split` | `data/annotations/{train,val}_list.txt` |
| v1/v2 的 mAP | `python -m src.eval.evaluate` | `artifacts/reports/metrics_v{1,2}.json`、`comparison.json` |
| 尺寸分层召回 | `python -m src.eval.analyze_errors` | `artifacts/reports/error_analysis_summary.txt`、`fn/fp_visualization.png` |
| 阈值扫描 | `python -m src.eval.conf_sweep` | `artifacts/reports/conf_sweep.{txt,png}` |
| 三变体的 mAP/吞吐/体积 | `python src/deploy/compare_export_variants.py` | 控制台汇总表 + `artifacts/logs/export_variant_comparison.log` |
| 裸网络前向基准 | `python -m src.deploy.benchmark` | 控制台（中位数 + 分布） |
| ONNX 导出与精度门限 | `python -m src.deploy.export_onnx` | `artifacts/checkpoints/best_model_v2.onnx` |
| 训练峰值显存 | `python src/train/probe_vram.py` | `artifacts/logs/vram_probe.log` |
| 小脸裁剪增强 A/B | `python src/train/probe_aug_crop.py` | `artifacts/logs/aug_ab.log` |
| 摄像头 read 耗时 | `python apps/vtube_bridge/probe_camera.py` | `artifacts/diag/camera_read_report.txt` |
| **VTS 连续稳定性** | `python apps/vtube_bridge/sample_vts_params.py` | `artifacts/diag/vts_stability_<时间>.{csv,json,txt}`，退出码 0/2 |
| VMC 出口端到端冒烟 | `python apps/vmc_link/e2e_check.py` | 控制台 |
| Unity 发送端 / 假接收端 | `python apps/unity_link/selfcheck.py` / `mock_receiver.py` | 控制台 |
| Unity C# 静态检查 | `apps/unity_link/CompileCheck/check.ps1` | 控制台 |

> **仓库里的脚本就是唯一的"文档来源"** —— 上表每个数字都能用仓库内代码重跑复现。
> 需要自己准备的外部资源见 4.3（均在仓库外）。

---

## 16. 文档导航与许可

### 16.1 对外发布的文档

**本仓库只发布本 README**（外加 `.gitignore`、`requirements.txt` 与代码本身）。
开发过程中的设计文档、差距台账、面试材料、决策与失误记录都在本地 `docs/` 下，**未纳入版本库**
（历史提交里仍有旧版本）。可对外复核的内容已集中写在本 README 内：指标与口径、三条出口的实测状态与被否决的结论、
已知限制与未完成项、复现命令与证据索引。

### 16.2 本地文档（`docs/`，不随仓库发布）

| 文件 | 内容 |
|:---|:---|
| `模拟面试项目全解.md` | **自包含答辩底稿**：项目全景 + 口径基准 + 高频问答 + 红线清单 |
| `面试讲解提纲.md` | 8–10 分钟讲解的节奏、时间分配与演示脚本 |
| `面试问答弹药库.md` | 分板块详答（模型 / 训练 / 实时 / 集成 / AI 协作 / 反思） |
| `简历项目经历.md` | 简历措辞（客户端与游戏客户端两版）、禁止写清单、JD 对照 |
| `AI协作复盘.md` | 显式讲 AI 协作的复盘稿（与简历口径分开使用） |
| `皮套联调问题与解决.md` | 联调排障手册（缺陷 → 根因 → 处置） |
| `archive/` | 历史文档：早期设计规格书、实施计划、退休版简历稿，以及**被本 README 取代的工程文档**（架构说明、使用指南、项目现状与差距） |
| `plans/` | 各阶段的实施计划（含被否决方向的评估记录） |

### 16.3 许可

本仓库当前**没有 LICENSE 文件**。若使用 Live2D 官方样本模型（如 VTube Studio 自带的 hiyori 等）
发布任何内容，**必须标注官方指定的版权声明**，且部分角色不得改动设计 ——
要点与官方原文摘录见本地文档 `apps/vtube_bridge/README.md`（未纳入版本库）。商用或长期直播应自制模型或购买授权。
