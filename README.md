# 高精度人脸检测与实时表情捕捉

基于 **YOLOv8n-face** 的实时人脸检测系统，覆盖「数据准备 → 训练 → 评估 → 导出 → 部署推理」
全链路，并在此之上做出一套**实时表情捕捉管线**（检测 → 裁剪 → MediaPipe 关键点 → 时序滤波），
把表情与头部姿态驱动到虚拟形象上。**由一人独立完成的工程实践项目。**

> **本 README 的写法**：所有指标都标注**测量口径**并给出可复现的命令。不同口径的数字不放在一起比较，
> 未经复现的数值不写。没做完的部分如实列在[已知限制](#13-已知限制与未完成项)。
> 仓库里的脚本就是唯一的"文档来源" —— 下面每一个数字都能用仓库内代码重跑出来。

---

## 目录

1. [项目概览](#1-项目概览)
2. [实测指标](#2-实测指标)
3. [环境与安装](#3-环境与安装)
4. [快速开始](#4-快速开始)
5. [目录结构](#5-目录结构)
6. [数据管线](#6-数据管线)
7. [训练](#7-训练)
8. [评估](#8-评估)
9. [导出与基准](#9-导出与基准)
10. [实时表情捕捉管线](#10-实时表情捕捉管线)
11. [另外两条出口：VMC/OSC 与 Unity UDP](#11-另外两条出口vmcosc-与-unity-udp)
12. [测试](#12-测试)
13. [已知限制与未完成项](#13-已知限制与未完成项)
14. [复现命令与证据索引](#14-复现命令与证据索引)
15. [技术栈](#15-技术栈)
16. [文档与许可](#16-文档与许可)

---

## 1. 项目概览

| 层 | 内容 | 状态 |
|:---|:---|:---|
| **检测模型** | YOLOv8n-face，WIDER Face 全量微调（v1 基线 100 epoch + v2 微调 30 epoch） | ✅ 已训练、已评估、已导出 |
| **部署推理** | `src/deploy/detect.py`：摄像头 / 视频 / 单图 / 文件夹四种输入；PyTorch `.pt`（CUDA）与 ONNX FP16 两种运行时 | ✅ 可运行 |
| **表情捕捉** | 自训练检测器定位人脸 → 裁剪 ROI → MediaPipe Face Landmarker（478 关键点 + 52 ARKit blendshape）→ 时序滤波 → 参数映射 | ✅ 已实现并实测 |
| **出口 ①：VTube Studio（主线）** | WebSocket Plugin API 注入 Live2D 参数 | ★ **端到端已跑通**，连续 **599.7 s** 稳定性实测通过 |
| **出口 ②：VMC/OSC（3D 备选）** | OSC over UDP → VSeeFace + VRM | ⚠️ 发送端已实测（回环 + 编码自检 + 一键端到端冒烟），**未与真机 VSeeFace 联调** |
| **出口 ③：Unity UDP（目标线）** | UDP JSON → Unity C# 接收端 → 占位物体（P0） | ⚠️ 代码已完成并过"桩编译"静态检查，**未在 Unity 中编译或运行** |

三条出口复用**同一条**捕捉管线，只是挂了不同的 sink（`--sink vts` / `vmc` / `unity`，可逗号组合），
**没有复制任何感知代码**。

---

## 2. 实测指标

### 2.1 检测精度（WIDER Face 验证集，3,226 张）

| 指标 | v1 基线（100 epoch） | v2 微调（+30 epoch） |
|:---|---:|---:|
| mAP50 | **0.66501** | 0.66404 |
| mAP50-95 | 0.35545 | **0.35618** |
| Precision | 0.849 | 0.846 |
| Recall | 0.595 | 0.595 |

来源：`artifacts/reports/metrics_v1.json` / `metrics_v2.json`，由 `src/eval/evaluate.py` 在 **best 权重**上重新测得。

> **口径说明（重要）**：上表是**全部 3,226 张验证图**上按 COCO 方式计算的 AP50，分母包含大量极小尺寸人脸。
> 历史设计文档中的目标 `0.76` 指的是 WIDER Face **Hard 子集**的 VOC AP —— **两者口径不同，不可直接比较**。
> 官方 Easy/Medium/Hard 分项指标需要 `widerface-evaluate` 包，而**本机网络不可用、装不上**，
> 所以该目标无法对账；`evaluate.py` 在该包缺失时把三个字段报为 `null`（**不再用 mAP50 冒充分项结果**），
> 替代分析见 2.2。
>
> **v2 与 v1 基本持平**：mAP50 −0.001、mAP50-95 +0.0007，属噪声量级。v2 只是在已收敛点附近做 30 epoch
> 低学习率微调，数据分布没变，**没有增益也不意外**。真正的闭环优化需要困难样本回流，这一步没有做。
>
> 注：训练日志 `results.csv` 末行数字略低（0.66349 / 0.66171），那是**最后一个 epoch** 的值，
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

> **这是本项目最重要的一条结论**：汇总召回 0.610 看起来偏低，但 **72.5% 的标注人脸小于 32 像素**
> （人群远景里的脸），模型在这些极小脸上召回仅 0.493。
> **排除极小脸后，≥32px 人脸的召回是 0.916。**
>
> 也就是说：分数低主要是**基准集本身极端**，而非模型在实际场景不可用。虚拟形象驱动场景中人脸占据
> 画面主要位置（Medium/Large 桶），召回 0.955–0.989。这也解释了原始 `0.76` 目标为何无法达成 ——
> Hard 子集几乎全是极小脸，**目标设定与实际应用场景错配**。
>
> 注：此处召回 0.610 与 2.1 表中 ultralytics 报的 0.595 **口径不同**（前者为 IoU≥0.5 贪心匹配 +
> `conf=0.25`，后者为 ultralytics 自身的置信度扫描）。两个数都真实，不可互相替换引用。

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

> **结论：默认的 `conf=0.25` 几乎正好在最优 F1 点上**（max-F1 点为 `conf=0.297`，F1 0.6985
> 对比 0.25 处的 0.6971，仅差 **0.0014**）。此前文档里"`conf=0.25` 对召回不友好"的说法
> **经实测被推翻**。
>
> - **AP50 与阈值无关**（始终 0.66404），阈值只是在同一条 P-R 曲线上选取工作点。
> - 降到 0.10：召回 **+0.065**，但精确率 **−0.233** —— 代价很大。
> - 对"宁可多检也不漏检"的场景（如人脸跟踪），0.15–0.20 是合理选择。
> - 本项目**未改动任何默认阈值**，因为默认值本身就是最优 F1 点。

### 2.4 导出变体对比（检测质量 + 端到端吞吐 + 体积）

`python src/deploy/compare_export_variants.py` —— 三个变体各跑**同一套 3,226 张验证集**的 mAP，
吞吐用 `model()` 计时（**含 letterbox + 前向 + 解码 + NMS**，即应用真正付出的代价）：

| 变体 | mAP50 | mAP50-95 | 吞吐（30 次中位） | 运行设备 | 体积 |
|:---|---:|---:|---:|:---|---:|
| **PyTorch `.pt`** | 0.66404 | 0.35618 | **10.31 ms / 97.0 FPS** | **CUDA（GPU）** | 5.95 MiB |
| ONNX **FP16** | 0.66382 | 0.35627 | 24.80 ms / 40.3 FPS | CPU | **5.88 MiB** |
| ONNX FP32 | 0.66379 | 0.35670 | ≈25 ms / ≈40 FPS | CPU | 11.70 MiB |

> **ONNX 精度的取舍已用实测解决**：FP16 相对 PyTorch 的 mAP50 差 **−0.0002**、mAP50-95 差
> **+0.0001**，即**检测质量无差别**；相对 FP32 体积减半、CPU 吞吐相同。
> 因此导出采用 **FP16**（`src/deploy/export_onnx.py` 的 `EXPORT_HALF = True`），
> 项目原定 ≤10 MB 目标随之重新满足（FP32 为 11.70 MiB，一度超标）。
>
> ⚠️ **ONNX 在本机只能跑 CPU**：`onnxruntime` 1.27 实际只提供 `CPUExecutionProvider` 与
> `AzureExecutionProvider`，**没有 CUDA**；安装 `onnxruntime-gpu` 需要网络，本机不可用。
> 所以 ONNX 比 PyTorch 慢约 **2.4 倍** —— 它在本机的价值是**可移植性**，不是速度。
> `src/deploy/benchmark.py` 与 ultralytics 现在都会明确打印实际生效的 provider，不再静默降级。
>
> ⚠️ **精度验证口径已修正**：`export_onnx.py` 的张量 MAE 门限按精度区分
> （FP32 `1e-4` / FP16 `1e-1`，见 `MAE_TOLERANCE`），实测 FP32 MAE = **0.000015**、FP16 = **0.050635**。
> 但**张量 MAE 比的是浮点表示而非模型行为** —— FP16 下 8400 个候选框的坐标只保留约 3 位有效数字，
> 单个坐标从 320.12345 漂到 320.1 就已贡献约 2e-2。**真正的验收判据是 mAP**（上表），MAE 只作参考。

### 2.5 纯网络前向基准（**另一口径**，勿与 2.4 混用）

`python -m src.deploy.benchmark` —— **只测网络前向**，不含 letterbox / 解码 / NMS：

| 项 | 实测（中位） |
|:---|---:|
| PyTorch（CUDA） | **10.31 ms / 97.0 FPS** |
| ONNX Runtime（CPU） | 24.80 ms / 40.3 FPS |
| 文件体积 | `.pt` 5.95 MiB ｜ `.onnx` 5.88 MiB |

> 脚本默认 50 次预热 + 200 次计时，并**打印中位数、最小值、p95、最大值与样本数**。
> ⚠️ **速度口径（重要）**：本机单次测量噪声极大 —— 同一调用在不同轮次测到 **4.8 – 18.8 ms**（跨度约 3 倍）。
> 此前多轮"数字对不上"的现象都源于此，而非代码变化。因此 `benchmark.py` 已改为报告分布而非单个均值。
> **引用任何速度数字时请注明测量次数与统计量，单次读数不可作为结论。**
>
> 2.4 与 2.5 的差异来自**是否包含前后处理**：2.5 是裸网络，2.4 是 `model()` 全流程。两者都真实，
> 但**不可交叉引用**。

---

## 3. 环境与安装

### 3.1 实测环境

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

> **OpenCV 的坑（已加护栏）**：`opencv-python` / `-headless` / `-contrib` 三个分发包
> **共用同一个 `site-packages/cv2/`，最后安装者生效**。本机实际生效的是
> **opencv-contrib-python 5.0.0.93**（`cv2.__version__` = `5.0.0`，而 `pip show opencv-python` 仍报 4.13.0.92）。
> `requirements.txt` 因此声明 `opencv-contrib-python`。
> ⚠️ **绝不能让 `opencv-python-headless` 成为最后安装的那个** —— 它没有 `imshow` / `waitKey` /
> `destroyAllWindows`，而 `src/deploy/detect.py` 用了它们，实时预览会直接崩，且报错指向 `cv2.imshow`
> 而非安装顺序。护栏测试：`tests/test_opencv_environment.py`（被 headless 覆盖时会**红**，不会 skip）。
>
> **显存注意（已实测）**：6 GB 卡上 `batch=8 + imgsz=640` 的峰值**保留**显存达 **6.18 GiB**，
> 已超过 6.00 GiB 物理上限，从第一轮起就在向共享内存溢出；100 epoch 长跑会因内存碎片继续增高
> （历史日志达 11.1 GiB），吞吐从 5.8 it/s 崩到 1.7 it/s。
> **配置已下调为 `batch=6`**（峰值保留 4.41 GiB，留 27% 余量）。实测脚本：`src/train/probe_vram.py`。

### 3.2 需要自己准备的三份大文件（**均不在仓库内**）

| 资源 | 大小 | 用途 | 获取方式 |
|:---|:---|:---|:---|
| WIDER Face 数据集 | ≈3 GB | 训练与评估 | `python -m src.data.download` **只打印手动下载指引**（脚本不执行下载），解压到 `data/raw/` |
| `face_landmarker.task` | 3.7 MB | MediaPipe 关键点与表情系数 | 见 `apps/vtube_bridge/thirdparty/MediaPipe/README.md` 里的一条 `Invoke-WebRequest` 命令，存到 `thirdparty/MediaPipe/models/` |
| `yolov8n.pt` | 6.2 MB | 训练起点（COCO 预训练权重） | ultralytics 首次训练时自动下载，或手动放入项目根目录 |

> 缺少 MediaPipe 模型时，桥接会以明确的 `FileNotFoundError` 失败（**不会静默降级**）。
> 顺带一个已解决的真实坑：本项目路径含中文字符，而 MediaPipe 的 C++ 层按系统区域设置打开文件，
> 非 ASCII 路径会失效 —— `face_landmarker.py` 的 `_safe_path()` 会自动把模型复制到纯 ASCII 的临时目录。

### 3.3 摄像头相关的实测约束

- 摄像头设备**只在交互式用户会话里可用**。受限的执行环境（如本项目的自动化助手 shell）中
  `Get-PnpDevice` 返回 0 个设备、`Get-CimInstance` 拒绝访问 —— **空结果不是否定证据**。
  → **凡是要打开摄像头的命令，必须由使用者本人在自己的终端运行。**
- 没有摄像头也能跑全链路：`--input` 同时接受**视频文件路径**，仓库根目录就随附了一个
  **`7f62f96bca5cffdfe2e0167bf3de3169.mp4`**（586 KiB，已入库），放完自动退出、退出码 0，
  因此可作无人值守的回归冒烟。

---

## 4. 快速开始

所有脚本都从**项目根目录**以模块方式运行；路径一律经 `src/paths.py` 解析。

### 4.1 实时摄像头检测（OpenCV 窗口）

```bash
python -m src.deploy.detect --input 0
# 按 q 退出，按 s 截图
```

### 4.2 视频 / 单图 / 文件夹

```bash
python -m src.deploy.detect --input video.mp4 --save output.mp4
python -m src.deploy.detect --input photo.jpg --save results/
python -m src.deploy.detect --input my_photos/ --save results/
python -m src.deploy.detect --input photo.jpg --no-show      # 不弹结果窗口（脚本/批处理/CI 用）
```

参数（`--help` 实测）：`--input` 摄像头序号 / 视频 / 图片 / 文件夹 ｜ `--model` 权重路径 ｜
`--imgsz`（默认 640）｜ `--conf`（默认 0.25）｜ `--save` ｜ `--no-show`。

### 4.3 表情驱动：VTube Studio 主线（无需摄像头即可验证链路）

```bash
# 前置：VTube Studio 里开启「开启API（允许安装插件）」，并加载一个主模型（如自带 hiyori）
#       首次运行 VTS 会弹授权窗，有 120 秒时间点"允许"
python apps/vtube_bridge/main.py --input 0 --sink vts --no-gui

# 没有摄像头：换成随仓库的演示视频，放完自动结束
python apps/vtube_bridge/main.py --input 7f62f96bca5cffdfe2e0167bf3de3169.mp4 --sink vts --no-gui
```

> `--no-gui` 必须带：**本机未安装 PyQt5**，不加这个开关桥接会因缺 Qt 直接报错退出。
> （PyQt5 **不在依赖清单内** —— 它只服务于内部调试面板，而该面板在本机从未运行过；如需要请自行安装。）

### 4.4 三条出口各自的"不装任何东西"自检

```bash
python apps/vmc_link/selfcheck.py       # VMC/OSC 编码是否符合规范（逐字节）
python apps/unity_link/selfcheck.py     # Unity UDP 字段完整性 + 序号单调
python apps/vmc_link/e2e_check.py       # 一条命令跑完整链路：真桥接 + 真演示视频 + UDP 收包
python apps/vtube_bridge/sample_vts_params.py   # VTS 连续稳定性采样（默认 600 s，退出码 0/2）
```

---

## 5. 目录结构

```
README.md                            本文件（仓库内唯一的说明文档）
requirements.txt                     依赖清单（含每条的实测口径注释）
configs/model.yaml                   训练与模型配置
7f62f96bca5cffdfe2e0167bf3de3169.mp4 随仓库发布的演示视频 —— 无摄像头时跑全链路用

src/                                 库代码（被导入，不直接运行）
├── paths.py                         ★ 统一路径解析：全项目唯一的路径来源
├── data/                            数据管线
│   ├── wider_annotations.py         ★ WIDER 标注解析的唯一实现（4 份重复实现已收敛到此）
│   ├── download.py                  数据集准备指引（**不执行下载**）
│   ├── qc.py                        标注质检 → data/annotations/quality_report.txt
│   ├── convert.py                   WIDER 标注 → YOLO labels/（训练真正依赖的产物）
│   ├── split.py                     生成 train_list.txt / val_list.txt
│   ├── loader.py                    数据集封装（供将来接入训练，当前为孤立模块）
│   ├── augment.py                   Albumentations 增强管线（★ 仅用于可视化，**不参与训练**）
│   └── vis_aug.py                   增强效果可视化（人工检查用）
├── train/                           训练
│   ├── config.py                    超参管理
│   ├── train.py                     v1 基线 100 epoch + v2 微调 30 epoch
│   ├── resume.py                    断点续训
│   ├── check_model.py               模型诊断（输出 shape / 导出可行性；只打印不断言）
│   ├── probe_vram.py                探针：各 batch 的峰值显存与吞吐
│   └── probe_aug_crop.py            探针：小脸裁剪放大的 A/B 实验（结论：不值得）
├── eval/                            评估
│   ├── evaluate.py                  mAP + P-R 曲线 + 分项（缺包时返回 null）
│   ├── analyze_errors.py            困难样本分析：按尺寸分层的召回/精确率 + TOP-N 可视化
│   └── conf_sweep.py                置信度阈值扫描（P/R/F1 随 conf 变化）
└── deploy/                          部署
    ├── detect.py                    实时检测统一入口（摄像头/视频/图片/文件夹，四种输入）
    ├── export_onnx.py               ONNX 导出（FP16 + simplify + opset 12）+ 精度验证
    ├── benchmark.py                 纯网络前向基准（**不含前后处理**）
    └── compare_export_variants.py   三变体对比：mAP + 端到端吞吐 + 体积（**决策用**）

apps/                                三条落地出口（共用同一条捕捉管线）
├── vtube_bridge/                    ★ 出口 ① 主线：VTube Studio + Live2D
│   ├── main.py                      CLI 入口（转发到下面的包）
│   ├── vtube_studio_bridge/
│   │   └── vtube_studio_bridge.py   全部实现：捕捉管线 + 三个 sink + 滤波 + 校准映射
│   ├── sample_vts_params.py         稳定性采样器（只读观察者，产出原始 CSV + 结论 + 报告）
│   ├── probe_camera.py              摄像头 read 耗时探针（12 组后端×分辨率×帧率×FOURCC）
│   ├── run_gui.bat                  Qt 调试面板启动脚本（可选，需自装 PyQt5）
│   ├── config/tracking_calibration.json  实测标定的参数映射范围（源范围/中心/上下限）
│   └── thirdparty/MediaPipe/        MediaPipe Face Landmarker 包装 + 模型存放位（模型需自备）
├── vmc_link/                        出口 ② 备选：VMC/OSC → VSeeFace + VRM
│   ├── monitor.py                   VMC 监听器：把 OSC 消息翻译成人话
│   ├── selfcheck.py                 编码规范自检（地址/类型/表情名/取值方向/顺序）
│   ├── osc_reader.py                最小 OSC 解码器（监听与自检共用，支持 bundle）
│   └── e2e_check.py                 一键端到端冒烟：真桥接 + 演示视频 + 真收包
└── unity_link/                      出口 ③ 目标线：Unity 接收端
    ├── mock_receiver.py             假 Unity：不装 Unity 就能验证发送端（仅标准库）
    ├── selfcheck.py                 发送端字段完整性 + 序号单调性自检
    ├── UnityProject/Assets/Scripts/ 3 个 C# 脚本（接收 / 映射 / 运行时诊断 HUD）
    └── CompileCheck/                桩编译静态检查（手写 UnityEngine 桩 + csc.csproj + check.ps1）

data/                                数据集（.gitignore 排除；仓库内只留 widerface.yaml）
artifacts/                           产出物
├── checkpoints/                     best_model_v1.pt / best_model_v2.pt / best_model_v2.onnx + 训练曲线
├── reports/                         metrics_*.json、conf_sweep.*、误差分析、尺寸分层汇总
├── diag/                            诊断证据：稳定性采样原始 CSV/JSON/TXT、摄像头 read 报告
└── logs/                            训练与实验日志（.gitignore 排除）
runs/                                ultralytics 默认输出目录（.gitignore 排除）
tests/                               单元测试：15 个测试文件、**155 个用例**（unittest，零额外依赖）+ fixtures
```

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
> **本机绝对路径**，换机器必然失效。**新环境的第一步就是 `python -m src.data.split` 重新生成**，
> 否则依赖它的三个脚本会走各自的降级分支（见第 8 节）。
>
> 设计文档中承诺的"自采 500 张侧脸/遮挡/暗光数据"**未采集**，最终只使用公开数据集。
>
> 另：WIDER 标注里每个脸都带 `blur / expression / illumination / occlusion / pose` 五个困难属性，
> 当前解析时被跳过（未用于训练）。这是"属性感知增强"这条优化方向的前提，见 13 节。

---

## 7. 训练

```bash
python -m src.train.train        # v1 基线 100 epoch + v2 微调 30 epoch
python -m src.train.resume       # 断点续训（从 artifacts/checkpoints/*/weights/last.pt 接续）
tensorboard --logdir artifacts/checkpoints --port 6006
```

实际使用的超参（以 `artifacts/checkpoints/*/args.yaml` 为准）：

| 参数 | 值 |
|:---|:---|
| batch / imgsz | 历史训练 **8 / 640**（见 `artifacts/checkpoints/*/args.yaml`）；**配置已改为 6 / 640**（显存实测见 3.1） |
| optimizer | AdamW，lr 1e-3 → 1e-5 余弦退火（v2 从 1e-4 起），warmup 3 epoch（v2 为 1） |
| 混合精度 | `amp=true` |
| 增强 | `mosaic=1.0`（v2 降至 0.5）、`hsv_*`、`fliplr=0.5`、`translate=0.1`、`scale=0.5`、`close_mosaic=10` |
| 其他 | `nbs=64`、`workers=2`、`seed=0`、`deterministic=true` |

> 训练用的是 **ultralytics 内置增强**。`src/data/augment.py` 里的 Albumentations 管线
> **没有接入训练流程**，只用于增强效果的可视化验证 —— 原因与论证见 13 节。
>
> `configs/model.yaml` 里有 3 个**死配置键**（`accumulation_steps`、`multi_scale`、`architecture`），
> 声明了但代码从不读取，属已知技术债。

---

## 8. 评估

```bash
python -m src.data.split          # 前置：生成 val_list.txt（缺失时下面两步会降级）
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

> 判据设计上的一条原则：**"没测"必须和"通过"区分开**。早期版本在这两处的表现分别是
> "用 mAP50 冒充分项指标"和"缺文件直接判定通过"，都已修掉。

---

## 9. 导出与基准

```bash
python -m src.deploy.export_onnx            # ONNX 导出（FP16 + simplify + opset 12，静态 shape）+ 精度验证
python -m src.deploy.benchmark              # 纯网络前向延迟与体积（不含前后处理，与上一行不同口径）
python src/deploy/compare_export_variants.py  # 三变体 mAP + 端到端吞吐 + 体积（决策依据）
```

导出产物直接落在 `artifacts/checkpoints/best_model_v2.onnx`（5.88 MiB），与 `.pt` 同级，
已是桥接的可选运行时之一。

---

## 10. 实时表情捕捉管线

### 10.1 数据流

```
摄像头帧
   │
   ▼ ① YOLOv8n-face 检测人脸框                             实测 ~7.8 ms（GPU，FP16）
   ▼ ② FaceBoxFilter：EMA(0.35) + 中值(5) + hold 3 帧 + 新人脸重置
   ▼ ③ 裁剪 ROI（外扩 1.45 倍）        ← 核心设计：先检测再裁剪
   ▼ ④ MediaPipe Face Landmarker                           实测 ~8.4 ms（CPU）
        478 关键点 + 52 ARKit blendshape
        + 变换矩阵 SVD 分解 → 头部欧拉角
        + 虹膜关键点归一化 → 视线方向
   ▼ ⑤ 三个独立滤波通道：
        ParameterFilter（表情 EMA 0.45）
        BlinkStabiliser（眨眼：非对称平滑 + 迟滞吸附 + 深度分档）
        smooth_position（位置）
   ▼ ⑥ 校准映射：按 VTS 返回的 min/max 裁剪 + 源范围采集 + 一键回中
   ▼ ⑦ Sink（三选一或并存）：
        VTuberStudioWorker  WebSocket → VTube Studio，30 FPS 注入，断线 10 s 重连
        VmcOscSink          OSC/UDP   → VSeeFace（逐帧发送）
        UnityUdpSink        UDP JSON  → Unity（逐帧发送，带 seq 去乱序）
   ▼
虚拟形象跟随表情、眨眼、转头与视线
```

### 10.2 "先检测再裁剪"为什么是关键

全图直接送关键点模型时人脸往往只占画面一小部分；用自训练检测器先框出人脸再裁剪放大，
**人脸在关键点模型输入里的占比大幅提升**，关键点定位与表情系数都更准。

外扩比例 1.45 是权衡值：裁太紧会切掉下巴/额头导致姿态估计不稳，裁太松则背景干扰回归。

### 10.3 实测状态（2026-09-16 真机联调）

| 项 | 实测结果 |
|:---|:---|
| 端到端 | ✅ `摄像头 → 桥接 → VTube Studio 插件 API（WebSocket）→ Live2D 形象` 已跑通；模型 **127 个输入参数中所需 16 个全部存在**（已用 `InputParameterListRequest` 核对） |
| 注入链路 | ✅ 写入特征值后立即回读，**5 项全部 MATCH**（`MouthOpen` / `FaceAngleX` / `EyeOpenLeft` / `EyeLeftX` / `MouthSmile`） |
| 跟随项 | ✅ 张嘴 / 眨眼（左右可分辨）/ 三轴转头 / 眼球视线 —— 均由使用者目视复验确认 |
| **连续稳定性** | ✅ **连续 599.734 s（10 分钟）：19/19 个 30 秒窗口均有参数在变化；掉线 / 采样失败 / 采样中断 / VTS 重启全部为 0** |
| 管线帧率 | **29.3 帧/秒** —— 经 12 组「后端 × 分辨率 × 帧率 × FOURCC」实测确认**已是内置摄像头 30 帧/秒的硬上限** |
| 未做 | **端到端延迟未实测**（缺同步外部刺激，故不给毫秒数）；**未录演示视频**（使用者决定） |

稳定性验收可复现：`python apps/vtube_bridge/sample_vts_params.py`
默认采 600 秒（要求 300 秒，留 5 分钟余量），原始 CSV **边采边写**，
输出 CSV + JSON 判据结论 + 报告三件套，**退出码 0 = 通过 / 2 = 未通过**。
本次证据：`artifacts/diag/vts_stability_20260916-150813.{csv,json,txt}`（2,090 行采样）。

> **判据要求"每个 30 秒窗口至少有一个参数在变化"，而不只是"连接没掉线"** ——
> 这不是过度设计：同一判据下**另一次同长度运行连接侧零掉线**，但人不在镜头前时
> t≈102 s 起参数就回落到 VTS 默认值（后 15 个窗口全无变化），被判 **FAIL（4/19）**。
> **只看连接状态会把那次误判成"稳定运行 10 分钟"。**
> 顺带实测到"离开画面再回来仍能跟随"。

### 10.4 帧率为什么停在 29.3

单帧耗时分解（实测）：`read=15.8  yolo=7.8  mediapipe=8.4  total=32.6 ms → 29.3 帧/秒`。

`probe_camera.py` 跑满 12 组组合后，**全部 `read ≈ 33.3 ms = 1000/30`** —— 即摄像头的帧间隔。
推论：

- 那 15.8 ms 的 `read` **不是可优化的开销**，而是"帧周期 33.3 ms − 算力 16.2 ms"的**等待时间**；
- `--imgsz`（320/480/640 实测无差别）、`--yolo-every`（跳帧只多出空闲）、`--camera-*`
  **都提高不了这台机器的帧率**；
- 管线自身算力约 16 ms ≈ 59 帧/秒，**有余量**，瓶颈在采集侧；
- 想要更高帧率**只能换摄像头**；
- 顺带发现**驱动会谎报帧率**：dshow 请求 @60 时 `CAP_PROP_FPS` 读回 60，实测仍 29.8 —— 
  所以探针必须同时打印「请求值」与「实际值」，只信请求值会得出错误结论。

保留 `--camera-*` 五个采集参数与 `probe_camera.py`（换摄像头/换机器时用得上），**本机无需设置**。

### 10.5 桥接参数

`python apps/vtube_bridge/main.py --help` 共 **36 个开关**（实测）。常用的几组：

| 组 | 参数 | 说明 |
|:---|:---|:---|
| 输入 | `--input` | 摄像头序号或视频路径（后者放完自动退出） |
| 出口 | `--sink` | `vts` / `vmc` / `unity`，可逗号组合；别名 `both`=vts+unity、`all`=全部 |
| 模型 | `--model` | 默认 `.pt`（CUDA + FP16）；也可指向 `.onnx`（本机只能 CPU） |
| 平滑 | `--expression-alpha`、`--head-pose-alpha`、`--eye-gaze-alpha`、`--smoothing` | 三个通道各自独立的 EMA 系数 |
| 人脸框 | `--bbox-alpha`、`--bbox-window`、`--hold-frames` | EMA / 中值窗口 / 保持帧数 |
| 眨眼 | `--blink-snap-below`、`--blink-deep-below`、`--blink-squint-frames` | 阈值设在实测的"真闭眼 0.30–0.365"与"眯眼 0.41–0.55"之间的空隙里 |
| 关键点 | `--mediapipe-model`、`--mediapipe-crop-scale`（默认 1.45） | ROI 外扩比例 |
| 采集 | `--camera-backend`、`--camera-width`、`--camera-height`、`--camera-fps`、`--camera-fourcc` | 本机无需设置 |
| 其他 | `--send-fps`（**只对 VTS WebSocket 出口生效**）、`--no-gui`、`--nohalf` | 两个 UDP 出口**逐帧发送、不受限频** |

> ⚠️ **`--send-fps` 对两个 UDP 出口不起作用**：早期实现过 30 帧/秒上限，实测它把 41 帧/秒的管线
> **砍到只剩 20 帧/秒**（240 帧只发 120 帧）—— 因为管线帧间隔 24.9 ms 小于限频间隔 33.3 ms，
> 于是"每帧都差一点、隔帧发一次"。对没有背压的 UDP 数据报，这个上限只有害处，已移除。

> ⚠️ **眨眼幅度受模型侧限制**：本机 640×480 / 人脸宽约 147 px 下，MediaPipe 的 `eyeBlink` 上限只有
> **约 0.75** —— 尽力闭眼保持 2 s，`EyeOpenLeft` 只到 0.251。已用项目自带的校准机制按实测范围
> 校准（`apps/vtube_bridge/config/tracking_calibration.json`，源范围 `0.25/0.26..0.99`）→ `0.251 → 0.001`。
> **该范围依赖当前相机/距离/光线，换条件需重测**；`--no-gui` 下没有 GUI 校准入口，只能改这个 JSON。

---

## 11. 另外两条出口：VMC/OSC 与 Unity UDP

### 11.1 VMC/OSC → VSeeFace（3D 备选，出口 ②）

```
摄像头 → 同一条捕捉管线 → VmcOscSink → OSC over UDP（默认 39540）→ VSeeFace → VRM 模型
```

每帧按规范顺序发送：`/VMC/Ext/T` → `/VMC/Ext/Blend/Val` × N → `/VMC/Ext/Bone/Pos` → `/VMC/Ext/Blend/Apply`。
表情映射 4 项：`EyeOpenLeft/Right`→`Blink_L/R`（**取反**，我们是"睁眼度"、协议是"闭眼度"）、
`MouthOpen`→`A`、`MouthSmile`→`Joy`（默认 VRM0 名，`--vmc-vrm1-names` 切 VRM1 名）。

```bash
# 终端 A：监听器
python apps/vmc_link/monitor.py
# 终端 B：只发 VMC（用随仓库的演示视频，无需摄像头）
python apps/vtube_bridge/main.py --input 7f62f96bca5cffdfe2e0167bf3de3169.mp4 --sink vmc --no-gui
# 或者一条命令跑完整冒烟
python apps/vmc_link/e2e_check.py
```

**已实测（用仓库内演示视频跑的真实数据）**：管线 40.2 帧/秒、发送 **240 帧**（= 管线帧率，
逐帧发送）、**1,681 个 OSC 包**（≈7 包/帧）、`Blink_L` 取反方向正确、头部骨骼确实在发、退出码 0。

⚠️ **未与真机 VSeeFace 联调**：本机没装 VSeeFace，验证只到"OSC 编码符合规范 + 本机回环收发正确"。
**头部骨骼的轴向符号未验证**（按 Unity `Quaternion.Euler` 的 ZXY 顺序实现并做了数学自检，
但真机左右/俯仰是否需要取反不确定）。只发 4 个表情（VRM0 预设无眉毛项）。

### 11.2 Unity 接收端（目标线，出口 ③）

```
python apps/vtube_bridge/main.py --sink unity --unity-port 39540
        │  UDP JSON，逐帧发送，16 个参数，带 seq
        ▼
Unity: FaceParamReceiver ──► FaceParamMapper ──► 占位物体（P0）/ 皮套（P1）
                                  ▲
                            FaceParamHud（运行时诊断面板）
```

```bash
# 不装 Unity 先验证发送端
python apps/unity_link/mock_receiver.py
python apps/vtube_bridge/main.py --input 0 --sink unity --unity-port 39540 --no-gui
python apps/unity_link/selfcheck.py
```

**已实测**：`seq` 1→240（= 视频帧数）、40.2 包/秒、`face_found` 全程 `true`、**乱序 0**、
参数逐帧变化；同一个 `MouthOpen=0.570` 在 VMC 监听器里显示为 `A=0.570` ——
**两个出口是同一份感知结果，互相印证**。

⚠️ **脚本从未在 Unity 里编译过**：本机没有 Unity，改用 `CompileCheck/` 里的手写 UnityEngine 桩 +
Roslyn `csc.exe` 做静态检查（结论：语法/成员名/类型自洽且未超过 C# 9）。
**"过了桩"不等于"在 Unity 里能编译"** —— 首次导入仍可能有报错，以 Unity Console 为准。
重新检查：`powershell -NoProfile -ExecutionPolicy Bypass -File apps/unity_link/CompileCheck/check.ps1`。
P0 的验收（占位物体跟随 5 条）**尚未在 Unity 中执行**。

> ⚠️ **端口冲突**：`--unity-port` 与 `--vmc-port` 默认都是 **39540**，两条线不要同时启用同一端口。

---

## 12. 测试

```bash
python -m unittest discover -s tests -v
# Ran 155 tests ... OK
```

**155 个用例 / 15 个测试文件**（`unittest`，零额外依赖）。覆盖分布：

| 领域 | 文件（用例数） |
|:---|:---|
| 数据解析与坐标 | `test_wider_annotations.py`(14)、`test_coordinates.py`(6) |
| 桥接时序滤波与映射 | `test_facebox_filter.py`(23)、`test_blink_stabiliser.py`(18)、`test_vts_parameter_mapping.py`(9)、`test_eye_open_sides.py`(5)、`test_head_rotation_direction.py`(5)、`test_eye_gaze_direction.py`(6) |
| 采集与跳帧 | `test_camera_capture_settings.py`(12)、`test_frame_skipping.py`(5) |
| 评估与诊断口径 | `test_error_metrics.py`(13)、`test_stage_timing_summary.py`(4) |
| 稳定性采样判据 | `test_vts_stability_sampler.py`(28) |
| 授权与运行环境护栏 | `test_vts_auth_wait.py`(4)、`test_opencv_environment.py`(3) |

> **回归测试是按"真实踩过的 bug"写的**，不是为覆盖率而写。几个例子：
> `test_opencv_environment.py` 会在 `cv2` 被 headless 版本覆盖时**变红**（不 skip）；
> `test_vts_stability_sampler.py` 里有"桥接启动前不算掉线"那个 bug 的回归；
> `test_vts_parameter_mapping.py` 锁住了"单向参数映射退化成常量 0 → 皮套整场闭着眼"那个必现 bug。
>
> **覆盖不到的地方（同样重要）**：**模型 IO、训练全流程、Unity C# 运行时、VMC 真机接收端**都没有测试。
> 训练数据路径无覆盖，是"接入定向增强前必须先补测试"的原因（见 13 节）。

---

## 13. 已知限制与未完成项

| # | 项 | 说明 |
|:--|:---|:---|
| 1 | **WIDER 官方分项评估** | 需要 `widerface-evaluate`，**本机网络不可用、装不上**，原始 `0.76` 目标无法对账。`evaluate.py` 返回 `null` 而非用 mAP50 冒充。替代方案：2.2 的尺寸分层表 |
| 2 | **小脸召回** | 汇总召回 0.610，其中 **72.5% 的 GT 人脸 <32px、召回仅 0.493**；≥32px 人脸召回 0.916。conf 扫描已完成，**默认 0.25 已接近最优 F1，阈值不是瓶颈** |
| 3 | **v2 无增益** | 与 v1 基本持平（mAP50 −0.001，mAP50-95 +0.0007），缺困难样本回流 |
| 4 | **增强模块定位** | `src/data/augment.py` **仅用于可视化，不参与训练**：其变换已被 ultralytics 内置覆盖，且缺 Mosaic。若要走数据侧提升小脸召回，方向是**属性感知增强**（需 Dataset 级改造，前提是先补上被跳过的属性解析）与**小脸放大裁剪**（**已做 A/B：收益 +0.004 mAP、代价精确率 −0.043、多 1,881 个误检，判定不值得**，脚本 `src/train/probe_aug_crop.py`）。更高杠杆的下一步是提高分辨率（`imgsz=960`），可用同一个 A/B 脚本只改 `--imgsz` 验证 |
| 5 | **自采数据缺失** | 设计中的 500 张侧脸/遮挡/暗光数据未采集 |
| 6 | **训练吞吐** | 已实测并修复配置（`batch=8` 峰值 6.18 GiB 超 6 GB 上限 → 改为 `batch=6`，4.41 GiB）。**尚未用新 batch 重训验证效果** |
| 7 | **测量噪声** | 同一调用实测跨度 **4.8–18.8 ms（约 3 倍）**，单次平均值无意义。`benchmark.py` 已改为报告中位数与分布。**已缓解，未根治** |
| 8 | **测试覆盖有限** | 155 个用例集中在纯函数与滤波逻辑；**模型 IO、训练全流程无覆盖** |
| 9 | **虚拟形象驱动两项缺口** | ① **端到端延迟未实测**（缺同步外部刺激）—— 故本仓库不给延迟毫秒数，只给"管线算力约 16 ms/帧 + 摄像头帧周期 33.3 ms"；② **无演示视频**（使用者决定），验收证据改为可复算的数字证据包 |
| 10 | **VMC 线未联调** | 未与真机 VSeeFace 对接；头部骨骼轴向符号未验证；只发 4 个表情 |
| 11 | **Unity 线未编译未运行** | 3 个 C# 脚本只过"桩编译"静态检查；P0 的 5 条验收尚未执行 |
| 12 | **VTS 音频唇同步未关** | 本机 `Config_LipsyncType = ULipSync` 会自己驱动 `MouthOpen`，与注入**直接打架**。实测注入后 +0.5 s 回读一致（该窗口内没被抢走），**长时间是否干扰未测**，仍建议关掉（设置 → 麦克风 → 对口型模式） |
| 13 | **死配置键** | `configs/model.yaml` 的 `accumulation_steps`、`multi_scale`、`architecture` 从未被代码读取 |
| 14 | **仓库无 LICENSE** | 当前没有版权/许可文件；Live2D 官方样本模型用于发布内容时**必须标注指定版权声明**（要点见本地开发文档 `apps/vtube_bridge/README.md`，未纳入版本库） |
| 15 | **收尾清理未做** | 工作区仍留有若干测试临时目录（`facetest_*` / `.facetest_tmp_*` / `_wtest_*`）与本地实验日志，其中部分未被 `.gitignore` 覆盖 |

### 运行方式建议（本机实测）

| 场景 | 建议 | 原因 |
|:---|:---|:---|
| **日常使用、追求速度** | **PyTorch `.pt` + CUDA** | 中位 10.31 ms / 97 FPS，比 ONNX（CPU）快约 2.4 倍 |
| 免装 PyTorch 的部署 | ONNX **FP16** | 5.88 MiB，mAP 与 PyTorch 无差别，但本机只能跑 CPU |
| **VTube Studio 桥接** | **默认已用 `.pt`（CUDA + FP16）** | 曾默认 `.onnx` → 落在 CPU、GPU 闲置；已修正，FP16 另获 7.4% 提速 |

---

## 14. 复现命令与证据索引

| 想复核什么 | 运行 | 产物 |
|:---|:---|:---|
| 155 个测试全绿 | `python -m unittest discover -s tests -v` | 控制台 |
| 数据集清单 | `python -m src.data.split` | `data/annotations/{train,val}_list.txt` |
| v1/v2 的 mAP | `python -m src.eval.evaluate` | `artifacts/reports/metrics_v{1,2}.json`、`comparison.json` |
| 尺寸分层召回 | `python -m src.eval.analyze_errors` | `artifacts/reports/error_analysis_summary.txt` 等 |
| 阈值扫描 | `python -m src.eval.conf_sweep` | `artifacts/reports/conf_sweep.{txt,png}` |
| 三变体的 mAP/吞吐/体积 | `python src/deploy/compare_export_variants.py` | 控制台汇总表 |
| 裸网络前向基准 | `python -m src.deploy.benchmark` | 控制台（中位数 + 分布） |
| ONNX 导出与精度门限 | `python -m src.deploy.export_onnx` | `artifacts/checkpoints/best_model_v2.onnx` |
| 训练峰值显存 | `python src/train/probe_vram.py` | `artifacts/logs/vram_probe*` |
| 小脸裁剪增强 A/B | `python src/train/probe_aug_crop.py` | `artifacts/logs/aug_ab*` |
| 摄像头 read 耗时 | `python apps/vtube_bridge/probe_camera.py` | `artifacts/diag/camera_read_report.txt` |
| **VTS 连续稳定性** | `python apps/vtube_bridge/sample_vts_params.py` | `artifacts/diag/vts_stability_<时间>.{csv,json,txt}`，退出码 0/2 |
| VMC 出口端到端冒烟 | `python apps/vmc_link/e2e_check.py` | 控制台 |
| Unity 发送端与接收端 | `python apps/unity_link/selfcheck.py` / `mock_receiver.py` | 控制台 |
| Unity C# 静态检查 | `apps/unity_link/CompileCheck/check.ps1` | 控制台 |

---

## 15. 技术栈

Python 3.12 ｜ PyTorch 2.5 ｜ ultralytics 8.4 ｜ OpenCV 5.0（`cv2.__version__` 实测值）｜
MediaPipe 1.0 ｜ ONNX Runtime 1.27（**本机无 CUDA provider**）｜
websocket-client 1.9 ｜ Matplotlib ｜ TensorBoard ｜
Unity 侧为 C#（仅静态检查，未在 Unity 中编译）

> 上面只列**实际用到**的东西（版本为本机 `pip show` 实测值）。已从技术栈与 `requirements.txt` 中移除：
> **PyQt5**（本机未安装、GUI 调试面板从未运行）、**scikit-learn**（全仓库从未 import）、
> **onnx-simplifier**（从未 import）。**Albumentations** 不在技术栈里：它只用于增强效果可视化、
> 不参与训练（见下方「已知限制」第 4 条）。
>
> GUI 调试面板（PyQt5）是**可选项**，需要自行安装（不在依赖清单内），故所有命令一律带 `--no-gui`。

---

## 16. 文档与许可

**本仓库只发布本 README 作为说明文档**（外加 `.gitignore`、`requirements.txt` 与代码本身）。
开发过程中的设计文档、变更记录、决策与失误记录、验收证据等已用 `git rm --cached` **移出版本库**
（本地文件保留，历史提交里仍有它们）。可对外复核的内容已集中写在本 README 内：
指标与口径、三条出口的实测状态与被否决的结论、已知限制与未完成项。

**仓库里的脚本就是"文档来源"** —— 上面每一个数字都能用仓库内代码重跑复现：
稳定性数据由 `apps/vtube_bridge/sample_vts_params.py` 产出（原始 CSV 落盘、可自行重算），
摄像头帧率上限由 `apps/vtube_bridge/probe_camera.py` 产出，
导出决策由 `src/deploy/compare_export_variants.py` 产出。

**许可**：本仓库当前**没有 LICENSE 文件**。若使用 Live2D 官方样本模型（如 VTube Studio 自带的
hiyori 等）发布任何内容，**必须标注官方指定的版权声明**，且部分角色不得改动设计 ——
详细要点与官方原文摘录见本地开发文档 `apps/vtube_bridge/README.md`（未纳入版本库）。
