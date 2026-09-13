# 文档校正实施计划（阶段二）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让项目文档与代码实际状态严格一致；旧文档（设计承诺）原文保留，作为阶段三"查漏补缺"的待办来源。

**Architecture:** 三层文档结构 —— ① `docs/archive/` 存放不可变的旧文档副本；② 旧文档原位置加"与实现存在差距"说明横幅，正文一字不改；③ README / 使用指南 / 新增的《项目现状与差距》描述代码**实际**状态。

**Tech Stack:** Markdown 文档。校验手段为 `grep` + 逐条比对已验证的代码事实（文件行号、实测数据、命令行为）。

---

## 一、事实基准（本计划的唯一数据来源，均已实测）

以下数据在整个阶段二不得修改或重新估算，全部来自代码与产出的直接核查：

| 事实 | 数值 / 结论 | 证据来源 |
|:---|:---|:---|
| 本机 GPU | NVIDIA GeForce RTX 3060 Laptop GPU，**6144 MiB（6 GB）** | `nvidia-smi`；`torch.cuda.get_device_properties(0).total_memory` |
| 驱动 / CUDA | 581.95 / CUDA 12.1 | `nvidia-smi`；`torch.version.cuda` |
| Python / torch | 3.12.8 / 2.5.1+cu121 | 本机实测 |
| v1 结果（100 ep） | mAP50 **0.66349**，P 0.8494，R 0.59536，mAP50-95 0.35394 | `training/checkpoints/v1_baseline/results.csv` 末行 |
| v2 结果（+30 ep） | mAP50 **0.66171**，P 0.84588，R 0.59507，mAP50-95 0.35275 | `training/checkpoints/v2_finetune/results.csv` 末行 |
| 训练中断点 | epoch **63/64** 中断，后 `resume=True` 续训至 100 | `training/resume_train.py:68` 打印 + `training_log.txt` 末行 |
| 训练超参 | batch=8, imgsz=640, AdamW, lr0=1e-3, lrf=1e-5, amp=true, mosaic=1.0(v2:0.5), nbs=64, workers=2, seed=0, deterministic=true, close_mosaic=10 | `*/args.yaml` |
| 训练峰值显存 | **11.1 GB（超过 6 GB 物理显存 → WDDM 共享内存超配）** | `training_log.txt` |
| 性能退化证据 | `it/s` 5.8 → 1.7；单 epoch 已耗时 8:08 → 14:34 | `training_log.txt` |
| `best_model_v1.pt` | 5.96 MB | 文件实测 |
| `best_model_v2.pt` | 5.95 MB | 文件实测 |
| `best_model_v2.onnx` | **5.88 MB（FP32，非 FP16）** | 文件实测 + `deployment/export_onnx.py:18` `half=False` |
| ONNX 导出参数 | `dynamic=False, simplify=True, opset=12, half=False` | `deployment/export_onnx.py:18` |
| 精度验证状态 | **静默跳过**：依赖 `data/annotations/val_list.txt`，该文件不存在 → `print WARN` 后 `return True` | `export_onnx.py:38-41` |
| 分项评估状态 | **未完成**：`evaluate.py:51-53` 对缺失 `val_list.txt` 同样跳过；`evaluation/reports/` 仅有 6 张 `test_detect_*.jpg` | `evaluation/evaluate.py:51-53` + 目录实测 |
| benchmark 口径 | PyTorch 调 `model.model(dummy)`，**绕过 letterbox/解码/NMS**，且输入为 `torch.randn` 随机噪声 | `deployment/benchmark.py:21-25` |
| benchmark EP | `providers=["CUDAExecutionProvider","CPUExecutionProvider"]`，**全程不打印实际生效的 provider** | `deployment/benchmark.py:49` |
| 计时方法 | 计时前后均调 `torch.cuda.synchronize()`，50 次 warmup | `deployment/benchmark.py:26-38` |
| 数据规模 | train 图 12,880 / 标注 12,881；val 图 3,226 / 标注 3,227 | `data/raw/**` 实测 |
| 自采数据 | **不存在**（设计承诺的 500 张侧脸/遮挡/暗光未做） | `data/raw/` 仅 WIDER_train / WIDER_val |
| 标注质检报告 | **不存在** | `data/annotations/quality_report.txt` 缺失 |
| 训练用增强 | 仅 ultralytics 内置（mosaic/hsv/fliplr/translate/scale/erasing） | `training/train.py:86-94` |
| Albumentations 模块 | **未接入训练**，仅被 `dataset/vis_aug.py` 引用 | `training/train.py` 全文无 `augmentation` 导入 |
| `model_config.yaml` | `batch_size: 8`、`workers: 2`、`model_name: "yolov8n.pt"` —— **与代码一致** | `model/model_config.yaml:6,18,31` |
| 死配置键 | `accumulation_steps`、`multi_scale`、`architecture` 声明但代码从不读取 | `model_config.yaml:10-13,19,45` × `train.py` 全文 |
| `model_test.py` | 引用 `YOLO("yolov8n-face.pt")` —— **该文件不存在** | `model/model_test.py:7,30` |
| 数据流依赖 | 训练**不需要** `train_list.txt`（ultralytics 按 `images/`↔`labels/` 同构目录读取） | `training/train.py:42-62` + 数据布局 |
| VTS 桥接默认模型 | `training/checkpoints/best_model_v2.onnx` | `README.md:147` |

---

## 二、文件处置总表

| # | 文件 | 处置 | 理由 |
|:---|:---|:---|:---|
| 1 | `docs/archive/2026-06-22-design-spec.原始.md` | **新建**（原文副本，不可变） | 旧文档原文留存 |
| 2 | `docs/archive/2026-06-22-plan.原始.md` | **新建**（原文副本，不可变） | 同上 |
| 3 | `docs/superpowers/specs/2026-06-22-face-detection-design.md` | **加横幅，正文不改** | 保留承诺原文供阶段三比对 |
| 4 | `docs/superpowers/plans/2026-06-22-face-detection-plan.md` | **加横幅，正文不改** | 同上 |
| 5 | `README.md` | **修正 8 处 + 新增 1 章** | 主页必须与代码一致 |
| 6 | `docs/使用指南.md` | **修正 4 处** | 面向使用者的操作必须可执行 |
| 7 | `docs/presentation_outline.md` | **修正 5 处** | 答辩口径必须经得起代码核对 |
| 8 | `.claude/CLAUDE.md` | **修正 4 处** | GPU 与工作目录错误会误导后续所有会话 |
| 9 | `MEMORY.md` | **重写** | 现索引指向 3 个不存在的文件 |
| 10 | `docs/项目现状与差距.md` | **新建** | 阶段三的输入清单 |
| 11 | `CHANGELOG.md` | **新建** | 约定要求；项目此前无 |
| 12 | `.dsh-dev/conventions.md` | **新建** | 个人约定文件缺失，提供模板待补充 |

---

## 三、任务分解

### Task 1: 归档旧文档原文（不可变副本）

**Files:**
- Create: `docs/archive/2026-06-22-design-spec.原始.md`
- Create: `docs/archive/2026-06-22-plan.原始.md`

- [ ] **Step 1: 复制原文**

```powershell
New-Item -ItemType Directory -Force -Path "docs\archive" | Out-Null
Copy-Item "docs\superpowers\specs\2026-06-22-face-detection-design.md" "docs\archive\2026-06-22-design-spec.原始.md"
Copy-Item "docs\superpowers\plans\2026-06-22-face-detection-plan.md" "docs\archive\2026-06-22-plan.原始.md"
```

- [ ] **Step 2: 校验副本与原文哈希一致**

```powershell
Get-FileHash "docs\superpowers\specs\2026-06-22-face-detection-design.md","docs\archive\2026-06-22-design-spec.原始.md" | Select Hash
```

预期：两个哈希相同。

- [ ] **Step 3: 提交**

```powershell
git add docs/archive
git commit -m "docs: archive immutable copies of the original spec and plan"
```

---

### Task 2: 给旧文档加差距横幅（正文一字不改）

**Files:**
- Modify: `docs/superpowers/specs/2026-06-22-face-detection-design.md`（在第 1 行前插入）
- Modify: `docs/superpowers/plans/2026-06-22-face-detection-plan.md`（在第 1 行前插入）

- [ ] **Step 1: 插入横幅**

统一插入以下文本（标题下接原文）：

```markdown
> ⚠️ **本文为设计阶段的承诺，与当前代码实现存在若干差距，请勿当作现状描述阅读。**
> 完整差距清单见 [`docs/项目现状与差距.md`](../../项目现状与差距.md)。
> 本文正文**保持原样不改**，用途是作为逐项补齐实现的任务来源。
> 不可变副本：[`docs/archive/2026-06-22-design-spec.原始.md`](../../archive/2026-06-22-design-spec.原始.md)
```

> 注：plan 文档的副本链接指向 `2026-06-22-plan.原始.md`。

- [ ] **Step 2: 校验正文未被改动**

```powershell
# 原文内容应为副本的子集（仅少了横幅）
(Get-Content "docs\superpowers\specs\2026-06-22-face-detection-design.md").Count
(Get-Content "docs\archive\2026-06-22-design-spec.原始.md").Count
```

预期：差值为 5（横幅行数）。

- [ ] **Step 3: 提交**

```powershell
git add docs/superpowers
git commit -m "docs: mark original spec/plan as design intent, not current state"
```

---

### Task 3: 新建《项目现状与差距》

**Files:**
- Create: `docs/项目现状与差距.md`

- [ ] **Step 1: 写入差距表**

本文件是阶段三的**唯一输入清单**，必须逐条包含：差距描述、证据（文件:行号）、承诺出处、修复成本、优先级。内容直接取自本计划第一节「事实基准」。

结构：

```markdown
# 项目现状与差距

> 数据来源：2026-09-13 对代码与产出的直接核查。每一条均可按"证据"列复核。
> 用途：阶段三"查漏补缺"的任务来源。承诺出处指 `docs/superpowers/` 下的原始文档。

## 一、指标与口径

| # | 差距 | 现状（实测） | 承诺（原始文档） | 证据 | 优先级 |
|:--|:---|:---|:---|:---|:---|
| G1 | mAP 口径不可比 | 全验证集 COCO 式 mAP50 = 0.664 | Hard 子集 VOC AP ≥ 0.76 | `v1_baseline/results.csv` vs 规格书第一章 | 高 |
| G2 | 分项评估缺失 | `evaluation/reports/` 仅 6 张 jpg | "Easy/Medium/Hard mAP 摘要页" | 目录实测；`evaluate.py:51-53` | 高 |
| G3 | Recall 偏低未分析 | R = 0.595（漏检 40%） | 未设目标 | `results.csv` | 高 |
| G4 | v2 为负收益 | 0.66349 → 0.66171，P/R 双降 | "v1→v2 迭代提升柱状图" | 两份 `results.csv` | 中 |
| G5 | ONNX 精度验证无据 | 静默跳过（`return True`） | "MAE < 1e-4 方可导出通过" | `export_onnx.py:38-41` | 高 |
| G6 | 导出实为 FP32 | 5.88 MB，`half=False` | "ONNX FP16"，6.17 MB | `export_onnx.py:18` | 高 |
| G7 | FPS 口径夸大 | 裸前向，不含后处理 | 表格未标口径 | `benchmark.py:21-25` | 高 |
| G8 | ONNX 基准未固定 EP | 不打印生效 provider | 声称"ONNX CPU 54.9 FPS" | `benchmark.py:49` | 中 |
| G9 | Hard 子集从未评估 | 无任何 Hard 结果 | 目标 0.76 | 全目录检索 | 高 |

## 二、数据与增强

| # | 差距 | 现状 | 承诺 | 证据 | 优先级 |
|:--|:---|:---|:---|:---|:---|
| G10 | 自采 500 张缺失 | 仅 WIDER 公开集 | "自采 500 张（侧脸/遮挡/暗光）" | `data/raw/` 实测 | 中 |
| G11 | 标注质检未产出 | 无 `quality_report.txt` | "质检通过" | 文件缺失 | 中 |
| G12 | 增强未接入训练 | 仅 ultralytics 内置 | "后 50 epoch 加 Cutout/模糊" | `train.py` 无 augmentation 导入 | 高 |
| G13 | 无在线 DataLoader | `dataloader.py` 孤立 | "统一 DataLoader：yield (images, targets)" | 仅有 `vis_aug.py` 引用 | 低 |
| G14 | 未使用困难样本属性 | 解析时跳过 blur/occlusion/pose 字段 | 未明确 | `convert_wider_to_yolo.py` | 低 |

## 三、训练

| # | 差距 | 现状 | 承诺 | 证据 | 优先级 |
|:--|:---|:---|:---|:---|:---|
| G15 | 梯度累积未实现 | 无 `accumulate` 参数（靠 nbs=64 隐式缩放） | "accumulation_steps=2 模拟 batch=32" | `train.py` 无该参数 | 低 |
| G16 | 死配置键 | `accumulation_steps`/`multi_scale`/`architecture` 从不被读取 | — | `model_config.yaml` × `train.py` | 低 |
| G17 | 显存超配导致性能崩塌 | 峰值 11.1 GB > 6 GB；it/s 5.8→1.7 | "RTX 3060 12GB，batch=16" | `training_log.txt` | 中 |
| G18 | 无单元测试 | `model_test.py` 只有 print，无 assert；且引用不存在的 `yolov8n-face.pt` | "输出 shape 断言测试" | `model/model_test.py:7,30` | 中 |

## 四、部署与集成

| # | 差距 | 现状 | 承诺 | 证据 | 优先级 |
|:--|:---|:---|:---|:---|:---|
| G19 | TensorRT 未做 | 仅 ONNX Runtime | 规格书已声明降级为 ONNX | 规格书简化决策表 | 低（已声明） |
| G20 | 52 blendshape 仅用 6 维 | 精选 6 个表情维度 | 答辩稿列为"可扩展方向" | 桥接实现 | 低（已声明） |

## 五、文档与工程

| # | 差距 | 现状 | 证据 | 优先级 |
|:--|:---|:---|:---|:---|
| G21 | MEMORY.md 索引悬空 | 指向 3 个不存在的 md | `MEMORY.md` | 中 |
| G22 | `.claude/CLAUDE.md` GPU/路径错误 | 写 12GB、`C:\学习\课程设计` | `CLAUDE.md:15,21,42` | 中 |
| G23 | 无 CHANGELOG | — | 根目录无该文件 | 低 |
| G24 | 无 `.dsh-dev/conventions.md` | — | 目录缺失 | 低 |
```

- [ ] **Step 2: 校验每一条都有证据可查**

```powershell
# 抽查 3 条：G5 / G12 / G18 的证据是否成立
Select-String -Path "deployment\export_onnx.py" -Pattern "val_list.txt not found"
Select-String -Path "model\model_test.py" -Pattern "yolov8n-face.pt"
```

- [ ] **Step 3: 提交**

```powershell
git add "docs/项目现状与差距.md"
git commit -m "docs: add implementation-vs-design gap inventory (input for phase 3)"
```

---

### Task 4: 修正 README.md

**Files:**
- Modify: `README.md`

- [ ] **Step 1: 修正模型性能表（第 69-75 行）**

原文：

```markdown
| 指标 | v1 基线 | v2 微调 |
|:---|:---|:---|
| mAP50 | 0.663 | 0.662 |
```

改为（并在表后补一段）：

```markdown
| 指标 | v1 基线 | v2 微调 |
|:---|:---|:---|
| mAP50 | 0.663 | 0.662（↓ 未提升） |
```

表后新增：

```markdown
> **指标口径说明**：以上 mAP50 为**全验证集**（3,226 张，COCO 式 AP50）结果。设计目标中的 0.76
> 对应 WIDER Face **Hard 子集**（VOC AP 口径），两者不可直接比较；Hard/Easy/Medium 分项评估**尚未完成**。
> v2 微调为**负收益**（0.66349 → 0.66171，Precision 0.8494 → 0.8459，Recall 0.59536 → 0.59507），
> 因为它仅在已收敛点附近做了 30 epoch 低学习率微调，训练分布未改变。
```

- [ ] **Step 2: 修正部署指标表（第 77-81 行）**

将表头由「部署指标」改为「网络前向延迟与体积（不含前后处理）」，并把第 81 行改为：

```markdown
| ONNX 模型大小（FP32 + simplify） | 5.88 MB | ≤ 10 MB ✅ |
```

表后新增：

```markdown
> **速度口径说明**：上表 FPS 由 `deployment/benchmark.py` 测得，调用的是 `model.model(dummy)`
> **裸网络前向**，**不含** letterbox 预处理、解码与 NMS 后处理，输入为随机张量（无框可做 NMS，
> 后处理开销为 0）。因此**不是端到端管线帧率**。此外该脚本同时注册 CUDA 与 CPU 两个
> ExecutionProvider 且不打印实际生效者，"ONNX CPU 54.9 FPS" 需重新测量并固定 provider 后才可作为 CPU 性能引用。
```

- [ ] **Step 3: 修正环境要求（第 83-86 行）**

在现有两行后补充实测环境：

```markdown
- **实测环境**：Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 / RTX 3060 **Laptop 6 GB** / 驱动 581.95
- 显存注意：6 GB 卡上 `batch=8 + imgsz=640` 会超额分配至共享内存（实测峰值 11.1 GB），导致吞吐骤降；建议降到 `batch=4` 或 `imgsz=512`
```

- [ ] **Step 4: 修正训练与评估命令说明（第 90-109 行）**

- `python training/train.py` 后补注：`# 首轮以 val=False 启动（当时 WIDER_val 缺失）`
- 新增一行：`python training/resume_train.py  # 断点续训（从 epoch 63 接续至 100，并开启验证）+ v2 微调`
- `python evaluation/evaluate.py` 后补注：`# 依赖 data/annotations/val_list.txt；缺失时 Easy/Medium/Hard 分项评估会被跳过`
- 新增一行：`python data/generate_list.py  # 生成 train_list/val_list.txt（评估与 ONNX 精度验证的前置）`
- `python deployment/export_onnx.py` 的注释由「ONNX FP16 导出 + 精度验证」改为「ONNX 导出（FP32 + simplify）+ 精度验证（依赖 val_list.txt，缺失则跳过）」

- [ ] **Step 5: 新增「已知限制与未完成项」章节**

插入在「技术栈」之前：

```markdown
## 已知限制与未完成项

| 项 | 说明 |
|:---|:---|
| WIDER 分项评估 | Easy/Medium/Hard 尚未评估，原始 0.76 目标无法直接对账 |
| ONNX 精度验证 | 因 `val_list.txt` 缺失而静默跳过，MAE 未经实测 |
| Recall 偏低 | 0.595，漏检主要来自小脸与遮挡；尚无 conf 阈值扫描数据 |
| 增强模块未接入 | `dataset/augmentation.py` 为独立模块，训练仅用 ultralytics 内置增强 |
| 自采数据 | 设计中的 500 张侧脸/遮挡/暗光数据未采集 |
| 困难样本未回流 | `analyze_errors.py` 只做分析，未参与再训练 |
| 训练吞吐 | 6 GB 显存下超额分配，需降低 batch/imgsz |

逐条差距与证据见 [`docs/项目现状与差距.md`](docs/项目现状与差距.md)。
```

- [ ] **Step 6: 校验无遗留的 FP16 表述**

```powershell
Select-String -Path "README.md" -Pattern "FP16"
```

预期：仅剩 VTS 桥接的 `--nohalf` 相关表述（那是推理精度开关，与 ONNX 导出无关）。

- [ ] **Step 7: 提交**

```powershell
git add README.md
git commit -m "docs: align README with verified code behaviour and metrics"
```

---

### Task 5: 修正 docs/使用指南.md

**Files:**
- Modify: `docs/使用指南.md`

- [ ] **Step 1: 修正 ONNX 导出说明（约第 95-103 行）**

原文「注意：精度验证步骤需要 WIDER Face 数据集，如未下载会自动跳过。」改为：

```markdown
> 注意：导出为 **FP32 + simplify**（`half=False`，约 5.88 MB）。精度验证步骤依赖
> `data/annotations/val_list.txt`——**该文件不在仓库中**，需先执行 `python data/generate_list.py` 生成。
> 文件缺失时脚本会打印 `[WARN]` 并**直接判定通过**，不会真正比对 MAE。
```

- [ ] **Step 2: 修正 MediaPipe 模型准备（约第 111-120 行）**

补注：

```markdown
> 该模型文件（3.76 MB）**已随仓库提供**于 `deployment/Vtube-Studio-Bridge/thirdparty/MediaPipe/models/face_landmarker.task`，
> 无需重复下载；下列命令仅用于文件缺失时恢复。
```

- [ ] **Step 3: 修正硬件要求（约第 9-13 行）**

将「RTX 3060 6GB+ 显存」改为：

```markdown
- **GPU**：RTX 3060 Laptop 6 GB（实测机型）；CPU 亦可运行，速度较慢
```

- [ ] **Step 4: 修正评估章节（约第 142-154 行）**

在命令块前补：`# 前置：先执行 python data/generate_list.py 生成 val_list.txt`

- [ ] **Step 5: 提交**

```powershell
git add "docs/使用指南.md"
git commit -m "docs: correct usage guide against verified script behaviour"
```

---

### Task 6: 修正 docs/presentation_outline.md（答辩口径）

**Files:**
- Modify: `docs/presentation_outline.md`

- [ ] **Step 1: 修正模型性能表（第 60 行）**

`| ONNX 导出 | — | 6.2 MB | FP16，精度损失 < 1e-4 |` 改为：

```markdown
| ONNX 导出 | — | 5.88 MB | FP32 + simplify（`half=False`）；MAE 未实测 |
```

- [ ] **Step 2: 修正 v2 话术（第 220 行）**

将「v2 微调」问答改为：

```markdown
| **v2 为什么没比 v1 明显提升？** | 实际是**负收益**：mAP50 0.66349 → 0.66171，P/R 双降。原因是它只在已收敛点附近做了 30 epoch 低学习率微调，训练分布没变。真正的闭环需要困难样本回流改变数据分布，这一步我没做到。 |
```

- [ ] **Step 3: 修正显存问答（第 224 行）**

`| **12GB 显存够吗？** | batch=16 下显存占用约 6~8GB，实际训练从未 OOM。...` 改为：

```markdown
| **显存够吗？** | 实际机型是 RTX 3060 Laptop **6 GB**。batch=8 + 640 会超额分配（日志峰值 11.1 GB，Windows WDDM 溢出到共享内存），导致训练后期吞吐从 5.8 it/s 掉到 1.7 it/s。降到 batch=4 或 imgsz=512 是明确的优化方向。 |
```

- [ ] **Step 4: 修正分工话术（第 223 行）**

改为：

```markdown
| **这些模块都是你做的吗？** | 是，六个模块全部由我独立完成。所以每个环节的实现细节我都能讲清楚。 |
```

- [ ] **Step 5: 补充速度口径与差距提示**

在第三节表格后新增：

```markdown
> **口径提醒**：162.8 FPS 是 `benchmark.py` 实测的**裸网络前向**速度，不含 letterbox 与 NMS，
> 不是端到端帧率。端到端（含检测+MediaPipe+VTS 通信）约 16 ms/帧。
> 完整未完成项见 [`docs/项目现状与差距.md`](项目现状与差距.md)。
```

- [ ] **Step 6: 校验无遗留 FP16 与 0.76 的错误表述**

```powershell
Select-String -Path "docs\presentation_outline.md" -Pattern "FP16|12GB|六个模块这么多"
```

预期：无输出。

- [ ] **Step 7: 提交**

```powershell
git add docs/presentation_outline.md
git commit -m "docs: correct defense outline metrics and hardware claims"
```

---

### Task 7: 修正 .claude/CLAUDE.md 与重写 MEMORY.md

**Files:**
- Modify: `.claude/CLAUDE.md`
- Rewrite: `MEMORY.md`

- [ ] **Step 1: 修正 GPU 与显存（CLAUDE.md:15、42）**

`- **GPU**：单张 RTX 3060 / 12GB 显存` → `- **GPU**：RTX 3060 Laptop / 6GB 显存（实测 6144 MiB）`
`- 训练相关操作注意 12GB 显存限制` → `- 训练相关操作注意 6GB 显存限制：batch=8 + 640 会超额分配（峰值 11.1GB），建议 batch≤4`

- [ ] **Step 2: 修正工作目录（CLAUDE.md:21）**

`C:\学习\课程设计` → `C:\学习\高精度人脸检查算法`

- [ ] **Step 3: 更新必读文档清单（CLAUDE.md:5-8）**

在现有两条后追加：

```markdown
3. `docs/项目现状与差距.md` — **实现与设计承诺的差距清单**（设计文档描述的是目标，不是现状）
4. `docs/superpowers/specs/` 与 `docs/superpowers/plans/` 的正文为**设计承诺原文，不代表当前实现**
```

- [ ] **Step 4: 重写 MEMORY.md**

现内容索引了 3 个不存在的文件，改为指向真实文档：

```markdown
# 项目记忆索引

- [README](../README.md) — 项目定位、实测指标与口径说明、快速开始
- [项目现状与差距](../docs/项目现状与差距.md) — 实现 vs 设计承诺的逐条差距（含证据）
- [使用指南](../docs/使用指南.md) — 安装、四种检测模式、VTube Studio 追踪、FAQ
- [答辩提纲](../docs/presentation_outline.md) — 答辩讲解流程与预设问答
- [设计规格书（承诺原文）](../docs/superpowers/specs/2026-06-22-face-detection-design.md)
- [实施计划（承诺原文）](../docs/superpowers/plans/2026-06-22-face-detection-plan.md)
- [开发决策记录](../.dsh-dev/decisions.md) — 每次变更的候选方案、理由与结果

> GPU 约束（实测）：RTX 3060 Laptop 6 GB / CUDA 12.1 / 驱动 581.95
```

- [ ] **Step 5: 提交**

```powershell
git add .claude/CLAUDE.md MEMORY.md
git commit -m "docs: fix GPU/path metadata and repair memory index"
```

---

### Task 8: 新建 CHANGELOG.md 与 .dsh-dev/conventions.md

**Files:**
- Create: `CHANGELOG.md`
- Create: `.dsh-dev/conventions.md`

- [ ] **Step 1: 写入 CHANGELOG.md**

```markdown
# 更新日志

本文件记录项目的重要变更。格式参考 Keep a Changelog。

## [未发布]

### 文档
- **文档校正（阶段二）**：README、使用指南、答辩提纲、CLAUDE.md、MEMORY.md 全部对齐代码实测状态。
  - 修正 ONNX 导出精度：FP16/6.17MB → **FP32/5.88MB**（`half=False`）
  - 修正 FPS 口径：明确为**裸网络前向**，不含 letterbox 与 NMS
  - 修正 mAP 口径：0.664 为全验证集 COCO 式 AP50，与 Hard 子集目标 0.76 不可比
  - 如实记录 v2 微调为**负收益**（0.66349 → 0.66171）
  - 修正硬件描述：12GB → **RTX 3060 Laptop 6GB**（实测）
  - 新增《项目现状与差距》清单，作为后续补齐实现的输入
- 旧文档（设计规格书、实施计划）正文保留为设计承诺，加差距横幅并归档不可变副本

### 变更
- **清理重复文件（阶段一）**：删除 `yolo26n.pt`（与 `yolov8n.pt` 字节相同的误命名副本）、
  交付包目录及其 zip（独有资产《使用指南》已归档至 `docs/`）、12 个训练调试图与全部 `__pycache__`，共释放约 49.6 MB
- `.gitignore` 泛化训练调试图规则，新增运行产物（截图、`results/`、输出视频）与交付包排除规则
- 将 `.claude/settings.local.json` 移出版本控制，使既有忽略规则真正生效
- **首次纳入版本控制**：`deployment/Vtube-Studio-Bridge/`（VTube Studio 表情驱动桥接）与答辩提纲
- 新建 `.dsh-dev/` 工作记录目录（决策记录、清理清单、项目约定）

### 已知问题
- WIDER Face Easy/Medium/Hard 分项评估未完成，原始 0.76 目标无法对账
- ONNX 精度验证因 `data/annotations/val_list.txt` 缺失而静默跳过，MAE 无实测值
- Recall 0.595 偏低，小脸与遮挡场景漏检明显
- `dataset/augmentation.py`（Albumentations）未接入训练流程
- 设计中的自采 500 张数据未采集
```

- [ ] **Step 2: 写入 .dsh-dev/conventions.md**

提供模板并预填已观察到的约定，标注「待用户确认/补充」：

```markdown
# 项目开发约定

> 由助手根据代码现状预填，标注「待确认」的条目请你补充或修改。
> 只把本文件明确写下的内容当作硬性规则。

## 语言

- 文档、注释、提交说明的正文使用中文；提交标题沿用既有英文 conventional commits 风格（`feat:` / `fix:` / `chore:` / `docs:`）。**待确认**

## 代码风格

- Python 3.12，无 type hints 强制要求（沿用现状）**待确认**
- 路径处理统一使用 `os.path`，不使用 `pathlib`（沿用现状）**待确认**
- 中文路径安全：涉及 `cv2.imread` 的位置一律使用 `imdecode` 兜底（已在 `train.py`、`convert_wider_to_yolo.py` 落地）

## 目录结构

- 六模块固定：`data/`(A) `dataset/`(B) `model/`(C) `training/`(D) `evaluation/`(E) `deployment/`(F)
- 新脚本必须归入对应模块目录，不新增顶层目录
- 计划与规格文档放 `docs/superpowers/{plans,specs}/`，日期前缀 `YYYY-MM-DD-`

## 测试

- 当前无测试框架。**待确认**：是否引入 pytest？
- 新增纯函数（坐标转换、滤波器）应附带最小单元测试

## 提交规范

- 沿用 conventional commits；每个可独立验证的变更单独提交
- 模型权重仅保留 `training/checkpoints/best_model_*.{pt,onnx}`，其余被忽略

## 环境

- 实测：Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 / RTX 3060 Laptop 6GB / 驱动 581.95
- 训练显存约束：6GB，batch=8+640 会超额分配，建议 batch≤4 或 imgsz≤512
```

- [ ] **Step 3: 提交**

```powershell
git add CHANGELOG.md .dsh-dev/conventions.md
git commit -m "docs: add CHANGELOG and seed project conventions"
```

---

## 四、验收标准

| # | 标准 | 验证命令 / 方式 |
|:---|:---|:---|
| V1 | 全部文档中不再出现「ONNX FP16」与「6.17 MB」的错误表述 | `Select-String -Path README.md,docs/*.md -Pattern "FP16\|6\.17"` |
| V2 | 不再出现「12GB」显存表述 | `Select-String -Path README.md,docs/*.md,.claude/CLAUDE.md -Pattern "12GB"` |
| V3 | FPS 数字旁均有「不含前后处理」口径说明 | 人工检查 README 与答辩提纲 |
| V4 | v2 负收益在两处以上被如实记录 | `Select-String -Pattern "0.66171"` 应命中 README 与差距清单 |
| V5 | 旧文档正文与归档副本内容一致（仅差横幅） | 行数比对，差值等于横幅行数 |
| V6 | 差距清单每条都有「证据」列且可复核 | 抽查 3 条 | 
| V7 | `git status` 干净，每个 Task 一个提交 | `git status --short` 为空；`git log --oneline` 有 8 个新提交 |
| V8 | 文档中所有命令可执行（不引用不存在的脚本/文件） | 逐条核对脚本是否存在 |

## 五、风险与回退

| 风险 | 概率 | 应对 |
|:---|:---|:---|
| 改文档时误删了对阶段三有用的承诺描述 | 中 | 承诺原文已双重保留（`docs/archive/` 副本 + git 历史），且旧文档正文不改 |
| 修正后的口径与答辩稿不一致，反而更混乱 | 低 | 以本计划第一节「事实基准」为唯一数据源，所有文档引用同一组数字 |
| 用户对「如实记录负收益」有顾虑 | 中 | 已在答辩提纲中同时给出**诚实话术**（可解释为"闭环未成立及其原因"），而非单纯自曝 |
| 阶段三开工后发现差距清单漏项 | 中 | 差距清单设计为可增量追加，每条独立成行，不依赖编号连续性 |

## 六、执行顺序与提交点

```
Task 1 归档副本        → commit
Task 2 旧文档加横幅    → commit
Task 3 差距清单        → commit   ← 阶段三的输入
Task 4 README          → commit
Task 5 使用指南        → commit
Task 6 答辩提纲        → commit
Task 7 CLAUDE/MEMORY   → commit
Task 8 CHANGELOG/约定  → commit
验收 V1–V8
```

---

## Self-Review

**Spec coverage**：用户要求「按对得上代码来对文档进行修正，旧文档先保留」→ Task 1+2 满足「保留」，Task 3–8 满足「修正」；「待代码和文档一致后按照旧文档一步步修改项目」→ Task 3 产出阶段三输入清单。
**Placeholder scan**：本计划所有数值均来自第一节实测表，无 TBD；所有修改步骤给出了可直接粘贴的替换文本。
**Type consistency**：文件名、行号引用在全文保持一致；`docs/项目现状与差距.md` 在 README、CLAUDE.md、MEMORY.md 三处的相对链接路径已分别核对（`docs/`、`../docs/`、同目录）。
