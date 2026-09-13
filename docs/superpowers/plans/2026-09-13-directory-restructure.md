# 目录重组实施计划（阶段 2A）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把项目从"六模块人头分工"结构重组为业界通用的 ML 仓库布局（`src/` 库代码 / `apps/` 独立应用 / `data/` 数据 / `artifacts/` 产出 / `configs/` 配置），并保证行为不变。

**Architecture:** 代码、数据、产出三者分离。新增 `src/paths.py` 作为**唯一**路径解析来源，取代散落在 6 个文件中的 `dirname(dirname(__file__))`。所有脚本改为以模块方式从项目根运行：`python -m src.<pkg>.<module>`。

**Tech Stack:** Python 3.12 / 现有依赖不变。迁移用 `git mv` 保留文件历史。验证手段：导入检查、路径断言、实跑推理与基准（**不重跑训练**）。

---

## 零、前置决定（已与用户确认）

| 项 | 决定 |
|:---|:---|
| 目录方案 | 方案 A：标准 ML 仓库布局 |
| 文档结构 | `docs/` 根 = 现状文档；`docs/archive/` = 历史文档（正文不改）；`docs/plans/` = 工作计划 |
| 课程设计产物 | `_template_full.txt`、`_temp_team.txt` **删除**；`presentation_outline.md` 改造为面试讲解提纲（内容改造属阶段 2B） |
| 项目定位 | **个人独立完成的工程实践项目**（兼具学习性质）；文档中不得再出现"课程设计/实训/答辩/评委/六人分工/外六内一" |
| 生成的数据集 yaml | **保持在 `data/annotations/widerface.yaml` 不动**。它由 `train.py` 动态写入、内容含 `path: data/raw` 相对路径，ultralytics 的相对路径解析行为未经实测验证，改动风险高收益低 |

---

## 一、完整文件映射表

| 原路径 | 新路径 | 说明 |
|:---|:---|:---|
| `data/download_widerface.py` | `src/data/download.py` | 注：**实为手动下载说明脚本，不执行下载** |
| `data/check_annotation.py` | `src/data/qc.py` | 标注质检 |
| `data/convert_wider_to_yolo.py` | `src/data/convert.py` | WIDER → YOLO 标注转换 |
| `data/generate_list.py` | `src/data/split.py` | 生成 train/val 清单 |
| `dataset/dataloader.py` | `src/data/loader.py` | 数据集封装 |
| `dataset/augmentation.py` | `src/data/augment.py` | Albumentations 管线（当前未接入训练） |
| `dataset/vis_aug.py` | `src/data/vis_aug.py` | 增强可视化（开发工具） |
| `training/config.py` | `src/train/config.py` | 超参管理 |
| `training/train.py` | `src/train/train.py` | v1 + v2 训练 |
| `training/resume_train.py` | `src/train/resume.py` | 断点续训 |
| `model/model_test.py` | `src/train/check_model.py` | 注：**当前引用不存在的 `yolov8n-face.pt`，实为坏的** |
| `model/model_config.yaml` | `configs/model.yaml` | 训练配置 |
| `evaluation/evaluate.py` | `src/eval/evaluate.py` | mAP 评估 |
| `evaluation/analyze_errors.py` | `src/eval/analyze_errors.py` | 困难样本分析 |
| `deployment/export_onnx.py` | `src/deploy/export_onnx.py` | ONNX 导出 |
| `deployment/realtime_detect.py` | `src/deploy/detect.py` | 实时检测统一入口 |
| `deployment/benchmark.py` | `src/deploy/benchmark.py` | 性能基准 |
| `deployment/Vtube-Studio-Bridge/**` | `apps/vtube_bridge/**` | 内部结构**不变**（保持 4 层深度） |
| `training/checkpoints/**` | `artifacts/checkpoints/**` | 权重与训练曲线 |
| `evaluation/reports/**` | `artifacts/reports/**` | 评估与检测输出图 |
| `training/training_log.txt` | `artifacts/logs/training_log.txt` | 被 gitignore |
| `data/raw/**`、`data/annotations/**` | **不动** | 数据位置不变 |
| `yolov8n.pt`（根） | **不动** | 被 gitignore、ultralytics 可自动下载；`configs/model.yaml` 以裸名引用，移动会引入 CWD 解析歧义 |
| `_template_full.txt`、`_temp_team.txt` | **删除** | 课程设计产物 |
| `docs/superpowers/specs/2026-06-22-face-detection-design.md` | `docs/archive/2026-06-22-设计规格书.md` | 历史文档 |
| `docs/superpowers/plans/2026-06-22-face-detection-plan.md` | `docs/archive/2026-06-22-实施计划.md` | 历史文档 |
| `docs/superpowers/`（空目录） | **删除** | 迁移后清空 |
| `docs/archive/*.原始.md`（2 份） | **删除** | 上一轮创建的冗余副本；改为直接移动原文件，一处一份 |
| `docs/superpowers/plans/2026-09-13-documentation-realignment.md` | `docs/plans/2026-09-13-文档校正.md` | 我的工作计划 |

**最终结构：**

```
configs/model.yaml
src/
├── __init__.py   paths.py
├── data/   __init__.py  download.py  qc.py  convert.py  split.py  loader.py  augment.py  vis_aug.py
├── train/  __init__.py  config.py  train.py  resume.py  check_model.py
├── eval/   __init__.py  evaluate.py  analyze_errors.py
└── deploy/ __init__.py  export_onnx.py  detect.py  benchmark.py
apps/vtube_bridge/          main.py  run_gui.bat  thirdparty/  vtube_studio_bridge/
data/                       raw/  annotations/
artifacts/                  checkpoints/  reports/  logs/
docs/                       archive/  plans/  + 现状文档
tests/                      （占位，阶段三填充）
yolov8n.pt  README.md  CHANGELOG.md  MEMORY.md  requirements.txt  .gitignore
```

---

## 二、任务分解

### Task R1: 建立骨架与统一路径模块

**Files:**
- Create: `src/__init__.py`、`src/paths.py`、`src/data/__init__.py`、`src/train/__init__.py`、`src/eval/__init__.py`、`src/deploy/__init__.py`、`tests/__init__.py`
- Create: `configs/`、`artifacts/`（目录）

- [ ] **Step 1: 写入 `src/paths.py`（唯一路径来源）**

```python
"""Central path resolution — the single source of truth for where things live.

Run scripts as modules from the project root, e.g.:

    python -m src.train.train
    python -m src.deploy.detect --input 0
"""
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configuration
CONFIG_DIR = os.path.join(PROJECT_ROOT, "configs")
MODEL_CONFIG_PATH = os.path.join(CONFIG_DIR, "model.yaml")

# Dataset
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
ANNO_DIR = os.path.join(DATA_DIR, "annotations")
TRAIN_IMAGES_DIR = os.path.join(RAW_DIR, "WIDER_train", "images")
VAL_IMAGES_DIR = os.path.join(RAW_DIR, "WIDER_val", "images")
WIDER_YAML = os.path.join(ANNO_DIR, "widerface.yaml")
TRAIN_LIST = os.path.join(ANNO_DIR, "train_list.txt")
VAL_LIST = os.path.join(ANNO_DIR, "val_list.txt")

# Outputs
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
CHECKPOINT_DIR = os.path.join(ARTIFACTS_DIR, "checkpoints")
REPORTS_DIR = os.path.join(ARTIFACTS_DIR, "reports")
LOGS_DIR = os.path.join(ARTIFACTS_DIR, "logs")

# Deployed weights
BEST_MODEL_V1_PT = os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")
BEST_MODEL_V2_PT = os.path.join(CHECKPOINT_DIR, "best_model_v2.pt")
BEST_MODEL_V2_ONNX = os.path.join(CHECKPOINT_DIR, "best_model_v2.onnx")
```

- [ ] **Step 2: 创建空的 `__init__.py` 与目录**

- [ ] **Step 3: 验证路径解析正确**

```powershell
python -c "from src import paths; print(paths.PROJECT_ROOT); print(paths.MODEL_CONFIG_PATH); print(paths.CHECKPOINT_DIR)"
```

预期：三条路径均指向项目绝对路径，且 `PROJECT_ROOT` 等于工作目录。

- [ ] **Step 4: 提交**

```powershell
git add src configs artifacts tests
git commit -m "refactor: add src/paths.py as single source of path resolution"
```

---

### Task R2: 迁移产物与配置目录

**Files:**
- Move: `model/model_config.yaml` → `configs/model.yaml`
- Move: `training/checkpoints/**` → `artifacts/checkpoints/**`
- Move: `evaluation/reports/**` → `artifacts/reports/**`
- Move: `training/training_log.txt` → `artifacts/logs/training_log.txt`

- [ ] **Step 1: git mv 三个目录**

```powershell
git mv model/model_config.yaml configs/model.yaml
git mv training/checkpoints artifacts/checkpoints
git mv evaluation/reports artifacts/reports
Move-Item training/training_log.txt artifacts/logs/training_log.txt
```

- [ ] **Step 2: 更新 `.gitignore`**

```gitignore
# Model weights — only ship best deployment models
*.pt
!artifacts/checkpoints/best_model_v1.pt
!artifacts/checkpoints/best_model_v2.pt
*.onnx
!artifacts/checkpoints/best_model_v2.onnx

# Dataset (large files, downloaded separately)
data/raw/
data/annotations/wider_face_split/

# Training intermediates (regenerated on each run)
artifacts/checkpoints/*/weights/
artifacts/checkpoints/**/train_batch*.jpg
artifacts/logs/
runs/
```

- [ ] **Step 3: 校验权重未被忽略**

```powershell
git check-ignore -v artifacts/checkpoints/best_model_v2.pt artifacts/checkpoints/best_model_v1.pt artifacts/checkpoints/best_model_v2.onnx
```

预期：**无输出**（三个文件均未被忽略）。

- [ ] **Step 4: 校验被忽略的仍被忽略**

```powershell
git check-ignore -v artifacts/checkpoints/v1_baseline/weights/last.pt artifacts/logs/training_log.txt
```

预期：两者均命中断言行。

- [ ] **Step 5: 提交**

```powershell
git add -A
git commit -m "refactor: move model config to configs/ and outputs to artifacts/"
```

---

### Task R3: 迁移数据模块（`data/` + `dataset/` → `src/data/`）

**Files:**
- Move: `data/download_widerface.py` → `src/data/download.py`
- Move: `data/check_annotation.py` → `src/data/qc.py`
- Move: `data/convert_wider_to_yolo.py` → `src/data/convert.py`
- Move: `data/generate_list.py` → `src/data/split.py`
- Move: `dataset/dataloader.py` → `src/data/loader.py`
- Move: `dataset/augmentation.py` → `src/data/augment.py`
- Move: `dataset/vis_aug.py` → `src/data/vis_aug.py`

- [ ] **Step 1: git mv 七个文件**

- [ ] **Step 2: 改写 `src/data/qc.py` 的路径常量**

```python
# 原：
ANNO_DIR = os.path.join(os.path.dirname(__file__), "annotations")
RAW_DIR = os.path.join(os.path.dirname(__file__), "raw")
REPORT_PATH = os.path.join(ANNO_DIR, "quality_report.txt")
```
```python
# 改：
from src.paths import ANNO_DIR, RAW_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
REPORT_PATH = os.path.join(ANNO_DIR, "quality_report.txt")
```

同时把该文件中所有 `os.path.join(RAW_DIR, "WIDER_train", "images")` 替换为 `TRAIN_IMAGES_DIR`，val 同理。**该脚本先前引用 `data/annotations/wider_face_split/*.txt`，路径保持不变即可成立。**

- [ ] **Step 3: 改写 `src/data/convert.py` 与 `src/data/split.py`**

```python
from src.paths import ANNO_DIR, RAW_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR, TRAIN_LIST, VAL_LIST
```

并把 `os.path.join(ANNO_DIR, "wider_face_split", ...)` 保留原样（该目录位置未变）。

- [ ] **Step 4: 改写 `src/data/download.py`**

```python
from src.paths import RAW_DIR, ANNO_DIR
```

并修正其 docstring：明确说明**本脚本不执行下载，仅打印手动下载指引**。

- [ ] **Step 5: 改写 `src/data/loader.py`**

```python
# 原：
ANNO_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "annotations")
RAW_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
```
```python
# 改：
from src.paths import ANNO_DIR, RAW_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
```

并删除文件末尾 `if __name__ == "__main__":` 块中的 `sys.path.insert` 两行（`-m` 运行已保证根目录在 `sys.path`）。

- [ ] **Step 6: 改写 `src/data/vis_aug.py`**

```python
from src.paths import REPORTS_DIR
OUTPUT_DIR = REPORTS_DIR
```
并把导入改为 `from src.data.loader import create_dataloader`，删除 `sys.path.insert` 两行。

- [ ] **Step 7: 验证导入全部成立**

```powershell
python -c "import src.data.qc, src.data.convert, src.data.split, src.data.loader, src.data.augment, src.data.vis_aug, src.data.download; print('data modules OK')"
```

预期：`data modules OK`，无 ImportError。

- [ ] **Step 8: 验证数据集路径解析命中真实文件**

```powershell
python -c "
from src import paths
import os
for name in ['TRAIN_IMAGES_DIR','VAL_IMAGES_DIR','ANNO_DIR','RAW_DIR']:
    p = getattr(paths, name); print(name, os.path.exists(p), p)
"
```

预期：四个路径 `True`。

- [ ] **Step 9: 提交**

```powershell
git add -A
git commit -m "refactor: move data pipeline and dataset code into src/data"
```

---

### Task R4: 迁移训练模块

**Files:**
- Move: `training/config.py` → `src/train/config.py`
- Move: `training/train.py` → `src/train/train.py`
- Move: `training/resume_train.py` → `src/train/resume.py`
- Move: `model/model_test.py` → `src/train/check_model.py`

- [ ] **Step 1: git mv 四个文件**

- [ ] **Step 2: 改写 `src/train/config.py`**

```python
from src.paths import MODEL_CONFIG_PATH
CONFIG_PATH = MODEL_CONFIG_PATH
```

- [ ] **Step 3: 改写 `src/train/train.py`**

```python
from src.paths import (
    ANNO_DIR, DATA_DIR, RAW_DIR, CHECKPOINT_DIR, WIDER_YAML,
)
from src.train.config import load_config, print_training_summary, WIDER_TRAIN_SIZE
```

删除 `sys.path.insert(PROJECT_ROOT)` 一行。`prepare_data_yaml()` 中：

- 写入目标改为 `WIDER_YAML`
- `rel_path` 仍用 `os.path.relpath(RAW_DIR, PROJECT_ROOT)`，其中 `PROJECT_ROOT` 改为从 `src.paths` 导入 —— **保持 `path: data/raw` 的相对形式不变**（已验证的训练行为，不得改动）

`os.path.join(CHECKPOINT_DIR, "v1_baseline", "weights", "best.pt")` 等拷贝逻辑中的 `CHECKPOINT_DIR` 自动指向新位置，无需改字面量。

- [ ] **Step 4: 改写 `src/train/resume.py`**

同 Step 3 的路径与导入替换（该文件与 `train.py` 的路径常量高度重复，改动一一对应）。

- [ ] **Step 5: 改写 `src/train/check_model.py`**

将 `YOLO("yolov8n-face.pt")` 改为 `YOLO("yolov8n.pt")`（后者存在且与 `configs/model.yaml` 的 `model_name` 一致）。
在 docstring 中标注：**本脚本为诊断脚本，非单元测试；`yolov8n-face.pt` 从未存在是原缺陷，已于本次修正**。

- [ ] **Step 6: 验证配置可加载、训练摘要可打印**

```powershell
python -m src.train.config
```

预期：打印 `Training Configuration Summary`，且 `Batch Size: 8`、`Mixed Precision: True`。

- [ ] **Step 7: 验证训练脚本路径解析（不启动训练）**

```powershell
python -c "
import ast, sys
src = open('src/train/train.py', encoding='utf-8').read()
tree = ast.parse(src)
# 断言：不再出现 dirname(dirname(__file__)) 与 sys.path.insert
assert 'sys.path.insert' not in src, 'still has sys.path hack'
assert 'dirname(os.path.dirname(__file__))' not in src, 'still has relative root hack'
print('train.py path handling migrated OK')
"
```

- [ ] **Step 8: 验证数据 yaml 生成逻辑（不启动训练）**

```powershell
python -c "
import src.train.train as t
p = t.prepare_data_yaml()
print(open(p, encoding='utf-8').read())
"
```

预期：生成 `data/annotations/widerface.yaml`，内容含 `path: data\raw`、`train: WIDER_train`、`val: WIDER_val`、`nc: 1`。

- [ ] **Step 9: 提交**

```powershell
git add -A
git commit -m "refactor: move training code into src/train and fix model_name reference"
```

---

### Task R5: 迁移评估模块

**Files:**
- Move: `evaluation/evaluate.py` → `src/eval/evaluate.py`
- Move: `evaluation/analyze_errors.py` → `src/eval/analyze_errors.py`

- [ ] **Step 1: git mv 两个文件**

- [ ] **Step 2: 改写 `src/eval/evaluate.py`**

```python
from src.paths import ANNO_DIR, DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR, WIDER_YAML
```

删除 `sys.path.insert` 一行。所有 `os.path.join(CHECKPOINT_DIR, "best_model_v1.pt")` 保持字面量（`CHECKPOINT_DIR` 已指向新位置）。

- [ ] **Step 3: 改写 `src/eval/analyze_errors.py`**

```python
from src.paths import ANNO_DIR, DATA_DIR, CHECKPOINT_DIR, REPORTS_DIR, RAW_DIR, TRAIN_IMAGES_DIR, VAL_IMAGES_DIR
from src.data.loader import parse_wider_annotation
```

删除 `sys.path.insert` 一行与原先的 `from dataset.dataloader import parse_wider_annotation`。

- [ ] **Step 4: 验证导入成立**

```powershell
python -c "import src.eval.evaluate, src.eval.analyze_errors; print('eval modules OK')"
```

- [ ] **Step 5: 提交**

```powershell
git add -A
git commit -m "refactor: move evaluation code into src/eval"
```

---

### Task R6: 迁移部署模块（含实跑验证）

**Files:**
- Move: `deployment/export_onnx.py` → `src/deploy/export_onnx.py`
- Move: `deployment/realtime_detect.py` → `src/deploy/detect.py`
- Move: `deployment/benchmark.py` → `src/deploy/benchmark.py`
- Modify: `.gitignore`（若 `deployment/` 清空则删目录）

- [ ] **Step 1: git mv 三个文件**

- [ ] **Step 2: 改写 `src/deploy/detect.py`**

```python
from src.paths import BEST_MODEL_V2_PT
DEFAULT_MODEL = BEST_MODEL_V2_PT
```

- [ ] **Step 3: 改写 `src/deploy/export_onnx.py`**

```python
from src.paths import ANNO_DIR, CHECKPOINT_DIR, VAL_LIST
```

- [ ] **Step 4: 改写 `src/deploy/benchmark.py`**

```python
from src.paths import CHECKPOINT_DIR
```

- [ ] **Step 5: 验证导入与 `--help`**

```powershell
python -c "import src.deploy.detect, src.deploy.benchmark, src.deploy.export_onnx; print('deploy modules OK')"
python -m src.deploy.detect --help
```

预期：`--help` 打印 `--input/--model/--imgsz/--conf/--save` 五个参数，且 `--model` 默认值为 `...\artifacts\checkpoints\best_model_v2.pt`。

- [ ] **Step 6: 实跑单图检测（端到端验证）**

```powershell
python -m src.deploy.detect --input artifacts/reports/test_detect_0.jpg --save artifacts/reports/_smoke
```

预期：打印检测到的人脸数与 FPS，`artifacts/reports/_smoke/` 下生成一张标注图。**这是本次重组最关键的验证点**——它同时验证了路径解析、模型加载、推理与输出写入。

- [ ] **Step 7: 清理冒烟产物**

```powershell
Remove-Item artifacts/reports/_smoke -Recurse -Force
```

- [ ] **Step 8: 实跑基准（验证权重路径与 ONNX 加载）**

```powershell
python -m src.deploy.benchmark
```

预期：打印 PyTorch 与 ONNX 的延迟/FPS 与模型体积（PyTorch ≈ 5.95 MB / ONNX ≈ 5.88 MB），并给出 FPS/Size 的 PASS/FAIL 判定。

- [ ] **Step 9: 提交**

```powershell
git add -A
git commit -m "refactor: move deployment code into src/deploy"
```

---

### Task R7: 迁移 VTS 桥接至 apps/

**Files:**
- Move: `deployment/Vtube-Studio-Bridge/**` → `apps/vtube_bridge/**`
- Modify: `apps/vtube_bridge/vtube_studio_bridge/vtube_studio_bridge.py`
- Modify: `apps/vtube_bridge/run_gui.bat`

- [ ] **Step 1: git mv 整个目录**

```powershell
git mv deployment/Vtube-Studio-Bridge apps/vtube_bridge
```

> **深度不变性**：桥接内部用 `Path(__file__).resolve().parents[1].parents[1]` 求项目根，依赖文件位于根下第 4 层。`apps/vtube_bridge/vtube_studio_bridge/xxx.py` **同为第 4 层**，故该行无需改动。

- [ ] **Step 2: 修正内部写死的 MediaPipe 模型路径**

```python
# 原：
DEFAULT_MEDIAPIPE_MODEL = PROJECT_ROOT / "deployment" / "Vtube-Studio-Bridge" / "thirdparty" / "MediaPipe" / "models" / "face_landmarker.task"
# 改：
DEFAULT_MEDIAPIPE_MODEL = PROJECT_ROOT / "apps" / "vtube_bridge" / "thirdparty" / "MediaPipe" / "models" / "face_landmarker.task"
```

同理修正 `DEFAULT_MODEL` 的注释文档串中若出现的旧路径。

- [ ] **Step 3: 修正 `run_gui.bat`**

原文件引用 `..\..\.venv\Scripts\pythonw.exe`，而**项目中不存在 `.venv`**（原缺陷）。改为使用 PATH 中的 python：

```bat
@echo off
setlocal
cd /d "%~dp0"
start "vtube_bridge" pythonw "main.py" --input 0 --send-fps 30 --landmarks
```

- [ ] **Step 4: 验证两个默认路径命中真实文件**

```powershell
python -c "
import sys; sys.path.insert(0, 'apps/vtube_bridge')
from vtube_studio_bridge.vtube_studio_bridge import DEFAULT_MODEL, DEFAULT_MEDIAPIPE_MODEL
print('model   ', DEFAULT_MODEL, DEFAULT_MODEL.exists())
print('mediapipe', DEFAULT_MEDIAPIPE_MODEL, DEFAULT_MEDIAPIPE_MODEL.exists())
"
```

预期：两条均为 `True`。

- [ ] **Step 5: 验证 `--help` 可用（同时验证 PyQt5 可导入）**

```powershell
python apps/vtube_bridge/main.py --help
```

预期：打印含 `--input/--vtshost/--vtsport/--mediapipe-model/--landmarks/--send-fps` 等参数的使用说明。
若因 PyQt5 缺失而失败，记录为**环境问题（非本次重组引入）**并继续。

- [ ] **Step 6: 确认 `deployment/` 已空并删除**

```powershell
Get-ChildItem deployment -Recurse -Force
Remove-Item deployment -Recurse -Force
```

- [ ] **Step 7: 提交**

```powershell
git add -A
git commit -m "refactor: move VTube Studio bridge to apps/ and fix broken launcher"
```

---

### Task R8: 文档结构归位与课程设计产物清理

**Files:**
- Move: `docs/superpowers/specs/...design.md` → `docs/archive/2026-06-22-设计规格书.md`
- Move: `docs/superpowers/plans/...plan.md` → `docs/archive/2026-06-22-实施计划.md`
- Move: `docs/superpowers/plans/2026-09-13-documentation-realignment.md` → `docs/plans/2026-09-13-文档校正.md`
- Delete: `docs/archive/2026-06-22-design-spec.原始.md`、`docs/archive/2026-06-22-plan.原始.md`、`docs/superpowers/`
- Delete: `_template_full.txt`、`_temp_team.txt`
- Create: `docs/archive/README.md`

- [ ] **Step 1: 移动旧文档**

```powershell
git mv docs/superpowers/specs/2026-06-22-face-detection-design.md docs/archive/2026-06-22-设计规格书.md
git mv docs/superpowers/plans/2026-06-22-face-detection-plan.md docs/archive/2026-06-22-实施计划.md
git mv docs/superpowers/plans/2026-09-13-documentation-realignment.md docs/plans/2026-09-13-文档校正.md
```

- [ ] **Step 2: 删除冗余副本与空目录**

```powershell
git rm docs/archive/2026-06-22-design-spec.原始.md docs/archive/2026-06-22-plan.原始.md
Remove-Item docs/superpowers -Recurse -Force
```

- [ ] **Step 3: 删除课程设计产物**

```powershell
git rm _template_full.txt _temp_team.txt
```

- [ ] **Step 4: 写入 `docs/archive/README.md`**

```markdown
# 归档文档

这里存放**历史文档**，正文保持原样、不再更新。

| 文档 | 说明 |
|:---|:---|
| `2026-06-22-设计规格书.md` | 2026-06-22 撰写的设计规格书。**描述的是设计承诺，不是当前实现。** |
| `2026-06-22-实施计划.md` | 同期的任务分解计划。**其中部分任务从未完成或已被其他方式取代。** |

## 为什么保留

项目采用"先建立可运行的原型，再逐步校正为可信文档"的方式推进。这些文档记录了当初的
目标与假设，是**逐项核对实现差距、补齐缺漏的输入清单**，因此不删除、不改写。

## 当前文档在哪

- 项目概览：[`../../README.md`](../../README.md)
- 架构说明：[`../架构说明.md`](../架构说明.md)
- 实现与设计的差距：[`../项目现状与差距.md`](../项目现状与差距.md)
- 使用指南：[`../使用指南.md`](../使用指南.md)

> ⚠️ 阅读旧文档时请勿将其中的指标、目录结构、模块分工当作现状。
```

- [ ] **Step 5: 校验文档结构**

```powershell
Get-ChildItem docs -Recurse -File | ForEach-Object { $_.FullName.Substring((Get-Location).Path.Length+1) }
Test-Path docs/superpowers, _template_full.txt, _temp_team.txt
```

预期：文件清单符合最终结构；`Test-Path` 三项均为 `False`。

- [ ] **Step 6: 提交**

```powershell
git add -A
git commit -m "docs: consolidate archives, drop coursework artifacts, add archive README"
```

---

### Task R9: 全链路回归验证

- [ ] **Step 1: 复现验收清单**

| # | 验收项 | 命令 | 预期 |
|:---|:---|:---|:---|
| V1 | 无残留旧路径常量 | `Select-String -Path src/**/*.py -Pattern "dirname\(os\.path\.dirname\(__file__\)\)"` | 无输出 |
| V2 | 无残留 `sys.path.insert` 根目录 hack | `Select-String -Path src/**/*.py -Pattern "sys\.path\.insert"` | 无输出 |
| V3 | 全部模块可导入 | `python -c "import src.data.qc, src.data.convert, src.data.split, src.data.loader, src.data.augment, src.train.config, src.eval.evaluate, src.eval.analyze_errors, src.deploy.detect, src.deploy.benchmark, src.deploy.export_onnx"` | 无 ImportError |
| V4 | 配置加载正常 | `python -m src.train.config` | 打印摘要，Batch Size 8 |
| V5 | 单图检测端到端可用 | `python -m src.deploy.detect --input artifacts/reports/test_detect_0.jpg` | 检测到人脸并显示 FPS |
| V6 | 基准测试可用 | `python -m src.deploy.benchmark` | 打印 FPS 与模型体积 |
| V7 | 桥接默认路径有效 | 见 Task R7 Step 4 | 两条 `True` |
| V8 | 权重仍被 Git 跟踪 | `git ls-files artifacts/checkpoints/*.pt artifacts/checkpoints/*.onnx` | 列出 3 个文件 |
| V9 | 被忽略项仍被忽略 | `git check-ignore -v artifacts/checkpoints/v1_baseline/weights/last.pt` | 命中断言行 |
| V10 | 工作树干净 | `git status --short` | 空 |

- [ ] **Step 2: 提交回归记录**

在 `.dsh-dev/decisions.md` 追加"阶段 2A 完成"条目，记录每条验收的实际结果（含失败的）。

---

## 三、风险与回退

| 风险 | 概率 | 影响 | 应对 |
|:---|:---|:---|:---|
| **训练无法重跑验证** | 高 | 训练脚本的改动未经实跑确认 | 用 AST 静态断言 + `prepare_data_yaml()` 实跑 + 导入检查覆盖；`path: data/raw` 的相对形式**保持原样不动** |
| ultralytics 相对路径解析行为变化 | 低 | 训练时找不到数据集 | 不改动 yaml 位置与 `path` 内容；`prepare_data_yaml()` 实跑输出可逐字比对 |
| 桥接 PyQt5 未安装导致 `--help` 失败 | 中 | 无法验证桥接 | 判定为环境问题而非回归；记录并继续 |
| `git mv` 后 `.gitignore` 例外规则失效 | 中 | 权重被忽略、仓库丢模型 | Task R2 Step 3/4 用 `git check-ignore -v` 双向校验 |
| 中间状态不可运行 | 确定 | 逐 Task 之间仓库处于半迁移状态 | 每个 Task 内自带验证；**不在全部完成前对外声称可用** |
| 重组范围过大 | 中 | 引入难以定位的问题 | 全程 `git mv` 保留历史；每个 Task 一次提交，可单独 revert |

**回退方式**：`git reset --hard <重组前提交>`（重组前基线为 `a42e884`），工作树与索引一并恢复。

---

## 四、影响面清单（已核实的路径/导入引用）

| 文件 | 需改内容 |
|:---|:---|
| `src/data/qc.py` | `ANNO_DIR`/`RAW_DIR` 常量 → `src.paths` |
| `src/data/convert.py` | 同上 |
| `src/data/split.py` | 同上 |
| `src/data/download.py` | 同上 + 修正误导性 docstring |
| `src/data/loader.py` | 2 处 `dirname(dirname(...))` + `sys.path.insert` |
| `src/data/vis_aug.py` | `OUTPUT_DIR` → `REPORTS_DIR`；导入改 `src.data.loader`；删 `sys.path.insert` |
| `src/train/config.py` | `CONFIG_PATH` → `MODEL_CONFIG_PATH` |
| `src/train/train.py` | 4 处路径 + `from training.config` → `from src.train.config` + 删 `sys.path.insert` |
| `src/train/resume.py` | 同上 |
| `src/train/check_model.py` | `yolov8n-face.pt` → `yolov8n.pt` |
| `src/eval/evaluate.py` | 3 处路径 + 删 `sys.path.insert` |
| `src/eval/analyze_errors.py` | 2 处路径 + 导入改 `src.data.loader` + 删 `sys.path.insert` |
| `src/deploy/detect.py` | `DEFAULT_MODEL` |
| `src/deploy/export_onnx.py` | 3 处路径 |
| `src/deploy/benchmark.py` | `CHECKPOINT_DIR` |
| `apps/vtube_bridge/.../vtube_studio_bridge.py` | `DEFAULT_MEDIAPIPE_MODEL`（`PROJECT_ROOT` 行不用改） |
| `apps/vtube_bridge/run_gui.bat` | `.venv` 路径 → `pythonw` |
| `.gitignore` | 权重例外与训练产物路径 |

---

## Self-Review

**Spec coverage**：用户要求「目录也重新组织」→ Task R1–R7 覆盖全部 21 个 Python 文件与 3 个产出目录；「删除课程设计产物」→ Task R8；「新旧文档不混放」→ Task R8。
**Placeholder scan**：所有目标路径、代码片段、验证命令均已给出实际内容，无 TBD。
**Type consistency**：`src.paths` 导出的常量名在全部任务中保持一致（`TRAIN_IMAGES_DIR`/`VAL_IMAGES_DIR`/`CHECKPOINT_DIR`/`REPORTS_DIR`/`WIDER_YAML`/`BEST_MODEL_V2_PT`）；`check_model.py` 与 `configs/model.yaml` 的 `model_name` 均为 `yolov8n.pt`。
**遗漏补正**：原计划未覆盖 `run_gui.bat`（引用不存在的 `.venv`）与 `download.py`（不执行下载）两处实际缺陷，已在 Task R7 与 Task R3 中纳入修正。
