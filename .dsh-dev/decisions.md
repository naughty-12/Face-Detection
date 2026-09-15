# 开发决策记录

> 记录每次更新的候选方案、选定理由与结果。新条目追加在最上方。

---

## 2026-09-13 阶段 2A：目录重组（人头分工结构 → 标准 ML 仓库布局）

### 背景

用户要求：① 修正"外六内一/六人分工"架构叙事（实际由一人独立完成，也非课程设计）；
② 目录本身也重新组织。定位澄清为**个人独立完成的学习型工程项目**。

### 决策 1：目录布局

| 候选方案 | 优缺点 | 结论 |
|:---|:---|:---|
| **A. 标准 ML 仓库布局**（`src/ apps/ configs/ data/ artifacts/`） | 代码/数据/产出三者分离，符合业界惯例；缺点是改动面最大 | **选定** |
| B. 保留原六目录只做正名 | 风险低，但顶层仍有 9 个目录，对单人项目过于碎片化 | 未选 |
| C. 按流水线顺序编号脚本 | 顺序直观，但插入一步就要全部重编号，长期维护差 | 未选 |

**理由**：单人项目的结构应体现**工程习惯**，而不是"人数分工"。要抹掉的是"六个人"的伪装，
而标准布局恰好用通行工程惯例替换它，比单纯改名更有说服力。

### 决策 2：以 `src/paths.py` 作为唯一路径来源

取代原先散落在 6 个文件中的 `os.path.dirname(os.path.dirname(__file__))` + `sys.path.insert`。
**理由**：文件一旦进入子包，按自身深度推算项目根必然算错（`src/deploy/detect.py` 会算出 `src/`）。
脚本统一改为从项目根以模块方式运行：`python -m src.<pkg>.<module>`。

### 决策 3：不动 `widerface.yaml` 的位置与相对路径形式

`data/annotations/widerface.yaml` 由训练脚本动态生成，内含 `path: data/raw`。
**理由**：ultralytics 对 data.yaml 中相对 `path` 的解析行为未经实测，改动风险高、收益低。
**验证**：迁移后 `prepare_data_yaml()` 的输出与迁移前**逐字一致**，`git diff` 为空。

### 决策 4：保持桥接文件的目录深度不变（4 层）

`deployment/Vtube-Studio-Bridge/vtube_studio_bridge/x.py` → `apps/vtube_bridge/vtube_studio_bridge/x.py`。
**理由**：桥接用 `Path(__file__).resolve().parents[1].parents[1]` 求项目根，依赖"文件位于根下第 4 层"。
新位置同为第 4 层，故该行无需改动。
**验证**：实测 `PROJECT_ROOT` 仍解析为项目根、`BRIDGE_ROOT` 为 `apps/vtube_bridge`，非推测。

### 决策 5：删除课程设计产物，旧文档移入归档

- 删除 `_template_full.txt`（哈理工课程设计报告模板）、`_temp_team.txt`（成员分工表）。
- 旧文档移入 `docs/archive/`，**正文不改**，另加归档说明 `docs/archive/README.md`。
- 删除上一轮创建的 `.原始` 冗余副本 —— 同一份旧文档只保留一处（避免重蹈"去重复"前的覆辙）。
- 用户已决定：交付包（目录 + zip）在项目定稿后重新生成，当前不存在。

### 决策 6：顺手修掉 6 个实际缺陷

| 缺陷 | 原状 | 修正 |
|:---|:---|:---|
| `check_model.py`（原 `model_test.py`） | 引用 `yolov8n-face.pt`，**该文件从未存在** | 改为 `yolov8n.pt`，与 `configs/model.yaml` 一致 |
| `download.py`（原 `download_widerface.py`） | 名称暗示会下载，实际只打印手动下载说明 | docstring 明确写"**不执行下载**" |
| `check_model.py` | 自称 unit test，却无一条断言 | 重新标注为**诊断脚本** |
| 桥接 `sys.path.insert` | 插入 `PROJECT_ROOT`，而 `thirdparty/` 实际在桥接目录内 —— **仅因 CWD 恰好正确才侥幸工作**（死代码） | 改为插入 `BRIDGE_ROOT` |
| 桥接 `DEFAULT_MEDIAPIPE_MODEL` | 写死 `deployment/Vtube-Studio-Bridge/thirdparty/...` 全路径 | 改为从 `BRIDGE_ROOT` 派生，今后移动不再失效 |
| `run_gui.bat` | 引用 `..\..\.venv\Scripts\pythonw.exe`，而**项目根本没有 `.venv`** | 改用 PATH 中的 `pythonw` |

### 验证结果：10 项验收全部通过

| # | 验收项 | 结果 |
|:---|:---|:---|
| V1 | 无手动根目录解析残留 | 0 处（`paths.py` 单一来源除外） |
| V2 | 无 `sys.path.insert` hack 残留 | 0 处 |
| V3 | 全部 16 个模块可导入 | 通过 |
| V4 | 训练配置可加载 | `Batch Size: 8` / `Mixed Precision: True` |
| V5 | 单图检测端到端 | `test_detect_0.jpg → 8 face(s)` |
| V6 | 性能基准 | PyTorch 146.9 FPS / ONNX 43.9 FPS，目标 PASS |
| V7 | 桥接两默认路径有效 | 两文件均存在 |
| V8 | 三个部署权重仍被 Git 跟踪 | 3 个 |
| V9 | 该忽略的仍被忽略 | 3 条命中 |
| V10 | 索引/工作树状态 | 正常 |

另：`prepare_data_yaml()` 输出与迁移前逐字一致；Git 全程将迁移识别为 `rename`（相似度 72–94%），
文件历史完整延续。

### 遗留问题与新发现（进入阶段 2B / 阶段三）

1. **文档中的命令路径已失效**：`README.md`、`docs/使用指南.md`、`docs/presentation_outline.md`
   仍写 `python deployment/realtime_detect.py` 等旧命令，**现在跑不通**。阶段 2B 修正。
2. **README 的 FPS 数字不可复现**：文档写 162.8 / 54.9 FPS，本次实测 **145–147 / 43.9–44.8 FPS**。
3. **ONNX 比 PyTorch 慢约 3 倍**（22.3 ms vs 6.9 ms），强烈提示 ONNX 会话未真正跑在 GPU 上；
   这是差距 **G8**（benchmark 不打印实际生效的 ExecutionProvider）的实证。

---

## 2026-09-13 阶段一：清理重复文件、确立仓库基线

### 背景

会话核查发现工作区存在多处重复/冗余文件，且 `git status` 被未跟踪产物污染（16MB 交付 zip、两个 mp4、临时 txt）。清理前先做全量哈希比对，避免误删。

### 决策 1：删除 `yolo26n.pt`（6.2 MB）

| 候选方案 | 优缺点 |
|:---|:---|
| A. 保留，视为 YOLO26 权重 | 安全，但占位且模型名与实际内容不符，会误导后续读者 |
| **B. 删除（选定）** | 已证实为 `yolov8n.pt` 的误命名副本，无任何引用 |

**理由**：SHA256 完全相同（`31E20DDE3DEF09E2CF938C7BE6FE23D9150BBBE503982AF13345706515F2EF95`）。全项目代码只引用 `yolov8n.pt`（`model/model_config.yaml:6`），无任何文件引用 `yolo26n`。
**结果**：已删除；如需恢复，复制 `yolov8n.pt` 即可。

### 决策 2：交付包（目录 + zip）整体移除，定稿后重新生成

| 候选方案 | 优缺点 |
|:---|:---|
| A. 删除 zip、保留目录 | 交付包可独立运行，但目录内的脚本/权重与根目录重复，会持续产生"两份真相" |
| B. 删除目录、保留 zip | 少了干扰，但 zip 无法编辑，后续修正还要解压 |
| **C. 目录与 zip 都删除，定稿后统一重新打包（选定，用户决定）** | 彻底消除双份真相；代价是当前交付包暂时不存在 |

**执行顺序（先保资产后删除）**：先把交付包中**唯一独有**的 `高精度人脸检测使用指南.md` 归档为 `docs/使用指南.md`，再删除整个目录与 zip。
**核对**：逐文件哈希比对确认交付包 29 个文件中，23 个与根目录字节相同、3 个为 `.pyc` 缓存、3 个根目录不存在（即《使用指南.md》+ 2 个 `.pyc`）。权重 `best_model_v1.pt`/`best_model_v2.pt`/`best_model_v2.onnx` 与根目录同名文件哈希一致，无信息损失。

### 决策 3：保留 WIDER Face 源压缩包（1.74 GB）

| 候选方案 | 优缺点 |
|:---|:---|
| A. 删除，释放 1.74 GB | 解压后的 images/labels 已完整可训练 |
| **B. 保留（选定）** | 失去离线重建能力，日后重建环境需重新下载 |

**理由**：离线可重建性 > 磁盘空间；这两个文件已被 `.gitignore` 排除，不污染仓库。

### 决策 4：清理中间产物，保留演示视频

**删除**：`training/checkpoints/{v1_baseline,v2_finetune}/train_batch*.jpg`（12 个，5.6 MB，ultralytics 训练调试图）、全部 `__pycache__/`（6 个目录）。
**理由**：二者均可在重训/运行时自动重建，且已被 `.gitignore` 覆盖。
**保留**：`output_result.mp4`（5.0 MB，检测输出演示素材）、`7f62f96b….mp4`（答辩提纲指定的演示输入）。

### 决策 5：GPU 机型以实测为准 —— 修正文档的 12GB 错误

**实测**（`nvidia-smi` + `torch.cuda.get_device_properties`）：

```
name:            NVIDIA GeForce RTX 3060 Laptop GPU
total_memory:    6144 MiB (6.00 GB)
driver_version:  581.95
CUDA:            12.1
```

**结论**：
- `model/model_config.yaml:3` 注释「RTX 3060 Laptop 6GB → batch_size=8」**正确**，应保留。
- `.claude/CLAUDE.md` 的「RTX 3060 / 12GB 显存」**错误**，阶段二修正。
- `高精度人脸检测使用指南.md` 的「6GB+ 显存」**正确**。

**附带发现（重要）**：`training/training_log.txt` 记录的训练峰值显存达 **11.1 GB，超过 6 GB 物理显存**。在 Windows WDDM 下这意味着 CUDA 已溢出到共享系统内存，与日志中观察到的性能崩塌一致：`it/s` 由 5.8 降至 1.7，单 epoch 已耗时由 `8:08` 涨到 `14:34`。即 **batch=8 + imgsz=640 在 6GB 卡上已处于超额分配状态**，存在明确的优化空间（降低 batch 或 imgsz、开启 `cache` 以减少 I/O 抖动）。列入阶段三待办。

### 决策 6：撤回上一轮的一处错误结论

上一轮曾判断「`model/model_config.yaml` 写 batch=16、workers=4，与代码不一致」。**该结论错误**——当时读的是 `docs/superpowers/plans/` 里的旧草案，而非实际配置文件。
**实际情况**：`model/model_config.yaml` 为 `batch_size: 8`、`workers: 2`、`model_name: "yolov8n.pt"`，**与 `train.py`/`resume_train.py` 一致**。
**遗留问题（轻微）**：该文件中的 `accumulation_steps`、`multi_scale`、`architecture` 三个键声明了但代码从不读取，属死配置。

### 决策 7：建立 `.dsh-dev/` 工作记录

**理由**：用户个人约定要求把选定方案、理由与结果留档，并支持会话中断后的断点续跑。此前项目无此目录。
**本次建立**：`decisions.md`（本文件）、`cleanup-log.md`（清理清单）、`conventions.md`（项目约定，待用户补充）。

---

## 三段式总体方案（用户确认）

| 阶段 | 内容 | 状态 |
|:---|:---|:---|
| **阶段一** | 清理重复文件，确立可追溯的仓库基线 | 进行中 |
| **阶段二** | **以代码为准**修正文档；旧文档原文保留并加醒目说明，另存不可变副本 | 待开始 |
| **阶段三** | **以旧文档为清单**逐步修改项目代码，查漏补缺 | 待开始 |

阶段二与阶段三的关键约定：旧文档不删除、不改写其承诺内容——它是阶段三的"待办清单来源"。修正后的新文档描述**代码实际状态**，两者差异单独成表。
