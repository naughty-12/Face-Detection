# 更新日志

本文件记录项目的重要变更。

## [未发布]

### 修复

- **单图检测会弹出阻塞窗口**：`src/deploy/detect.py` 在单图模式下无条件调用 `cv2.imshow()`
  并进入 `waitKey` 循环，直到按键或手动关闭窗口才返回。这使脚本无法用于批处理与自动化。
  新增 `--no-show` 开关；默认的交互行为不变。

### 变更

- **目录重组（阶段 2A）**：从按"人头分工"划分的六目录结构改为标准工程布局。

  | 旧路径 | 新路径 |
  |:---|:---|
  | `data/*.py`、`dataset/` | `src/data/` |
  | `training/`、`model/model_test.py` | `src/train/`（`model_test.py` → `check_model.py`） |
  | `evaluation/` | `src/eval/` |
  | `deployment/*.py` | `src/deploy/`（`realtime_detect.py` → `detect.py`） |
  | `deployment/Vtube-Studio-Bridge/` | `apps/vtube_bridge/` |
  | `model/model_config.yaml` | `configs/model.yaml` |
  | `training/checkpoints/`、`evaluation/reports/` | `artifacts/checkpoints/`、`artifacts/reports/` |

  - 新增 `src/paths.py` 作为**全项目唯一的路径来源**，取代原先散落在 6 个文件中的
    `dirname(dirname(__file__))` + `sys.path.insert` 写法（该写法在脚本进入子包后必然算错项目根）。
  - 所有脚本改为从项目根以模块方式运行：`python -m src.<pkg>.<module>`。
  - Git 全程记录为 `rename`（相似度 72–100%），文件历史完整保留。

- **文档结构分层**：`docs/` 根为现状文档，`docs/archive/` 为历史设计文档（正文不改），
  `docs/plans/` 为实施计划。移除 `docs/superpowers/`。

### 移除

- 删除课程设计相关产物：`_template_full.txt`（课程报告模板）、`_temp_team.txt`（成员分工表）。
  本项目为个人独立完成，不存在人员分工。
- 删除重复文件：`yolo26n.pt`（与 `yolov8n.pt` 字节相同且无任何引用）、交付包目录及其 zip
  （唯一独有资产《使用指南》已先归档到 `docs/`）、12 个训练调试图、全部 `__pycache__`，
  共释放约 49.6 MB。

### 修复（随目录重组一并处理）

| 缺陷 | 原状 | 修正 |
|:---|:---|:---|
| `check_model.py` | 引用 `yolov8n-face.pt`，**该文件从未存在** | 改为 `yolov8n.pt`，与 `configs/model.yaml` 一致 |
| `download.py` | 名称暗示会下载，实际只打印手动下载说明 | docstring 明确写出不执行下载 |
| `check_model.py` | 自称 unit test，却没有任何断言 | 重新标注为诊断脚本 |
| `apps/vtube_bridge/run_gui.bat` | 引用 `..\..\.venv\Scripts\pythonw.exe`，而**项目没有 `.venv`** | 改用 PATH 中的 `pythonw` |
| 桥接 `sys.path` 设置 | 插入 `PROJECT_ROOT`，但 `thirdparty/` 实际位于桥接目录内 —— 仅因当前工作目录恰好正确才侥幸工作 | 改为插入 `BRIDGE_ROOT` |
| 桥接 `DEFAULT_MEDIAPIPE_MODEL` | 写死整条 `deployment/Vtube-Studio-Bridge/...` 路径 | 改为从 `BRIDGE_ROOT` 派生 |

### 文档

- 新增《架构说明》，按功能职责（而非人员分工）描述模块划分、数据流与设计取舍。
- 新增《项目现状与差距》，逐条记录实现与设计承诺的差异，每条附证据。
- 重写 README：所有指标标注测量口径，并如实列出未完成项。
- `.gitignore` 扩展：泛化训练调试图规则，新增运行产物（截图、`results/`、输出视频）
  与重新生成的交付包排除规则；将 `.claude/settings.local.json` 移出版本控制使既有忽略规则生效。

### 首次纳入版本控制

- `apps/vtube_bridge/`（VTube Studio 表情驱动）—— 此前从未提交，项目最核心的应用部分。

### 已知问题

- WIDER Face Easy/Medium/Hard **分项评估未完成**，原始 0.76 目标无法对账。
- ONNX 精度验证因 `data/annotations/val_list.txt` 缺失而**静默跳过**，"MAE < 1e-4" 无实测支撑。
- **ONNX 前向比 PyTorch 慢约 3 倍**（22.3 ms vs 6.9 ms），且 benchmark 不打印实际生效的
  ExecutionProvider，运行设备待确认。
- Recall 仅 0.595，小脸与遮挡场景漏检明显。
- v2 微调为**负收益**（mAP50 0.66349 → 0.66171），缺困难样本回流。
- `src/data/augment.py`（Albumentations）未接入训练流程。
- WIDER 标注解析存在 **4 份重复实现**，边界处理不一致。
- 设计中的自采 500 张数据未采集。
- 6 GB 显存下训练超额分配（峰值 11.1 GB），吞吐显著下降。

---

## [2026-09-15] 测量与修复

### 新增（项目此前从未有过的实测数据）

- **ONNX 导出精度按检测质量裁定**，而非张量误差。同一组权重在全部 3,226 张验证图上对比：

  | 变体 | mAP50 | mAP50-95 | P | R | 吞吐 | 设备 | 体积 |
  |---|---:|---:|---:|---:|---:|:---|---:|
  | PyTorch `.pt` | 0.66404 | 0.35618 | 0.8479 | 0.5937 | 87.8–147 FPS | CUDA | 5.95 MiB |
  | ONNX **FP16** | 0.66382 | 0.35627 | 0.8448 | 0.5959 | ~36 FPS | CPU | **5.88 MiB** |
  | ONNX FP32 | 0.66379 | 0.35670 | 0.8446 | 0.5959 | ~36 FPS | CPU | 11.70 MiB |

  结论：**FP16 的检测质量与 PyTorch 无差别**（mAP50 −0.0002 / mAP50-95 +0.0001），体积减半。
  导出改为 `half=True`，项目 ≤10 MB 目标重新满足（FP32 为 11.70 MiB，一度超标）。

- **按人脸尺寸分层的检测分析**（39,708 张标注人脸）：汇总召回 0.610，但 **72.5% 的 GT 人脸
  <32px、召回仅 0.493**；**≥32px 人脸召回 0.916**。这修正了"召回偏低是模型短板"的判断 ——
  主因是 WIDER 基准中压倒性的极小脸，而非实际场景不可用。
- **运行环境实测**：`onnxruntime` 1.27 仅提供 `CPUExecutionProvider`（无 CUDA，且因无网络
  无法安装 `onnxruntime-gpu`），因此 ONNX 比 PyTorch/CUDA 慢 **2.4–3.3 倍**。
- 为 v1/v2 保存的 best 权重重新测得 mAP，取代此前引用的训练日志末行（最后 epoch）数字。

### 性能

- **VTube Studio 桥接默认模型由 `.onnx` 改为 `.pt`**。`onnxruntime` 在本机只暴露
  `CPUExecutionProvider`，所以桥接此前一直在**CPU 上做 YOLO 推理（中位 24.8 ms）而 GPU 闲置**。
  改为 `.pt` 后走 CUDA 并自动启用 FP16。无头验证：`device=0`、`format=PyTorch`、`use_half=True`、检出 8 张人脸。
- **实测确认 FP16 默认值正确**（交替 A/B，5 轮 × 15 样本，同进程）：
  `half=True` 中位 **12.52 ms / 79.9 FPS**，`half=False` 中位 **13.52 ms / 73.9 FPS**，FP16 快 **7.4%**。

### 测量方法

- **发现本机测量噪声约为 ±50%**：同一调用跨度达 **4.8 – 18.8 ms**（约 3 倍）。
  本会话中所有"同一数字对不上"的现象都源于此，而非代码或环境变化。
- `src/deploy/benchmark.py` 因此不再只报一个平均值，改为输出
  **中位数 + 最小值 / p95 / 最大值 + 样本数**，并在开头说明原因。
  最新中位结果：PyTorch **10.31 ms / 97.0 FPS**，ONNX **24.80 ms / 40.3 FPS（CPU）**；
  体积检查随 FP16 生效而重新 **PASS**（`≤10MB`），此前 FP32 为 11.70 MiB 属 **FAIL**。
- 结论：**任何单次读数不可作为结论**，引用速度必须注明测量次数与统计量。

### 修复

- `split.py` 解析 WIDER 标注时崩溃（`num_faces=0` 的假框行未跳过），导致 `train_list.txt` /
  `val_list.txt` **从未成功生成**，进而使 ONNX 精度验证与分项评估双双静默失效。
- `export_onnx.py` 把 ONNX 拷贝到自身路径，抛 `SameFileError` —— 导出脚本无法运行。
- 清单文件按平台默认编码（GBK）写入，中文绝对路径被破坏；写入与读取统一为 UTF-8。
- `evaluate.py` 的 P-R 曲线查找 `PR_curve.png`（ultralytics 实际写 `BoxPR_curve.png`）且无 else
  分支，**静默不产出任何文件**；同时消除每个模型的第二次冗余 val（4 轮 → 2 轮）。
- `evaluate.py` 在官方评测包缺失时把 mAP50 填进 Easy/Medium/Hard 三个字段**冒充真实结果**；
  现返回 `null` 并显式说明不可用。
- `analyze_errors.py` 的误差分布图建立在 **20 个样本**上（且是"最小的 20 个漏检"）；重建为
  基于全部错误的统计，输出**数字表格**，并改为分析实际部署的 v2 权重。
- `validate_precision()` 在 `val_list.txt` 缺失时 `return True`（判定通过），把跳过当通过；
  现明确报告 `[SKIPPED]` 且不视为通过。
- 精度判据由固定 `1e-4` 改为按精度区分（FP16 `1e-1` / FP32 `1e-4`），并说明张量 MAE 的含义边界。
- 单图检测弹出阻塞窗口（`cv2.imshow` + `waitKey` 循环），无法用于批处理 → 新增 `--no-show`。
- WIDER 标注解析的 **4 份重复实现**（`convert` / `qc` / `split` / `loader`）合并为
  `src/data/wider_annotations.py` 单一来源；`qc.py` 另有一处 `cv2.imread(...).shape` 缺 None 检查。

### 文档

- README：新增按尺寸分层表与运行方式建议；修正 v2 结论（持平，而非负收益）、ONNX 为 FP16，
  补充单位（MB/MiB）与测量次数的说明。
- 使用指南、面试讲解提纲：同步上述全部修正，含 ONNX 精度判据的完整解释。
- 《项目现状与差距》新增第七节：状态更新、9 个新差距、可复现命令。
