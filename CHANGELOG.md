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
