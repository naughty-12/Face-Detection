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

### 新增：置信度阈值扫描（G3）

`src/eval/conf_sweep.py` —— **不重跑推理**：检测 AP 与阈值无关，`model.val()` 本来就计算完整
P-R 曲线并沿置信度轴暴露 P/R/F1，因此一次验证即可读出所有工作点。

| conf | precision | recall | F1 |
|---:|---:|---:|---:|
| 0.10 | 0.5812 | 0.6746 | 0.6244 |
| 0.20 | 0.7560 | 0.6308 | 0.6877 |
| **0.25（默认）** | 0.8139 | 0.6097 | **0.6971** |
| 0.30 | 0.8581 | 0.5887 | 0.6983 |
| 0.50 | 0.9537 | 0.4909 | 0.6482 |

max-F1 点为 `conf=0.297`、F1 0.6985。

**结论：默认 `conf=0.25` 几乎正是最优 F1 点**（相差 0.0014）。此前文档中"`conf=0.25` 对召回
不友好"的说法**被实测推翻**。AP50 与阈值无关，阈值只用于选取工作点；本项目未改动任何默认阈值。

### 新增：单元测试（G18）

`tests/` 共 **56 个测试**，使用标准库 `unittest`（本机无网络，装不了 pytest），覆盖
WIDER 标注解析、WIDER→YOLO 坐标转换、尺寸分层结论所依赖的 IoU / 分桶几何，以及**桥接的时序
滤波**（`FaceBoxFilter` 的 EMA/中值/hold/reset、`ParameterFilter`、人脸框到 VTS 位置的映射）。
fixture 采用**仓库内提交的文件**而非运行时临时文件——受限沙箱会拒绝写入新建目录，
而**被跳过的测试不是安全网**。

针对桥接滤波的**判别性用例**：中值抑制（`alpha=1` 隔离中值滤波，稳定窗口内 6 像素的偏移必须被
丢弃）、以及远距离跳变重置（`alpha=0.05` 时应**精确跳到**新脸而不是缓慢靠近）。

### 修复（由新测试推导出来）

- **`parse_samples()` 存在生成器/`with` bug**：它在 `with open(...)` 内 `return parse_lines(f)`，
  而 `parse_lines` 是生成器，函数体在迭代时才执行，此时文件已关闭 →
  `ValueError: I/O operation on closed file`。**该 bug 会使 `src/data/split.py` 失效。**
  已改为先读完全部行再委托；并新增"同一 fixture 读取两次"的回归测试。
- **空行容忍只做了一半**：docstring 声称空行可容忍，实际只在图片名之前跳过，
  **图片名与数量之间、数量与框之间都不跳**。两处现已一致。

### 性能（G17）：训练显存实测，`batch` 由 8 下调为 6

新增 `src/train/probe_vram.py`（1 epoch / 5% 数据切片 / workers=0），实测峰值显存：

| batch | 峰值**分配** | 峰值**保留** | 占 6.00 GiB 物理 |
|---:|---:|---:|---:|
| 4 | 1.20 GiB | 3.58 GiB | 60% |
| 6 | 1.75 GiB | **4.41 GiB** | 73% |
| 8 | 2.29 GiB | **6.18 GiB** | **超过上限** |

关键：**分配量只有保留量的约 1/2.5**，与显存上限比较的应是**保留量**。`batch=8` 单轮即
6.18 GiB，已超过 6.00 GiB 物理上限，从第一轮起就向共享内存溢出；100 epoch 长跑因内存碎片
继续增高（历史日志达 11.1 GiB），吞吐从 5.8 it/s 崩到 1.7 it/s。

**`configs/model.yaml` 的 `batch_size` 已由 8 改为 6**（4.41 GiB，留 27% 余量）；若需更大余量可再降
到 4。**尚未用新 batch 重训验证实际收益。**

> 本次探针的**耗时不具可比性**：`batch=4` 那轮包含 `yolo26n.pt` 下载与首次初始化，且 161 步的
> 短跑无法体现长跑碎片化。只有显存数据有效。

### 环境：出网由沙箱控制，且不稳定

受限模式下 `github.com` 与 `pypi.org` **均不可达**；提权后可达但**不稳定** —— 同一会话中 pip 能
拉取仓库，`git clone` 却以 `Connection was reset` 失败；官方评测包的 `setup.py` 也无法在当前
setuptools 下构建。

因此 **G9/G2 的官方分项指标（Easy/Medium/Hard）无法在本环境可靠取得**，维持尺寸分层的替代方案
（它直接指出失效模式，比按基准自带难度分桶更有信息量）。

顺带确认：`yolo26n.pt` 由 **ultralytics 的 AMP 检查自动从 GitHub 下载**（本次为 5.29 MB），
不是项目产物 —— 此前删除的那个（6.2 MB，与 `yolov8n.pt` 字节相同）确属误命名副本。

### 新增：`src/deploy/compare_export_variants.py`

把"三种导出变体的 mAP 与吞吐对比"固化为可复用脚本（原先是一次性探针），
其结论已写入 README 与《决策记录》。

### 变更：Albumentations 管线的定位（G12，采用方案 B）

`src/data/augment.py` 从未进入训练，只被 `vis_aug.py` 引用。经评估**不接入训练**，定位为
**增强可视化工具**并写入文件头说明。理由：

- 其全部变换已被 ultralytics 内置增强覆盖（`fliplr` / `hsv_*` / `randaugment` 含模糊 /
  `erasing` 即 Cutout / letterbox）；
- 且**缺 Mosaic**（ultralytics 在检测上最有价值的增强）→ **替换会严格变差，叠加会让翻转与
  HSV 双重施加**。

顺带纠正：设计文档承诺的"后 50 epoch 加 Cutout/模糊"**其实早已被 ultralytics 默认值满足**。

**若将来要做数据侧提召回，重点是两条**（已写入 `docs/项目现状与差距.md` 第八节）：

1. **属性感知增强** —— 用 WIDER 标注自带的脸级 `blur` / `illumination` / `occlusion` / `pose`
   属性做定向增强；ultralytics 看不到这些字段（对应 G14）。依据：≥32px 人脸召回已达 0.916，
   剩余失误集中在困难子集。
2. **小脸放大裁剪** —— 随机裁出含小人脸的区域放大到 640。依据：72.5% 的 GT 人脸 <32px、召回 0.493。

真实成本：关闭重叠变换、自行实现 Mosaic、打通 ultralytics 的 Dataset / Transform 层。
**第一步应是补训练数据路径的测试**，而非直接改 `augment.py`。

文档同步：README、`docs/架构说明.md`、`docs/项目现状与差距.md`（新增第八节）、
`.dsh-dev/decisions.md`。

### 修复：`CoarseDropout` 参数被静默忽略

做出 G12 决策后校验时发现，`src/data/augment.py` 的 `CoarseDropout` 用的是 **Albumentations 1.x
的参数名**（`max_holes` / `max_height` / `max_width`），而项目安装的是 **2.0.8**。这些参数
**只产生一条 `UserWarning` 后被忽略**，实际生效的是默认值：

| | 预期（1.x 写法） | 实际生效（2.x 默认） |
|:---|:---|:---|
| 洞的数量 | 最多 8 个 | `num_holes_range=(1, 2)` |
| 洞的尺寸 | ≤32 px | `hole_height_range=(0.1, 0.2)`（即 10~20%） |

已改用 2.x API（`num_holes_range=(1, 8)` / `hole_height_range=(1, 32)` /
`hole_width_range=(1, 32)`），并用 `-W error::UserWarning` 验证构造时不再抛警告。
`requirements.txt` 相应收紧为 `albumentations>=2.0.0`。

同时把 `widerface-evaluate` 标注为**可选依赖**并说明：项目实际汇报的指标不依赖它
（缺失时 Easy/Medium/Hard 报 `null`，分项分析由按尺寸分层替代）。

### 文档订正：定向增强的载体（`docs/项目现状与差距.md` 新增 8.5）

查证 ultralytics 8.4.75 源码后，订正了此前两处判断：

1. **载体不是"Albumentations vs 自定义 Dataset"**。`Albumentations` 是库；真实载体是
   **`augmentations` 超参钩子**（`v8_transforms()` 已内插 `Albumentations(transforms=hyp.augmentations)`，
   一行配置即可，且**会自动同步标注框**）与**自定义 Dataset**（数天级）。
2. **"必须自行实现 Mosaic"是错的**：钩子运行在 Mosaic / affine / MixUp **之后**，Mosaic 原样保留。

并记录了钩子的一个隐蔽陷阱：`contains_spatial` 靠**硬编码类名白名单**判断，**自定义变换类名不在表内
→ 被当作纯像素变换 → 框静默不同步、标注被破坏**。

据此给出两条定向增强与载体的对应：**属性感知增强必须走自定义 Dataset**（标注属性进不了钩子的
`image + bboxes + class_labels` 通道，且需先补 G14 的属性解析）；**小脸放大裁剪可走配置级钩子**
（用白名单内的 `BBoxSafeRandomCrop` + `LongestMaxSize`，无需自定义类）。

推进顺序因此改为：**先用配置级钩子零代码验证小脸放大裁剪是否真能提升召回，有效再投入 Dataset 级改造。**

### 实验：小脸裁剪放大 —— 代理方案不成立（G12 / 差距清单 8.6）

新增 `src/train/probe_aug_crop.py`（A/B 框架：两 arm 均从 `best_model_v2.pt` 微调 3 epoch，
唯一差别是 `augmentations`）。实测结果：

| 尺寸桶 | base | crop | Δ |
|:---|---:|---:|---:|
| **Tiny（<32px）** | 0.494 | **0.501** | +0.007 |
| Small（32–96px） | 0.902 | 0.897 | −0.005 |
| Medium（96–256px） | 0.959 | 0.952 | −0.007 |
| Large（>256px） | 0.991 | 0.986 | −0.005 |
| 总体召回 | 0.610 | 0.614 | +0.004 |
| **精确率** | **0.808** | **0.765** | **−0.043** |
| 预测总数 | 29,993 | 31,874 | +1,881 |

**多命中 147 张人脸的代价是多输出 1,881 个预测（约 13:1）** → 代理方案无效：收益不显著而代价明确，
**不足以支撑 Dataset 级改造**。机制推测：裁剪放大抬高了模型对大脸的置信度分布，推理时过度自信。

**这不能完全否定"定向"版本**（本实验用的是随机裁剪，不做"含小脸"的定向选择），但结论明确。
局限：仅微调 3 epoch、单次运行无种子重复、`Resize` 会拉伸长宽比、测的是随机而非定向裁剪。

**下一步的高杠杆方向**：既然根因是"<32px 的脸在 640 输入下不足 10px"，直接提高分辨率
（`imgsz=960`）作用于同一根因，且同一个 A/B 脚本只改 `--imgsz` 即可验证。

### 过程中修掉的三个问题

- `probe_aug_crop.py` 传入整个 config 却直接索引 `cfg["optimizer"]`（该键在 `cfg["training"]` 下）→ `KeyError`；
- **`fraction < 1` 会强制重建标签缓存**（`base.py` 在 `get_labels()` 之前截断 `im_files`，哈希失配）；
  更关键的是 `fraction` 取的是**排序列表前缀而非随机抽样** —— WIDER 按事件类目录排序，
  `fraction=0.25` 等于只保留前 25% 的事件类别，**不是代表性样本**，据此比较召回会得出错误结论。
  已改为 `fraction=1.0`；
- 磁盘上的标签缓存曾被 `fraction=0.05` 的探针写成**子集缓存**，导致后续全量训练全部哈希失配 →
  一次性提权重建后恢复正常。

### 文档：新增《简历项目经历》撰写稿（游戏客户端 / 客户端开发方向）

新增 `docs/简历项目经历.md`。定位是**工作文档，不是简历本身**。首轮稿偏"技术专业化"，
经反馈后按**「先过筛选，再谈专业」**重写：简历的读者顺序是 HR/关键词初筛 → 技术面，
一份稿子同时讨好两边的结果是两边都过不了。

- **区分两个筛子**：① 关键词初筛的死因是**岗位关键词零重合**（JD 要 Unity/C#/C++，稿子全是
  WIDER/mAP/ONNX/letterbox）；② 技术面第一眼觉得"投错岗了"。对策相反 —— 客户端版**一个 mAP
  都不出现**，算法口径降为备用版。
- **新增 JD 关键词对照表**：列出 7 项可对应的真实经历（实时帧预算、多线程、性能定位、工具开发、
  SDK 协议对接、手感调优、工程规范）与 2 项**硬缺口（C#/Unity、C++）**，并指出补 Unity +
  Live2D Cubism SDK 是这份项目性价比最高的改造。
- **新增"术语翻译对照表"**：同一件事的算法说法 ↔ 客户端说法（如"三级时序滤波"→"解决抖动、
  闪烁、切换滑动三类手感问题"）。
- **A/B 两版桥接措辞**：皮套**尚未真机验证**，故给出现可用（"已落地 + 联调进行中"）与跑通后
  （"已完成并验证"）两版句子及替换对照表，避免提前宣称已验证。
- **禁止写清单**：逐条列出会当场翻车的表述（mAP 0.76、162.8 FPS、ONNX 加速、自采 500 张、
  六人分工、"~6 ms"检测延迟等）及其实测反证。
- **顺手发现的两处不一致**：① README/面试提纲/架构说明的"**~6 ms** 检测单帧"与
  `docs/使用指南.md` 实测**中位 12.5 ms** 冲突，简历应改用 benchmark 有日志支撑的 10.3 ms
  （此项**未改**，需实测后统一）；② `MEMORY.md` 的精度 **0.663 / 0.662 为旧值**，与 README 及
  `artifacts/reports/metrics_*.json` 的 0.66501 / 0.66404 不一致 —— **已修正**，同时订正同表
  "训练超参 batch=8"（配置已改 6）与"主要短板"三项已过时表述（ONNX 精度验证已补、"v2 为负收益"
  应为止平）。

### 文档订正：简历稿的角色定位于"决策者"（去除"自研"类表述）

经用户澄清：本项目是 **AI 辅助产出的产品，用户担任决策者**，此前简历稿按"实现者"口径写
（"自研滤波方案""自研调试面板""独立完成从 0 到 1""我写了 56 个测试"）属**定位错误，非措辞问题**。
`docs/简历项目经历.md` 据此重写：

- 新增「角色重定义」表：逐条给出"实现者口径 → 决策者口径"的替换
  （自研 → 确定/选定/否决/验收；独立完成 → 主导）；并把"**决定不自建检测模型**"
  改写为产品判断（成熟任务不重复投入，预算集中在管线工程）。
- 新增「决策者必答：你的决策清单」：从 `.dsh-dev/decisions.md` 抽取 **10 条带证据的真实决策**
  （判据由张量 MAE 改为 mAP、"未测量不算通过"、否决通用增强接入训练、batch 8→6、
  桥接默认 .onnx→.pt、推翻此前 2 条结论等），每条附"候选方案 + 依据 + 结果"的出处，
  作为"决策者"身份的唯一硬支撑。
- 新增「AI 辅助要不要写进简历」：给出不显式（默认，投游戏客户端）与显式（投 AI 产品方向）
  两条路及各自风险，并要求**面试时主动交底**。
- 禁止写清单首条改为 **"自研 / 独立完成 / 从 0 到 1 实现 / 我写了 X"**，理由是角色定位而非措辞。

### 文档：简历稿新增「正式版」与旧稿审计（5 处与仓库文档矛盾）

用户提供的早前 Claude 稿（项目描述 + 技术栈 + 技术特点）经逐条核对，**16 条中 5 处 ❌ 与仓库
文档直接矛盾**，会与 README「已知限制」、CHANGELOG「已知问题」、差距表自相打架。据此：

- 新增 **3.4 正式版**：沿用旧稿三段式结构与 Vibe Coding 定位，事实校正为 —— ONNX 定位为
  可移植性而非部署主路径；增强交由 ultralytics 内置（未接第三方管线）；**batch 8 → 6 的显存
  实测依据**取代"12GB 小卡 + 梯度累积"；按尺寸分档统计的是**召回**而非 mAP，官方 Easy/Medium/
  Hard 分项未取得一律报 `null`；478 关键点；分步可复现流程取代"一条命令跑通"。
- 新增 **第十一节 旧稿审计**：16 条逐条判定与证据，重点标注 5 处 ❌ ——
  ① "依托 ONNX Runtime 完成推理部署"（本机无 CUDA provider，桥接已改回 `.pt`）；
  ② "梯度累积在 12GB 小卡上模拟大 batch"（6 GB；`accumulation_steps` 是死键）；
  ③ "按尺寸分 Easy/Medium/Hard 三级评测 mAP"（难度子集 ≠ 尺寸分档，且未算 mAP、官方分项为 null）；
  ④ "468 个 3D 关键点"（应为 478）；⑤ "一条命令跑通全流程"（下载需手动，无 pipeline 脚本）。
  另标注 4 处 ⚠️：多模态术语误用、Albumentations 未接入训练、越界框剔除无实现、训练无模糊增强、
  "静默降级"措辞反了。
- 记录一条判断：**这类矛盾损失的不是单条加分，而是整份简历的可信度**，而"每个数字都标口径、
  未完成如实列出"恰是本项目唯一的差异化优势。
