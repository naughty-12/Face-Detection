# 清理与变更清单（阶段一）

> 执行日期：2026-09-13
> 所有删除项均已确认可由现有文件重建，或已先归档独有资产。

## 一、归档（保留，仅移动位置）

| 原路径 | 新路径 | 大小 | 说明 |
|:---|:---|:---|:---|
| `高精度人脸检测/高精度人脸检测使用指南.md` | `docs/使用指南.md` | 5,652 B | 交付包中**唯一独有**资产；阶段二将按代码实际情况修正 |

## 二、删除（重复 / 可重建）

| 路径 | 大小 | 判定依据 | 重建方式 |
|:---|:---|:---|:---|
| `yolo26n.pt` | 6.2 MB | SHA256 与 `yolov8n.pt` 完全相同；无代码引用 | 复制 `yolov8n.pt` |
| `高精度人脸检测/`（整个目录） | 含 18.6 MB 权重 | 29 文件中 23 个与根目录字节相同、3 个为 `.pyc`；独有资产已归档 | 项目定稿后重新打包 |
| `高精度人脸检测.zip` | 16.2 MB | 36 条目 = 交付目录快照 | 重新打包 |
| `training/checkpoints/v1_baseline/train_batch*.jpg` | 6 个 / 2.9 MB | ultralytics 训练调试图，已 gitignore | 重训自动生成 |
| `training/checkpoints/v2_finetune/train_batch*.jpg` | 6 个 / 2.7 MB | 同上 | 重训自动生成 |
| `deployment/__pycache__/` | — | 编译缓存 | 运行时自动生成 |
| `deployment/Vtube-Studio-Bridge/thirdparty/MediaPipe/__pycache__/` | — | 编译缓存 | 运行时自动生成 |
| `training/__pycache__/` | — | 编译缓存 | 运行时自动生成 |
| （随目录一并删除）`高精度人脸检测/…/__pycache__/` ×3 | — | 编译缓存 | 运行时自动生成 |

**合计释放磁盘**：约 36.4 MB（不含交付包内权重则约 17.8 MB）；交付包相关共约 34.8 MB。

## 三、明确保留（经核验**不是**重复文件）

| 路径 | 原因 |
|:---|:---|
| `data/annotations/wider_face_split/` 下 7 个官方文件 | 根目录 `data/annotations/` 下**没有**同名副本，不存在重复 |
| `training/checkpoints/best_model_v1.pt` / `best_model_v2.pt` / `best_model_v2.onnx` | 与 `*/weights/best.pt` 哈希相同，但为 `train.py` 显式拷贝的 public 路径，被 `README.md` 与 VTS 桥接默认参数引用 |
| `training/checkpoints/v1_baseline/weights/last.pt` | `resume_train.py` 断点续训必需 |
| `data/raw/WIDER_train.zip`、`WIDER_val.zip` | 1.74 GB，离线重建能力（用户决定保留） |
| `output_result.mp4` | 5.0 MB，检测输出演示素材（用户决定保留） |
| `7f62f96bca5cffdfe2e0167bf3de3169.mp4` | 演示用输入视频（原答辩稿指定；现为《面试讲解提纲》的演示素材） |
| `_template_full.txt`、`_temp_team.txt` | 当时判定为"报告模板与分工表素材"，建议归档保留；**后续经确认改为删除** —— 本项目为个人独立完成，不存在人员分工，课程设计产物已无用途 |
| `yolov8n.pt`、`_wheels/` | 训练预训练权重来源；离线安装 wheel |
| `MEMORY.md` | 索引文件（其指向的 3 个 md 实际不存在，阶段二处理） |

## 四、后续动作

- [x] `.gitignore` 补充规则，防止运行产物（截图、`results/`、输出视频、重新生成的交付包）再次污染仓库
- [x] Git 提交基线快照，为后续文档修正提供恢复点
- [x] 旧文档原文归档（`docs/archive/`，正文不改，附归档说明）
- [x] 文档校正至与代码一致（含目录重组为 `src/ apps/ configs/ data/ artifacts/`）
- [ ] 阶段三：以 [`docs/项目现状与差距.md`](../docs/项目现状与差距.md) 为清单逐步查漏补缺（30 条）

> **记录变更说明**：本文件记录的是**阶段一清理当时的判定**。其中 `_template_full.txt` 与
> `_temp_team.txt` 两项的处置后来发生变更 —— 当时建议归档保留，最终确认**删除**
> （理由：本项目为个人独立完成，不存在人员分工，课程设计产物已无用途）。
> 已在上表中标注，此处保留原始判定以便追溯。
