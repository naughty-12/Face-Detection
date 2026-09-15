# 项目开发约定

> 由助手根据当前代码现状预填，标注 **「待确认」** 的条目请你补充或修改。
> **只把本文件明确写下的内容当作硬性规则**，其余以"最小改动、可读、可验证"为准。

---

## 语言

- 文档、代码注释使用中文；标识符（变量、函数、文件名）使用英文。
- 提交信息：标题用英文 conventional commits 前缀（`feat:` / `fix:` / `refactor:` / `docs:` / `chore:`），
  正文可用中文或英文。**（与现有 22 次提交历史一致；待确认是否改为全中文）**

## 代码风格

- Python 3.12，**不使用 type hints**（沿用现状）。**待确认**
- 路径处理统一使用 `os.path`，不使用 `pathlib`（沿用现状）。
  **例外**：`apps/vtube_bridge` 使用 `pathlib`（既有实现，不强制改造）。
- 字符串格式化统一 f-string。
- 中文路径安全：任何 `cv2.imread` 调用点都必须考虑 Windows 非 ASCII 路径问题，
  优先使用 `imdecode` 兜底（已在 `src/train/train.py`、`src/data/convert.py` 落地）。
- 每个文件顶部写一行 docstring 说明职责。

## 目录结构

```
src/          库代码（被导入，不直接运行）
apps/         可独立运行的完整应用
configs/      配置文件
data/         数据集（raw/ 与 annotations/）
artifacts/    产出（checkpoints/ reports/ logs/）
docs/         现状文档；archive/ 历史文档；plans/ 实施计划
tests/        测试
```

- **硬性规则**：所有路径必须从 `src/paths.py` 导入，**禁止**任何模块用 `__file__` 自行推算项目根。
- 脚本一律从项目根以模块方式运行：`python -m src.<pkg>.<module>`。
- 新脚本归入 `src/` 下对应的功能子包；不得新增顶层目录。
- 计划文档放 `docs/plans/`，命名 `YYYY-MM-DD-<主题>.md`。

## 数据与产出

- 数据集（`data/raw/`、`data/annotations/wider_face_split/`）不提交，由使用者自行下载。
- 模型权重只提交部署用的三个文件：`artifacts/checkpoints/best_model_v{1,2}.pt`、`best_model_v2.onnx`。
- 训练中间产物（`*/weights/`、`train_batch*.jpg`、`artifacts/logs/`）不提交。
- 运行产物（截图、`results/`、`output_result.mp4`）不提交。
- 交付包在项目定稿后统一生成，不纳入仓库。

## 测试

- 当前 `tests/` 为空，**无测试框架**。**待确认**：是否引入 pytest？
- 约定：新增**纯函数**（坐标转换、滤波器、标注解析）应附带最小单元测试。
- 涉及真实推理或训练的验证，优先用"小样本实跑 + 明确预期输出"的方式，而不是跑完整流程。

## 提交规范

- 一个可独立验证的变更一次提交；提交信息说明**改了什么**与**为什么**。
- 迁移/重命名类改动使用 `git mv` 以保留文件历史。
- 破坏性操作（删除、覆盖）前先说明影响面；能备份先备份。
- 每次更新完成后同步更新文档，并把本次条目追加到 `CHANGELOG.md`。
- 决策记录写入 `.dsh-dev/decisions.md`（候选方案、选定理由、结果）。

## 环境

- 实测：Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 / **RTX 3060 Laptop 6 GB** / 驱动 581.95 / CUDA 12.1
- 训练显存约束：6 GB。`batch=8 + imgsz=640` 会超额分配（实测峰值 11.1 GB）并导致吞吐骤降，
  建议 `batch≤4` 或 `imgsz≤512`。
- 交互式脚本应提供无头开关（如 `--no-show`），保证可在脚本与 CI 中运行。

## 指标与口径

- **硬性规则**：任何指标必须标注**测量口径**（数据集范围、计算方式、是否含前后处理、运行设备）。
- 禁止把口径不同的数字并列比较（例：全验证集 COCO 式 AP50 与 Hard 子集 VOC AP）。
- 未实测的数字不得写入文档；测试脚本不得在跳过验证时输出"通过"。
