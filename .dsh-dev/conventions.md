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

- 当前 `tests/` 有 **15 个测试文件 / 155 个用例**，用 `unittest`（零额外依赖）运行：
  `python -m unittest discover -s tests -t .`。**仍然没有 pytest**，也没有 lint / format / type check 配置。
- 约定：新增**纯函数**（坐标转换、滤波器、标注解析）应附带最小单元测试。
  回归测试按"真实踩过的 bug"写：护栏类测试要**失败而不是 skip**（例：`cv2` 被 headless 覆盖时必须变红）。
- 涉及真实推理或训练的验证，优先用"小样本实跑 + 明确预期输出"的方式，而不是跑完整流程。
- 未覆盖区（改这些地方时人工验证）：**模型 IO、训练数据路径、Unity C# 运行时、VMC 真机接收端**。

## 提交规范

- 一个可独立验证的变更一次提交；提交信息说明**改了什么**与**为什么**。
- 迁移/重命名类改动使用 `git mv` 以保留文件历史。
- 破坏性操作（删除、覆盖）前先说明影响面；能备份先备份。
- 每次更新完成后同步更新文档，并把本次条目追加到 `CHANGELOG.md`。
- 决策记录写入 `.dsh-dev/decisions.md`（候选方案、选定理由、结果）；
  **工作失误写入 `.dsh-dev/mistakes.md`**（错在哪、影响、如何发现、修复与残留）。
- **对外表述（简历、README、答辩稿）只能按已验证的层次来写**：设计 / 源码 / 运行 / 有日志与截图
  四层中，只按实际达到的那一层措辞。"仓库里有这个文件"**不等于**"已经跑过"。见 `mistakes.md`。

## 环境

- 实测：Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 / **RTX 3060 Laptop 6 GB** / 驱动 581.95 / CUDA 12.1
- 训练显存约束：6 GB。`batch=8 + imgsz=640` 会超额分配（实测峰值 11.1 GB）并导致吞吐骤降，
  建议 `batch≤4` 或 `imgsz≤512`。
- 交互式脚本应提供无头开关（如 `--no-show`），保证可在脚本与 CI 中运行。
- **取证红线（2026-09-16 实测）**：受限 shell 里 **`Get-PnpDevice` / `Get-CimInstance` /
  `Get-NetTCPConnection` 都不可信** —— 分别表现为返回 0 个设备、拒绝访问、
  **连 445 端口（PID 4，确定在监听）都报 0 条**。端口与设备一律改用
  `netstat -ano`、`pnputil /enum-devices /connected`、注册表，或**真实连接 / 真实打开**来验证。
- **硬性规则：任何"没有 / 不存在 / 未监听"的结论，必须先自证这次查询是有效的**（正对照：
  先用同一工具看一个已知为真的对象）。空结果**不是**否定证据。见 `mistakes.md` 失误 4 与 5。
- **涉及摄像头（或其它本机设备）的命令必须由使用者本人执行**：受限 shell 取不到摄像头设备，
  在那里判断"设备坏没坏"只会得到错误结论。
- **远端与推送（2026-09-22 实测）**：远端有四个 —— `origin` = gitee（**唯一有跟踪分支的**）、
  `github` / `github-ssh` = GitHub、`fastgit`（域名 `hub.fastgit.xyz` **已停服，勿用**）。
  推送实测有**两个坑**：① `.git/config` 里配了代理 `http://127.0.0.1:7897`，但**当时该代理并未监听**
  （`netstat` 查不到 7897；同一工具对已知为真的 445 端口查得到，故该否定结论有效），走代理会
  `Failed to connect to ... via 127.0.0.1`；② 直连时受限 shell 被 schannel 拦下
  （`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS`），**提权后同一条命令即可成功**。
  可用写法：`git -c http.proxy= -c https.proxy= push <remote> <branch>`；
  GitHub 直连不稳时再加 `-c http.postBuffer=157286400 -c http.lowSpeedLimit=0 -c http.lowSpeedTime=999999`
  （首次 `HTTP 408` 用这组参数重试一次成功）。**改用 `http.sslBackend=openssl` 不可行**
  （git 会拉起 `sh.exe`，受限 shell 下报 `couldn't create signal pipe, Win32 error 5`）。

## 指标与口径

- **硬性规则**：任何指标必须标注**测量口径**（数据集范围、计算方式、是否含前后处理、运行设备）。
- 禁止把口径不同的数字并列比较（例：全验证集 COCO 式 AP50 与 Hard 子集 VOC AP）。
- 未实测的数字不得写入文档；测试脚本不得在跳过验证时输出"通过"。
