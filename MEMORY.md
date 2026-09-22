# 项目记忆索引

- [README](../README.md) — **唯一的项目文档**：定位与角色、指标口径基准、架构、环境与快速开始、实时管线、三条出口、工程规范、被否决的结论、已知限制、证据索引
- [架构说明（已归档）](../docs/archive/架构说明.md) — 模块划分、数据流、依赖方向、设计取舍与技术债（**已被 README §3 取代**）
- [项目现状与差距（已归档）](../docs/archive/项目现状与差距.md) — 2026-09-15 的差距台账（**已被 README §14 取代**）
- [使用指南（已归档）](../docs/archive/使用指南.md) — 安装、四种检测模式、VTS 追踪、FAQ（**已被 README §4/§5 取代**）
- [模拟面试项目全解](../docs/模拟面试项目全解.md) — **自包含答辩底稿**：项目全景 + 口径基准 + 高频问答 + 红线清单
- [面试讲解提纲](../docs/面试讲解提纲.md) — 讲解流程与预设问答
- [简历项目经历](../docs/简历项目经历.md) — 简历稿（游戏客户端方向）、皮套跑通前后的措辞、禁止写清单
- [皮套落地三线方案](../docs/plans/2026-09-15-皮套落地三线.md) — **线 A 2D（VTS+Live2D）/ 线 B Unity / 线 C 保底**，含验收标准与 Live2D 授权要点
- [线 A 操作手册（2D）](../apps/vtube_bridge/README.md) — VTube Studio 安装、插件 API、官方样本模型与版权要求、命令与排障
- [皮套联调：问题与解决](../docs/皮套联调问题与解决.md) — **2026-09-16 首次真机联调**：8 个问题的现象/根因/证据/解决（含 4 个代码 bug）+ 未验证项 + 回归防护清单（新增 22 个测试）
- [3D 备选手册（VMC）](../apps/vmc_link/README.md) — VMC/OSC → VSeeFace 的步骤、编码规范与实测数据
- [变更记录](../CHANGELOG.md) — 重要变更
- [开发约定](../.dsh-dev/conventions.md) — 硬性规则（路径、目录、指标口径）
- [决策记录](../.dsh-dev/decisions.md) — 每次变更的候选方案、理由与结果
- [失误记录](../.dsh-dev/mistakes.md) — **助手犯过的错误**：错在哪、影响、如何发现与修复、残留问题
- [历史设计文档](../docs/archive/README.md) — **描述的是当初的目标，不是现状**

## 关键事实速查

| 项 | 值 |
|:---|:---|
| 定位 | **个人主导、AI 辅助实现**的工程实践项目；角色是决策者（选型、架构、判据、验收），无人员分工（见 README §1.2） |
| **线 A 状态（皮套）** | **★ 端到端已跑通**（2026-09-16 真机联调）：`摄像头 → 桥接 → 注入 VTube Studio → hiyori 跟随`，60 s 高频采样中 `MouthOpen` / `FaceAngleX` / `FaceAngleY` / `EyeLeftX` 持续变化。管线 **29.3 帧/秒 = 内置摄像头的硬上限**（`probe_camera.py` 12 组组合全部 `read ≈ 33.3 ms = 1000/30`）。联调共修掉 **7 个问题**、新增 **94 个回归测试**（全量 **155 / OK**）。**验收进展**：7 条中 1–4 使用者目视确认 ✅、5 连续稳定性 ✅（**2026-09-16 实跑 600 s：连续 599.7 s、19/19 个 30 s 窗口参数在变化、掉线与采样失败与中断与 VTS 重启全 0**；原始 CSV/JSON 已落盘 `artifacts/diag/vts_stability_20260916-150813.*`，别人可自己重算。此前那次 7 分钟跑只有 ≈5.0 分钟且**没留原始文件**，已被这次取代）、6 仅帧率有口径（**端到端延迟未实测**）、7 演示视频**按使用者决定不做**（证据 = 采样 CSV + VTS 日志 + 测试）。→ **不得写"7 条验收全部通过"**。全过程见 [`docs/皮套联调问题与解决.md`](../docs/皮套联调问题与解决.md) |
| **稳定性判据与采样器（实测）** | 第 5 条的证据工具 = `apps/vtube_bridge/sample_vts_params.py`（默认采 600 s，**原始 CSV 边采边写** + JSON 结论 + 报告，退出码 0/2）。**`connectedPlugins` 来自 `StatisticsRequest`，数的是活着的 API 会话数（含采样器自己）**：2 探针+桥接 = **3**、1 探针+桥接 = **2**、**只有采样器 = 1 = 桥接还没起来（不是掉线）**；同响应 `allowedPlugins` 恒 = 1。读参数用 `ParameterValueRequest` 且键名是 **`name`**（传 `id` → `APIError 500`），**一次只返回一个参数**；复用桥接 token 不弹授权窗。**2026-09-16 两次 10 分钟实跑构成对照**：① 人不在镜头前那次判 **FAIL**（连接侧 599.9 s 全 0 掉线，但 t≈102 s 起参数回落到 VTS 默认值 = `faceFound:false`，后 15 个窗口全 BAD）；② **人入镜那次判 PASS**（连续 599.7 s、**19/19** 窗口参数在变化、全 0 异常）。→ **只看连接日志会把①误判成"稳定 10 分钟"**，这就是判据的价值。另外实测到**离开画面再回来仍能跟随**：15:03:34–15:03:54 采样为 VTS 默认值（无人），15:07:57 再采已恢复（`MouthOpen 0.223→0.258`、`FaceAngleX -0.42→2.27`） |
| GPU（实测） | RTX 3060 Laptop，**6 GB**，驱动 581.95，CUDA 12.1 |
| 环境 | Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 |
| 精度 | 全验证集 mAP50 = **0.66501**（v1）/ **0.66404**（v2），**非** Hard 子集口径。来源 `artifacts/reports/metrics_v{1,2}.json`（best 权重重测，非训练日志末行） |
| 训练超参 | 历史训练 batch=8；**配置已改 batch=6**（实测 batch=8 峰值保留 6.18 GiB > 6 GB 上限）。imgsz=640, AdamW, lr 1e-3→1e-5, amp, nbs=64, workers=2, seed=0 |
| 主要短板 | Recall 0.595（其中 <32px 占 GT 的 72.5%、召回仅 0.493）；WIDER 分项评估因无网络未取得（返回 `null`，非 mAP 冒充）；v1→v2 基本持平（非负收益） |
| OpenCV（实际生效） | **opencv-contrib-python 5.0.0.93**（`cv2.__version__` = 5.0.0，而 `pip show opencv-python` 仍报 4.13.0.92）。`opencv-python` / `-headless` / `-contrib` 三个分发包**共用同一个 `site-packages/cv2/`，最后安装者生效** → 护栏 `tests/test_opencv_environment.py`（缺 GUI 符号即红，不 skip）；`requirements.txt` 声明已改为 `opencv-contrib-python>=4.5.0` |
| 摄像头（实测） | **Integrated Webcam**（`USB\VID_0C45&PID_6A14&MI_00`，Camera 类，`Started`）。⚠️ **受限 shell（助手）取不到该设备**（`Get-PnpDevice` 返回 0 个设备、`Get-CimInstance` 拒绝访问）→ **凡是要开摄像头的命令必须由使用者本人运行**。使用者终端实测：`cv2.VideoCapture(0)` 与 `CAP_DSHOW` **均 `opened True`** |
| VTube Studio | **已安装可运行**：`C:\steam1\steam\steamapps\common\VTube Studio`（AppID 1325860，v1.35.10），**自带 5 个主模型**（hiyori / akari / tororo / hijiki / wanko）。**API 已开**（`Config_StartAPI = True`，`netstat` 确认 `0.0.0.0:8001` 监听）。**主模型已加载**：`ModelLoadRequest` → `modelLoaded=true, modelName=hiyori`（`modelID 5b786375d675445cb5301bb032dc333c`）。**VTS 侧注入链路已证明可用**（见下行） |
| VTS 侧已验证的事实 | ① **参数名风险排除**：hiyori 有 127 个输入参数，桥接要用的 **16 个全部存在**；② **注入回读 PASS**：注入 `MouthOpen 0.77 / FaceAngleX 22.5 / EyeOpenLeft 0.25 / EyeLeftX 0.6 / MouthSmile 0.9` 后回读**全部 MATCH**；③ **端到端已跑通**（2026-09-16，使用者终端跑桥接）：60 s 采样实测 `MouthOpen` range 0.313、`FaceAngleX` 13.8、`FaceAngleY` 31.8、`EyeLeftX` 0.333 **均在变化**；④ **曾有一个必现 bug 已修**：`_map_value` 的单向参数映射退化，把 `EyeOpenLeft/Right` 压成常量 0 → 皮套整场闭着眼（相机图本身正常，实测 `EyeOpenLeft=0.967`）。见 CHANGELOG 与 `tests/test_vts_parameter_mapping.py` |
| 方向与增益约定（实测） | ① **头部 yaw 取负**：`FaceAngleX = -yaw`（实测"左右摇头方向相反"，2026-09-16）；② **眼球 X/Y 不取反**：两次目视反馈互相矛盾，判定真凶是"头方向反"（看侧方时头也转）；若头正了眼仍反则单独翻；③ 点头（`FaceAngleY=-pitch`）/歪头（`FaceAngleZ=+roll`）**未被反馈过，待复验**；④ **头部三轴按 ±30 校准**：源 ±45 → VTS ±30 原为 0.67 倍，校准后 1:1（`raw 20 -> 20`，**+50%**）；⑤ **方向只能由使用者目视判定**，数值一致（input 与 Param 同号）不能证明方向对；⑥ 眨眼实测：慢速闭紧能到 `ParamEyeLOpen 0.017`（完全闭），**快眨只到 0.09–0.12 就回弹** —— 受 `--expression-alpha 0.45` EMA 与 VTS 模型侧 `Smoothing: 10` 限制 |
| 眨眼幅度（实测口径） | 本机 640×480 / 人脸宽约 147 px 下，**MediaPipe 的 `eyeBlink` 上限只有 ~0.75**：尽力闭眼保持 2 s，`EyeOpenLeft` 只到 **0.251**（=合并眼），`ParamEyeLOpen` 到 0.480。已用本项目校准机制按实测范围 `0.25/0.26..0.99` 校准（`apps/vtube_bridge/config/tracking_calibration.json`）→ `0.251 -> 0.001`。**该范围依赖当前相机/距离/光线，换条件需重测**；`--no-gui` 下无 GUI 校准入口，只能手写该 JSON |
| 帧率瓶颈（实测结论） | **本机内置摄像头硬上限 30 帧/秒**：`probe_camera.py` 跑满 12 组组合（msmf/dshow × 640x480/1280x720 × @30/@60 × 默认/MJPG），**全部 read ≈ 33.3 ms = 1000/30**，实测 28.3–30.0 帧/秒；@60 与 MJPG 请求**被驱动静默忽略**，且驱动会**谎报**帧率（dshow 请求 @60 → `CAP_PROP_FPS` 回 60，实测 29.8）。故 summary 里的 `read=15.8 ms` **不是可优化开销**，而是"帧周期 33.3 − 算力 16.2"的等待。`--imgsz`/`--yolo-every`/`--camera-*` 都无效；**超 30 帧/秒只能换摄像头**（管线算力 ≈16 ms ≈ 59 帧/秒 有余量） |
| 眨眼平滑与阈值（实测） | 眼睛走 `BlinkStabiliser`：**非对称平滑**（确实在闭时 α=0.8，其余 0.45）+ **带迟滞吸附** + **按深度分档**。**实测三档 raw**：眨眼谷底/真闭眼 **0.300–0.365**、**眯眼 0.410–0.550**、睁眼 0.63–0.97 → 阈值设进空隙：`--blink-snap-below 0.40`、`--blink-deep-below 0.33`、`--blink-squint-frames 6`（兜底）。**测量档**：`--blink-snap-below 0 --blink-deep-below 0` 关掉吸附后，VTS 回读值可反推 raw。**刻意不抬高校准下限**（会放大睁眼抖动） |
| VTS 侧需注意 | ① 开插件 API 的开关是 `Config_StartAPI`（界面原文「**开启API（允许安装插件）**」）；② `Config_Webcam_Auto_Start = False`，故**面捕不必手动关**；③ ⚠️ **拖放无法加载主模型**：`.json` 是 VTS 的禁用拖放类型（`Blocked file types: ".pdf", ".txt", ".html", ".json"`），拖文件夹只会得到「Live2D 挂件」→ **主模型必须用界面模型选择或 API `ModelLoadRequest`**；④ 唇同步：`Config_LipsyncType = ULipSync`，注意 `Config_UseMicrophone` 已是 `False` 但 uLipSync 仍在跑，要改的是「对口型模式」；⑤ ⚠️ **首次运行要点 VTS 弹窗**：`AuthenticationTokenRequest` 现已单独用 `AUTH_TOKEN_TIMEOUT = 120 s`（旧代码统一 1 s 超时 → 桥接必然在点击前断开，token 永远缓存不下来；2026-09-16 已修，见 CHANGELOG） |
| 受限 shell 取证红线 | `Get-PnpDevice`（返回 0 个设备）、`Get-CimInstance`（拒绝访问）、**`Get-NetTCPConnection`（连 445 端口都报 0 条）在本环境均不可信**。请改用 `pnputil /enum-devices /connected`、注册表、**`netstat -ano`**、真实 TCP 连接。**空结果不是否定证据** —— 见 `mistakes.md` 失误 4 与 5 |

## 运行方式

```bash
python -m src.deploy.detect --input 0        # 实时检测
python -m src.train.train                    # 训练
python -m src.eval.evaluate                  # 评估
python apps/vtube_bridge/main.py --input 0   # VTube Studio 表情驱动
```

脚本必须以 `python -m` 方式从项目根运行；路径一律经 `src/paths.py` 解析。
