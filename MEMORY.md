# 项目记忆索引

- [README](../README.md) — 项目定位、实测指标与口径说明、快速开始、已知限制
- [架构说明](../docs/架构说明.md) — 模块划分、数据流、依赖方向、设计取舍与技术债
- [项目现状与差距](../docs/项目现状与差距.md) — 实现 vs 设计承诺的逐条差距（每条附证据）
- [使用指南](../docs/使用指南.md) — 安装、四种检测模式、VTube Studio 追踪、FAQ
- [面试讲解提纲](../docs/面试讲解提纲.md) — 讲解流程与预设问答
- [变更记录](../CHANGELOG.md) — 重要变更
- [开发约定](../.dsh-dev/conventions.md) — 硬性规则（路径、目录、指标口径）
- [决策记录](../.dsh-dev/decisions.md) — 每次变更的候选方案、理由与结果
- [历史设计文档](../docs/archive/README.md) — **描述的是当初的目标，不是现状**

## 关键事实速查

| 项 | 值 |
|:---|:---|
| 定位 | 个人独立完成的工程实践项目（**无人员分工**） |
| GPU（实测） | RTX 3060 Laptop，**6 GB**，驱动 581.95，CUDA 12.1 |
| 环境 | Windows / Python 3.12.8 / PyTorch 2.5.1+cu121 |
| 精度 | 全验证集 mAP50 = 0.663（v1）/ 0.662（v2），**非** Hard 子集口径 |
| 训练超参 | batch=8, imgsz=640, AdamW, lr 1e-3→1e-5, amp, nbs=64, workers=2, seed=0 |
| 主要短板 | Recall 0.595；分项评估未做；ONNX 精度验证被跳过；v2 为负收益 |

## 运行方式

```bash
python -m src.deploy.detect --input 0        # 实时检测
python -m src.train.train                    # 训练
python -m src.eval.evaluate                  # 评估
python apps/vtube_bridge/main.py --input 0   # VTube Studio 表情驱动
```

脚本必须以 `python -m` 方式从项目根运行；路径一律经 `src/paths.py` 解析。
