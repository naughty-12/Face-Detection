# 高精度人脸检测算法及实践 — 项目上下文

## 必读文档（每次操作前）

在进行任何编码、文件修改、项目操作之前，**必须先读取以下文档**：

1. `docs/superpowers/specs/2026-06-22-face-detection-design.md` — 设计规格书（架构、技术栈、指标、里程碑）
2. `docs/superpowers/plans/2026-06-22-face-detection-plan.md` — 实施计划（详细任务分解）

## 关键约束

- **开发者**：单人（文档按六人分工呈现）
- **GPU**：单张 RTX 3060 / 12GB 显存
- **工期**：4 周
- **框架**：YOLOv8n-face + Ultralytics
- **Python**：3.8+
- **PyTorch**：≥1.10

## 工作目录

`C:\学习\课程设计`

## 目录结构

```
project/
├── data/            # A — 数据采集与标注
├── dataset/         # B — 数据增强与加载
├── model/           # C — 模型结构设计
├── training/        # D — 训练与调优
├── evaluation/      # E — 评估与错误分析
├── deployment/      # F — 部署与集成
├── docs/            # 设计文档与计划
└── .claude/         # 项目配置
```

## 操作原则

- 始终遵循设计规格书中的技术选型和架构决策
- 新代码匹配现有目录结构中的模块归属
- 编码前确认对应模块的注意事项（见规格书第七章）
- 训练相关操作注意 12GB 显存限制
