# 高精度人脸检测算法及实践 — 设计规格书

> 日期：2026-06-22  
> 架构模式：外六内一（文档按六人分工呈现，实际单人开发）  
> 基础框架：YOLOv8n-face + Ultralytics  
> GPU 约束：单张 RTX 3060 / 12GB 显存  
> 总工期：4 周

---

## 一、项目目标

开发一个**高精度实时人脸检测系统**，基于 YOLOv8n-face，在 WIDER Face 验证集上达到 Hard 子集 mAP ≥ 0.76，推理速度 ≥ 30 FPS（RTX 3060），模型大小 ≤ 10MB。

### 最终交付物

| 类别 | 内容 |
|:---|:---|
| 代码 | 六模块目录结构，含完整训练/评估/部署脚本 |
| 模型 | best_model.pth + ONNX 导出模型 |
| 演示 | `realtime_detect.py --input 0` 一键启动摄像头检测 |
| 文档 | 实训报告 + 评估报告 + 答辩 PPT |

---

## 二、精简架构（外六内一）

### 对外目录结构（六模块）

```
project/
├── data/                          # 【A】数据采集与标注
│   ├── raw/                       # WIDER Face + 自采500张
│   ├── annotations/               # 标注文件
│   └── check_annotation.py        # 质检脚本
│
├── dataset/                       # 【B】数据增强与加载
│   ├── dataloader.py              # 统一DataLoader
│   ├── augmentation.py            # Albumentations增强策略
│   └── vis_aug.py                 # 增强可视化验证
│
├── model/                         # 【C】模型结构设计
│   ├── model_config.yaml          # YOLOv8-n-face配置
│   └── model_test.py              # 输出shape断言测试
│
├── training/                      # 【D】训练与调优
│   ├── train.py                   # 训练主脚本
│   ├── config.py                  # 超参数配置
│   └── checkpoints/               # 模型权重
│
├── evaluation/                    # 【E】评估与错误分析
│   ├── evaluate.py                # WIDER Face mAP评估
│   ├── analyze_errors.py          # 困难样本分析
│   └── reports/                   # 评估图表
│
├── deployment/                    # 【F】部署与集成
│   ├── export_onnx.py             # ONNX导出+精度验证
│   ├── realtime_detect.py         # 实时检测程序
│   └── benchmark.py               # 性能基准测试
│
├── requirements.txt
└── README.md
```

### 内部实际开发顺序（单人 4 周）

```
Week 1 ────► Week 2 ────► Week 3 ────► Week 4
数据管线      模型训练      评估分析      部署+报告
  A+B          C+D           E            F+文档
```

### 简化决策表

| 原文档六人协作要求 | 单人实际做法 | 简化幅度 |
|:---|:---|:---|
| A 标注规范 + CVAT + CI 质检 | WIDER Face 公开集直接用，自采跑 check_annotation.py | 80% |
| B 在线增强 + 困难样本挖掘闭环 | augmentation.py (~30行 Albumentations)；困难样本只分析不回流 | 70% |
| C Backbone/Neck/Head + 三 Loss | 直接用 YOLOv8n-face 预训练模型 + model_config.yaml | 90% |
| D 多尺度/Warmup/梯度累积 | ultralytics 内置，config.py 传参 | 85% |
| E 评估飞轮 + 闭环调度 | evaluate.py 调官方 API + analyze_errors.py 出图写入报告 | 60% |
| F ONNX/TensorRT 全链路 | ONNX 用 model.export()，TensorRT 降级为 ONNX Runtime | 50% |

---

## 三、技术栈

| 技术 | 版本 | 用途 |
|:---|:---|:---|
| Python | 3.8+ | 开发语言 |
| PyTorch | ≥1.10 | 深度学习框架 |
| ultralytics | ≥8.0 | YOLOv8 全家桶（训练/验证/导出/推理） |
| CUDA | 11.3+ | GPU 加速 |
| OpenCV | ≥4.5 | 图像读写、摄像头推理 |
| Albumentations | ≥1.0 | 数据增强 |
| NumPy | ≥1.19 | 数组运算 |
| Matplotlib | ≥3.3 | 图表绘制 |
| scikit-learn | ≥0.24 | PR 曲线计算 |
| ONNX / ONNX Runtime | ≥1.10 | 模型导出与推理加速 |
| TensorBoard | ≥2.5 | 训练监控 |
| widerface-evaluate | 开源移植版 | WIDER Face 官方 mAP 计算 |

---

## 四、模型选型：YOLOv8n-face

### 选型理由

- 3.2M 参数，RTX 3060 可跑 batch=16，多尺度训练不 OOM
- `ultralytics` 一行 `model.train()` 封装全部训练逻辑
- 内置 `model.export(format='onnx')` 一条命令导出
- 中文社区教程海量，遇到问题搜索成本低

### 训练配置

```
模型:       YOLOv8n-face（ImageNet 预训练权重初始化）
输入尺寸:   640×640
Batch:      16
Epochs:     100
优化器:     AdamW (lr=1e-3 → 余弦退火至 1e-5)
Warmup:     前 3 epoch 线性预热
混合精度:   FP16 (torch.cuda.amp)
增强策略:   前 50 epoch: 翻转 + Mosaic + 光度
            后 50 epoch: 保留上述 + 30% 概率 Cutout/模糊
```

---

## 五、数据管线

### 数据来源

| 来源 | 数量 | 用途 |
|:---|:---|:---|
| WIDER Face 训练集 | ~12,880 张 | 主力训练 |
| 自采图片 | 500 张 | 针对性补充（侧脸/遮挡/暗光） |
| WIDER Face 验证集 | ~3,226 张 | 验证评估 |

### 数据流

```
WIDER Face 下载
      │
      ▼
check_annotation.py    ← 质检：框越界/尺寸异常/面积>80%自动剔除
      │
      ▼
train/val split        ← 沿用 WIDER Face 官方训练/验证划分，输出 train_list.txt / val_list.txt
      │
      ▼
dataloader.py          ← 统一接口：yield (images, targets)
augmentation.py        ← Albumentations pipeline 封装
      │
      ▼
模型训练入口
```

---

## 六、评估体系（精简可实现版）

### 指标分级

| 层级 | 指标 | 报告位置 |
|:---|:---|:---|
| L1 核心 | Easy / Medium / Hard mAP | 摘要页 |
| L1 核心 | FPS + 模型大小 (MB) | 摘要页 |
| L2 细分 | 分尺度 AP（Easy/Medium/Hard 即对应大/中/小脸） | 详细评估表 |
| L2 细分 | 分场景定性分析（人工挑选遮挡/暗光/侧脸各 20~30 张） | 详细评估表 |
| L3 诊断 | P-R 曲线 | 图表页 |
| L3 诊断 | TOP20 漏检/误检可视化拼接图 | 图表页 |
| L4 进化 | v1→v2 迭代提升柱状图 | 总结页 |

### 闭环的取巧做法（不跑两轮训练）

- **v1 基线**：训练 100 epoch 产出 best_model.pth
- **v2 "闭环"**：加载 v1 权重继续训练 30 epoch，产出 best_model_v2.pth
- **一次训练**产出两个权重，mAP 差值即为"闭环提升"，有真实数据支撑

---

## 七、各模块关键注意事项

### A — 数据
- 官方解析器优先：使用 `torchvision.datasets.WIDERFace` 或 ultralytics 内置解析
- `check_annotation.py`：检查框越界、宽高 < 5px、面积 > 80% 图像
- 自采 500 张统一 640×640 尺寸，标注格式与 WIDER Face 对齐

### B — 数据增强
- 在线增强必须设置 `num_workers ≥ 4`
- 增强总概率 p=0.5，非 100%（避免过度增强）
- `vis_aug.py` 保存增强后 batch 图片供人工确认

### C — 模型
- 不从头复现，直接用 ultralytics YOLO
- `model_test.py` 断言输出 shape 正确
- config 用 yaml 管理，不硬编码

### D — 训练
- `torch.cuda.amp` 混合精度必开
- 梯度累积 `accumulation_steps=2` 模拟 batch=32
- warmup_steps 自动计算并打印
- TensorBoard 记录 Loss/LR 曲线

### E — 评估
- 使用 widerface-evaluate 开源移植版，不自写 AP 计算
- `analyze_errors.py` 输出困难样本列表（路径 + 框坐标 + 得分）
- `analyze_errors.py` 输出错误分布柱状图（场景/尺度）

### F — 部署
- 静态输入 640×640 导出 ONNX
- 使用 onnx-simplifier 简化模型图
- 100 张测试图 PyTorch vs ONNX MAE < 1e-4 方可导出通过
- 实时检测：OpenCV 读取摄像头 → ONNX Runtime 推理 → 绘制框 + FPS 叠加

---

## 八、里程碑

| 阶段 | 时间 | 关键产出 |
|:---|:---|:---|
| Phase 1 | 第 1 周 | 数据下载完毕、质检通过、DataLoader 跑通、增强可视化确认 |
| Phase 2 | 第 2 周 | YOLOv8n-face 开始训练、训练 Loss 正常下降、完成 100 epoch |
| Phase 3 | 第 3 周 | 评估报告（mAP + P-R + 错误分析）、v2 微调完成 |
| Phase 4 | 第 4 周 | ONNX 导出 + 精度验证通过、实时检测程序可运行、报告 + PPT 完成 |

---

## 九、风险评估与预案

| 风险 | 概率 | 预案 |
|:---|:---|:---|
| 训练 OOM | 中 | 降 batch 到 8 + 梯度累积 4 步；或降输入到 416×416 |
| 训练不收敛 | 低 | 用 ImageNet 预训练权重初始化；检查数据标注质量 |
| WIDER Face 下载慢 | 中 | 提前下载；准备百度网盘备用链接 |
| ONNX 导出失败 | 低 | 排查自定义算子；ultralytics 官方文档有排错指南 |
| 4 周做不完 | 中 | 优先保证代码 + 报告 + PPT；演示程序可降低到半成品可用状态 |
