# 高精度人脸检测算法及实践

基于 YOLOv8n-face 的高精度实时人脸检测系统。

## 快速开始

```bash
pip install -r requirements.txt
python realtime_detect.py --input 0
```

## 项目结构

| 模块 | 负责内容 | 目录 |
|:---|:---|:---|
| A | 数据采集与标注 | data/ |
| B | 数据增强与加载 | dataset/ |
| C | 模型结构设计 | model/ |
| D | 训练与调优 | training/ |
| E | 评估与错误分析 | evaluation/ |
| F | 部署与集成 | deployment/ |

## 模型性能

| 指标 | 值 |
|:---|:---|
| WIDER Face Hard mAP | TBD |
| 推理速度 (RTX 3060) | TBD FPS |
| 模型大小 | TBD MB |
