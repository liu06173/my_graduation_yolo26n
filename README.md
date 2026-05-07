# YOLO26n + VisDrone2019 无人机地面目标检测

基于 **Ultralytics YOLO26n** 模型，使用天津大学开源的 **VisDrone2019** 数据集训练的无人机视角目标检测项目。

## 项目特性

- **最新YOLO26n架构**: NMS-Free端到端检测，推理速度更快
- **VisDrone2019数据集**: 10类无人机视角目标 (行人、车辆等)
- **一键操作**: train / pause / resume / eval / export
- **工程化设计**: 支持断点续训、多GPU训练、模型导出

## 快速开始

### 1. 克隆仓库

```bash
git clone <your-repo-url> YOLO26n_VisDrone
cd YOLO26n_VisDrone
```

### 2. 一键配置环境

```bash
make setup
# 或: bash setup.sh
```

这将自动完成:
- 创建Python虚拟环境
- 安装PyTorch (自动检测CUDA)
- 安装Ultralytics及所有依赖

### 3. 准备数据集

**方式A — 手动下载 (推荐)**

由于数据集约25GB，建议手动下载后转换:

```bash
# 1. 从以下任一来源下载 VisDrone2019-DET:
#    - 官方: https://github.com/VisDrone/VisDrone-Dataset
#    - Kaggle: https://www.kaggle.com/datasets/jamMan/visdrone2019-det

# 2. 运行转换脚本
python tools/prepare_visdrone.py --convert /path/to/VisDrone2019-DET

# 3. 检查数据完整性
python tools/prepare_visdrone.py --check
```

**方式B — 自动下载提示**

```bash
make download-data
```

### 4. 开始训练

```bash
make train
```

## 命令大全

### 训练控制

| 命令 | 说明 |
|------|------|
| `make train` | 新建训练 (自动下载yolo26n.pt预训练权重) |
| `make resume` | 从最近checkpoint恢复训练 |
| `make pause` | 优雅暂停 (Ctrl+C同理，会自动保存last.pt) |
| `make status` | 查看训练状态 |
| `make logs` | 查看训练日志 |

### 高级训练选项

```bash
# 指定GPU和batch size
bash scripts/train.sh --device 0,1 --batch 64

# 指定epochs和分辨率
bash scripts/train.sh --epochs 500 --imgsz 1280

# 从指定权重恢复
bash scripts/train.sh --resume path/to/checkpoint.pt
```

### 评估与推理

```bash
# 评估模型mAP
make eval

# 图片推理
make infer SRC=test.jpg

# 视频推理
bash scripts/infer.sh video.mp4

# 批量推理
bash scripts/infer.sh data/visdrone/images/val
```

### 模型导出

```bash
make export          # ONNX
make export-engine  # TensorRT
```

## 项目结构

```
.
├── README.md                      # 本文档
├── Makefile                       # 命令入口 (make train / make resume ...)
├── setup.sh                       # 一键环境配置
├── requirements.txt               # Python依赖
├── .gitignore
├── configs/
│   ├── visdrone.yaml              # VisDrone2019 数据配置 (10类)
│   └── hyp_visdrone.yaml          # 训练超参数 (针对无人机优化)
├── scripts/
│   ├── train.sh                   # 训练脚本 (支持 --resume)
│   ├── eval.sh                    # 评估脚本
│   ├── infer.sh                   # 推理脚本
│   ├── export.sh                  # 模型导出脚本
│   └── download_data.sh           # 数据集下载指南
├── tools/
│   ├── prepare_visdrone.py        # VisDrone数据格式转换
│   └── train_ctrl.py              # 训练管理 (pause/status/kill/logs)
├── data/
│   └── visdrone/                  # 数据集目录 (需自行下载)
│       ├── images/
│       │   ├── train/
│       │   └── val/
│       └── labels/
│           ├── train/
│           └── val/
├── ultralytics/                   # YOLO26源代码
└── runs/                          # 训练输出 (自动生成)
    └── detect/
        └── train/
            ├── weights/           # best.pt, last.pt
            ├── results.csv        # 训练指标记录
            └── *.png              # 可视化图表
```

## VisDrone2019 数据集

| 属性 | 值 |
|------|-----|
| 来源 | 天津大学机器学习与数据挖掘实验室 |
| 训练集 | 6,471 张 |
| 验证集 | 548 张 |
| 测试集 | 1,610 张 |
| 类别 | 10类 |
| 分辨率 | ~2000×1500 |

**类别列表:**

| ID | 名称 | 英文 |
|----|------|------|
| 0 | 行人 | pedestrian |
| 1 | 人群 | people |
| 2 | 自行车 | bicycle |
| 3 | 小汽车 | car |
| 4 | 面包车 | van |
| 5 | 卡车 | truck |
| 6 | 三轮车 | tricycle |
| 7 | 带蓬三轮车 | awning-tricycle |
| 8 | 公交车 | bus |
| 9 | 摩托车 | motor |

## 训练策略建议

### 分阶段训练

```bash
# 阶段1: 冻结骨干网络，训练检测头 (前10 epoch)
bash scripts/train.sh --epochs 10
# 在 hyp_visdrone.yaml 中设置 freeze=10

# 阶段2: 全参数微调 (10-200 epoch)
bash scripts/train.sh --resume --epochs 200

# 阶段3: 关闭强数据增强，精调 (最后50 epoch)
# 在 hyp_visdrone.yaml 中设置 close_mosaic=50
bash scripts/train.sh --resume --epochs 250
```

### 典型训练参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 300 | 小目标检测需更多轮次 |
| batch | 32 | 可根据显存调整 |
| imgsz | 640 | 保持宽高比resize |
| lr0 | 0.01 | YOLO26推荐初始学习率 |
| optimizer | MuSGD | YOLO26专属优化器 |
| warmup_epochs | 3 | 预热稳定训练 |
| mosaic | 1.0 | 小目标检测关键增强 |
| close_mosaic | 10 | 最后10 epoch关闭mosaic |

## 训练监控

训练开始后，在另一个终端:

```bash
# 查看实时状态
make status

# 查看最新日志
make logs

# TensorBoard可视化
tensorboard --logdir runs/detect/train
```

## YOLO26n 性能参考

| 指标 | 预期值 (VisDrone val) |
|------|---------------------|
| mAP@0.5 | ~35-38% |
| mAP@0.5:0.95 | ~20-24% |
| 推理速度 | ~200 FPS (NVIDIA GPU) |
| 模型大小 | ~5 MB |

*注: VisDrone数据集小目标密集，mAP通常低于COCO。YOLO26n的P2版本可进一步提升小目标性能。*

## 常见问题

**Q: 训练中断后如何恢复？**
```bash
make resume  # 自动查找last.pt
```

**Q: 如何更换为其他YOLO26模型？**
```bash
# 修改 scripts/train.sh 中的 MODEL 变量为:
# yolo26s.pt / yolo26m.pt / yolo26l.pt / yolo26x.pt
```

**Q: 如何使用P2版本 (更好的小目标检测)？**
```bash
# 修改 train.sh: 将 yolo26n.pt 改为 yolo26n-p2.pt
# 并在 hyp_visdrone.yaml 中设置 imgsz=1280
```

**Q: CPU训练太慢怎么办？**
VisDrone数据量大，建议使用GPU。如果只有CPU:
```bash
bash scripts/train.sh --device cpu --batch 4 --epochs 50
```

## License

本项目代码基于 AGPL-3.0 许可证。VisDrone2019数据集请遵循其原始许可协议。

---

**模型配置**: YOLO26n | **数据集**: VisDrone2019 (天津大学) | **框架**: Ultralytics 8.4+
