# YOLO26n + VisDrone2019 无人机地面目标跟踪

基于 **Ultralytics YOLO26n** 的无人机地面目标检测与跟踪系统，使用天津大学开源的 **VisDrone2019** 数据集。

## 项目特性

- **双模型架构**: Baseline检测模型 + TrackingYOLO26 JDE跟踪模型
- **最新YOLO26n**: NMS-Free端到端检测，MuSGD优化器
- **JDE跟踪**: 检测+Re-ID嵌入一体化，端到端多目标跟踪
- **VisDrone2019**: 10类无人机视角目标，支持DET和MOT格式
- **一键操作**: train / pause / resume / eval / export
- **Cloud Studio 支持**: 三步启动，GPU云端训练
- **bypy 数据获取**: 百度网盘命令行下载

---

## 双模型

| 模型 | 命令 | 能力 | 用途 |
|------|------|------|------|
| **YOLO26n Baseline** | `make start-train` | 目标检测 | 基准性能对比 |
| **TrackingYOLO26** | `make train-tracking` | 检测 + Re-ID跟踪 | JDE多目标跟踪 |

> 研究报告: `docs/research_report.md` — 8000字学术论文，30篇真实参考文献

---

## Cloud Studio 云端训练

### Step 1 — 克隆 + 配置环境

```bash
git clone https://github.com/liu06173/my_graduation_yolo26n.git
cd my_graduation_yolo26n
make env-setup
```

自动完成: Conda环境创建 → PyTorch(CUDA)安装 → Ultralytics安装 → bypy配置

### Step 2 — 准备数据集

**首次使用先授权bypy:**

```bash
bypy info
# 1. 复制打开的链接到浏览器 → 登录百度账号
# 2. 复制授权码粘贴回终端
```

**下载并解压:**

```bash
# 下载压缩包到 data/
bypy downfile /task3/VisDrone2019-MOT-train.zip ./data/
bypy downfile /task3/VisDrone2019-MOT-val.zip   ./data/
bypy downfile /task3/VisDrone2019-MOT-test.zip  ./data/   # 可选

# 一键解压 + 转YOLO格式 + 验证
make prepare-data
```

### Step 3 — 开始训练

```bash
# 训练 Baseline 检测模型
make start-train

# 训练 TrackingYOLO26 跟踪模型
make train-tracking
```

**后台训练（断开SSH不影响）:**

```bash
nohup bash scripts/start_train.sh > runs/train.log 2>&1 &
nohup bash scripts/train_tracking.sh > runs/tracking.log 2>&1 &

# 查看进度
tail -f runs/train.log
tail -f runs/tracking.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

**高级选项:**

```bash
# P2模型（更好小目标检测）
bash scripts/start_train.sh --p2

# 恢复中断的训练
make start-train --resume
make train-tracking --resume

# 自定义参数
bash scripts/start_train.sh --batch 64 --epochs 500 --imgsz 1280
```

---

## 本地快速开始

```bash
git clone https://github.com/liu06173/my_graduation_yolo26n.git
cd my_graduation_yolo26n
make setup                        # 环境配置
make prepare-data                 # 准备数据
make train                        # 训练 baseline
make train-tracking               # 训练跟踪模型
```

---

## 命令大全

### 环境与数据

| 命令 | 说明 |
|------|------|
| `make env-setup` | Cloud Studio环境 (conda + CUDA + bypy) |
| `make setup` | 本地环境 (venv) |
| `make prepare-data` | 解压 data/*.zip + 转YOLO格式 + 验证 |
| `make prepare-data-zip ZIP=x.zip` | 处理指定压缩包 |

### 训练控制

| 命令 | 说明 |
|------|------|
| `make train` / `make start-train` | 训练 Baseline 检测模型 |
| `make resume` | 恢复 Baseline 训练 |
| `make train-tracking` | 训练 TrackingYOLO26 跟踪模型 |
| `make train-tracking-resume` | 恢复跟踪模型训练 |
| `make pause` | 优雅暂停 (Ctrl+C) |
| `make status` | 查看训练状态 |
| `make logs` | 查看训练日志 |

### 评估与推理

| 命令 | 说明 |
|------|------|
| `make eval` | 评估模型mAP |
| `make infer SRC=test.jpg` | 单张图片推理 |
| `bash scripts/infer.sh video.mp4` | 视频推理 |
| `make export` | 导出ONNX |
| `make export-engine` | 导出TensorRT |

---

## 项目结构

```
.
├── README.md                      # 本文档
├── Makefile                       # 命令入口
├── setup.sh                       # 本地环境配置
├── requirements.txt               # Python依赖
├── environment.yml                # Conda环境
├── Dockerfile                     # Docker环境
├── .gitignore
├── configs/
│   ├── visdrone.yaml              # VisDrone2019 数据配置 (10类)
│   └── hyp_visdrone.yaml          # 训练超参数
├── models/                        # ★ 改进模型
│   ├── tracking_model.py          #   TrackingYOLO26 (JDE跟踪)
│   └── tracking_loss.py           #   TripletLoss + CenterLoss
├── scripts/
│   ├── setup_env.sh               # Cloud Studio环境配置
│   ├── prepare_data.sh            # 解压 + 转换 + 验证
│   ├── start_train.sh             # Baseline训练
│   ├── train_tracking.sh          # ★ 跟踪模型训练
│   ├── train.sh                   # 本地训练
│   ├── eval.sh                    # 评估
│   ├── infer.sh                   # 推理
│   └── export.sh                  # 模型导出
├── tools/
│   ├── prepare_visdrone.py        # VisDrone标注转YOLO (DET+MOT)
│   ├── train_tracking.py          # ★ 跟踪模型训练脚本
│   └── train_ctrl.py              # 训练管理
├── docs/
│   ├── research_report.md         # ★ 学术论文格式研究报告
│   └── help/                      # 日常问题记录
│       ├── git.md
│       ├── env.md
│       └── tools.md
├── data/
│   ├── *.zip                      # 手动下载的压缩包
│   ├── VisDrone_raw/              # 解压后的原始数据
│   └── visdrone/                  # YOLO格式数据集
├── ultralytics/                   # YOLO26源代码
└── runs/                          # 训练输出
    ├── detect/train/              # Baseline训练结果
    └── tracking/                  # 跟踪模型训练结果
```

★ = 本次新增

---

## VisDrone2019 数据集

| 属性 | 值 |
|------|-----|
| 来源 | 天津大学机器学习与数据挖掘实验室 |
| 训练集 | 6,471 张 (DET) / 56个序列 (MOT) |
| 验证集 | 548 张 (DET) / 7个序列 (MOT) |
| 测试集 | 1,610 张 |
| 类别 | 10类 |
| 分辨率 | ~2000×1500 |

**类别:** pedestrian(行人) / people(人群) / bicycle(自行车) / car(小汽车) / van(面包车) / truck(卡车) / tricycle(三轮车) / awning-tricycle(带蓬三轮车) / bus(公交车) / motor(摩托车)

---

## 训练建议

### 两阶段训练（推荐）

```bash
# 阶段1: 冻结backbone，训练head (前10 epoch)
# hyp_visdrone.yaml 中 freeze=10
bash scripts/train.sh --epochs 10

# 阶段2: 全参数微调
bash scripts/train.sh --resume --epochs 300
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 300 | 小目标需更多轮次 |
| batch | 32 | 16G显存→64 |
| imgsz | 640 | P2版本→1280 |
| mosaic | 1.0 | 小目标关键增强 |
| close_mosaic | 10 | 最后10 epoch关闭 |

---

## TrackingYOLO26 — 改进模型说明

在原版YOLO26n基础上新增:

| 模块 | 说明 |
|------|------|
| **P2检测层** | stride=4, 160×160, 增强极小目标 |
| **EmbedHead** | 并行Re-ID分支, 128维嵌入向量 |
| **SEBlock** | 通道注意力, 抑制背景噪声 |
| **MEF** | 多尺度嵌入融合 (P2-P5) |
| **TripletLoss** | Hard mining 三元组损失 |
| **CenterLoss** | 类内紧凑性约束 |

详见 `docs/research_report.md`

---

## 常见问题

**Q: 训练中断后如何恢复？**
```bash
make resume                # Baseline
make train-tracking-resume # 跟踪模型
```

**Q: 如何更换模型尺寸？**
```bash
# 修改脚本中的 MODEL: yolo26n.pt → yolo26s.pt / yolo26m.pt
```

**Q: 新加了数据zip怎么办？**
```bash
make prepare-data  # 再跑一次，已解压的自动跳过
```

**Q: 两个模型能同时训练吗？**
```bash
# 可以，输出目录不同，互不冲突
# runs/detect/  vs  runs/tracking/
```

## License

本项目代码基于 AGPL-3.0 许可证。VisDrone2019 数据集请遵循其原始许可协议。

---

**模型**: YOLO26n + TrackingYOLO26 | **数据集**: VisDrone2019 (天津大学) | **框架**: Ultralytics 8.4+
