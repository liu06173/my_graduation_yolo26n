# YOLO26s UAV 无人机目标检测与跟踪

基于 **Ultralytics YOLO26** 的无人机地面目标检测与跟踪系统，使用天津大学开源的 **VisDrone2019** 数据集。

## 分支版本

| 分支 | 模型 | 版本 | 说明 |
|------|------|------|------|
| `master` | YOLO26n | v0.2 | 基础检测+JDE跟踪，训练速度优化 |
| `yolov26s` | YOLO26s | v0.2 | YOLO26s 替换 nano，精度更高 |
| **`yolov26s-uav`** | **YOLO26s-P2** | **v0.3** | **★ UAV追踪改进版：5个新模块 + P2检测头** |

## 项目特性

- **5个新模块**: ECA、CoordAtt、DySample、WeightedConcat、C3k2_ECA
- **P2检测头**: 4尺度检测 (P2/4-P5/32)，小目标检测能力翻倍
- **改进ReID**: ECA+CoordAtt 嵌入头，跟踪关联更精准
- **VisDrone2019**: 10类无人机视角目标，支持DET和MOT格式
- **JDE跟踪**: 检测+Re-ID嵌入一体化，端到端多目标跟踪
- **速度优化**: cache=True, rect=True, 每轮3-5分钟

---

## 快速开始 (yolov26s-uav 分支)

```bash
# 克隆分支
git clone --branch yolov26s-uav --depth 1 git@github.com:liu06173/my_graduation_yolo26n.git
cd my_graduation_yolo26n

# 安装
cd ultralytics && pip install -e . --no-deps && cd ..
```

### 训练 P2 检测模型

```bash
yolo detect train \
  model=yolo26s-p2-tracking.yaml \
  pretrained=yolo26s.pt \
  data=configs/visdrone.yaml \
  epochs=100 \
  imgsz=640 \
  batch=18 \
  device=0 \
  workers=12 \
  cache=True
```

### 训练 JDE 跟踪模型

```bash
python tools/train_tracking.py \
  --model yolo26s-p2-tracking.yaml \
  --weights yolo26s.pt \
  --data configs/visdrone.yaml \
  --epochs 100 \
  --batch 16 \
  --imgsz 640 \
  --embed_dim 128
```

### 后台训练 + 监控

```bash
nohup bash -c "yolo detect train model=yolo26s-p2-tracking.yaml pretrained=yolo26s.pt data=configs/visdrone.yaml epochs=100 imgsz=640 batch=18 device=0 workers=12 cache=True" > runs/train.log 2>&1 &

tail -f runs/train.log     # 查看日志
watch -n 1 nvidia-smi       # 监控GPU
python tools/train_ctrl.py status  # 训练状态
```

### 恢复训练

```bash
yolo detect train \
  model=yolo26s-p2-tracking.yaml \
  pretrained=yolo26s.pt \
  data=configs/visdrone.yaml \
  resume=True
```

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

## v0.3 新增模块 (yolov26s-uav)

| 模块 | 论文 | 参数 | 作用 |
|------|------|------|------|
| **ECA** | ECA-Net (ECCV 2020) | ~0 | 1D卷积通道注意力，自适应核大小 |
| **CoordAtt** | Coordinate Attention (CVPR 2021) | 轻量 | 位置感知通道注意力，保留空间坐标 |
| **WeightedConcat** | BiFPN / EfficientDet (CVPR 2020) | 每输入1标量 | 学习加权特征融合，替代无脑Concat |
| **DySample** | DySample (ICCV 2023) | 轻量 | 内容感知动态上采样，替代nearest |
| **C3k2_ECA** | — | ~0 | ECA增强的C3k2变体 |
| **ECABottleneck** | — | ~0 | ECA增强的Bottleneck |

### 模型架构

```
Backbone: Conv → C3k2_ECA → CoordAtt → Conv → C3k2_ECA → CoordAtt →
          Conv → C3k2 → Conv → C3k2 → SPPF → C2PSA
Neck:     DySample↑ + WeightedConcat (FPN 4-scale) →
          Conv↓ + WeightedConcat (PAN 4-scale)
Head:     Detect(P2/4, P3/8, P4/16, P5/32)
```

4个检测尺度: P2(stride 4) / P3(stride 8) / P4(stride 16) / P5(stride 32)

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
│   ├── tracking_model.py          #   TrackingYOLO26 (JDE跟踪, ECA+CoordAtt)
│   └── tracking_loss.py           #   TripletLoss + CenterLoss
├── ultralytics/ultralytics/
│   ├── cfg/models/26/
│   │   ├── yolo26.yaml            #   标准YOLO26配置
│   │   ├── yolo26-p2.yaml         #   P2检测头配置
│   │   └── yolo26-p2-tracking.yaml #  ★ v0.3 P2+新模块配置
│   └── nn/modules/
│       ├── conv.py                #   ★ ECA, CoordAtt, DySample, WeightedConcat
│       └── block.py               #   ★ ECABottleneck, C3k2_ECA
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

## v0.3 模块技术说明

### ECA (Efficient Channel Attention)
- 用自适应1D卷积替代FC层，~0额外参数
- 核大小公式: `k = |log2(C)/2 + 1/2|_odd`
- 集成于 C3k2_ECA 和 ECABottleneck

### CoordAtt (Coordinate Attention)
- 将2D全局池化分解为(H,1)+(1,W)两个1D编码
- 保留空间位置信息，适合UAV俯视场景

### DySample (Dynamic Upsampling)
- 轻量卷积生成内容感知采样点偏移
- 比 nearest 更精准保留小目标边界细节

### WeightedConcat (BiFPN 加权融合)
- 学习每个输入特征的标量权重 (softmax归一化)
- 初始权重=1，退化为普通Concat，训练后自适应

---

## TrackingYOLO26 — 改进模型说明

在原版YOLO26n基础上新增:

| 模块 | 说明 |
|------|------|
| **P2检测层** | stride=4, 160×160, 增强极小目标 |
| **EmbedHead** | 并行Re-ID分支, 128维嵌入向量 (v0.3: ECA+CoordAtt) |
| **SEBlock** → **ECA** | ~0参数通道注意力替代SE |
| **CoordAtt** | 位置感知注意力, 增强ReID特征 |
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

**分支**: yolov26s-uav (v0.3) | **模型**: YOLO26s-P2 + TrackingYOLO26 | **数据集**: VisDrone2019 (天津大学) | **框架**: Ultralytics 8.4+
