# YOLO26n + VisDrone2019 无人机地面目标检测

基于 **Ultralytics YOLO26n** 模型，使用天津大学开源的 **VisDrone2019** 数据集训练的无人机视角目标检测项目。

## 项目特性

- **最新YOLO26n架构**: NMS-Free端到端检测，推理速度更快
- **VisDrone2019数据集**: 10类无人机视角目标 (行人、车辆等)
- **一键操作**: train / pause / resume / eval / export
- **工程化设计**: 支持断点续训、多GPU训练、模型导出
- **Cloud Studio 支持**: 三步启动，GPU云端训练

---

## Cloud Studio 云端训练（三步启动）

### Step 1 — 克隆 + 配置环境

```bash
git clone https://github.com/liu06173/my_graduation_yolo26n.git
cd my_graduation_yolo26n
make env-setup
```

自动完成: Conda环境创建 → PyTorch(CUDA)安装 → Ultralytics安装 → bypy配置

### Step 2 — 下载数据集

**首次使用先授权bypy:**

```bash
bypy info
# 1. 复制打开的链接到浏览器
# 2. 登录百度账号后复制授权码
# 3. 粘贴回终端
```

**手动下载数据集并解压:**

```bash
# 下载压缩包到 data/ 目录
bypy downfile /task3/VisDrone2019-MOT-train.zip ./data/
bypy downfile /task3/VisDrone2019-MOT-val.zip   ./data/
# 有测试集也一起下
bypy downfile /task3/VisDrone2019-MOT-test.zip  ./data/

# 一键解压 + 转YOLO格式 + 验证
make prepare-data
```

脚本自动完成: 检测 `.zip` → 解压 → VisDrone原始格式转YOLO格式 → 数据完整性验证

### Step 3 — 开始训练

```bash
make start-train
```

**后台训练（断开SSH不影响）:**

```bash
nohup bash scripts/start_train.sh > runs/train.log 2>&1 &

# 查看进度
tail -f runs/train.log

# 查看GPU使用
watch -n 1 nvidia-smi
```

**高级选项:**

```bash
# 使用P2模型（更好小目标检测）
bash scripts/start_train.sh --p2

# 恢复中断的训练
make start-train --resume

# 自定义参数
bash scripts/start_train.sh --batch 64 --epochs 500 --imgsz 1280
```

---

## 本地快速开始

### 1. 克隆仓库

```bash
git clone https://github.com/liu06173/my_graduation_yolo26n.git
cd my_graduation_yolo26n
```

### 2. 配置环境

```bash
make setup
```

自动完成: 创建虚拟环境 → 安装PyTorch(自动检测CUDA) → 安装Ultralytics及依赖

### 3. 准备数据集

```bash
# 手动下载VisDrone压缩包到 data/ 目录
# （百度网盘 / Google Drive / Kaggle 任选）

# 一键解压 + 清洗 + 验证
make prepare-data
```

### 4. 开始训练

```bash
make train
```

## 命令大全

### 环境与数据

| 命令 | 说明 |
|------|------|
| `make env-setup` | Cloud Studio环境配置 (conda + CUDA + bypy) |
| `make setup` | 本地环境配置 (venv) |
| `make prepare-data` | 解压 data/*.zip + 转YOLO格式 + 验证 |
| `make prepare-data-zip ZIP=xxx.zip` | 处理指定压缩包 |

### 训练控制

| 命令 | 说明 |
|------|------|
| `make train` | 新建训练 |
| `make start-train` | 一键训练 (Cloud Studio，支持 --resume --p2) |
| `make resume` | 从最近checkpoint恢复训练 |
| `make pause` | 优雅暂停 (Ctrl+C同理，自动保存last.pt) |
| `make status` | 查看训练状态 |
| `make logs` | 查看最近50行训练日志 |

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

| 命令 | 说明 |
|------|------|
| `make eval` | 评估模型mAP |
| `make infer SRC=test.jpg` | 单张图片推理 |
| `bash scripts/infer.sh video.mp4` | 视频推理 |

### 模型导出

```bash
make export          # ONNX
make export-engine  # TensorRT
```

## 项目结构

```
.
├── README.md                      # 本文档
├── Makefile                       # 命令入口
├── setup.sh                       # 本地环境配置
├── requirements.txt               # Python依赖
├── environment.yml                # Conda环境 (Cloud Studio)
├── Dockerfile                     # Docker环境
├── .gitignore
├── configs/
│   ├── visdrone.yaml              # VisDrone2019 数据配置 (10类)
│   └── hyp_visdrone.yaml          # 训练超参数 (无人机优化)
├── scripts/
│   ├── setup_env.sh               # Cloud Studio环境配置
│   ├── prepare_data.sh            # 解压zip + 转YOLO格式 + 验证
│   ├── start_train.sh             # 一键训练 (--resume / --p2)
│   ├── train.sh                   # 本地训练
│   ├── eval.sh                    # 评估
│   ├── infer.sh                   # 推理
│   └── export.sh                  # 模型导出
├── tools/
│   ├── prepare_visdrone.py        # VisDrone标注转YOLO格式
│   └── train_ctrl.py              # 训练管理 (pause/status/kill/logs)
├── docs/
│   └── help/                      # 日常问题记录
│       ├── git.md                 # Git常见问题
│       ├── env.md                 # 环境配置
│       └── tools.md               # 工具使用
├── data/
│   ├── *.zip                      # 手动下载的压缩包
│   ├── VisDrone_raw/              # 解压后的原始数据
│   └── visdrone/                  # YOLO格式数据集
│       ├── images/train/          # 训练图片
│       ├── images/val/            # 验证图片
│       ├── images/test/           # 测试图片 (可选)
│       ├── labels/train/          # 训练标注
│       ├── labels/val/            # 验证标注
│       └── labels/test/           # 测试标注 (可选)
├── ultralytics/                   # YOLO26源代码
└── runs/                          # 训练输出 (自动生成)
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

## 训练建议

### 分阶段训练

```bash
# 阶段1: 冻结骨干网络，训练检测头 (前10 epoch)
# 在 hyp_visdrone.yaml 中设置 freeze=10
bash scripts/train.sh --epochs 10

# 阶段2: 全参数微调
bash scripts/train.sh --resume --epochs 200

# 阶段3: 关闭Mosaic，精调 (最后50 epoch)
# hyp_visdrone.yaml 中 close_mosaic=10
bash scripts/train.sh --resume --epochs 250
```

### 关键参数

| 参数 | 值 | 说明 |
|------|-----|------|
| epochs | 300 | 小目标检测需更多轮次 |
| batch | 32 | 根据显存调整 (16G → 32) |
| imgsz | 640 | 保持宽高比resize |
| mosaic | 1.0 | 小目标关键增强 |
| close_mosaic | 10 | 最后10 epoch关闭 |

## 训练监控

```bash
make status              # 查看状态
make logs                # 查看日志
tensorboard --logdir runs/detect/train  # TensorBoard
```

## 常见问题

**Q: 训练中断后如何恢复？**
```bash
make resume
```

**Q: 如何更换模型尺寸？**
```bash
# 修改 scripts/train.sh: yolo26n.pt → yolo26s.pt / yolo26m.pt ...
```

**Q: 如何使用P2版本（小目标更优）？**
```bash
bash scripts/start_train.sh --p2
```

**Q: 新加了测试集zip怎么办？**
```bash
make prepare-data  # 再跑一次，自动增量处理
```

## License

本项目代码基于 AGPL-3.0 许可证。VisDrone2019数据集请遵循其原始许可协议。

---

**模型配置**: YOLO26n | **数据集**: VisDrone2019 (天津大学) | **框架**: Ultralytics 8.4+
