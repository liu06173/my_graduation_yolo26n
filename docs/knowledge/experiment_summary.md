# YOLO无人机航拍检测实验全记录

> VisDrone2019-MOT · Person + Vehicle 双类 · RTX 4060 Ti 8GB  
> 训练日期：2026-05-09 ~ 2026-05-11

---

## 一、实验环境

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA RTX 4060 Ti 8GB |
| CPU | Intel (Windows 11) |
| PyTorch | 2.5.1+cu121 |
| Ultralytics | 8.4.47 (本地fork) |
| 数据集 | VisDrone2019-MOT → YOLO 2类 |
| 训练集 | 24,198张 |
| 验证集 | 2,846张 |
| 类别 | 0: person (328K) + 1: vehicle (592K) |
| 数据盘 | D: SSD (D:/yolo26_cache/) |

---

## 二、全部实验排行

| 排名 | 模型 | 输入 | 最佳mAP@50 | 最佳轮次 | 参数量 | GFLOPs | 备注 |
|:----:|------|------|:---------:|:-------:|--------|--------|------|
| 1 | **YOLOv11n** | 640 | **0.548** | E6 | 2.59M | 6.4 | 最终模型 |
| 2 | YOLOv12n | 640 | 0.545 | E3 | 2.57M | 6.5 | A2C2F注意力 |
| 3 | YOLO26n-640 | 640 | 0.540 | E6 | 2.50M | 5.8 | 大输入有效 |
| 4 | YOLO26n-512 | 512 | 0.511 | E14 | 2.50M | 5.8 | 基准线 |
| 5 | P2-Only | 512 | 0.500 | E20 | 2.52M | 7.5 | P2头仅+2万参数 |
| 6 | P2-Tracking | 512 | 0.459 | E22 | 2.80M | 11.0 | 注意力全家桶→负优化 |
| 7 | YOLOv11n+ECA | 640 | 0.387 | E1 | 2.59M | 6.4 | ECA负优化 |

```
YOLOv11n-640          ████████████████████████████████ 0.548  ← 冠军
YOLOv12n-640          ███████████████████████████████  0.545
YOLO26n-640           █████████████████████████████    0.540
YOLO26n-512           ██████████████████████████       0.511  ← 基准
P2-Only-512           █████████████████████████        0.500
P2-Tracking-512       ██████████████████████           0.459
```

---

## 三、逐实验详情

### 3.1 基准：YOLO26n-512 (baseline)

**配置**：yolo26n.pt, imgsz=512, batch=8, SGD, cos_lr, 60轮

| Epoch | mAP@50 | mAP@50-95 | P | R | Box Loss | Cls Loss |
|-------|:------:|:---------:|:----:|:----:|:--------:|:--------:|
| 1 | 0.350 | 0.152 | 0.435 | 0.410 | 2.441 | 1.894 |
| 5 | 0.498 | 0.235 | 0.656 | 0.469 | 2.107 | 1.107 |
| 10 | 0.500 | 0.248 | 0.697 | 0.454 | 1.976 | 0.979 |
| 14 | **0.511** | 0.252 | 0.701 | 0.466 | 1.922 | 0.925 |
| 30 | 0.497 | 0.248 | 0.705 | 0.458 | 1.783 | 0.823 |

**结论**：14轮达峰值后进入平台。验证损失从第1轮就持平，说明VisDrone 2类数据信息量有限。

---

### 3.2 最优：YOLOv11n-640 (champion)

**配置**：yolo11n.pt, imgsz=640, batch=4, SGD, cos_lr, 60轮

| Epoch | mAP@50 | mAP@50-95 | P | R |
|-------|:------:|:---------:|:----:|:----:|
| 1 | **0.512** | 0.255 | 0.638 | 0.493 |
| 2 | 0.534 | 0.267 | 0.677 | 0.500 |
| 5 | 0.544 | 0.271 | 0.719 | 0.497 |
| 6 | **0.548** | 0.270 | 0.719 | 0.496 |

**关键突破**：
- Epoch 1 就超越YOLO26n-512的14轮最佳（0.512 vs 0.511）
- Recall从0.466提升到0.497（+3.1%）
- 182层精简架构，比YOLO26n少78层但效果更好
- 448/499权重成功迁移（90%）

---

### 3.3 YOLOv12n-640 (A2C2F注意力)

**配置**：yolo12n.pt, imgsz=640, batch=4, SGD, cos_lr, 60轮

| Epoch | mAP@50 | mAP@50-95 | P | R |
|-------|:------:|:---------:|:----:|:----:|
| 1 | 0.535 | 0.265 | 0.668 | 0.501 |
| 2 | 0.544 | 0.271 | 0.704 | **0.507** |
| 3 | **0.545** | 0.275 | 0.684 | 0.508 |

**关键发现**：
- Recall 0.507 是全部实验最高（A2C2F对小目标Recall有效）
- 272层，包含5个A2C2f模块，640/691权重迁移（93%）
- A2C2F = Attention-to-Context Channel Fusion（区域注意力+多头MLP）

---

### 3.4 输入尺寸消融：YOLO26n-512 vs 640

| 指标 | 512 | 640 | 提升 |
|------|:---:|:---:|:----:|
| 最佳mAP@50 | 0.511 | **0.540** | +2.9% |
| 最佳Recall | 0.466 | **0.502** | +3.6% |
| GFLOPs | 5.8 | 5.8 | 0 |
| Batch size | 8 | 4 | 减半 |

**结论**：imgsz=640是最有效的单项优化。Recall突破0.50是质的提升。

---

### 3.5 架构消融

| 实验 | 改动 | mAP@50 | vs基准 | 结论 |
|------|------|:------:|:------:|------|
| Baseline | 无 | 0.511 | — | — |
| P2-Only | +P2/4检测头 | 0.500 | -1.1% | P2头不加注意力略逊 |
| P2-Tracking | +P2+ECA+CoordAtt+DySample+WeightedConcat | 0.459 | -5.2% | 注意力全家桶→严重负优化 |
| YOLOv11n+ECA | YOLOv11+ECA | 0.387 | — | ECA破坏YOLOv11权重 |

**核心教训**：在2类小数据上，额外注意力模块破坏预训练权重分布，导致性能下降。YOLOv12的A2C2F成功在于它是**原生模块**（预训练权重包含），而非事后添加。

---

## 四、关键经验

### 4.1 有效优化
1. **imgsz 512→640**：mAP +2.9%，Recall +3.6%，零成本
2. **换架构**：YOLOv11n/v12n 比 YOLO26n 好3-5%
3. **原生模块 > 外挂模块**：A2C2F（YOLOv12原生）有效，ECA/P2（外挂）无效

### 4.2 无效/负优化
1. ECA通道注意力 → 破坏预训练权重
2. CoordAtt坐标注意力 → 未见提升
3. DySample动态上采样 → 未见提升
4. WeightedConcat加权融合 → 未见提升
5. P2检测头（单独使用）→ 略逊于基准

### 4.3 训练稳定性
- 所有模型在10-20轮后进入平台（mAP不再上升）
- VisDrone 2类数据信息上限约 mAP@50=0.55
- 训练进程频繁崩溃（Windows/PyTorch稳定性），但checkpoint可恢复
- D盘SSD缓存显著加速数据加载

---

## 五、最终模型

**YOLOv11n-640**：mAP@50=0.548，2.59M参数，6.4GFLOPs

```
权重路径: D:/yolo26_cache/runs/detect/train_yolo11n/weights/best.pt
网页平台: webapp/models/yolov26n_baseline_2cls/best.pt
```

后续待定：YOLOv12n+DecoupledDetect（训练中，302层/2.66M参数）

---

## 六、产出清单

| 产出 | 路径 |
|------|------|
| 数据集 | D:/yolo26_cache/data/visdrone/ |
| 训练输出 | D:/yolo26_cache/runs/detect/ |
| 检测平台 | `webapp/` (http://localhost:5800) |
| 知识库 | `docs/knowledge/` |
| 学术论文 | `docs/knowledge/academic_paper.pdf` |
| 答辩PPT | `docs/knowledge/graduation_defense.pptx` |
| 训练指标详解 | `docs/knowledge/training_metrics.md` |
| P2架构文档 | `docs/knowledge/yolo26_p2_architecture.html` |
| YOLOv12论文总结 | `docs/knowledge/yolov12_paper.html` |
| YOLOv12原始论文 | `docs/knowledge/YOLOv12改进算法.pdf` |
| 模型权重 | `webapp/models/yolov26n_baseline_2cls/best.pt` |
| MOT转换脚本 | `scripts/convert_mot2yolo.py` |
| 训练脚本 | `scripts/train_2cls.py` |

---

## 七、论文参考

| 论文 | 模块 | 会议 |
|------|------|------|
| ECA-Net | ECA通道注意力 | ECCV 2020 |
| Coordinate Attention | CoordAtt | CVPR 2021 |
| DySample | 动态上采样 | ICCV 2023 |
| EfficientDet | BiFPN/WeightedConcat | CVPR 2020 |
| YOLOv12 | C3K2 + A2C2F | Scientific Reports 2025 |
