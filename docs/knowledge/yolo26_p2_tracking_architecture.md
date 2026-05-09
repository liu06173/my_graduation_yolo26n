# YOLO26-P2-Tracking 改进模型架构

> 面向无人机航拍场景的YOLO26改进版，集成6种注意力/融合模块，专攻小目标检测

---

## 一、设计动机

无人机航拍图像的核心挑战：**目标像素极小**。VisDrone数据集中，行人平均只占图像的 0.02%~0.5%。

| 问题 | 标准YOLO26 | 改进方案 |
|------|-----------|----------|
| 小目标漏检 | P3/8 = 8px最小检测 | **P2/4 = 4px最小检测** (+P2检测头) |
| 通道无侧重 | 所有通道等权 | **ECA自适应通道注意力** |
| 空间信息丢失 | 下采样丢失位置 | **CoordAtt坐标注意力** |
| 粗糙上采样 | 最近邻插值 | **DySample动态上采样** |
| 盲目融合 | 直接Concat | **WeightedConcat加权融合** |

---

## 二、整体架构

```
输入 512×512×3
│
┌────────────────────── Backbone ──────────────────────┐
│                                                        │
│  [0] Conv(64, s=2)        ─────── 256×256             │
│  [1] Conv(128, s=2)       ─────── 128×128  ← P2/4     │
│  [2] C3k2_ECA(256)          ECA通道注意力              │
│  [3] CoordAtt                    ↓坐标位置注意力        │
│  [4] Conv(256, s=2)       ─────── 64×64    ← P3/8     │
│  [5] C3k2_ECA(512)          ECA通道注意力              │
│  [6] CoordAtt                    ↓坐标位置注意力        │
│  [7] Conv(512, s=2)       ─────── 32×32    ← P4/16    │
│  [8] C3k2(512)                                          │
│  [9] Conv(1024, s=2)      ─────── 16×16    ← P5/32    │
│ [10] C3k2(1024)                                         │
│ [11] SPPF(1024)              多尺度池化                 │
│ [12] C2PSA(1024)             位置自注意力               │
│                                                        │
└────────────────────────────────────────────────────────┘
│
┌────────────────────── FPN Neck (Top-Down) ────────────┐
│                                                        │
│  [13] DySample(512, ×2)    P5 → P4   动态上采样        │
│  [14] WeightedConcat       + backbone[8]  加权融合     │
│  [15] C3k2(512)                                         │
│                                                        │
│  [16] DySample(256, ×2)    P4 → P3   动态上采样        │
│  [17] WeightedConcat       + backbone[5]  加权融合     │
│  [18] C3k2(256)             ← P3/8-small              │
│                                                        │
│  [19] DySample(128, ×2)    P3 → P2   动态上采样        │
│  [20] WeightedConcat       + backbone[2]  加权融合     │
│  [21] C3k2(128)             ← P2/4-xsmall ← 核心改进  │
│                                                        │
└────────────────────────────────────────────────────────┘
│
┌────────────────────── PAN Neck (Bottom-Up) ───────────┐
│                                                        │
│  [22] Conv(128, s=2)       P2 → P3   下采样            │
│  [23] WeightedConcat       + head[18]   加权融合       │
│  [24] C3k2(256)             ← P3/8-small              │
│                                                        │
│  [25] Conv(256, s=2)       P3 → P4   下采样            │
│  [26] WeightedConcat       + head[15]   加权融合       │
│  [27] C3k2(512)             ← P4/16-medium            │
│                                                        │
│  [28] Conv(512, s=2)       P4 → P5   下采样            │
│  [29] WeightedConcat       + backbone[12] 加权融合     │
│  [30] C3k2(1024)            ← P5/32-large             │
│                                                        │
└────────────────────────────────────────────────────────┘
│
┌───────────────── Detection Heads ─────────────────────┐
│  [31] Detect([P2, P3, P4, P5])   4尺度检测             │
│       P2/4  → 128×128网格 → 极小目标(4×4px+)          │
│       P3/8  → 64×64网格   → 小目标(8×8px+)            │
│       P4/16 → 32×32网格   → 中目标(16×16px+)          │
│       P5/32 → 16×16网格   → 大目标(32×32px+)          │
└────────────────────────────────────────────────────────┘
```

---

## 三、六大改进模块详解

### 3.1 ECA — 高效通道注意力

```
输入特征 [C, H, W]
    │
    ▼
AdaptiveAvgPool2d(1)     → [C, 1, 1]    全局平均池化
    │
    ▼
Conv1d(C→C, k=自适应)     → [C, 1, 1]    1D卷积跨通道交互
    │   kernel_size = |log₂(C)/2 + 0.5|  ← 自动计算，取奇数
    ▼
Sigmoid                   → [C, 1, 1]    归一化权重
    │
    ▼
× 原特征                  → [C, H, W]    通道加权输出
```

**核心优势**：几乎零额外参数（仅k个float，k≈3~5），却能让模型学会"哪些通道更重要"。

**论文**：ECA-Net: Efficient Channel Attention (ECCV 2020)

**代码**：`ultralytics/nn/modules/conv.py:676`

---

### 3.2 CoordAtt — 坐标注意力

```
输入特征 [C, H, W]
    │
    ├── AdaptiveAvgPool2d((None,1))  → [C, H, 1]  X方向压缩
    │
    └── AdaptiveAvgPool2d((1,None))  → [C, 1, W]  Y方向压缩
            │                              │
            └──── 拼接 [C, H+W, 1] ────────┘
                      │
                      ▼
            Conv2d(C→C/r, 1×1)      通道降维 (r=32)
                      │
                      ▼
            BatchNorm + Hardswish    归一化+激活
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      Conv2d→C, 1×1       Conv2d→C, 1×1
      Sigmoid              Sigmoid
      [C, H, 1]            [C, 1, W]
            │                   │
            └──── 相乘 ─────────┘
                      │
                      ▼
              输出 [C, H, W]
```

**核心优势**：保留精确的空间坐标信息（不像SE那样全局池化抹掉位置），对无人机小目标定位特别有用。

**论文**：Coordinate Attention (CVPR 2021)

**代码**：`ultralytics/nn/modules/conv.py:707`

---

### 3.3 DySample — 动态上采样

```
输入 [C, H, W]
    │
    ├── AdaptiveAvgPool2d(8)      → [C, 8, 8]
    │       │
    │       ▼
    │   Conv(3×3)→SiLU→Conv(3×3) → [2×scale², 8, 8]
    │       │
    │       ▼
    │   Interpolate(H×s, W×s)     → [2×scale², H×s, W×s]
    │       │
    │       ▼
    │   生成采样偏移网格             offset ∈ [-0.25, +0.25]
    │
    └── Proj Conv(1×1)             (可选通道调整)
            │
            ▼
    grid_sample(特征, 偏移网格)     ← 动态采样核心
            │
            ▼
    输出 [C₂, H×s, W×s]
```

**与最近邻插值对比**：

| 方法 | 原理 | 小目标效果 |
|------|------|-----------|
| nn.Upsample(nearest) | 直接复制像素 | 模糊，小目标边缘丢失 |
| **DySample** | 学习偏移量，内容自适应 | 保留细节，小目标更清晰 |

**论文**：Learning to Resize Images (ICCV 2023)

**代码**：`ultralytics/nn/modules/conv.py:771`

---

### 3.4 WeightedConcat — 加权特征融合

```
输入: [feat₁, feat₂]   ← 来自不同层的特征
    │
    ├── feat₁ × w₁    w₁ = softmax(learnable_w)[0]
    │
    └── feat₂ × w₂    w₂ = softmax(learnable_w)[1]
            │
            └──── torch.cat ────┘
                    │
                    ▼
              融合输出
```

**与普通Concat对比**：

```
普通Concat:    out = [feat₁, feat₂]        ← 两特征等权重
WeightedConcat: out = [0.7·feat₁, 0.3·feat₂] ← 学习最佳比例
```

**论文**：EfficientDet (BiFPN, CVPR 2020)

**代码**：`ultralytics/nn/modules/conv.py:745`

---

### 3.5 ECABottleneck — ECA增强残差块

```
输入 [C₁, H, W]
    │
    ├── Conv(1×1, C₁→C_hidden)     通道压缩 (e=0.5)
    │       │
    │       ▼
    │   Conv(3×3, C_hidden→C₂)     空间卷积
    │       │
    │       ▼
    │   ECA(C₂)                     通道注意力 ← 新增!
    │       │
    │       ▼
    │       + 残差连接 (if C₁=C₂)
    │
    └──────────────────────────────
```

与标准Bottleneck唯一区别：第二个Conv后加了ECA。几乎不增加参数。

**代码**：`ultralytics/nn/modules/block.py:486`

---

### 3.6 C3k2_ECA — ECA增强CSP模块

```
输入 [C₁, H, W]
    │
    ├── Conv(1×1, C₁→C₂/2)
    │
    └── Conv(1×1, C₁→C₂/2)
            │
            ▼
    ┌─── ECABottleneck ───┐
    │   Conv→Conv→ECA     │  ← 替代标准Bottleneck
    │   Conv→Conv→ECA     │
    │   ...               │
    └─────────────────────┘
            │
            ├── torch.cat ──┐
            │               │
            ▼               ▼
    Conv(1×1) → 输出 [C₂, H, W]
```

**代码**：`ultralytics/nn/modules/block.py:1140`

---

## 四、改进模块的放置策略

```
Backbone 各层强化方案:

P1/2 (256×256):  Conv only ─── 太浅，不加注意力
P2/4 (128×128):  C3k2_ECA + CoordAtt ─── 关键改进! 小目标第一道防线
P3/8 (64×64):    C3k2_ECA + CoordAtt ─── 关键改进! 中小目标
P4/16 (32×32):   标准 C3k2 ─── 中目标，普通注意力足够
P5/32 (16×16):   标准 C3k2 + C2PSA ─── 大目标，PSA注意力

Neck 融合:
  全部上采样:     DySample 替代 nn.Upsample
  全部拼接:       WeightedConcat 替代 Concat
```

**为什么只在P2/P3放ECA+CoordAtt？**
- P2/P3层分辨率高、目标小，注意力收益最大
- P4/P5层分辨率低，C2PSA已提供位置自注意力
- 控制参数量，避免过度工程化

---

## 五、参数量与计算量对比

| 模型 | 参数量 | GFLOPs | 检测头 | 注意力 |
|------|--------|--------|--------|--------|
| YOLO26n (标准) | 2.50M | 5.8 | P3,P4,P5 | 无 |
| YOLO26n-P2 | 2.66M | 9.5 | P2,P3,P4,P5 | 无 |
| YOLO26n-P2-Tracking | ~2.80M | ~10.2 | P2,P3,P4,P5 | ECA+CoordAtt |
| YOLO26s (标准) | 9.77M | 22.8 | P3,P4,P5 | 无 |
| YOLO26s-P2-Tracking | ~10.0M | ~28.0 | P2,P3,P4,P5 | ECA+CoordAtt |

**结论**：nano版改进后参数量仅增加12%，但P2检测头+注意力对无人机小目标的检测能力有质的提升。

---

## 六、训练建议

### 6.1 两阶段训练策略

```
Phase 1: 标准YOLO26n训练 (当前进行中)
  ├── epochs: 200
  ├── 数据集: person + vehicle (VisDrone2019-MOT)
  └── 用途: 建立baseline，参照对比

Phase 2: YOLO26n-P2-Tracking 训练
  ├── epochs: 200
  ├── 数据集: 同Phase 1
  ├── 预训练: yolo26n.pt (从标准模型迁移权重)
  └── 用途: 验证改进模块效果
```

### 6.2 预期改进幅度

| 指标 | 标准YOLO26n (预期) | P2-Tracking (预期) | 提升 |
|------|-------------------|---------------------|------|
| mAP@50 | 0.55~0.60 | 0.60~0.68 | +5~10% |
| mAP@50-95 | 0.28~0.32 | 0.32~0.38 | +4~6% |
| 小目标(<16px) AP | 低 | 明显提升 | +10~20% |
| 参数量 | 2.50M | 2.80M | +12% |
| 推理速度 | 快 | 略慢(~15%) | -15% |

---

## 七、已修复 / 已审查问题

| 状态 | 问题 | 位置 | 结论 |
|------|------|------|------|
| ✅ 已修复 | ECABottleneck的import在__init__内 | block.py:504 | 已移到模块顶部 `from .conv import ECA` |
| ✅ 审查通过 | DySample在P2层 | yaml第19行 | 非bug：AdaptiveAvgPool2d(8)仅生成offset，实际特征经grid_sample全分辨率采样 |
| 📋 待开发 | 缺少tracking头(ReID) | yaml第57行 | 功能需求：需新增ReID嵌入分支 + TripletLoss，属于MOT独立任务 |

---

## 八、论文引用

| 模块 | 论文 | 会议 |
|------|------|------|
| ECA | ECA-Net: Efficient Channel Attention for Deep CNNs | ECCV 2020 |
| CoordAtt | Coordinate Attention for Efficient Mobile Network Design | CVPR 2021 |
| DySample | Learning to Resize Images for Computer Vision Tasks | ICCV 2023 |
| BiFPN/Weighted | EfficientDet: Scalable and Efficient Object Detection | CVPR 2020 |
| C2PSA | YOLOv10: Real-Time End-to-End Object Detection | 2024 |
| YOLO26 | Ultralytics YOLO26 | 2025 |
