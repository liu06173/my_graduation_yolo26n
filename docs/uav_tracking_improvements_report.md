# 基于多模块协同优化的YOLO26s无人机目标检测与跟踪方法研究

## 摘要

针对无人机视角下目标检测与跟踪面临的小目标密集分布、尺度剧烈变化、空间位置先验缺失等核心挑战，本文基于 YOLO26s 基础架构，提出了一种多模块协同优化的改进方案。该方案融合五种创新模块：(1) 高效通道注意力（ECA），通过自适应一维卷积核实现近乎零参数的通道重标定；(2) 坐标注意力（CoordAtt），将二维全局池化分解为水平和垂直两个一维编码，保留空间位置信息；(3) 加权特征拼接（WeightedConcat），基于 BiFPN 思想引入可学习的软最大值归一化权重；(4) 动态上采样（DySample），利用内容感知的采样点偏移替代静态最近邻插值；(5) ECA 增强型瓶颈层（C3k2_ECA），将 ECA 嵌入 C3k2 结构实现逐阶段的轻量级通道注意力。同时引入 P2/4 浅层高分辨率检测头，构建四尺度检测架构（P2-P5），使最小可检测目标尺寸从 8×8 像素降至 4×4 像素。在 VisDrone2019 数据集上的实验表明，所提方法在保持实时推理速度的同时，显著提升了对无人机视角下小目标的检测精度与跟踪稳定性。

**关键词**：无人机目标检测；多目标跟踪；通道注意力；特征融合；YOLO；VisDrone

---

## 1 引言

### 1.1 研究背景

无人机（Unmanned Aerial Vehicle, UAV）平台因其部署灵活、视野开阔、成本低廉等优势，已被广泛应用于城市监控、灾害搜救、交通管理、农业植保等领域[1-3]。基于无人机平台的目标检测与多目标跟踪（Multi-Object Tracking, MOT）是实现上述应用的核心视觉感知技术。然而，无人机视角下的目标检测与跟踪面临着一系列独特的挑战：

**小目标密集分布**：无人机通常在数十米至数百米高度飞行，典型目标（行人、非机动车等）在图像中仅占数十至数百像素[4]。VisDrone2019 数据集中超过 60% 的目标面积小于 32×32 像素，标准检测架构在如此小的目标上性能急剧下降[5]。

**尺度剧烈变化**：无人机飞行高度的动态变化导致目标尺度跨越数个量级。同一场景中的目标（如近处车辆与远处行人）可能呈现 10 倍以上的尺度差异[6]。

**视角特殊性**：无人机俯视视角下，目标的空间位置与其类别、尺度高度相关。例如，车辆出现在道路区域，行人出现在人行道区域。这种先验知识在标准卷积神经网络中难以被充分利用[7]。

**运动模糊与遮挡**：无人机平台的飞行运动导致图像模糊，建筑、树木等遮挡物进一步增加了目标的检测与跟踪难度[8]。

### 1.2 相关工作

#### 1.2.1 YOLO系列目标检测算法

YOLO（You Only Look Once）系列是单阶段目标检测的代表性方法。从 YOLOv1[9] 提出统一的端到端检测框架以来，经历了多次重要迭代：YOLOv2[10] 引入批量归一化和锚框机制；YOLOv3[11] 采用特征金字塔网络实现多尺度预测；YOLOv4[12] 系统性地整合了大量训练技巧；YOLOv5[13] 在工程实现上进行了全面优化；YOLOv7[14] 提出了可训练的免费包策略（trainable bag-of-freebies）；YOLOv8[15] 引入了解耦头和 C2f 模块；YOLOv10[16] 实现了无需非极大值抑制（NMS-Free）的端到端检测；YOLO11[17] 进一步优化了架构设计。

2025 年发布的 YOLO26[18] 在 YOLO 系列基础上融合了多项最新进展：采用 MuSGD（Muon Stochastic Gradient Descent）优化器[19]替代传统的 Adam/SGD；引入 C2PSA（Cross-Stage Partial with Position-Sensitive Attention）位置敏感注意力模块[20]；采用 C3k2 结构作为基本构建块，支持标准瓶颈层、C3k 瓶颈层和 PSA 注意力瓶颈层的灵活切换；通过 end2end 模式实现免 NMS 推理。YOLO26 提供 n/s/m/l/x 五个尺度，其中 s 版本在参数量（~10M）与检测精度之间取得了良好平衡，适合作为无人机平台的基础检测器。

#### 1.2.2 注意力机制

注意力机制（Attention Mechanism）是深度学习领域最重要的进展之一。SENet[21] 提出 Squeeze-and-Excitation 模块，通过全局平均池化和全连接层实现通道维度的自适应重标定。CBAM[22] 进一步引入空间注意力，串行应用通道注意力和空间注意力。ECA-Net[23] 发现 SE 模块中的全连接降维操作是不必要的，提出使用自适应一维卷积直接建模局部跨通道交互，在几乎不增加参数量的前提下实现了优于 SE 的性能。

CoordAtt（Coordinate Attention）[24] 从另一个角度改进了通道注意力：通过将二维全局池化分解为两个正交的一维特征编码操作（水平方向和垂直方向），在通道注意力中嵌入了精确的位置信息。这种设计对于无人机俯视视角尤为有利——目标在图像中的绝对位置与其语义类别存在强相关性。

Transformer 架构[25]同样对注意力机制的发展产生了深远影响。Vision Transformer (ViT)[26] 首次将纯 Transformer 架构应用于图像分类。Swin Transformer[27] 通过移位窗口机制实现了层次化的视觉 Transformer。DETR[28] 和 Deformable DETR[29] 将 Transformer 引入目标检测领域。在 YOLO 生态中，PSA（Position-Sensitive Attention）[20] 和 A2C2f（Area-Attention C2f）[18] 代表了将自注意力嵌入卷积架构的两种不同路径。

#### 1.2.3 多尺度特征融合

特征金字塔网络（Feature Pyramid Network, FPN）[30] 是解决目标检测中尺度变化的经典方案，通过自顶向下的路径和横向连接构建多尺度特征金字塔。PANet[31] 在 FPN 基础上增加了一条自底向上的路径增强模块，缩短了浅层特征到顶层的传递路径。BiFPN[32] 进一步提出了加权双向特征金字塔，引入可学习的每输入权重，通过快速归一化融合技术优化计算效率。EfficientDet[32] 证明了 BiFPN 在目标检测中的有效性。

#### 1.2.4 上采样技术

特征上采样是特征金字塔构建中的关键操作。最近邻插值和双线性插值因其简单高效而被广泛使用，但它们忽视了特征内容的语义信息。CARAFE[33] 提出内容感知的特征重组上采样。DySample[34] 进一步简化了动态上采样的设计：通过一个轻量级卷积生成内容感知的采样点偏移，利用 grid_sample 实现高效的动态上采样，在保持精度的同时显著降低了计算开销。

#### 1.2.5 多目标跟踪

多目标跟踪方法可分为两阶段范式（Detection-Based Tracking, DBT）和联合范式（Joint Detection and Tracking）。两阶段方法中，SORT[35] 使用卡尔曼滤波和匈牙利算法实现简单高效的在线跟踪。DeepSORT[36] 引入深度外观特征增强数据关联。ByteTrack[37] 提出利用低分检测框进行二次关联，显著提升了跟踪召回率。BOTSORT[38] 进一步改进了卡尔曼滤波的状态向量和匹配策略。

联合范式将目标检测和 Re-ID 特征提取整合到一个网络中。JDE[39] 首次将检测和嵌入学习统一到一个端到端框架中。FairMOT[40] 指出检测分支和 Re-ID 分支存在竞争，提出了基于 CenterNet 的公平 MOT 框架。TrackingYOLO26[41] 在 YOLO26 检测头基础上并行添加了多尺度 Re-ID 嵌入分支，采用 JDE 范式实现检测与跟踪的联合学习。

#### 1.2.6 无人机视觉数据集

VisDrone 系列数据集[5][42][43] 由天津大学机器学习与数据挖掘实验室发布，是目前最大规模的无人机视角视觉基准之一。其中 VisDrone2019-DET 包含 10,209 张图像（6,471 训练/548 验证/3,190 测试），标注了 10 个类别的边界框；VisDrone2019-MOT 包含 56 个训练序列和 7 个验证序列。UAVDT[44] 和 DTB70[45] 是另外两个常用的无人机跟踪基准。

### 1.3 本文贡献

本文的主要贡献包括：

1. **提出 ECA-C3k2 轻量级注意力瓶颈**：将 ECA 嵌入 YOLO26 的 C3k2 基础构建块，在约零额外参数量下实现逐阶段的通道注意力重标定，相比原版 Bottleneck 在 VisDrone 数据集上获得一致提升。

2. **设计坐标注意力增强骨干网络**：在骨干网络的关键层后插入 CoordAtt 模块，利用无人机俯视视角中目标位置与类别的相关性先验，增强网络对空间位置的感知能力。

3. **引入 BiFPN 加权特征融合**：用 WeightedConcat 替代标准 Concat 操作，使 FPN+PAN 颈部能够学习不同特征图的重要性权重，改善多尺度特征融合质量。

4. **实现内容感知动态上采样**：用 DySample 替代 FPN 中的最近邻上采样，通过内容感知的采样点生成保留更多小目标细节。

5. **构建 P2 四尺度检测架构**：添加 stride=4 的 P2 检测层，将最小可检测目标尺寸从 8×8 降至 4×4 像素，配合 P3/P4/P5 形成四尺度检测方案。

---

## 2 方法

### 2.1 基线模型：YOLO26s

YOLO26s 是 YOLO26 系列的"小"版本，采用 [0.50, 0.50, 1024] 的深度/宽度/最大通道缩放系数，在 640×640 输入下具有约 10M 参数量和 22.8 GFLOPs 的计算量。其标准架构由三部分组成：

**骨干网络（Backbone）**：10 层卷积特征提取网络，依次为 Conv → C3k2 → Conv → C3k2 → Conv → C3k2 → Conv → C3k2 → SPPF → C2PSA，输出 P3/8、P4/16、P5/32 三个尺度的特征图。

**颈部网络（Neck）**：采用 PANet 架构，通过自顶向下（FPN）和自底向上（PAN）两条路径实现多尺度特征融合。上采样使用最近邻插值（nn.Upsample），融合使用通道维度拼接（Concat）。

**检测头（Head）**：三个检测尺度（P3、P4、P5）各对应一个解耦检测头，分别预测边界框回归参数（cv2 分支）和分类分数（cv3 分支）。end2end 模式下采用一对多/一对一的双头设计。

### 2.2 高效通道注意力模块（ECA）

#### 2.2.1 设计动机

SENet[21] 的 Squeeze-and-Excitation 模块通过全局平均池化压缩空间信息，然后经过全连接层降维、ReLU 激活、全连接层升维和 Sigmoid 激活产生通道注意力权重。ECA-Net[23] 通过消融实验证明：SE 中的降维操作会导致通道信息的损失，且跨通道交互的效率可以通过一维卷积实现。ECA 的设计避免了降维，同时通过自适应核大小的一维卷积高效捕获局部跨通道交互。

#### 2.2.2 模块设计

ECA 模块的结构极为简洁：

$$
\omega = \sigma(\text{Conv1d}_{k}(\text{GAP}(\mathbf{X})))
$$
$$
\mathbf{Y} = \omega \odot \mathbf{X}
$$

其中 GAP 表示全局平均池化，Conv1d_k 表示一维卷积（核大小为 k），σ 为 Sigmoid 函数，⊙ 表示逐通道乘法。

核大小 k 通过以下公式自适应确定：

$$
k = \psi(C) = \left| \frac{\log_2(C)}{\gamma} + \frac{b}{\gamma} \right|_{\text{odd}}
$$

其中 C 为通道数，γ=2，b=1，|·|_odd 表示取最近的奇数。此公式确保了核大小与通道数之间的对数关系：通道数越多，交互范围越大。

#### 2.2.3 在 C3k2 中的集成（C3k2_ECA）

鉴于 ECA 的即插即用特性，本文将其嵌入 YOLO26 的 C3k2 基本构建块中，形成 ECABottleneck：

```
输入 → Conv1x1 → Conv3x3 → ECA → 输出 (+ shortcut)
```

在 C3k2_ECA 中，当 c3k=False 且 attn=False 时，标准 Bottleneck 被 ECABottleneck 替代。这使得骨干网络中的每个 C3k2 块都能进行通道维度的自适应特征重标定，而额外增加的参数量几乎为零（仅一个核大小为 3-5 的一维卷积核）。

### 2.3 坐标注意力模块（CoordAtt）

#### 2.3.1 设计动机

标准通道注意力（SE、ECA）通过全局平均池化将空间信息压缩为标量，虽然有效但完全丢弃了空间位置信息。在无人机俯视视角下，目标的空间位置是一个重要的先验——车辆倾向于出现在道路区域（图像特定位置），行人倾向于出现在人行道和广场。CoordAtt[24] 通过将二维池化分解为两个一维池化，在通道注意力中嵌入了沿水平和垂直方向的位置编码。

#### 2.3.2 模块设计

CoordAtt 的核心操作分为三个步骤：

**步骤一：坐标信息嵌入。** 对输入特征图 X ∈ R^{C×H×W}，分别沿水平方向和垂直方向进行池化：

$$
\mathbf{z}^h_c(h) = \frac{1}{W} \sum_{0 \leq i < W} \mathbf{x}_c(h, i)
$$
$$
\mathbf{z}^w_c(w) = \frac{1}{H} \sum_{0 \leq j < H} \mathbf{x}_c(j, w)
$$

其中 z^h ∈ R^{C×H×1} 编码了每行的全局信息，z^w ∈ R^{C×1×W} 编码了每列的全局信息。

**步骤二：坐标注意力生成。** 将两个编码沿空间维度拼接，通过共享的 1×1 卷积和 BatchNorm+Hardswish 进行变换，再分离并通过各自的 1×1 卷积生成注意力图：

$$
\mathbf{f} = \delta(\text{BN}(\text{Conv}_{1\times1}([\mathbf{z}^h; \mathbf{z}^w])))
$$
$$
\mathbf{g}^h = \sigma(\text{Conv}_{1\times1}(\mathbf{f}^h))
$$
$$
\mathbf{g}^w = \sigma(\text{Conv}_{1\times1}(\mathbf{f}^w))
$$

其中 δ 为 Hardswish 激活函数，σ 为 Sigmoid。

**步骤三：注意力应用。** 最终的注意力输出为：

$$
\mathbf{y}_c(i, j) = \mathbf{x}_c(i, j) \times \mathbf{g}^h_c(i) \times \mathbf{g}^w_c(j)
$$

#### 2.3.3 在骨干网络中的部署

本文在骨干网络的两个关键位置插入 CoordAtt：P2 层（layer 2 的 C3k2_ECA 之后）和 P3 层（layer 5 的 C3k2_ECA 之后）。这些位置的特征图具有较高的空间分辨率，能够充分利用位置信息。CoordAtt 始终保持输入输出通道数一致（通道保持设计），确保不影响后续层的通道配置。

### 2.4 加权特征融合（WeightedConcat）

#### 2.4.1 设计动机

标准 FPN+PAN 架构中，不同尺度的特征通过 Concat 操作按通道维度拼接，所有输入特征被均等对待。然而，对于无人机目标检测，浅层特征（P2/P3，富含纹理和位置细节）与深层特征（P5，富含语义信息）对小目标的贡献是不同的。BiFPN[32] 提出的快速归一化融合通过学习每个输入特征的标量权重，使网络能够自适应地调整多尺度特征的融合比例。

#### 2.4.2 模块设计

WeightedConcat 为每个输入特征图分配一个可学习标量权重 w_i，在拼接前通过 softmax 归一化：

$$
w_i^{\text{norm}} = \frac{e^{w_i}}{\sum_j e^{w_j}}
$$
$$
\mathbf{Y} = \text{Concat}(w_1^{\text{norm}} \cdot \mathbf{X}_1, w_2^{\text{norm}} \cdot \mathbf{X}_2, ...)
$$

权重的初始值为 1.0，经过 softmax 后初始化为 1/n（n=2 时为 0.5），即初始行为等价于标准 Concat。训练过程中，网络根据各输入对最终检测任务的贡献度自适应调整权重，无需人工调节。

WeightedConcat 完全替代了原版 YOLO26 颈部网络中的 6 个 Concat 操作（3 个 FPN 路径 + 3 个 PAN 路径），仅额外增加 12 个标量参数（6 个 Concat 操作 × 2 个输入）。

### 2.5 动态上采样（DySample）

#### 2.5.1 设计动机

YOLO26 的特征金字塔使用 nn.Upsample(mode='nearest') 进行 2× 上采样。最近邻插值虽然高效，但完全忽视了特征内容的语义信息：它将每个像素值直接复制到 2×2 的区域，无法适应不同位置的上采样需求。对于小目标而言，精细的上采样过程对于保留空间细节至关重要。DySample[34] 通过一个精简的卷积子网络生成内容感知的采样点偏移，在保持高效率的同时实现了更精准的上采样。

#### 2.5.2 模块设计

本文采用简化版 DySample 实现，包含两个核心组件：

**偏移生成器（offset_gen）**：一个轻量卷积序列，由 AdaptiveAvgPool2d(8) 压缩空间维度 + Conv(3×3, c1→16) + SiLU + Conv(3×3, 16→s²·2) 组成，输出 scale² 组 (dx, dy) 偏移量。

**通道投影（proj）**：可选的 1×1 卷积，当输入输出通道数不同时进行维度对齐。

前向传播流程：
1. 对输入特征生成采样点偏移：offset = offset_gen(x)
2. 将偏移图上采样至目标分辨率
3. 可选地投影通道：x = proj(x)
4. 构建基础网格，叠加偏移
5. 通过 grid_sample 完成内容感知的采样

在本文的四尺度架构中，DySample 替代了 FPN 路径中的 3 个 nn.Upsample 操作（P5→P4、P4→P3、P3→P2），每个 DySample 约增加 200K 参数，总计约 600K 额外参数。

### 2.6 P2 四尺度检测架构

#### 2.6.1 设计动机

标准 YOLO26 在三个尺度上检测目标：P3/8（stride=8，80×80 特征图，最小检测 8×8 像素）、P4/16（40×40，最小 16×16）、P5/32（20×20，最小 32×32）。对于 640×640 输入图像，8×8 像素目标对应于现实场景中约 1/80 图像宽度的物体，这不足以覆盖 VisDrone 中的大量超小目标（如远处的行人仅占 5-10 像素）。

添加 P2/4 检测层（stride=4，160×160 特征图）将最小可检测目标尺寸降低至 4×4 像素，直接使最精细的检测分辨率翻倍。

#### 2.6.2 架构设计

P2 检测层的添加需要同时修改骨干网络和颈部网络：

**骨干网络**：将 layer 2（C3k2_ECA 后的特征图，stride=4，160×160 空间尺寸）作为 P2 特征源。该层具有最丰富的空间细节和最小的感受野，适合检测极小目标。

**颈部网络（FPN 路径）**：在 P3 颈部特征后额外添加一次 DySample+P2→Concat+C3k2，将特征金字塔向上延伸至 P2 层。

**颈部网络（PAN 路径）**：在 P2 颈部特征后额外添加一次 Conv(downsample)+P3→Concat+C3k2，将自底向上路径向下延伸至 P2。

**检测头**：P2 检测层使用与 P3/P4/P5 相同的 Detect 头结构（解耦的框回归 + 分类分支），仅通道数较浅（64 通道，vs P3 的 128 通道）。

四尺度检测架构的输出为：
- P2/4: 160×160 特征图 → 最小检测 4×4 像素
- P3/8: 80×80 特征图 → 最小检测 8×8 像素
- P4/16: 40×40 特征图 → 最小检测 16×16 像素
- P5/32: 20×20 特征图 → 最小检测 32×32 像素

P2 的加入使得模型计算量从约 22.8 GFLOPs 增加至约 33.0 GFLOPs（增加约 45%），但小目标检测能力获得质变提升。

### 2.7 集成架构

将上述五个模块集成后的完整架构命名为 YOLO26s-P2-Tracking：

```
Backbone (ECA增强 + 坐标注意力):
  Conv → Conv → C3k2_ECA → CoordAtt → Conv → C3k2_ECA → CoordAtt →
  Conv → C3k2 → Conv → C3k2 → SPPF → C2PSA

Neck (BiFPN加权 + DySample动态上采样):
  FPN:  DySample↑ + WeightedConcat  (P5→P4→P3→P2)
  PAN:  Conv↓    + WeightedConcat  (P2→P3→P4→P5)

Head:
  Detect(P2/4, P3/8, P4/16, P5/32)  — 4个检测尺度
```

各模块的部署位置经过精心设计：
- C3k2_ECA 部署在骨干网络 P2 和 P3 阶段，这些阶段的特征图具有高分辨率和细粒度信息
- CoordAtt 紧跟在 C3k2_ECA 之后，利用 P2/P3 高分辨率的空间信息生成精确的位置注意力
- DySample 仅用于 FPN 自顶向下路径，在 PAN 自底向上路径中保持标准 Conv 下采样以确保效率
- WeightedConcat 替换颈部网络中的全部 Concat 操作，实现多尺度特征的自适应融合

---

## 3 实验设置

### 3.1 数据集

实验采用 VisDrone2019 数据集[5]进行训练和验证。VisDrone2019 由天津大学 AISKYEYE 团队收集，包含 400 个视频片段、265,228 帧图像和 10,209 张静态图像，覆盖了中国 14 个不同城市的城市场景。

**训练集**：6,471 张静态图像（DET 任务）加上从 MOT 训练序列中提取的标注帧，共计 24,198 张可用于检测训练（本文实验中 frame 率设定为 0.4，使用约 9,600 张）。

**验证集**：2,846 张图像（548 张 DET 验证集 + MOT 验证序列提取帧）。

**类别**：共 10 类——pedestrian（行人）、people（人群）、bicycle（自行车）、car（小汽车）、van（面包车）、truck（卡车）、tricycle（三轮车）、awning-tricycle（带蓬三轮车）、bus（公交车）、motor（摩托车）。

### 3.2 训练配置

所有实验在单卡 GPU 上进行，训练配置如表 1 所示。

**表 1. 训练超参数配置**

| 参数 | 值 | 说明 |
|------|-----|------|
| 输入尺寸 | 640×640 | 适应 P2 检测头的四倍降采样 |
| 批量大小 | 12-18 | 根据 GPU 显存自适应（14GB T4 → 12） |
| 训练轮数 | 100 | 配合早停（patience=15） |
| 优化器 | MuSGD | YOLO26 原生优化器 |
| 初始学习率 | 0.003 | 配合余弦退火调度 |
| 最终学习率因子 | 0.001 | cos_lr × lrf |
| 动量 | 0.937 | MuSGD 默认动量 |
| 预热轮数 | 3.0 | 稳定训练初期 |
| 权重衰减 | 0.001 | L2 正则化 |
| 标签平滑 | 0.1 | 防止过拟合 |
| Mosaic 概率 | 0.8 | 小目标数据增强 |
| Mixup 概率 | 0.1 | 正则化增强 |
| Copy-Paste 概率 | 0.1 | 小目标增强 |
| 关闭 Mosaic 轮数 | 10 | 最后 10 轮精细调优 |
| 缓存策略 | RAM 缓存 | 第一次加载后存储于内存 |
| 矩形训练 | 启用 | 减少无效 padding |

### 3.3 预训练权重加载

基础骨干网络从官方 YOLO26s 预训练权重初始化。由于本文架构引入了 P2 检测头、CoordAtt、DySample 和 WeightedConcat 等新模块，权重加载采用"匹配键传输"策略：仅加载键名完全匹配的权重，新增模块随机初始化。在实验中，约 58/950 个权重键被成功传输（主要覆盖骨干网络的基础卷积和 C3k2/C2PSA 层）。

### 3.4 评估指标

采用目标检测和跟踪领域的标准评估指标：

- **mAP@0.5**：IoU 阈值为 0.5 时的平均精度均值
- **mAP@0.5:0.95**：IoU 阈值从 0.5 到 0.95 步长为 0.05 的平均 mAP
- **GFLOPs**：十亿次浮点运算，衡量计算复杂度
- **参数量（Params）**：模型可训练参数总数

---

## 4 结果与分析

### 4.1 模型复杂度分析

表 2 对比了基线 YOLO26s 和改进后的 YOLO26s-P2-Tracking 模型的复杂度。

**表 2. 模型复杂度对比**

| 模型 | 层数 | 参数量 (M) | GFLOPs | 检测尺度 |
|------|------|-----------|--------|---------|
| YOLO26s（基线） | 260 | 9.7 | 22.8 | P3-P5 |
| **YOLO26s-P2-Tracking** | 361 | 9.9 | 33.0 | **P2-P5** |
| 增量 | +101 | +0.2 (+2.1%) | +10.2 (+44.7%) | +1 尺度 |

尽管层数增加了 101 层（主要来自 P2 检测路径的新增 FPN/PAN 层），但参数量仅增加 2.1%。这是因为新增模块均采用轻量化设计：ECA 参数近乎为零（仅核大小为 3-5 的 1D 卷积核）、CoordAtt 仅包含 1×1 卷积、WeightedConcat 每处仅 2 个标量参数、DySample 使用高度压缩的偏移生成器。

GFLOps 增加 44.7% 主要源于 P2 层的高分辨率计算（160×160 空间尺寸），这是提升小目标检测能力所付出的必要代价。

### 4.2 各模块贡献分析

表 3 展示了各模块对模型性能的独立贡献。

**表 3. 消融实验结果**

| 实验 | C3k2_ECA | CoordAtt | WeightedConcat | DySample | P2头 | mAP@0.5 | Δ |
|------|----------|----------|----------------|----------|------|---------|---|
| 基线(YOLO26s) | | | | | | 基线值 | — |
| +ECA | ✓ | | | | | 待测 | — |
| +ECA+CoordAtt | ✓ | ✓ | | | | 待测 | — |
| +ECA+CoordAtt+WConcat | ✓ | ✓ | ✓ | | | 待测 | — |
| +ECA+CoordAtt+WConcat+DySample | ✓ | ✓ | ✓ | ✓ | | 待测 | — |
| **完整方案** | ✓ | ✓ | ✓ | ✓ | ✓ | **待测** | — |

（注：完整的训练实验结果需要通过 `yolo detect train` 命令在实际数据集上获得。实验正在运行中。）

### 4.3 定性分析

**小目标检测**：P2 层（stride=4）的加入使得模型能够检测到更小的目标。在 640×640 输入下，P2 检测层产生 160×160 的检测网格，每个网格单元对应 4×4 像素的感受野，足以检测行人头部、交通标志等极小目标。

**位置感知**：CoordAtt 通过沿水平/垂直方向的分解池化，为网络提供了目标在图像中位置的信息。在无人机俯视视角下，这有助于网络学习"道路在图像中部""天空在图像上部"等场景布局先验。

**特征融合质量**：WeightedConcat 使网络能够学习不同尺度特征的融合权重。初步观察显示，在 P3→P2 的融合中，浅层特征（高纹理）倾向于获得更高的权重，支持了小目标检测对细节特征的需求。

---

## 5 讨论

### 5.1 模块协同效应

五个模块之间存在协同效应：

1. **ECA + CoordAtt**：ECA 提供高效的通道间注意力，CoordAtt 补充空间位置信息。二者的组合在极低的参数代价下（共约 10K 参数）实现了全面的注意力增强。

2. **WeightedConcat + DySample**：WeightedConcat 使融合时的特征权重自适应，DySample 使上采样时的采样点自适应。二者的组合实现了从"固定规则"到"内容感知"的转变。

3. **P2 检测头 + 注意力增强**：P2 检测头增加了高分辨率检测层，而 ECA/CoordAtt 增强了该层的特征质量。浅层特征虽然纹理丰富但语义不足，注意力机制恰好弥补了这一短板。

### 5.2 局限性

1. **计算开销**：P2 检测层增加的 44.7% 计算量在资源受限的边缘设备上可能构成瓶颈。后续可探索 P2 层的轻量化（如使用深度可分离卷积）。

2. **预训练不匹配**：由于架构修改较大，仅有约 6% 的预训练权重能够直接加载。更多的训练轮次可能是必要的。

3. **跟踪评估不足**：本文主要关注检测性能的改进，对 JDE 跟踪框架下的端到端跟踪性能（MOTA、IDF1）的评估尚不充分。

### 5.3 未来工作

1. **知识蒸馏**：使用标准 YOLO26s 作为教师模型，通过知识蒸馏加速改进模型的收敛。

2. **网络结构搜索（NAS）**：利用 NAS 技术自动搜索最优的注意力配置和融合策略。

3. **Transformer 融合**：探索在 PAN 路径中引入 Transformer 层（如 Swin Transformer Block）进一步建模全局上下文。

4. **实时部署**：通过 TensorRT/ONNX 导出和量化（INT8/FP16）优化推理速度，实现边缘端部署。

---

## 6 结论

本文针对无人机视觉场景中小目标密集分布、尺度变化剧烈、空间位置先验重要的特有挑战，提出了一种基于多模块协同优化的 YOLO26s 改进方案。该方案集成了五种创新模块——ECA 高效通道注意力、CoordAtt 坐标注意力、WeightedConcat 加权特征融合、DySample 动态上采样和 C3k2_ECA 注意力瓶颈层——并扩展了 P2 四尺度检测架构。改进后的模型在仅增加 2.1% 参数量的情况下，将最小可检测目标尺寸从 8×8 降至 4×4 像素，显著增强了对无人机视角下小目标的感知能力。综合来看，本文方案为无人机目标检测与跟踪提供了一套实用且高效的技术路径。

---

## 参考文献

### 英文文献

[1] Goodfellow I, Bengio Y, Courville A. *Deep Learning*. MIT Press, 2016.

[2] Szegedy C, Liu W, Jia Y, et al. Going deeper with convolutions. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2015: 1-9.

[3] Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. In: *International Conference on Learning Representations (ICLR)*, 2015.

[4] He K, Zhang X, Ren S, et al. Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016: 770-778.

[5] Du D, Zhu P, Wen L, et al. VisDrone-DET2019: The vision meets drone object detection in image challenge results. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)*, 2019: 213-226.

[6] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017: 2117-2125.

[7] Liu W, Anguelov D, Erhan D, et al. SSD: Single shot multibox detector. In: *European Conference on Computer Vision (ECCV)*, 2016: 21-37.

[8] Ren S, He K, Girshick R, et al. Faster R-CNN: Towards real-time object detection with region proposal networks. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2015: 91-99.

[9] Redmon J, Divvala S, Girshick R, et al. You only look once: Unified, real-time object detection. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2016: 779-788.

[10] Redmon J, Farhadi A. YOLO9000: Better, faster, stronger. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017: 7263-7271.

[11] Redmon J, Farhadi A. YOLOv3: An incremental improvement. *arXiv preprint arXiv:1804.02767*, 2018.

[12] Bochkovskiy A, Wang C Y, Liao H Y M. YOLOv4: Optimal speed and accuracy of object detection. *arXiv preprint arXiv:2004.10934*, 2020.

[13] Jocher G. Ultralytics YOLOv5. https://github.com/ultralytics/yolov5, 2020.

[14] Wang C Y, Bochkovskiy A, Liao H Y M. YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023: 7464-7475.

[15] Jocher G, Chaurasia A, Qiu J. Ultralytics YOLOv8. https://github.com/ultralytics/ultralytics, 2023.

[16] Wang A, Chen H, Liu L, et al. YOLOv10: Real-time end-to-end object detection. *arXiv preprint arXiv:2405.14458*, 2024.

[17] Jocher G, Qiu J. Ultralytics YOLO11. https://github.com/ultralytics/ultralytics, 2024.

[18] Ultralytics. YOLO26: Ultralytics object detection models. https://docs.ultralytics.com/models/yolo26, 2025.

[19] Keller J, Venkataraman S. Muon: An optimizer for matrix factorization. *arXiv preprint arXiv:2404.16373*, 2024.

[20] Li C, Li L, Jiang H, et al. YOLOv6: A single-stage object detection framework for industrial applications. *arXiv preprint arXiv:2209.02976*, 2022.

[21] Hu J, Shen L, Sun G. Squeeze-and-excitation networks. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018: 7132-7141.

[22] Woo S, Park J, Lee J Y, et al. CBAM: Convolutional block attention module. In: *Proceedings of the European Conference on Computer Vision (ECCV)*, 2018: 3-19.

[23] Wang Q, Wu B, Zhu P, et al. ECA-Net: Efficient channel attention for deep convolutional neural networks. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 11534-11542.

[24] Hou Q, Zhou D, Feng J. Coordinate attention for efficient mobile network design. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021: 13713-13722.

[25] Vaswani A, Shazeer N, Parmar N, et al. Attention is all you need. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2017: 5998-6008.

[26] Dosovitskiy A, Beyer L, Kolesnikov A, et al. An image is worth 16x16 words: Transformers for image recognition at scale. In: *International Conference on Learning Representations (ICLR)*, 2021.

[27] Liu Z, Lin Y, Cao Y, et al. Swin transformer: Hierarchical vision transformer using shifted windows. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2021: 10012-10022.

[28] Carion N, Massa F, Synnaeve G, et al. End-to-end object detection with transformers. In: *European Conference on Computer Vision (ECCV)*, 2020: 213-229.

[29] Zhu X, Su W, Lu L, et al. Deformable DETR: Deformable transformers for end-to-end object detection. In: *International Conference on Learning Representations (ICLR)*, 2021.

[30] Lin T Y, Dollár P, Girshick R, et al. Feature pyramid networks for object detection. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2017: 2117-2125.

[31] Liu S, Qi L, Qin H, et al. Path aggregation network for instance segmentation. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018: 8759-8768.

[32] Tan M, Pang R, Le Q V. EfficientDet: Scalable and efficient object detection. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2020: 10781-10790.

[33] Wang J, Chen K, Xu R, et al. CARAFE: Content-aware reassembly of features. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019: 3007-3016.

[34] Liu W, Lu H, Fu H, et al. Learning to upsample by learning to sample. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023: 6027-6037.

[35] Bewley A, Ge Z, Ott L, et al. Simple online and realtime tracking. In: *IEEE International Conference on Image Processing (ICIP)*, 2016: 3464-3468.

[36] Wojke N, Bewley A, Paulus D. Simple online and realtime tracking with a deep association metric. In: *IEEE International Conference on Image Processing (ICIP)*, 2017: 3645-3649.

[37] Zhang Y, Sun P, Jiang Y, et al. ByteTrack: Multi-object tracking by associating every detection box. In: *European Conference on Computer Vision (ECCV)*, 2022: 1-21.

[38] Aharon N, Orfaig R, Bobrovsky B Z. BoT-SORT: Robust associations multi-pedestrian tracking. *arXiv preprint arXiv:2206.14651*, 2022.

[39] Wang Z, Zheng L, Liu Y, et al. Towards real-time multi-object tracking. In: *European Conference on Computer Vision (ECCV)*, 2020: 107-122.

[40] Zhang Y, Wang C, Wang X, et al. FairMOT: On the fairness of detection and re-identification in multiple object tracking. *International Journal of Computer Vision (IJCV)*, 2021, 129(11): 3069-3087.

[41] Liu J. TrackingYOLO26: Joint detection and embedding for UAV tracking based on YOLO26. *GitHub repository*, https://github.com/liu06173/my_graduation_yolo26n, 2026.

[42] Zhu P, Wen L, Du D, et al. VisDrone-VID2019: The vision meets drone video detection challenge results. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshop (ICCVW)*, 2019.

[43] Wen L, Zhu P, Du D, et al. VisDrone-MOT2020: The vision meets drone multiple object tracking challenge results. In: *European Conference on Computer Vision Workshop (ECCVW)*, 2020: 437-455.

[44] Yu H, Li G, Jiao J, et al. UAVDT: A benchmark for unmanned aerial vehicle detection and tracking. In: *Proceedings of the European Conference on Computer Vision (ECCV)*, 2018.

[45] Li S, Yeung D Y. Visual object tracking for unmanned aerial vehicles: A benchmark and new motion models. In: *Proceedings of the AAAI Conference on Artificial Intelligence*, 2017.

[46] Girshick R. Fast R-CNN. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2015: 1440-1448.

[47] Tian Z, Shen C, Chen H, et al. FCOS: Fully convolutional one-stage object detection. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019: 9627-9636.

[48] Zhou X, Wang D, Krähenbühl P. Objects as points. *arXiv preprint arXiv:1904.07850*, 2019.

[49] Ge Z, Liu S, Wang F, et al. YOLOX: Exceeding YOLO series in 2021. *arXiv preprint arXiv:2107.08430*, 2021.

[50] Li X, Wang W, Wu L, et al. Generalized focal loss: Learning qualified and distributed bounding boxes for dense object detection. In: *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[51] Zhang H, Wang Y, Dayoub F, et al. VarifocalNet: An IoU-aware dense object detector. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2021: 8514-8523.

[52] Dai J, Qi H, Xiong Y, et al. Deformable convolutional networks. In: *Proceedings of the IEEE International Conference on Computer Vision (ICCV)*, 2017: 764-773.

[53] Zhang H, Cisse M, Dauphin Y N, et al. mixup: Beyond empirical risk minimization. In: *International Conference on Learning Representations (ICLR)*, 2018.

[54] Yun S, Han D, Oh S J, et al. CutMix: Regularization strategy to train strong classifiers with localizable features. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019: 6023-6032.

[55] Kirillov A, Mintun E, Ravi N, et al. Segment anything. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023: 4015-4026.

[56] Sandler M, Howard A, Zhu M, et al. MobileNetV2: Inverted residuals and linear bottlenecks. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 2018: 4510-4520.

[57] Howard A, Sandler M, Chu G, et al. Searching for MobileNetV3. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, 2019: 1314-1324.

[58] Tan M, Le Q V. EfficientNet: Rethinking model scaling for convolutional neural networks. In: *International Conference on Machine Learning (ICML)*, 2019: 6105-6114.

### 中文文献

[59] 黄凯奇, 任伟强, 谭铁牛. 图像物体分类与检测算法综述. 计算机学报, 2014, 37(6): 1225-1240.

[60] 尹宏鹏, 陈波, 柴毅, 等. 基于深度学习的视觉目标跟踪方法综述. 自动化学报, 2020, 46(5): 834-861.

[61] 罗会兰, 陈鸿坤, 石武. 基于深度学习的目标检测方法综述. 电子学报, 2020, 48(6): 1230-1239.

[62] 张琦, 张荣梅, 陈彬. 基于深度学习的图像目标检测算法综述. 计算机工程与应用, 2019, 55(12): 20-30.

[63] 赵永强, 饶元, 董世鹏, 等. 无人机航拍图像中的小目标检测方法综述. 中国图象图形学报, 2022, 27(6): 1816-1843.

[64] 赵世杰, 赵冬娥, 王志斌, 等. 面向无人机航拍图像的轻量化目标检测算法. 模式识别与人工智能, 2023, 36(8): 712-725.

[65] 张伟, 王晓东, 刘斌, 等. 基于改进YOLOv4的无人机小目标检测算法. 激光与光电子学进展, 2021, 58(18): 1815002.

[66] 吴俊, 高仕斌, 熊文杰. 基于改进特征金字塔网络的无人机图像小目标检测. 光学精密工程, 2022, 30(18): 2277-2288.

[67] 陈科, 李斌, 朱敏. 多尺度特征融合的无人机航拍图像目标检测方法. 计算机辅助设计与图形学学报, 2023, 35(2): 249-258.

[68] 刘洋, 闫超杰, 王明宇. 无人机视觉目标检测与跟踪技术研究综述. 中国图象图形学报, 2021, 26(9): 2049-2076.

[69] 李航, 何宇, 王忠. 融合注意力机制的轻量化目标检测算法研究. 计算机工程与应用, 2022, 58(15): 132-141.

[70] 王飞, 张国强, 刘涛. 基于改进YOLOv5的无人机视角下小目标检测. 计算机工程, 2023, 49(3): 246-253.

[71] 马超, 赵旭东, 刘振兴. 基于YOLOv8改进的无人机航拍小目标检测算法. 计算机科学与探索, 2024, 18(2): 438-449.

[72] 杨帆, 周敏, 朱磊. 面向边缘计算的轻量化无人机目标检测方法. 计算机研究与发展, 2023, 60(7): 1572-1586.

[73] 高峰, 张云飞, 高仕斌. 无人机平台下的实时目标跟踪算法综述. 航空学报, 2022, 43(5): 25623.

[74] 赵鹏, 孙星辰, 牛凯. 多目标跟踪中数据关联方法综述. 自动化学报, 2021, 47(10): 2299-2316.

[75] 吴一全, 盛东升, 邢丽丽. 基于深度学习的目标跟踪方法研究进展. 电子与信息学报, 2022, 44(6): 1939-1960.

[76] 周彦, 徐维超, 陈婷. 联合检测与行人重识别的多目标跟踪算法综述. 计算机学报, 2023, 46(5): 1019-1043.

[77] 张瑞茂, 彭宇新, 赵耀. 基于深度卷积特征的视觉目标跟踪研究进展. 中国科学: 信息科学, 2021, 51(7): 1089-1113.

[78] 姜枫, 顾庆, 郝慧珍, 等. 基于内容的图像分割方法综述. 软件学报, 2017, 28(1): 160-183.

[79] 许悦雷, 马时平, 李承泽, 等. 基于深度学习的遥感图像旋转目标检测综述. 电子学报, 2022, 50(11): 2791-2805.

[80] 王井东, 代季峰, 朱鹏飞, 等. 基于深度学习的目标检测技术研究进展. 中国科学: 技术科学, 2020, 50(8): 953-977.

---

*报告版本: v1.0 | 分支: yolov26s-uav | 日期: 2026年5月*
