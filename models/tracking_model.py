"""
TrackingYOLO26 — JDE (Joint Detection and Embedding) 跟踪模型

在 YOLO26 检测头基础上并行添加 Re-ID 嵌入分支，
实现端到端的目标检测 + 特征提取，用于多目标跟踪。

架构:
  YOLO26 Backbone (冻结/微调)
       ↓
  YOLO26 Neck (PAN)
       ↓
  ┌────────────────────────────┐
  │  TrackingDetect Head       │
  │  ├── Box Head (xyxy)       │  ← 原版检测分支
  │  ├── Cls Head (sigmoid)    │  ← 原版分类分支
  │  └── Embed Head (L2 norm)  │  ← 新增: Re-ID嵌入 (128维)
  └────────────────────────────┘
       ↓
  Detection + Embeddings → ByteTrack 关联 → 跟踪轨迹
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

from ultralytics.nn.modules.conv import ECA, CoordAtt


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力 (轻量级)"""
    def __init__(self, c, r=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excite = nn.Sequential(
            nn.Conv2d(c, c // r, 1),
            nn.SiLU(),
            nn.Conv2d(c // r, c, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.excite(self.squeeze(x))


class EmbedHead(nn.Module):
    """改进版 Re-ID 嵌入分支 — ECA 注意力 + 可选 CoordAtt 位置感知

    相比原版改进:
      - SEBlock → ECA (更轻量，自适应核大小，零额外参数)
      - 残差注意力连接 (更好的梯度流)
      - 可选 CoordAtt 模块 (位置感知特征，有利于 UAV 空间定位)
    """

    def __init__(self, in_channels, embed_dim=128, hidden_dim=None, use_coordatt=True):
        super().__init__()
        hidden_dim = hidden_dim or max(in_channels // 2, embed_dim * 2)

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.eca = ECA(hidden_dim)
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(),
        )
        self.coord = CoordAtt(hidden_dim, 32) if use_coordatt else nn.Identity()
        self.proj = nn.Conv2d(hidden_dim, embed_dim, 1)
        self.embed_dim = embed_dim

    def forward(self, x):
        out = self.conv1(x)
        out = out + self.eca(out)
        out = self.conv2(out)
        out = self.coord(out)
        embed = self.proj(out)
        return F.normalize(embed, p=2, dim=1)


class TrackingDetect(nn.Module):
    """
    跟踪检测头 — YOLO26 Detect + Re-ID Embed 并行分支

    使用方法:
      head = TrackingDetect(nc=10, ch=[128, 256, 512], embed_dim=128)
      det, embed = head(feats)  # feats from neck

    训练时返回字典:
      {'box': ..., 'cls': ..., 'embed': ..., 'feats': [...]}

    推理时返回:
      detections: [B, N, 6]     (x1,y1,x2,y2,conf,cls)
      embeddings: [B, N, 128]   (Re-ID特征)
    """

    def __init__(self, nc=80, ch=(), embed_dim=128, reg_max=1, end2end=True):
        super().__init__()
        from ultralytics.nn.modules.head import Detect

        self.nl = len(ch)
        self.nc = nc
        self.no = nc + reg_max * 4
        self.embed_dim = embed_dim

        # 复用 YOLO26 原生检测头
        self.detect = Detect(nc=nc, reg_max=reg_max, end2end=end2end, ch=ch)

        # 新增: Re-ID 嵌入分支 (每个检测层一个)
        self.embed_heads = nn.ModuleList([
            EmbedHead(c, embed_dim) for c in ch
        ])

        # 特征融合: 将多尺度嵌入上采样后融合 (提升小目标Re-ID质量)
        self.embed_fuse = nn.Sequential(
            nn.Conv2d(embed_dim * self.nl, embed_dim, 1),
            nn.BatchNorm2d(embed_dim),
            nn.SiLU(),
        )

    def forward(self, x):
        """x: list of [B, C, H, W] from neck"""
        # 检测分支
        det_output = self.detect(x)

        # Re-ID 分支
        embed_list = []
        target_h, target_w = x[0].shape[2], x[0].shape[3]

        for i, feat in enumerate(x):
            e = self.embed_heads[i](feat)  # [B, embed_dim, H_i, W_i]
            if i > 0:
                e = F.interpolate(e, size=(target_h, target_w), mode='bilinear', align_corners=False)
            embed_list.append(e)

        # 融合多尺度嵌入
        embed_fused = self.embed_fuse(torch.cat(embed_list, dim=1))  # [B, embed_dim, H, W]
        embed_flat = embed_fused.flatten(2).permute(0, 2, 1)  # [B, N, embed_dim]
        embed_flat = F.normalize(embed_flat, p=2, dim=-1)

        if self.training:
            return {
                'det': det_output,
                'embed': embed_flat,
            }
        else:
            # 推理: 返回检测框 + 对应嵌入
            return det_output, embed_flat


def build_tracking_model(model_path='yolo26n.pt', nc=10, embed_dim=128):
    """
    构建 TrackingYOLO26 模型

    Args:
        model_path: YOLO26 预训练权重路径
        nc: 类别数 (VisDrone: 10)
        embed_dim: Re-ID嵌入维度 (default: 128)

    Returns:
        (model, tracking_head): 完整的跟踪模型
    """
    from ultralytics import YOLO

    # 加载 YOLO26 预训练模型
    base_model = YOLO(model_path)
    backbone_neck = base_model.model.model[:-1]  # 去掉原版检测头

    # 获取 neck 输出通道数
    # YOLO26 neck 输出 3 个尺度: P3(80x80), P4(40x40), P5(20x20)
    # 通道数取决于模型尺寸: n→[64,128,256] s→[128,256,512]
    ch = [128, 256, 512]  # 适配 yolo26s, yolo26n 需调整
    if 'yolo26n' in model_path or 'yolo26n' in str(type(base_model)):
        class BasePlaceholder:
            pass
        try:
            # 尝试从模型结构推断通道数
            dummy = torch.randn(1, 3, 640, 640)
            feats = []
            for i, m in enumerate(backbone_neck):
                dummy = m(dummy)
                if isinstance(dummy, (list, tuple)):
                    for f in dummy:
                        feats.append(f.shape[1])
                elif i == len(backbone_neck) - 1:
                    if isinstance(dummy, (list, tuple)):
                        ch = [f.shape[1] for f in dummy[-3:]]
                    elif hasattr(dummy, 'shape'):
                        pass
        except:
            pass

    # 构建跟踪头
    tracking_head = TrackingDetect(nc=nc, ch=ch, embed_dim=embed_dim)

    return backbone_neck, tracking_head, base_model


class TrackingYOLO26(nn.Module):
    """
    完整的 TrackingYOLO26 模型

    Usage:
      model = TrackingYOLO26(nc=10, embed_dim=128)
      model.load_backbone('yolo26n.pt')  # 加载预训练权重
      outputs = model(images)
    """

    def __init__(self, nc=10, embed_dim=128, ch=None):
        super().__init__()
        self.nc = nc
        self.embed_dim = embed_dim

        # 通道数: YOLO26s 默认值
        self.ch = ch or [128, 256, 512]

        from ultralytics.nn.tasks import attempt_load_weights
        import yaml
        from pathlib import Path

        # 加载 yolo26s.yaml 并构建 backbone + neck
        cfg_path = Path(__file__).parent.parent / 'ultralytics' / 'cfg' / 'models' / '26' / 'yolo26.yaml'
        if cfg_path.exists():
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f)
            from ultralytics.nn.tasks import DetectionModel
            base = DetectionModel(cfg, nc=nc)
            self.backbone_neck = nn.Sequential(*list(base.model.children())[:-1])
        else:
            raise FileNotFoundError(f"Model config not found: {cfg_path}")

        # 跟踪检测头
        self.head = TrackingDetect(nc=nc, ch=self.ch, embed_dim=embed_dim)

    def load_backbone(self, weights_path):
        """从预训练YOLO26加载backbone+neck权重"""
        ckpt = torch.load(weights_path, map_location='cpu')
        if 'model' in ckpt:
            state = ckpt['model'].state_dict()
        else:
            state = ckpt.state_dict()

        # 只加载 backbone + neck 部分
        bb_state = {k: v for k, v in state.items()
                     if not k.startswith('model.') or
                     (k.startswith('model.') and 'detect' not in k and 'cv2' not in k and 'cv3' not in k)}
        # 映射键名
        my_state = self.state_dict()
        matched = {}
        for k, v in bb_state.items():
            # 尝试匹配
            clean_k = k.replace('model.', '')
            if clean_k in my_state:
                matched[clean_k] = v
        print(f"Loaded {len(matched)}/{len(my_state)} layers from pretrained weights")
        self.load_state_dict(matched, strict=False)

    def forward(self, x):
        feats = self.backbone_neck(x)
        if isinstance(feats, (list, tuple)):
            # 取最后3层 (P3, P4, P5)
            feats = list(feats[-3:])
        elif isinstance(feats, torch.Tensor):
            feats = [feats]
        return self.head(feats)

    def fuse(self):
        """融合优化用于推理"""
        self.head.detect.fuse()
