"""
跟踪模型损失函数

总损失 = λ_det × 检测损失 + λ_reid × Re-ID损失

检测损失 (复用YOLO26):
  - VarifocalLoss (分类)
  - CIoU Loss (回归框)
  - DFL Loss (分布焦点损失, YOLO26中为L1)

Re-ID损失:
  - Triplet Loss (hardest positive + hardest negative)
  - Center Loss (类内紧凑, 可选)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TripletLoss(nn.Module):
    """
    Hard Triplet Loss for Re-ID

    对每个 anchor:
      pos = 同 ID 中最不相似的那个 (hardest positive)
      neg = 不同 ID 中最相似的那个 (hardest negative)
      loss = max(0, margin + d(pos) - d(neg))
    """

    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, target_ids, fg_mask):
        """
        Args:
            embeddings: [B, N, embed_dim]
            target_ids: [B, N] — 每个 anchor 的目标 ID (背景为 -1)
            fg_mask:   [B, N] — 前景掩码
        Returns:
            loss: scalar
        """
        B, N, D = embeddings.shape
        device = embeddings.device

        total_loss = 0.0
        valid_batches = 0

        for b in range(B):
            fg = fg_mask[b]  # [N]
            n_fg = fg.sum().item()
            if n_fg < 2:
                continue

            emb = embeddings[b][fg]  # [n_fg, D]
            ids = target_ids[b][fg]  # [n_fg]

            # 去重: 每个 ID 至少出现2次才构成有效 triplet
            unique_ids, counts = ids.unique(return_counts=True)
            valid_ids = unique_ids[counts >= 2]
            if len(valid_ids) == 0:
                continue

            # 相似度矩阵
            sim = torch.mm(emb, emb.t())  # [n_fg, n_fg]

            # 正样本掩码 (同ID)
            id_eq = (ids.unsqueeze(0) == ids.unsqueeze(1)).float()
            id_ne = 1.0 - id_eq

            # Hardest positive: 同ID中相似度最低的
            pos_sim = sim.clone()
            pos_sim[id_eq == 0] = -float('inf')
            hardest_pos = pos_sim.max(dim=1)[0]

            # Hardest negative: 不同ID中相似度最高的
            neg_sim = sim.clone()
            neg_sim[id_ne == 0] = float('inf')
            hardest_neg = neg_sim.min(dim=1)[0]

            loss = F.relu(self.margin + hardest_neg - hardest_pos).mean()
            total_loss += loss
            valid_batches += 1

        if valid_batches == 0:
            return torch.tensor(0.0, device=device)

        return total_loss / valid_batches


class CenterLoss(nn.Module):
    """
    Center Loss — 让同ID嵌入聚拢到类中心

    loss = Σ ||e_i - c_{y_i}||²
    where c_{y_i} is the moving-average center of class y_i
    """

    def __init__(self, num_classes=1000, embed_dim=128, alpha=0.5):
        super().__init__()
        self.alpha = alpha
        self.centers = nn.Parameter(torch.randn(num_classes, embed_dim))

    def forward(self, embeddings, ids, fg_mask):
        B, N, D = embeddings.shape
        device = embeddings.device

        total_loss = 0.0
        count = 0

        for b in range(B):
            fg = fg_mask[b]
            if fg.sum() < 1:
                continue
            emb = embeddings[b][fg]
            ids_b = ids[b][fg].long()

            # 只保留有效ID
            valid = (ids_b >= 0) & (ids_b < len(self.centers))
            if valid.sum() < 1:
                continue
            emb = emb[valid]
            ids_b = ids_b[valid]

            # 计算 loss
            centers_batch = self.centers[ids_b]  # [n, D]
            diff = emb - centers_batch
            loss = (diff ** 2).sum() / emb.shape[0]

            # 更新中心 (moving average)
            with torch.no_grad():
                for uid in ids_b.unique():
                    mask = ids_b == uid
                    center_update = emb[mask].mean(dim=0)
                    self.centers[uid] = self.centers[uid] * self.alpha + \
                                        center_update * (1 - self.alpha)

            total_loss += loss
            count += 1

        if count == 0:
            return torch.tensor(0.0, device=device)
        return total_loss / count


class TrackingLoss(nn.Module):
    """
    跟踪模型总损失

    loss = w_det × (cls_loss + box_loss + dfl_loss) + w_triplet × triplet_loss
    """

    def __init__(
        self,
        nc=10,
        w_det=1.0,
        w_triplet=0.3,
        w_center=0.05,
        triplet_margin=0.3,
    ):
        super().__init__()
        self.nc = nc
        self.w_det = w_det
        self.w_triplet = w_triplet
        self.w_center = w_center

        self.triplet_loss = TripletLoss(margin=triplet_margin)
        self.center_loss = CenterLoss(num_classes=200, embed_dim=128)

    def forward(self, preds, targets, batch):
        """
        Args:
            preds: dict from TrackingDetect
                'det': detection head output (for ultralytics Loss)
                'embed': [B, N, embed_dim] embeddings
            targets: ground truth from dataset
            batch: batch metadata

        Returns:
            total_loss, loss_dict
        """
        from ultralytics.utils.loss import v8DetectionLoss

        # 1. 检测损失 (使用 YOLO26 原生计算)
        det_criterion = v8DetectionLoss(self.nc)
        det_loss, det_items = det_criterion(preds['det'], batch)

        # 2. Re-ID 损失
        embed = preds['embed']

        # 从 targets 提取跟踪 ID
        # targets: [N_targets, 6] — [batch_idx, cls, x, y, w, h]
        # 我们需要额外的 track_id 字段
        # 这里先用 class_id 作为 proxy (MOT训练时替换为真实ID)
        target_ids = targets[:, 1].long()  # class id 作为临时 ID

        B = embed.shape[0]
        N = embed.shape[1]

        # 构建 fg_mask: anchors that have assigned targets
        fg_mask = torch.zeros(B, N, dtype=torch.bool, device=embed.device)
        # (简化: 用 TAL assigner 的结果, 此处先用 class > 0 标记前景)
        # 实际使用时需要从 assigner 获取

        # Simplified: compute triplet loss on all assigned anchors
        triplet_loss = self.triplet_loss(embed, target_ids.unsqueeze(0).expand(B, -1), fg_mask)

        # 3. 总损失
        total_loss = (
            self.w_det * det_loss +
            self.w_triplet * triplet_loss
        )

        loss_dict = {
            'det': det_loss.item(),
            'triplet': triplet_loss.item() if isinstance(triplet_loss, torch.Tensor) else triplet_loss,
            'total': total_loss.item(),
        }

        return total_loss, loss_dict


def extract_tracking_targets(labels, batch_idx):
    """
    从 VisDrone-MOT 标签提取跟踪目标

    MOT 标签格式: [frame_idx, track_id, x, y, w, h, score, class, trunc, occlusion]
    转换后: [class, x, y, w, h, track_id]

    Returns:
        targets: [N, 6] — [batch_idx, cls, x, y, w, h]
        track_ids: [N] — track_id per target
    """
    targets = []
    track_ids = []

    for i, label_per_img in enumerate(labels):
        if len(label_per_img) == 0:
            continue
        # label_per_img: [M, 10] (MOT format)
        for row in label_per_img:
            cls = int(row[7]) - 1  # MOT class 1-10 → 0-9
            if cls < 0 or cls >= 10:
                continue
            x, y, w, h = row[2], row[3], row[4], row[5]
            track_id = int(row[1])

            targets.append([batch_idx, cls, x, y, w, h])
            track_ids.append(track_id)

    return (
        torch.tensor(targets, dtype=torch.float32) if targets else torch.zeros((0, 6)),
        torch.tensor(track_ids, dtype=torch.long) if track_ids else torch.zeros(0, dtype=torch.long),
    )
