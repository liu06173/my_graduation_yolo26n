#!/usr/bin/env python3
"""
TrackingYOLO26 训练脚本

用法:
  python tools/train_tracking.py --data configs/visdrone.yaml --epochs 300
  python tools/train_tracking.py --resume runs/tracking/weights/last.pt
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.cuda import amp
from tqdm import tqdm

# 将项目根目录加入路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.tracking_model import TrackingYOLO26, TrackingDetect
from models.tracking_loss import TrackingLoss


def parse_args():
    parser = argparse.ArgumentParser(description="TrackingYOLO26 Training")
    parser.add_argument("--data", type=str, default="configs/visdrone.yaml",
                        help="Dataset config")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
    parser.add_argument("--weights", type=str, default="yolo26n.pt",
                        help="Pretrained YOLO26 weights")
    parser.add_argument("--embed-dim", type=int, default=128,
                        help="Re-ID embedding dimension")
    parser.add_argument("--project", type=str, default="runs/tracking")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--save-period", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    # Device
    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # Model
    print("[Model] Building TrackingYOLO26...")
    model = TrackingYOLO26(nc=10, embed_dim=args.embed_dim)
    model = model.to(device)

    # Load pretrained backbone
    if args.weights and os.path.exists(args.weights):
        print(f"[Model] Loading pretrained backbone from {args.weights}")
        model.load_backbone(args.weights)
    elif args.resume and os.path.exists(args.resume):
        print(f"[Model] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
    else:
        print("[Model] Training from scratch (backbone only, head initialized randomly)")

    # Data
    print("[Data] Loading dataset...")
    from ultralytics.data import build_dataloader
    from ultralytics.utils import load_yaml

    data_cfg = load_yaml(args.data)
    # Use ultralytics data pipeline
    from ultralytics.data.dataset import YOLODataset

    train_dataset = YOLODataset(
        img_path=data_cfg['train'],
        imgsz=args.imgsz,
        batch_size=args.batch,
        augment=True,
        data=data_cfg,
        task='detect',
    )

    train_loader = build_dataloader(
        train_dataset,
        batch=args.batch,
        workers=args.workers,
        shuffle=True,
    )

    # Optimizer
    from ultralytics.optim.muon import MuSGD
    param_groups = [
        {'params': model.head.parameters(), 'lr': args.lr, 'use_muon': True, 'momentum': 0.95},
        {'params': model.backbone_neck.parameters(), 'lr': args.lr * 0.1, 'use_muon': True, 'momentum': 0.95},
    ]
    optimizer = MuSGD(param_groups, lr=args.lr, muon=0.5, sgd=0.5)
    scaler = amp.GradScaler(enabled=device.type == 'cuda')

    # Loss
    criterion = TrackingLoss(nc=10, w_det=1.0, w_triplet=0.3)

    # Training loop
    os.makedirs(os.path.join(args.project, 'weights'), exist_ok=True)
    best_loss = float('inf')
    start_epoch = 0

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        start_epoch = ckpt.get('epoch', 0)

    print(f"\n[Training] Starting from epoch {start_epoch + 1}/{args.epochs}")
    print(f"  Dataset: {len(train_dataset)} images")
    print(f"  Batch:   {args.batch}")
    print(f"  Image:   {args.imgsz}×{args.imgsz}")
    print(f"  GPU:     {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_det = 0.0
        epoch_trip = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for batch_idx, batch in enumerate(pbar):
            images = batch['img'].to(device).float() / 255.0
            # batch['batch_idx'] has the batch index for each target

            # Forward
            with amp.autocast(enabled=device.type == 'cuda'):
                preds = model(images)
                total_loss, loss_dict = criterion(preds, batch)

            # Backward
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            epoch_loss += loss_dict['total']
            epoch_det += loss_dict['det']
            epoch_trip += loss_dict['triplet']

            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.3f}",
                'det': f"{loss_dict['det']:.3f}",
                'trip': f"{loss_dict['triplet']:.4f}",
            })

        n = len(train_loader)
        avg_loss = epoch_loss / n
        avg_det = epoch_det / n
        avg_trip = epoch_trip / n

        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f} det={avg_det:.4f} trip={avg_trip:.4f}")

        # Save
        if (epoch + 1) % args.save_period == 0:
            ckpt_path = os.path.join(args.project, 'weights', f'epoch_{epoch + 1}.pt')
            torch.save({
                'epoch': epoch + 1,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)

        # Best
        last_path = os.path.join(args.project, 'weights', 'last.pt')
        torch.save({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, last_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.project, 'weights', 'best.pt')
            torch.save({'epoch': epoch + 1, 'model': model.state_dict()}, best_path)

    print(f"\n[Training] Complete. Best loss: {best_loss:.4f}")
    print(f"  Weights saved to {args.project}/weights/")


if __name__ == "__main__":
    main()
