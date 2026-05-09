#!/usr/bin/env python3
"""VisDrone 2类检测训练 — 实时中文汇报 + 每轮分析"""
import sys, os, time, json
from pathlib import Path
from datetime import datetime

import torch
from ultralytics import YOLO

# ─── 训练配置 ───
CONFIG = {
    "model": "yolo26n.pt",
    "data": "configs/visdrone_2cls.yaml",
    "epochs": 60,
    "imgsz": 512,
    "batch": 8,
    "device": 0,
    "workers": 4,
    "cache": "disk",
    "lr0": 0.01,
    "lrf": 0.01,
    "momentum": 0.937,
    "weight_decay": 0.0005,
    "warmup_epochs": 3,
    "warmup_momentum": 0.8,
    "cos_lr": True,
    "close_mosaic": 10,
    "patience": 15,
    "save": True,
    "save_period": 10,
    "amp": False,
    "fraction": 1.0,
    "project": "D:/yolo26_cache/runs/detect",
    "name": "train_2cls",
    "exist_ok": True,
    "optimizer": "SGD",
}

CN_METRIC = {
    "train/box_loss": "框损失", "train/cls_loss": "分类损失", "train/dfl_loss": "DFL损失",
    "metrics/precision(B)": "精确率", "metrics/recall(B)": "召回率",
    "metrics/mAP50(B)": "mAP@50", "metrics/mAP50-95(B)": "mAP@50-95",
    "val/box_loss": "验证框损失", "val/cls_loss": "验证分类损失", "val/dfl_loss": "验证DFL损失",
    "lr/pg0": "学习率", "lr/pg1": "学习率1", "lr/pg2": "学习率2",
}

BEST = {"epoch": 0, "mAP50": 0, "mAP50_95": 0, "recall": 0, "precision": 0}


def on_fit_epoch_end(trainer):
    """每个epoch结束后回调 — 中文实时汇报"""
    epoch = trainer.epoch + 1
    total = trainer.epochs
    metrics = trainer.metrics

    # 提取指标
    box_loss = metrics.get("train/box_loss", 0)
    cls_loss = metrics.get("train/cls_loss", 0)
    dfl_loss = metrics.get("train/dfl_loss", 0)
    precision = metrics.get("metrics/precision(B)", 0)
    recall = metrics.get("metrics/recall(B)", 0)
    mAP50 = metrics.get("metrics/mAP50(B)", 0)
    mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0)
    lr = trainer.optimizer.param_groups[0]["lr"]
    gpu_mem = torch.cuda.memory_reserved() / 1e9

    # 更新最佳
    if mAP50 > BEST["mAP50"]:
        BEST["epoch"] = epoch
        BEST["mAP50"] = mAP50
        BEST["mAP50_95"] = mAP50_95
        BEST["recall"] = recall
        BEST["precision"] = precision

    # ─── 实时中文汇报 ───
    pct = epoch / total * 100
    bar_len = 20
    filled = int(bar_len * epoch / total)
    bar = "█" * filled + "░" * (bar_len - filled)

    print(f"\n{'='*60}")
    print(f"  📊 Epoch {epoch:3d}/{total}  [{bar}] {pct:.0f}%")
    print(f"  {'─'*50}")
    print(f"  🔧 训练损失 | 框:{box_loss:.4f}  分类:{cls_loss:.4f}  DFL:{dfl_loss:.4f}")
    print(f"  🎯 验证指标 | 精确率:{precision:.4f}  召回率:{recall:.4f}")
    print(f"  🏆 mAP      | mAP@50: {mAP50:.4f}   mAP@50-95: {mAP50_95:.4f}")
    print(f"  💡 学习率   | {lr:.6f}   显存: {gpu_mem:.2f}GB")

    # 分析走势
    if mAP50 == BEST["mAP50"] and epoch > 1:
        print(f"  ⭐ 当前最佳！(历史最佳: Epoch{BEST['epoch']} mAP@50={BEST['mAP50']:.4f})")

    # 判断是否进入平台
    if epoch > 10:
        if mAP50_95 < 0.01 and BEST["mAP50_95"] < 0.01:
            print(f"  ⚠️  mAP仍接近0，模型可能还在收敛初期")

    if precision > 0.95 and recall < 0.1:
        print(f"  ⚠️ 高精确率低召回 — 模型过于保守，置信度阈值可能偏高")

    if recall > 0.95 and precision < 0.1:
        print(f"  ⚠️ 高召回率低精确率 — 大量误检，模型可能混淆类别")

    print(f"{'='*60}\n")


def on_train_end(trainer):
    """训练结束回调"""
    print(f"\n{'='*60}")
    print(f"  🎉 训练完成！")
    print(f"  {'='*60}")
    print(f"  最佳结果 (Epoch {BEST['epoch']}):")
    print(f"    mAP@50:    {BEST['mAP50']:.4f}")
    print(f"    mAP@50-95: {BEST['mAP50_95']:.4f}")
    print(f"    精确率:     {BEST['precision']:.4f}")
    print(f"    召回率:     {BEST['recall']:.4f}")
    print(f"  {'='*60}\n")


def main():
    print("=" * 60)
    print("  🚁 YOLO26n VisDrone 人物检测训练")
    print("  类别: 人物(person) 单类检测 (VisDrone-VID)")
    print("=" * 60)
    print(f"  PyTorch: {torch.__version__}")
    print(f"  GPU:     {torch.cuda.get_device_name(0)}")
    print(f"  显存:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  启动时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("─" * 60)

    for k, v in CONFIG.items():
        print(f"  {k}: {v}")
    print("=" * 60)

    t0 = time.time()

    # 加载模型
    print("⏳ 加载预训练模型 yolo26n.pt ...")
    model = YOLO(CONFIG["model"])
    print("✅ 模型加载完成，开始训练...")
    print()

    # 添加回调
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
    model.add_callback("on_train_end", on_train_end)

    # 开始训练
    model.train(**{k: v for k, v in CONFIG.items() if k != "model"})

    elapsed = time.time() - t0
    print(f"\n⏱️ 总训练耗时: {elapsed/60:.1f} 分钟")
    print(f"📁 权重保存在: D:/yolo26_cache/runs/detect/train_2cls/weights/")

    # ─── 最终分析 ───
    best_pt = Path("D:/yolo26_cache/runs/detect/train_2cls/weights/best.pt")
    if best_pt.exists():
        print("\n📋 在验证集上评估最佳模型...")
        val_results = model.val(data=CONFIG["data"], split="val", device=CONFIG["device"])
        print(f"\n  ✅ 验证集最终结果:")
        print(f"    mAP@50:    {val_results.box.map50:.4f}")
        print(f"    mAP@50-95: {val_results.box.map:.4f}")
        print(f"    精确率:     {val_results.box.mp:.4f}" if hasattr(val_results.box, 'mp') else "")
        print(f"    召回率:     {val_results.box.mr:.4f}" if hasattr(val_results.box, 'mr') else "")


if __name__ == "__main__":
    main()
