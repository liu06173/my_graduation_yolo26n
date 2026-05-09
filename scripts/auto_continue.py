#!/usr/bin/env python3
"""自动衔接训练：60轮结束→自动恢复跑至200轮，无需人工干预"""
import time, sys, os
from pathlib import Path
from datetime import datetime

CSV = Path("D:/yolo26_cache/runs/detect/train_2cls/results.csv")
LAST_PT = Path("D:/yolo26_cache/runs/detect/train_2cls/weights/last.pt")
LOCK = Path("D:/yolo26_cache/runs/detect/train_2cls/.phase2_lock")

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)

log("=== 自动衔接守护启动 ===")
log("Phase 1: 等待当前60轮训练完成...")

# 等待 epoch 60 出现
while True:
    if not CSV.exists():
        time.sleep(60)
        continue
    try:
        lines = CSV.read_text().strip().splitlines()
        if len(lines) < 2:
            time.sleep(60)
            continue
        last_line = lines[-1].split(",")
        epoch = int(last_line[0])
        mAP50 = float(last_line[7])
        log(f"  当前进度: Epoch {epoch}/60  mAP@50:{mAP50:.4f}")
        if epoch >= 60:
            log("Phase 1 完成! 60轮训练结束")
            break
    except Exception as e:
        log(f"  检查异常: {e}")
    time.sleep(120)

# 等待文件释放
time.sleep(10)

# 检查锁文件，防止重复执行
if LOCK.exists():
    log("Phase 2 已在执行中，退出")
    sys.exit(0)
LOCK.write_text("running")

log("=== Phase 2: 启动 61→200 轮衔接训练 ===")
log(f"  模型: {LAST_PT}")
log(f"  目标: 200 epochs (实际运行 Epoch 61-200)")

from ultralytics import YOLO

model = YOLO(str(LAST_PT))
model.train(
    resume=True,
    epochs=200,
    device=0,
    patience=30,
    save_period=10,
    cos_lr=True,
)

# 验证
lines = CSV.read_text().strip().splitlines()
last = lines[-1].split(",")
epoch = int(last[0])
mAP50 = float(last[7])
mAP50_95 = float(last[8])
log(f"=== 全部完成! Epoch {epoch}/200  mAP@50:{mAP50:.4f}  mAP@50-95:{mAP50_95:.4f} ===")
LOCK.unlink()
