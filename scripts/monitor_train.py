#!/usr/bin/env python3
"""训练监控 — 在第5/10/30/60轮自动汇报，每个epoch打印进度"""
import time, sys
from pathlib import Path

CSV = Path("D:/yolo26_cache/runs/detect/train_2cls/results.csv")
MILESTONES = {5, 10, 30, 60}
reported = set()
last_epoch = 0

def flush_print(*args, **kwargs):
    print(*args, **kwargs, flush=True)

flush_print("Monitor started - milestones: 5, 10, 30, 60")
flush_print("=" * 55)

while True:
    if not CSV.exists():
        time.sleep(60)
        continue

    try:
        # 先复制再读，避免锁住YOLO写入
        import shutil, tempfile
        tmp = Path(tempfile.gettempdir()) / "train_monitor_tmp.csv"
        try:
            shutil.copy2(str(CSV), str(tmp))
            lines = tmp.read_text().strip().splitlines()
        except:
            time.sleep(60)
            continue
        if len(lines) < 2:
            time.sleep(60)
            continue

        last_line = lines[-1].split(",")
        epoch = int(last_line[0])
        mAP50 = float(last_line[7])
        mAP50_95 = float(last_line[8])
        precision = float(last_line[5])
        recall = float(last_line[6])
        box_loss = float(last_line[2])
        cls_loss = float(last_line[3])
        elapsed_min = float(last_line[1]) / 60

        # 每个新epoch打印简要进度
        if epoch > last_epoch:
            prev = int(lines[last_epoch].split(",")[7]) * 100 if last_epoch > 0 else 0
            now = mAP50 * 100
            delta = now - prev
            arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
            flush_print(f"  [{elapsed_min:5.0f}min] Epoch {epoch:2d}/60 | mAP@50: {mAP50:.4f} ({arrow}{abs(delta):.1f}) | P:{precision:.3f} R:{recall:.3f} | box:{box_loss:.4f} cls:{cls_loss:.4f}")
            last_epoch = epoch

        # 里程碑详细报告
        if epoch in MILESTONES and epoch not in reported:
            remaining = (elapsed_min / epoch) * (60 - epoch)
            print()
            flush_print("=" * 55)
            flush_print(f"  MILESTONE REPORT - Epoch {epoch}/60")
            flush_print("=" * 55)
            flush_print(f"  mAP@50:      {mAP50:.4f}  ({mAP50*100:.1f}%)")
            flush_print(f"  mAP@50-95:   {mAP50_95:.4f}  ({mAP50_95*100:.1f}%)")
            flush_print(f"  Precision:   {precision:.4f}")
            flush_print(f"  Recall:      {recall:.4f}")
            flush_print(f"  Box Loss:    {box_loss:.4f}")
            flush_print(f"  Cls Loss:    {cls_loss:.4f}")
            flush_print(f"  Elapsed:     {elapsed_min:.0f} min ({elapsed_min/60:.1f}h)")
            flush_print(f"  Est Remain:  {remaining:.0f} min ({remaining/60:.1f}h)")
            flush_print("=" * 55)
            print()
            reported.add(epoch)

        if epoch >= 60:
            flush_print("Training complete!")
            break

    except Exception as e:
        flush_print(f"  [monitor error: {e}]")

    time.sleep(90)
