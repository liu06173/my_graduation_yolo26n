#!/usr/bin/env python3
"""
VisDrone SOT (Single Object Tracking) 快速评估脚本
使用 OpenCV trackers (CSRT/KCF/MOSSE)，无需GPU训练，即开即跑
"""
import cv2, os, time, sys
import numpy as np
from pathlib import Path

DATA_ROOT = "data/sot_subset/VisDrone2019-SOT-val"
TRACKERS = {
    "CSRT": cv2.TrackerCSRT_create,
    "KCF": cv2.TrackerKCF_create,
}


def load_sequence(seq_name):
    """Load images and ground truth annotations for a sequence."""
    img_dir = Path(DATA_ROOT) / "sequences" / seq_name
    ann_file = Path(DATA_ROOT) / "annotations" / f"{seq_name}.txt"

    if not img_dir.exists():
        return None, None, None

    # Load annotations: x,y,w,h per line
    gts = []
    with open(ann_file) as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 4:
                gts.append([float(x) for x in parts[:4]])

    # Load images (sorted by name)
    images = sorted(img_dir.glob("*.jpg"))
    return images, gts, seq_name


def eval_tracker(tracker_name, images, gts):
    """Evaluate a single tracker on a sequence."""
    if len(images) < 2:
        return None

    tracker = TRACKERS[tracker_name]()

    # Init tracker with first frame GT
    first_img = cv2.imread(str(images[0]))
    gt0 = gts[0]  # x, y, w, h
    bbox = tuple(int(v) for v in gt0)
    tracker.init(first_img, bbox)

    results = [gt0]  # first frame is GT
    fps_list = []

    for i in range(1, len(images)):
        img = cv2.imread(str(images[i]))
        t0 = time.time()
        ok, bbox = tracker.update(img)
        fps = 1.0 / max(time.time() - t0, 0.001)
        fps_list.append(fps)

        if ok:
            results.append(list(bbox))
        else:
            results.append(results[-1])  # use last known position

    # Compute metrics
    gts_arr = np.array(gts)
    res_arr = np.array(results)

    # Center error
    gt_centers = gts_arr[:, :2] + gts_arr[:, 2:] / 2
    res_centers = res_arr[:, :2] + res_arr[:, 2:] / 2
    center_errors = np.linalg.norm(gt_centers - res_centers, axis=1)
    precision_20 = np.mean(center_errors < 20)

    # IoU
    ious = []
    for gt, res in zip(gts_arr, res_arr):
        gt_box = [gt[0], gt[1], gt[0] + gt[2], gt[1] + gt[3]]
        res_box = [res[0], res[1], res[0] + res[2], res[1] + res[3]]
        # Intersection
        xi1, yi1 = max(gt_box[0], res_box[0]), max(gt_box[1], res_box[1])
        xi2, yi2 = min(gt_box[2], res_box[2]), min(gt_box[3], res_box[3])
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        # Union
        gt_area = gt[2] * gt[3]
        res_area = res[2] * res[3]
        union = gt_area + res_area - inter
        ious.append(inter / union if union > 0 else 0)

    success = np.mean(np.array(ious) > 0.5)
    avg_fps = np.mean(fps_list) if fps_list else 0

    return {
        "precision@20": precision_20,
        "success@0.5": success,
        "avg_center_err": np.mean(center_errors),
        "avg_iou": np.mean(ious),
        "avg_fps": avg_fps,
        "frames": len(images),
    }


def main():
    print("=" * 60)
    print(" VisDrone SOT 单目标跟踪评估")
    print("=" * 60)

    # Get available sequences
    seq_dir = Path(DATA_ROOT) / "sequences"
    seqs = sorted([d.name for d in seq_dir.iterdir() if d.is_dir()])
    print(f"\n可用序列: {len(seqs)} 个")
    for s in seqs:
        n_frames = len(list((seq_dir / s).glob("*.jpg")))
        print(f"  {s}: {n_frames} frames")

    # Evaluate each sequence with each tracker
    print(f"\n{'='*60}")
    print(f" {'序列':<30} {'Tracker':>6} {'Prec@20':>8} {'Succ@0.5':>8} {'CtrErr':>7} {'FPS':>6}")
    print(f" {'-'*30} {'-'*6} {'-'*8} {'-'*8} {'-'*7} {'-'*6}")

    all_results = {}
    for seq in seqs:
        images, gts, name = load_sequence(seq)
        if images is None:
            continue
        # Use subset: every 3rd frame for speed
        images = images[::3]
        gts = gts[::3]

        for tname in TRACKERS:
            res = eval_tracker(tname, images, gts)
            if res:
                print(f" {name:<30} {tname:>6} {res['precision@20']:>7.1%} {res['success@0.5']:>7.1%} {res['avg_center_err']:>6.1f}px {res['avg_fps']:>5.1f}")
                all_results[f"{name}/{tname}"] = res

    # Summary
    print(f"\n{'='*60}")
    print(" 平均性能汇总")
    print(f" {'='*60}")
    for tname in TRACKERS:
        precs = [r["precision@20"] for k, r in all_results.items() if tname in k]
        succs = [r["success@0.5"] for k, r in all_results.items() if tname in k]
        fps_vals = [r["avg_fps"] for k, r in all_results.items() if tname in k]
        if precs:
            print(f" {tname:>6}: Prec@20={np.mean(precs):.1%}  Succ@0.5={np.mean(succs):.1%}  FPS={np.mean(fps_vals):.1f}")

    print(f"\n[完成] 基于 OpenCV 内置跟踪器，无需 GPU 训练")


if __name__ == "__main__":
    main()
