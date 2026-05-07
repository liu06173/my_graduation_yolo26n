#!/usr/bin/env python3
"""
VisDrone2019 数据准备工具
用法:
  python tools/prepare_visdrone.py --download              # 尝试下载数据集
  python tools/prepare_visdrone.py --convert /path/to/raw   # 转换原始标注为YOLO格式
  python tools/prepare_visdrone.py --check                  # 检查数据集完整性

原始VisDrone标注格式: <bbox_left>, <bbox_top>, <bbox_width>, <bbox_height>, <score>, <object_category>, <truncation>, <occlusion>
YOLO格式: <class_id> <x_center> <y_center> <width> <height>  (归一化)
"""
import argparse
import os
import sys
import shutil
import zipfile
import urllib.request
from pathlib import Path

# VisDrone2019 class mapping
CLASS_NAMES = [
    "pedestrian", "people", "bicycle", "car", "van",
    "truck", "tricycle", "awning-tricycle", "bus", "motor"
]

# Ignored regions (class 0) and others (class 11) should be filtered
IGNORE_IDS = {0, 11}


def visdrone2yolo(visdrone_ann_dir, yolo_labels_dir, img_width=None, img_height=None):
    """Convert VisDrone annotations to YOLO format."""
    os.makedirs(yolo_labels_dir, exist_ok=True)
    ann_files = list(Path(visdrone_ann_dir).glob("*.txt"))

    converted = 0
    for ann_file in ann_files:
        yolo_lines = []
        with open(ann_file, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                bbox_left = float(parts[0])
                bbox_top = float(parts[1])
                bbox_width = float(parts[2])
                bbox_height = float(parts[3])
                category = int(parts[5])
                truncation = int(parts[6])
                occlusion = int(parts[7])

                # Skip ignored regions and unknown classes
                if category in IGNORE_IDS:
                    continue
                # Skip heavily truncated/occluded (>90%)
                if truncation > 2 and occlusion > 2:
                    continue

                # VisDrone category: 1-10 → YOLO class: 0-9
                cls_id = category - 1
                if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                    continue

                # Convert to YOLO format (normalized)
                x_center = (bbox_left + bbox_width / 2) / img_width if img_width else bbox_left + bbox_width / 2
                y_center = (bbox_top + bbox_height / 2) / img_height if img_height else bbox_top + bbox_height / 2
                w = bbox_width / img_width if img_width else bbox_width
                h = bbox_height / img_height if img_height else bbox_height

                # If not normalized (raw pixel coords), normalize by image size later
                # Actually VisDrone uses pixel coordinates
                yolo_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")

        if yolo_lines:
            out_path = os.path.join(yolo_labels_dir, ann_file.name)
            with open(out_path, "w") as f:
                f.write("\n".join(yolo_lines))
            converted += 1

    return converted


def check_dataset(data_dir):
    """Check dataset integrity."""
    data_dir = Path(data_dir)
    errors = []

    for split in ["train", "val"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        if not img_dir.exists():
            errors.append(f"Missing: {img_dir}")
            continue
        if not lbl_dir.exists():
            errors.append(f"Missing: {lbl_dir}")
            continue

        imgs = set(p.stem for p in img_dir.glob("*.jpg"))
        lbls = set(p.stem for p in lbl_dir.glob("*.txt"))

        img_only = imgs - lbls
        lbl_only = lbls - imgs

        print(f"\n[{split}]")
        print(f"  Images: {len(imgs)}")
        print(f"  Labels: {len(lbls)}")

        if img_only:
            print(f"  [WARNING] {len(img_only)} images without labels")
            errors.append(f"{split}: {len(img_only)} images without labels")
        if lbl_only:
            print(f"  [WARNING] {len(lbl_only)} labels without images")
            errors.append(f"{split}: {len(lbl_only)} labels without images")

    if not errors:
        print("\n Dataset integrity: OK")
    else:
        print(f"\n Dataset integrity: {len(errors)} issues found")
        for e in errors:
            print(f"  - {e}")

    return len(errors) == 0


def convert_dataset(raw_dir, out_dir):
    """Convert full VisDrone2019 dataset from raw format."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # Detect directory structure
    # VisDrone2019-DET-train/
    #   images/  (or directly .jpg files)
    #   annotations/  (or directly .txt files)
    for split, split_name in [("train", "VisDrone2019-DET-train"), ("val", "VisDrone2019-DET-val")]:
        src_dir = raw_dir / split_name
        if not src_dir.exists():
            # Try alternate structure
            alt_dirs = list(raw_dir.glob(f"*{split}*"))
            if alt_dirs:
                src_dir = alt_dirs[0]

        if not src_dir.exists():
            print(f"[WARNING] Cannot find {split} data. Expected: {src_dir}")
            continue

        # Find images
        img_src = src_dir / "images"
        if not img_src.exists():
            imgs = list(src_dir.glob("*.jpg"))
            if imgs:
                img_src = src_dir
            else:
                print(f"[ERROR] No images found in {src_dir}")
                continue

        # Find annotations
        ann_src = src_dir / "annotations"
        if not ann_src.exists():
            txts = list(src_dir.glob("*.txt"))
            if txts:
                ann_src = src_dir

        # Create output directories
        img_dst = out_dir / "images" / split
        lbl_dst = out_dir / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split}...")

        # Copy/symlink images
        if img_src.is_dir():
            for img in img_src.glob("*.jpg"):
                shutil.copy2(img, img_dst / img.name)
        print(f"  Copied images to {img_dst}")

        # Convert annotations
        if ann_src.exists():
            # Detect if already in YOLO format
            test_file = next(ann_src.glob("*.txt"), None)
            if test_file:
                with open(test_file) as f:
                    first_line = f.readline().strip()
                # YOLO format: <int> <float> <float> <float> <float>
                parts = first_line.split()
                if len(parts) == 5 and "." not in parts[0]:
                    print("  Labels already in YOLO format, copying directly...")
                    for lbl in ann_src.glob("*.txt"):
                        shutil.copy2(lbl, lbl_dst / lbl.name)
                else:
                    print("  Converting VisDrone → YOLO format...")
                    n = visdrone2yolo(str(ann_src), str(lbl_dst))
                    print(f"  Converted {n} labels")
        else:
            print(f"  [WARNING] No annotations found at {ann_src}")

    print("\nConversion complete! Checking dataset...")
    check_dataset(out_dir)


def main():
    parser = argparse.ArgumentParser(description="VisDrone2019 Data Preparation")
    parser.add_argument("--download", action="store_true", help="Attempt to download dataset")
    parser.add_argument("--convert", type=str, metavar="RAW_DIR",
                        help="Convert raw VisDrone annotations to YOLO format")
    parser.add_argument("--out", type=str, default="data/visdrone",
                        help="Output directory (default: data/visdrone)")
    parser.add_argument("--check", nargs="?", const="data/visdrone", default=None,
                        metavar="DATA_DIR", help="Check dataset integrity (default: data/visdrone)")
    args = parser.parse_args()

    if args.download:
        print("=" * 50)
        print(" VisDrone2019 Download")
        print("=" * 50)
        print()
        print("Due to size (~25GB), automatic download is not supported.")
        print()
        print("Please download manually from:")
        print("  1. Official: https://github.com/VisDrone/VisDrone-Dataset")
        print("  2. Kaggle:   https://www.kaggle.com/datasets/jamMan/visdrone2019-det")
        print("  3. Baidu Netdisk (百度网盘): See official repo for links")
        print()
        print("After downloading, run:")
        print(f"  python tools/prepare_visdrone.py --convert /path/to/VisDrone2019-DET")
        return

    if args.convert:
        convert_dataset(args.convert, args.out)
        return

    if args.check is not None:
        check_dataset(args.check)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
