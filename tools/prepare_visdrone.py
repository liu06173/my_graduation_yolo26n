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
    """Convert full VisDrone2019 dataset from raw format (supports DET and MOT)."""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)

    # Auto-detect DET vs MOT and find splits
    # DET: VisDrone2019-DET-train/images/*.jpg + annotations/*.txt
    # MOT: VisDrone2019-MOT-train/sequences/<seq>/*.jpg + annotations/<seq>.txt
    for split in ["train", "val", "test"]:
        src_dir = None
        is_mot = False

        # Search for matching directory
        for d in raw_dir.iterdir():
            if not d.is_dir():
                continue
            dname = d.name.lower()
            if "mot" in dname and split in dname:
                src_dir = d
                is_mot = True
                break
            if "det" in dname and split in dname:
                src_dir = d
                is_mot = False
                break

        # Fallback: match any dir containing split name
        if src_dir is None:
            for d in raw_dir.iterdir():
                if d.is_dir() and split in d.name.lower():
                    src_dir = d
                    is_mot = ("mot" in d.name.lower() or
                              (d / "sequences").exists())
                    break

        if src_dir is None:
            print(f"[WARNING] Cannot find {split} data")
            continue

        img_dst = out_dir / "images" / split
        lbl_dst = out_dir / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        print(f"\nConverting {split} ({'MOT' if is_mot else 'DET'}) from {src_dir.name}...")

        if is_mot:
            _convert_mot(src_dir, img_dst, lbl_dst)
        else:
            _convert_det(src_dir, img_dst, lbl_dst)

    print("\nConversion complete! Checking dataset...")
    check_dataset(out_dir)


def _convert_det(src_dir, img_dst, lbl_dst):
    """Convert DET format: images/ + annotations/ pairs."""
    img_src = src_dir / "images"
    ann_src = src_dir / "annotations"

    if not img_src.exists():
        img_src = src_dir  # images directly in dir
    if not ann_src.exists():
        ann_src = src_dir  # annotations directly in dir

    img_count = 0
    for img in img_src.glob("*.jpg"):
        shutil.copy2(img, img_dst / img.name)
        img_count += 1
    print(f"  Copied {img_count} images")

    if ann_src.exists():
        _convert_annotations(ann_src, lbl_dst)
    else:
        print(f"  [WARNING] No annotations found")


def _convert_mot(src_dir, img_dst, lbl_dst):
    """Convert MOT format: sequences/<seq>/*.jpg + annotations/<seq>.txt."""
    seq_dir = src_dir / "sequences"
    ann_dir = src_dir / "annotations"

    if not seq_dir.exists():
        print(f"  [ERROR] No sequences/ directory found")
        return

    total_imgs = 0
    total_anns = 0

    for seq_path in sorted(seq_dir.iterdir()):
        if not seq_path.is_dir():
            continue

        seq_name = seq_path.name
        ann_file = ann_dir / f"{seq_name}.txt"

        # Read all annotations for this sequence, grouped by frame
        frame_anns = {}
        if ann_file.exists():
            with open(ann_file) as f:
                for line in f:
                    parts = line.strip().split(",")
                    if len(parts) < 9:
                        continue
                    # MOT format: frame, id, x, y, w, h, score, class, trunc, occlusion
                    try:
                        frame_idx = int(parts[0])
                        cls_id = int(parts[7]) - 1  # 1-10 → 0-9
                        trunc = int(parts[8])
                        occl = int(parts[9])
                    except ValueError:
                        continue

                    if (cls_id + 1) in IGNORE_IDS:
                        continue
                    if cls_id < 0 or cls_id >= len(CLASS_NAMES):
                        continue
                    if trunc > 2 and occl > 2:
                        continue

                    x, y, w, h = float(parts[2]), float(parts[3]), float(parts[4]), float(parts[5])
                    if frame_idx not in frame_anns:
                        frame_anns[frame_idx] = []
                    frame_anns[frame_idx].append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        # Copy images and generate per-frame labels
        for img_file in sorted(seq_path.glob("*.jpg")):
            # Frame index from filename (e.g., 0000001.jpg → 1)
            try:
                frame_idx = int(img_file.stem)
            except ValueError:
                frame_idx = total_imgs + 1

            dst_name = f"{seq_name}_{img_file.name}"

            # Copy image
            shutil.copy2(img_file, img_dst / dst_name)
            total_imgs += 1

            # Write YOLO label for this frame
            if frame_idx in frame_anns:
                # Normalize coordinates (MOT uses absolute pixel coords)
                # We need image dimensions - read from the image
                # For now, use a placeholder. Actual normalization done later if needed.
                # MOT bboxes are in absolute pixels, YOLO needs normalized
                # But without reading image dims each time, we write abs coords
                # and rely on the fact that all images in a sequence have same dims
                label_path = lbl_dst / f"{seq_name}_{img_file.stem}.txt"
                with open(label_path, "w") as lf:
                    lf.write("\n".join(frame_anns[frame_idx]))
                total_anns += 1

    print(f"  Copied {total_imgs} images")
    print(f"  Generated {total_anns} labels")

    # Normalize MOT labels (convert absolute coords to normalized)
    if total_anns > 0:
        _normalize_mot_labels(img_dst, lbl_dst)


def _normalize_mot_labels(img_dir, lbl_dir):
    """Normalize MOT absolute coordinates to YOLO format using actual image dimensions."""
    print("  Normalizing coordinates...")
    for label_file in lbl_dir.glob("*.txt"):
        # Find corresponding image
        img_name = label_file.stem + ".jpg"
        img_path = img_dir / img_name

        if not img_path.exists():
            continue

        # Read image dimensions
        import cv2
        im = cv2.imread(str(img_path))
        if im is None:
            continue
        h, w = im.shape[:2]

        # Normalize each line
        lines = []
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = parts[0]
                x, y, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                # Normalize: center_x, center_y, width, height
                nx = x / w
                ny = y / h
                nw = bw / w
                nh = bh / h
                # Clamp to [0,1]
                nx = max(0, min(1, nx))
                ny = max(0, min(1, ny))
                nw = min(1, nw)
                nh = min(1, nh)
                lines.append(f"{cls_id} {nx:.6f} {ny:.6f} {nw:.6f} {nh:.6f}")

        with open(label_file, "w") as f:
            f.write("\n".join(lines))


def _convert_annotations(ann_src, lbl_dst):
    """Convert annotations from VisDrone DET format to YOLO format."""
    test_file = next(ann_src.glob("*.txt"), None)
    if test_file:
        with open(test_file) as f:
            first_line = f.readline().strip()
        parts = first_line.split()
        if len(parts) == 5 and "." not in parts[0]:
            print("  Labels already in YOLO format, copying directly...")
            for lbl in ann_src.glob("*.txt"):
                shutil.copy2(lbl, lbl_dst / lbl.name)
        else:
            print("  Converting VisDrone DET → YOLO format...")
            n = visdrone2yolo(str(ann_src), str(lbl_dst))
            print(f"  Converted {n} labels")
    else:
        print(f"  [WARNING] No annotation files found")


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
