#!/usr/bin/env bash
# ============================================================
#  VisDrone2019 数据集下载和准备脚本
#  用法: bash scripts/download_data.sh
#  数据来源: https://github.com/VisDrone/VisDrone-Dataset
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

DATA_DIR="data/visdrone"
mkdir -p "$DATA_DIR"

echo "=========================================="
echo " VisDrone2019 Dataset Download"
echo "=========================================="
echo ""
echo "VisDrone2019 contains:"
echo "  - Train: 6,471 images"
echo "  - Val:   548 images"
echo "  - Test:  1,610 images"
echo "  - 10 classes (pedestrian, car, van, truck, bus, etc.)"
echo ""

# ---------- 方法1: 自动下载 ----------
echo "[Method 1] Download via Python script..."
python tools/prepare_visdrone.py --download 2>/dev/null && {
    echo "Done!"
    exit 0
}

# ---------- 方法2: 手动下载提示 ----------
echo ""
echo "[Method 2] Manual download steps:"
echo "=========================================="
echo ""
echo "1.  Download VisDrone2019-DET from:"
echo "    https://github.com/VisDrone/VisDrone-Dataset"
echo ""
echo "2.  Extract and organize into:"
echo "    data/visdrone/"
echo "    ├── images/"
echo "    │   ├── train/    (6,471 jpg files)"
echo "    │   ├── val/      (548 jpg files)"
echo "    │   └── test/     (1,610 jpg files) [optional]"
echo "    └── labels/"
echo "        ├── train/    (6,471 txt files)"
echo "        └── val/      (548 txt files)"
echo ""
echo "3.  Or run: python tools/prepare_visdrone.py --convert /path/to/VisDrone2019-DET"
echo "    to auto-convert from raw VisDrone format to YOLO format."
echo ""
