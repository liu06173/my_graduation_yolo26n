#!/usr/bin/env bash
# ============================================================
#  TrackingYOLO26 跟踪模型训练
#  用法:
#    bash scripts/train_tracking.sh          # 新建训练
#    bash scripts/train_tracking.sh --resume # 恢复训练
#    bash scripts/train_tracking.sh --p2     # P2小目标版本
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# 激活环境
if command -v conda &> /dev/null && conda env list 2>/dev/null | grep -q "yolo26_uav"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate yolo26_uav
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

MODE="new"
ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume) MODE="resume"; shift ;;
        --p2)     ARGS="$ARGS --imgsz 1280"; shift ;;
        *)        ARGS="$ARGS $1"; shift ;;
    esac
done

echo "=========================================="
echo " TrackingYOLO26 Training"
echo "=========================================="
echo "  Mode:  $MODE"
echo "  NOTE:  Ensure VisDrone-MOT data is ready"
echo "=========================================="
echo ""

if [ "$MODE" = "resume" ]; then
    CKPT=$(find runs/tracking -name "last.pt" 2>/dev/null | sort -r | head -1)
    if [ -z "$CKPT" ]; then
        echo "[ERROR] No checkpoint found"
        exit 1
    fi
    exec python tools/train_tracking.py --resume "$CKPT" $ARGS
else
    exec python tools/train_tracking.py --epochs 300 --batch 32 $ARGS
fi
