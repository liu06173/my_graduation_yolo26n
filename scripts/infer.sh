#!/usr/bin/env bash
# ============================================================
#  模型推理脚本
#  用法:
#    bash scripts/infer.sh image.jpg                    # 单张图片
#    bash scripts/infer.sh video.mp4                    # 视频推理
#    bash scripts/infer.sh data/visdrone/images/val     # 批量推理
#    bash scripts/infer.sh --webcam                     # 摄像头实时推理
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---------- 激活环境 ----------
[ -f "venv/bin/activate" ] && source venv/bin/activate
[ -f "venv/Scripts/activate" ] && source venv/Scripts/activate

# ---------- 查找权重 ----------
WEIGHT="runs/detect/train/weights/best.pt"
if [ ! -f "$WEIGHT" ]; then
    WEIGHT=$(find runs/detect -name "best.pt" 2>/dev/null | sort -r | head -1)
    if [ -z "$WEIGHT" ]; then
        echo "[WARNING] No trained weights found, using pretrained yolo26n.pt"
        WEIGHT="yolo26n.pt"
    fi
fi

SOURCE="${1:-}"
if [ -z "$SOURCE" ]; then
    echo "Usage: bash scripts/infer.sh <image|video|dir|--webcam>" >&2
    echo "  e.g.  bash scripts/infer.sh test.jpg" >&2
    echo "  e.g.  bash scripts/infer.sh video.mp4" >&2
    echo "  e.g.  bash scripts/infer.sh data/visdrone/images/val" >&2
    echo "  e.g.  bash scripts/infer.sh --webcam" >&2
    exit 1
fi

echo "=========================================="
echo " YOLO26n Inference"
echo "=========================================="
echo "  Weights: $WEIGHT"
echo "  Source:  $SOURCE"
echo "=========================================="
echo ""

exec yolo detect predict \
    model="$WEIGHT" \
    source="$SOURCE" \
    imgsz=640 \
    conf=0.25 \
    iou=0.45 \
    device=0 \
    save=True \
    show=False \
    project=runs/detect \
    name=predict \
    exist_ok=True
