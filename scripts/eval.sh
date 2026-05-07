#!/usr/bin/env bash
# ============================================================
#  模型评估脚本
#  用法:
#    bash scripts/eval.sh                           # 评估最新权重
#    bash scripts/eval.sh runs/detect/train/weights/best.pt  # 指定权重
#    bash scripts/eval.sh best.pt --device 0 --batch 64
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---------- 激活环境 ----------
[ -f "venv/bin/activate" ] && source venv/bin/activate
[ -f "venv/Scripts/activate" ] && source venv/Scripts/activate

# ---------- 查找权重 ----------
WEIGHT="${1:-runs/detect/train/weights/best.pt}"
shift 2>/dev/null || true

if [ ! -f "$WEIGHT" ]; then
    # Try to find best.pt
    WEIGHT=$(find runs/detect -name "best.pt" 2>/dev/null | sort -r | head -1)
    if [ -z "$WEIGHT" ]; then
        echo "[ERROR] No model weights found. Train first: bash scripts/train.sh" >&2
        exit 1
    fi
fi

echo "=========================================="
echo " YOLO26n VisDrone2019 Evaluation"
echo "=========================================="
echo "  Weights: $WEIGHT"
echo "=========================================="
echo ""

exec yolo detect val \
    model="$WEIGHT" \
    data=configs/visdrone.yaml \
    imgsz=640 \
    batch=64 \
    device=0 \
    save_json=True \
    save_hybrid=True \
    plots=True \
    project=runs/detect \
    name=eval \
    exist_ok=True \
    "$@"
