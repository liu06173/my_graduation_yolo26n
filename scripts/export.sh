#!/usr/bin/env bash
# ============================================================
#  模型导出脚本 (ONNX / TensorRT / TFLite / CoreML)
#  用法:
#    bash scripts/export.sh                    # 导出ONNX
#    bash scripts/export.sh --format engine    # 导出TensorRT
#    bash scripts/export.sh --format tflite    # 导出TFLite
#    bash scripts/export.sh --format onnx --half  # FP16 ONNX
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
        echo "[ERROR] No trained weights found. Train first: bash scripts/train.sh" >&2
        exit 1
    fi
fi

FORMAT="onnx"
HALF=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --format) FORMAT="$2"; shift 2 ;;
        --half)   HALF="half=True"; shift ;;
        *)        shift ;;
    esac
done

echo "=========================================="
echo " YOLO26n Export"
echo "=========================================="
echo "  Weights: $WEIGHT"
echo "  Format:  $FORMAT"
echo "  FP16:    ${HALF:-False}"
echo "=========================================="
echo ""

exec yolo export \
    model="$WEIGHT" \
    format="$FORMAT" \
    imgsz=640 \
    device=0 \
    $HALF
