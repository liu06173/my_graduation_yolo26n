#!/usr/bin/env bash
# ============================================================
#  YOLO26n VisDrone2019 训练脚本
#  用法:
#    bash scripts/train.sh              # 新建训练
#    bash scripts/train.sh --resume     # 从最近checkpoint恢复
#    bash scripts/train.sh --resume best.pt  # 从指定权重恢复
#    bash scripts/train.sh --device 0,1 --batch 64  # 多卡训练
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

# ---------- 参数解析 ----------
RESUME=false
RESUME_WEIGHT=""
DEVICE="0"
BATCH=32
EPOCHS=300
IMG_SIZE=640
CUSTOM_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)
            RESUME=true
            shift
            # check if next arg is a weight file
            if [[ $# -gt 0 && "$1" != --* ]]; then
                RESUME_WEIGHT="$1"
                shift
            fi
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --batch)
            BATCH="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --imgsz)
            IMG_SIZE="$2"
            shift 2
            ;;
        *)
            CUSTOM_ARGS="$CUSTOM_ARGS $1"
            shift
            ;;
    esac
done

# ---------- 激活环境 ----------
[ -f "venv/bin/activate" ] && source venv/bin/activate
[ -f "venv/Scripts/activate" ] && source venv/Scripts/activate

# ---------- 检查环境 ----------
python -c "import torch; import ultralytics" 2>/dev/null || {
    echo "[ERROR] Environment not setup. Run: bash setup.sh" >&2
    exit 1
}

# ---------- 检查数据 ----------
if [ ! -d "data/visdrone/images/train" ]; then
    echo "[WARNING] VisDrone dataset not found!"
    echo "  Please run: bash scripts/download_data.sh"
    echo "  Or manually place VisDrone2019 images in data/visdrone/"
    echo ""
fi

# ---------- 显示配置 ----------
echo "=========================================="
echo " YOLO26n VisDrone2019 Training"
echo "=========================================="
echo "  Model:      yolo26n"
echo "  Dataset:    VisDrone2019 (10 classes)"
echo "  Device:     $DEVICE"
echo "  Batch:      $BATCH"
echo "  Epochs:     $EPOCHS"
echo "  Image Size: $IMG_SIZE"
echo "  Resume:     $RESUME"
echo "=========================================="
echo ""

# ---------- 训练命令 ----------
if [ "$RESUME" = true ]; then
    echo "[Mode] RESUME training"
    if [ -n "$RESUME_WEIGHT" ]; then
        echo "  Resuming from: $RESUME_WEIGHT"
        RESUME_ARG="model=$RESUME_WEIGHT resume=True"
    else
        # Auto-find latest checkpoint
        if [ -f "runs/detect/train/weights/last.pt" ]; then
            echo "  Resuming from: runs/detect/train/weights/last.pt"
            RESUME_ARG="model=runs/detect/train/weights/last.pt resume=True"
        elif [ -f "runs/detect/train2/weights/last.pt" ]; then
            echo "  Resuming from: runs/detect/train2/weights/last.pt"
            RESUME_ARG="model=runs/detect/train2/weights/last.pt resume=True"
        else
            echo "[ERROR] No checkpoint found to resume from." >&2
            echo "  Start fresh training: bash scripts/train.sh" >&2
            exit 1
        fi
    fi
else
    echo "[Mode] NEW training"
    RESUME_ARG="model=yolo26n.pt"
fi

# ---------- 启动训练 ----------
exec yolo detect train \
    $RESUME_ARG \
    data=configs/visdrone.yaml \
    epochs=$EPOCHS \
    imgsz=$IMG_SIZE \
    batch=$BATCH \
    device=$DEVICE \
    workers=8 \
    project=runs/detect \
    name=train \
    exist_ok=True \
    save=True \
    save_period=10 \
    val=True \
    plots=True \
    $CUSTOM_ARGS
