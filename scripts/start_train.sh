#!/usr/bin/env bash
# ============================================================
#  一键启动训练 (适用于Cloud Studio / 远程服务器)
#  用法:
#    bash scripts/start_train.sh              # 默认参数训练
#    bash scripts/start_train.sh --resume     # 恢复训练
#    bash scripts/start_train.sh --p2         # 使用P2模型 (小目标优化)
#    nohup bash scripts/start_train.sh > train.log 2>&1 &  # 后台训练
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# ---------- 参数 ----------
MODE="new"
MODEL="yolo26n.pt"
EPOCHS=300
BATCH=32
IMGSZ=640
DEVICE="0"

while [[ $# -gt 0 ]]; do
    case $1 in
        --resume)    MODE="resume"; shift ;;
        --p2)        MODEL="yolo26n-p2.pt"; shift ;;
        --batch)     BATCH="$2"; shift 2 ;;
        --epochs)    EPOCHS="$2"; shift 2 ;;
        --imgsz)     IMGSZ="$2"; shift 2 ;;
        --device)    DEVICE="$2"; shift 2 ;;
        *)           shift ;;
    esac
done

echo -e "${GREEN}=========================================="
echo -e " YOLO26n 无人机目标检测训练"
echo -e "=========================================="
echo -e "  模型:     $MODEL"
echo -e "  模式:     $MODE"
echo -e "  Epochs:   $EPOCHS"
echo -e "  Batch:    $BATCH"
echo -e "  分辨率:   $IMGSZ"
echo -e "  GPU:      $DEVICE"
echo -e "==========================================${NC}"

# ---------- 激活环境 ----------
if command -v conda &> /dev/null && conda env list | grep -q "yolo26_uav"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate yolo26_uav
    echo -e "${GREEN}  环境: conda/yolo26_uav${NC}"
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
    echo -e "${GREEN}  环境: venv${NC}"
fi

# ---------- 检查数据 ----------
if [ ! -d "data/visdrone/images/train" ]; then
    echo -e "${RED}[ERROR] VisDrone数据集未就绪${NC}"
    echo "  请先运行: bash scripts/fetch_data.sh"
    exit 1
fi

# ---------- 检查GPU ----------
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'" 2>/dev/null || {
    echo -e "${RED}[ERROR] CUDA不可用，请检查GPU环境${NC}"
    echo "  如果确实无GPU，使用: bash scripts/start_train.sh --device cpu --batch 4"
    exit 1
}

# ---------- 显示GPU信息 ----------
echo ""
echo -e "${YELLOW}  GPU状态:${NC}"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv 2>/dev/null | head -3 || true
echo ""

# ---------- 启动训练 ----------
if [ "$MODE" = "resume" ]; then
    # 找最近的checkpoint
    CKPT=$(find runs/detect -name "last.pt" 2>/dev/null | sort -r | head -1)
    if [ -z "$CKPT" ]; then
        echo -e "${RED}[ERROR] 未找到checkpoint，无法恢复训练${NC}"
        exit 1
    fi
    echo -e "${YELLOW}  从checkpoint恢复: $CKPT${NC}"

    exec yolo detect train \
        model="$CKPT" resume=True \
        data=configs/visdrone.yaml \
        epochs=$EPOCHS \
        imgsz=$IMGSZ \
        batch=$BATCH \
        device=$DEVICE \
        workers=8 \
        project=runs/detect \
        name=train \
        exist_ok=True \
        save=True \
        save_period=10 \
        val=True \
        plots=True
else
    exec yolo detect train \
        model=$MODEL \
        data=configs/visdrone.yaml \
        epochs=$EPOCHS \
        imgsz=$IMGSZ \
        batch=$BATCH \
        device=$DEVICE \
        workers=8 \
        project=runs/detect \
        name=train \
        exist_ok=True \
        save=True \
        save_period=10 \
        val=True \
        plots=True
fi
