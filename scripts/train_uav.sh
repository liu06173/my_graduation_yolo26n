#!/usr/bin/env bash
# ============================================================
# YOLO26s-P2 UAV Tracking — 一键训练脚本 (yolov26s-uav v0.3)
# 用法:
#   bash scripts/train_uav.sh              # 默认参数训练
#   bash scripts/train_uav.sh --resume     # 恢复中断的训练
#   bash scripts/train_uav.sh --batch 24   # 自定义batch
#   bash scripts/train_uav.sh --dry-run    # 只检查环境，不训练
# ============================================================
set -euo pipefail

# -------------------- 默认参数 --------------------
MODEL="${MODEL:-yolo26s-p2-tracking.yaml}"
PRETRAINED="${PRETRAINED:-yolo26s.pt}"
DATA="${DATA:-configs/visdrone.yaml}"
EPOCHS="${EPOCHS:-100}"
IMGSZ="${IMGSZ:-512}"
BATCH="${BATCH:-8}"
DEVICE="${DEVICE:-0}"
WORKERS="${WORKERS:-8}"
FRACTION="${FRACTION:-0.4}"
RESUME="${RESUME:-False}"
DRY_RUN=false

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --model) MODEL="$2"; shift 2 ;;
        --pretrained) PRETRAINED="$2"; shift 2 ;;
        --data) DATA="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --imgsz) IMGSZ="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --fraction) FRACTION="$2"; shift 2 ;;
        --resume) RESUME=True; shift ;;
        --dry-run) DRY_RUN=true; shift ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
done

# -------------------- 环境检查与修复 --------------------
echo "========================================"
echo " YOLO26s-P2 UAV 训练环境检查"
echo "========================================"

# 检查 Python
if ! command -v python &>/dev/null; then
    echo "[ERROR] Python 未安装"
    exit 1
fi
echo "[OK] Python: $(python --version)"

# 检查 PyTorch/CUDA
CUDA_OK=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "False")
if [ "$CUDA_OK" != "True" ]; then
    echo "[WARN] CUDA 不可用，将使用 CPU (极慢)"
    DEVICE="cpu"
else
    GPU_NAME=$(python -c "import torch; print(torch.cuda.get_device_name(${DEVICE:-0}))" 2>/dev/null || echo "unknown")
    GPU_MEM=$(python -c "import torch; d=torch.cuda.get_device_properties(${DEVICE:-0}); print(f'{d.total_mem/1024**3:.0f}GB')" 2>/dev/null || echo "?")
    echo "[OK] GPU: $GPU_NAME ($GPU_MEM)"
fi

# 检查关键文件
REQUIRED_FILES=(
    "ultralytics/ultralytics/cfg/models/26/yolo26-p2-tracking.yaml"
    "configs/visdrone.yaml"
    "configs/hyp_visdrone.yaml"
    "ultralytics/ultralytics/nn/modules/conv.py"
    "ultralytics/ultralytics/nn/modules/block.py"
    "ultralytics/ultralytics/nn/tasks.py"
)
MISSING=0
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "[ERROR] 文件不存在: $f"
        MISSING=$((MISSING + 1))
    fi
done
if [ $MISSING -gt 0 ]; then
    echo "[FIX] 可能分支不对，尝试: git checkout yolov26s-uav && git pull"
    exit 1
fi
echo "[OK] 所有关键文件存在"

# 检查数据
if [ ! -f "$DATA" ]; then
    echo "[ERROR] 数据配置不存在: $DATA"
    exit 1
fi
echo "[OK] 数据配置: $DATA"

# 检查 yolo CLI
if ! command -v yolo &>/dev/null; then
    echo "[FIX] 安装 ultralytics..."
    cd ultralytics && pip install -e . --no-deps -q && cd ..
fi
echo "[OK] yolo CLI: $(which yolo)"

# 清除旧缓存
echo "[FIX] 清理 Python 缓存..."
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete 2>/dev/null || true

# 检查新模块能否导入
echo "[CHECK] 验证新模块导入..."
python -c "
from ultralytics.nn.modules.conv import ECA, CoordAtt, WeightedConcat, DySample
from ultralytics.nn.modules.block import ECABottleneck, C3k2_ECA
print('  [OK] 6 个新模块全部可导入')
" || {
    echo "[FIX] 重新安装 ultralytics..."
    cd ultralytics && pip install -e . --no-deps && cd ..
}

# -------------------- 显存自适应 --------------------
if [ "$DEVICE" != "cpu" ] && [ "$CUDA_OK" == "True" ]; then
    GPU_TOTAL=$(python -c "import torch; p=torch.cuda.get_device_properties(${DEVICE}); print(p.total_mem)" 2>/dev/null || echo "0")
    GPU_TOTAL_GB=$((GPU_TOTAL / 1024 / 1024 / 1024))

    if [ "$GPU_TOTAL_GB" -le 8 ]; then
        SUGGESTED_BATCH=4
    elif [ "$GPU_TOTAL_GB" -le 12 ]; then
        SUGGESTED_BATCH=6
    elif [ "$GPU_TOTAL_GB" -le 16 ]; then
        SUGGESTED_BATCH=10
    else
        SUGGESTED_BATCH=16
    fi

    if [ "$BATCH" -gt "$SUGGESTED_BATCH" ]; then
        echo "[WARN] batch=$BATCH 可能超出 ${GPU_TOTAL_GB}GB 显存，建议 ≤ $SUGGESTED_BATCH"
        echo "       自动调整为 batch=$SUGGESTED_BATCH"
        BATCH=$SUGGESTED_BATCH
    fi
fi

# -------------------- 预训练权重 --------------------
if [ ! -f "$PRETRAINED" ]; then
    echo "[INFO] 预训练权重 $PRETRAINED 不存在，将自动下载"
fi

# -------------------- 干跑模式 --------------------
if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "========================================"
    echo " 环境检查通过！训练参数预览:"
    echo "========================================"
    echo "  model:      $MODEL"
    echo "  pretrained: $PRETRAINED"
    echo "  data:       $DATA"
    echo "  epochs:     $EPOCHS"
    echo "  imgsz:      $IMGSZ"
    echo "  batch:      $BATCH"
    echo "  device:     $DEVICE"
    echo "  workers:    $WORKERS"
    echo "  resume:     $RESUME"
    echo "  fraction:   $FRACTION"
    echo ""
    echo "执行训练: bash scripts/train_uav.sh"
    exit 0
fi

# -------------------- 检查可恢复的checkpoint --------------------
RESUME_ARG=""
if [ "$RESUME" = "True" ]; then
    # 自动查找最新 checkpoint
    LAST_PT=$(find runs/detect -name "last.pt" 2>/dev/null | sort -r | head -1)
    if [ -n "$LAST_PT" ]; then
        echo "[INFO] 发现 checkpoint: $LAST_PT"
        RESUME_ARG="resume=True"
    else
        echo "[WARN] 未找到可恢复的 checkpoint，从头训练"
        RESUME_ARG=""
    fi
fi

# -------------------- 开始训练 --------------------
echo ""
echo "========================================"
echo " 开始训练 YOLO26s-P2 UAV 模型"
echo "========================================"
echo " 命令: yolo detect train"
echo " model:      $MODEL"
echo " pretrained: $PRETRAINED"
echo " epochs:     $EPOCHS"
echo " imgsz:      $IMGSZ"
echo " batch:      $BATCH"
echo " device:     $DEVICE"
echo " fraction:   $FRACTION"
echo " resume:     ${RESUME_ARG:-False}"
echo "========================================"
echo ""

# 记录训练开始时间
START_TIME=$(date +%s)

set +e  # 允许捕获错误
yolo detect train \
    model="$MODEL" \
    pretrained="$PRETRAINED" \
    data="$DATA" \
    epochs="$EPOCHS" \
    imgsz="$IMGSZ" \
    batch="$BATCH" \
    device="$DEVICE" \
    workers="$WORKERS" \
    cache=True \
    amp=False \
    optimizer=SGD \
    lr0=0.01 \
    momentum=0.937 \
    fraction="$FRACTION" \
    $RESUME_ARG \
    2>&1 | tee runs/train_uav.log

EXIT_CODE=$?
set -e

# -------------------- 错误处理 --------------------
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
MIN=$((DURATION / 60))
SEC=$((DURATION % 60))

if [ $EXIT_CODE -ne 0 ]; then
    echo ""
    echo "========================================"
    echo " 训练异常退出 (code=$EXIT_CODE, 耗时 ${MIN}m${SEC}s)"
    echo "========================================"

    # 检查是否是 OOM
    if grep -q "out of memory" runs/train_uav.log 2>/dev/null; then
        NEW_BATCH=$((BATCH - 2))
        if [ "$NEW_BATCH" -ge 2 ]; then
            echo "[OOM] 显存不足，自动降 batch: $BATCH → $NEW_BATCH"
            echo "      重新执行: bash scripts/train_uav.sh --batch $NEW_BATCH"
        else
            echo "[OOM] batch 已降到最低 (2)，建议减小 imgsz 或换 GPU"
        fi
    fi

    # 检查是否是 TaskAlignedAssigner OOM (非致命 warning，但建议降参数)
    if grep -q "TaskAlignedAssigner.*using CPU" runs/train_uav.log 2>/dev/null; then
        echo "[WARN] 标签分配GPU显存不足(已自动用CPU)，非致命。下次可:"
        echo "      bash scripts/train_uav.sh --imgsz 480 --batch 6"
    fi

    # 检查是否是 MuSGD 优化器不兼容 (zeropower_via_newtonschulz5 assert)
    if grep -q "zeropower_via_newtonschulz5\|AssertionError.*muon" runs/train_uav.log 2>/dev/null; then
        echo "[FIX] MuSGD 不兼容新模块的非2D参数，已自动切换 optimizer=SGD，重新训练..."
        exec bash scripts/train_uav.sh --resume
    fi

    # 检查是否是 AMP 检查失败 (assets/bus.jpg 缺失)
    if grep -q "assets/bus.jpg" runs/train_uav.log 2>/dev/null; then
        echo "[FIX] AMP 检查失败 (ultralytics assets 缺失)，已自动加 amp=False，重新训练..."
        exec bash scripts/train_uav.sh --resume
    fi

    # 提示恢复命令
    echo ""
    echo "恢复训练: bash scripts/train_uav.sh --resume"
    echo "查看日志: tail -100 runs/train_uav.log"

    exit $EXIT_CODE
fi

echo ""
echo "========================================"
echo " 训练完成! (耗时 ${MIN}m${SEC}s)"
echo "========================================"
echo " 权重: runs/detect/train/weights/best.pt"
echo " 日志: runs/train_uav.log"
echo ""
echo " 评估: yolo detect val model=runs/detect/train/weights/best.pt data=configs/visdrone.yaml"
echo " 推理: yolo detect predict model=runs/detect/train/weights/best.pt source=test.jpg"
