#!/usr/bin/env bash
# ============================================================
#  YOLO26n + VisDrone2019 一键环境配置脚本
#  适用: Linux / macOS / Windows(WSL)
#  用法: bash setup.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " YOLO26n + VisDrone2019 环境配置"
echo "=========================================="

# ---------- 1. 检测Python环境 ----------
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "[ERROR] Python not found. Please install Python 3.10+"
    exit 1
fi

echo "[1/5] Python: $($PYTHON --version)"

# ---------- 2. 创建虚拟环境(可选) ----------
if [ ! -d "venv" ]; then
    echo "[2/5] Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "[2/5] Virtual environment already exists"
fi

source venv/bin/activate 2>/dev/null || source venv/Scripts/activate 2>/dev/null

# ---------- 3. 安装PyTorch ----------
echo "[3/5] Installing PyTorch..."
python -c "import torch" 2>/dev/null || {
    if command -v nvidia-smi &> /dev/null; then
        echo "  -> Installing CUDA PyTorch..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 2>/dev/null || \
        pip install torch torchvision 2>/dev/null
    else
        echo "  -> Installing CPU PyTorch..."
        pip install torch torchvision 2>/dev/null
    fi
}

# ---------- 4. 安装Ultralytics ----------
echo "[4/5] Installing ultralytics from local source..."
cd "$SCRIPT_DIR/ultralytics"
pip install -e . 2>/dev/null || pip install ultralytics
cd "$SCRIPT_DIR"

# ---------- 5. 安装其他依赖 ----------
echo "[5/5] Installing additional dependencies..."
pip install \
    opencv-python-headless>=4.9.0 \
    numpy>=1.23.0 \
    matplotlib>=3.7.0 \
    pandas>=2.0.0 \
    pyyaml>=6.0 \
    tqdm>=4.65.0 \
    scipy>=1.11.0 \
    psutil>=5.9.0 \
    tensorboard>=2.13.0 \
    seaborn>=0.12.0 \
    2>/dev/null

# ---------- 验证 ----------
echo ""
echo "=========================================="
echo " 环境验证"
echo "=========================================="
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.cuda.is_available()} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"})')
"
python -c "import ultralytics; print(f'  Ultralytics: {ultralytics.__version__}')" 2>/dev/null || echo "  Ultralytics: installed from local source"

echo ""
echo "=========================================="
echo " 配置完成!"
echo "=========================================="
echo ""
echo " 下一步:"
echo "  1. 下载VisDrone2019数据集:"
echo "     bash scripts/download_data.sh"
echo ""
echo "  2. 开始训练:"
echo "     bash scripts/train.sh"
echo ""
