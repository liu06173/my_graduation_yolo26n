#!/usr/bin/env bash
# ============================================================
#  Cloud Studio / 服务器 环境一键配置脚本
#  支持: conda / venv / 直接安装 三种方式
#  用法: bash scripts/setup_env.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}=========================================="
echo -e " YOLO26n 无人机目标跟踪 — 环境配置"
echo -e "==========================================${NC}"

# ---------- GPU 检测 ----------
echo -e "\n${YELLOW}[1/5] 检测GPU环境...${NC}"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
    HAS_GPU=true
else
    echo "  未检测到NVIDIA GPU (将使用CPU模式)"
    HAS_GPU=false
fi

# ---------- Conda 环境 ----------
if command -v conda &> /dev/null; then
    echo -e "\n${YELLOW}[2/5] 通过Conda创建环境...${NC}"
    if conda env list | grep -q "yolo26_uav"; then
        echo "  环境已存在，跳过创建"
    else
        conda env create -f environment.yml
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate yolo26_uav

# ---------- venv 降级 ----------
elif command -v python3 &> /dev/null; then
    echo -e "\n${YELLOW}[2/5] 通过venv创建环境...${NC}"
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
    pip install -r requirements.txt

else
    echo -e "${RED}[ERROR] 未找到Python，请先安装Python 3.10+${NC}"
    exit 1
fi

# ---------- PyTorch CUDA 验证 ----------
echo -e "\n${YELLOW}[3/5] 验证PyTorch + CUDA...${NC}"
python -c "
import torch
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU:     {torch.cuda.get_device_name(0)}')
    print(f'  VRAM:    {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
"

# ---------- Ultralytics ----------
echo -e "\n${YELLOW}[4/5] 安装Ultralytics...${NC}"
pip install ultralytics 2>/dev/null && echo "  Ultralytics installed from PyPI" || {
    # 从本地源码安装
    if [ -d "ultralytics" ]; then
        cd ultralytics && pip install -e . && cd ..
        echo "  Ultralytics installed from local source"
    fi
}

# ---------- bypy 配置 ----------
echo -e "\n${YELLOW}[5/5] 配置bypy (百度网盘)...${NC}"
if command -v bypy &> /dev/null; then
    echo "  bypy 已安装，版本: $(bypy --version)"
    echo ""
    echo "  ⚠ 首次使用需要授权:"
    echo "    1. 运行: bypy info"
    echo "    2. 在浏览器打开提示的链接"
    echo "    3. 复制授权码粘贴回终端"
else
    pip install bypy && echo "  bypy 安装完成"
fi

# ---------- 完成 ----------
echo -e "\n${GREEN}=========================================="
echo -e " 环境配置完成!"
echo -e "==========================================${NC}"
echo ""
echo -e "  下一步:"
echo -e "    bash scripts/fetch_data.sh    # 下载VisDrone数据集"
echo -e "    bash scripts/start_train.sh   # 一键开始训练"
echo ""
