#!/usr/bin/env bash
# ============================================================
#  VisDrone2019 数据获取脚本 (bypy 百度网盘)
#  用法:
#    bash scripts/fetch_data.sh                    # 交互式下载+转换
#    bash scripts/fetch_data.sh --remote /VisDrone  # 指定网盘路径
#    bash scripts/fetch_data.sh --skip-download     # 跳过下载，仅转换
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

REMOTE_DIR="/VisDrone2019"
SKIP_DOWNLOAD=false
RAW_DIR="data/VisDrone_raw"
OUT_DIR="data/visdrone"

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_DIR="$2"; shift 2 ;;
        --skip-download) SKIP_DOWNLOAD=true; shift ;;
        *) shift ;;
    esac
done

echo -e "${GREEN}=========================================="
echo -e " VisDrone2019 数据获取"
echo -e "==========================================${NC}"

# ---------- 激活环境 ----------
if command -v conda &> /dev/null && conda env list | grep -q "yolo26_uav"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate yolo26_uav
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ---------- 下载 ----------
if [ "$SKIP_DOWNLOAD" = false ]; then
    echo -e "\n${YELLOW}[1/3] 通过bypy下载数据集...${NC}"

    # 检查bypy是否可用
    if ! command -v bypy &> /dev/null; then
        echo -e "${RED}[ERROR] bypy未安装，请先运行: bash scripts/setup_env.sh${NC}"
        exit 1
    fi

    # 测试bypy授权
    if ! bypy info 2>/dev/null | grep -q "Quota"; then
        echo -e "${RED}[ERROR] bypy未授权。请运行 'bypy info' 并按提示完成授权${NC}"
        echo "  1. bypy info"
        echo "  2. 复制打开的链接到浏览器"
        echo "  3. 登录百度账号"
        echo "  4. 复制授权码贴回终端"
        exit 1
    fi

    echo "  远程路径: $REMOTE_DIR"
    echo "  本地目录: $RAW_DIR"

    # 列出远程目录确认路径正确
    echo ""
    echo "  远程目录内容:"
    bypy list "$REMOTE_DIR" 2>/dev/null || {
        echo -e "${RED}  找不到远程目录 $REMOTE_DIR${NC}"
        echo "  请确认VisDrone数据集在百度网盘中的路径"
        echo "  例如: bash scripts/fetch_data.sh --remote /数据集/VisDrone2019"
        exit 1
    }

    mkdir -p "$RAW_DIR"

    # 尝试找训练集和验证集
    echo ""
    echo "  开始下载 (约8.5GB，视网速可能需要30-60分钟)..."

    # 下载训练集
    TRAIN_DIR=""
    for name in "VisDrone2019-DET-train" "DET-train" "train"; do
        if bypy list "$REMOTE_DIR" 2>/dev/null | grep -q "$name"; then
            TRAIN_DIR="$name"
            break
        fi
    done

    if [ -n "$TRAIN_DIR" ]; then
        echo "  下载训练集: $REMOTE_DIR/$TRAIN_DIR → $RAW_DIR/$TRAIN_DIR"
        bypy downdir "$REMOTE_DIR/$TRAIN_DIR" "$RAW_DIR/$TRAIN_DIR" \
            --processes 4 --downloader aria2 2>/dev/null || {
            # aria2不可用时用默认下载器
            bypy downdir "$REMOTE_DIR/$TRAIN_DIR" "$RAW_DIR/$TRAIN_DIR"
        }
    fi

    # 下载验证集
    VAL_DIR=""
    for name in "VisDrone2019-DET-val" "DET-val" "val"; do
        if bypy list "$REMOTE_DIR" 2>/dev/null | grep -q "$name"; then
            VAL_DIR="$name"
            break
        fi
    done

    if [ -n "$VAL_DIR" ]; then
        echo "  下载验证集: $REMOTE_DIR/$VAL_DIR → $RAW_DIR/$VAL_DIR"
        bypy downdir "$REMOTE_DIR/$VAL_DIR" "$RAW_DIR/$VAL_DIR"
    fi

    echo -e "  下载完成"
else
    echo -e "\n${YELLOW}[1/3] 跳过下载 (--skip-download)${NC}"
fi

# ---------- 转换 ----------
echo -e "\n${YELLOW}[2/3] 转换为YOLO格式...${NC}"

if [ ! -d "$RAW_DIR" ]; then
    echo -e "${RED}[ERROR] 原始数据目录不存在: $RAW_DIR${NC}"
    echo "  请先下载数据或指定正确的目录"
    echo "  用法: bash scripts/fetch_data.sh"
    exit 1
fi

python tools/prepare_visdrone.py --convert "$RAW_DIR" --out "$OUT_DIR"

# ---------- 验证 ----------
echo -e "\n${YELLOW}[3/3] 验证数据集完整性...${NC}"
python tools/prepare_visdrone.py --check

# ---------- 完成 ----------
echo -e "\n${GREEN}=========================================="
echo -e " 数据准备完成!"
echo -e "==========================================${NC}"
echo ""
echo -e "  训练集图片: $(find $OUT_DIR/images/train -name '*.jpg' 2>/dev/null | wc -l)"
echo -e "  验证集图片: $(find $OUT_DIR/images/val -name '*.jpg' 2>/dev/null | wc -l)"
echo ""
echo -e "  下一步: bash scripts/start_train.sh"
echo ""
