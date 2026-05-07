#!/usr/bin/env bash
# ============================================================
#  VisDrone2019 数据准备脚本（手动下载版）
#  流程: 检测压缩包 → 解压 → 转换为YOLO格式 → 验证
#
#  用法:
#    bash scripts/prepare_data.sh                  # 处理data/下所有zip
#    bash scripts/prepare_data.sh --zip xxx.zip    # 处理指定zip
#    bash scripts/prepare_data.sh --skip-extract   # 跳过解压，仅转换
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$SCRIPT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

RAW_DIR="data/VisDrone_raw"
OUT_DIR="data/visdrone"
TARGET_ZIP=""
SKIP_EXTRACT=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --zip) TARGET_ZIP="$2"; shift 2 ;;
        --skip-extract) SKIP_EXTRACT=true; shift ;;
        *) shift ;;
    esac
done

echo -e "${GREEN}=========================================="
echo -e " VisDrone2019 数据准备"
echo -e "==========================================${NC}"

# ---------- 激活环境 ----------
if command -v conda &> /dev/null && conda env list 2>/dev/null | grep -q "yolo26_uav"; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate yolo26_uav
elif [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# ---------- Step 1: 检测和解压 ----------
if [ "$SKIP_EXTRACT" = false ]; then
    echo -e "\n${YELLOW}[1/4] 检测 data/ 目录下的压缩包...${NC}"

    # 找要处理的zip文件
    if [ -n "$TARGET_ZIP" ]; then
        ZIP_FILES=("$TARGET_ZIP")
    else
        # 自动搜索 data/ 下所有 zip
        mapfile -t ZIP_FILES < <(find data/ -maxdepth 1 -name "*.zip" -type f 2>/dev/null || true)
    fi

    if [ ${#ZIP_FILES[@]} -eq 0 ]; then
        echo -e "${RED}  未找到任何 .zip 文件${NC}"
        echo ""
        echo "  请手动将VisDrone数据集下载到 data/ 目录:"
        echo "    bypy downfile /task3/VisDrone2019-MOT-train.zip ./data/"
        echo ""
        echo "  支持的数据集:"
        echo "    VisDrone2019-DET   (目标检测)"
        echo "    VisDrone2019-MOT   (多目标跟踪)"
        echo ""
        if [ -d "$RAW_DIR" ] && [ "$(ls -A "$RAW_DIR" 2>/dev/null)" ]; then
            echo -e "${YELLOW}  检测到已有原始数据，跳过解压直接转换...${NC}"
            SKIP_EXTRACT=true
        else
            exit 1
        fi
    fi

    if [ "$SKIP_EXTRACT" = false ]; then
        mkdir -p "$RAW_DIR"

        for zip_file in "${ZIP_FILES[@]}"; do
            fname=$(basename "$zip_file")
            echo ""

            # 先探测zip内的顶层目录名
            top_dir=$(unzip -l "$zip_file" 2>/dev/null | awk 'NR>3 && NF==4 {print $NF}' | head -1 | cut -d'/' -f1)
            if [ -z "$top_dir" ]; then
                top_dir=$(unzip -l "$zip_file" 2>/dev/null | awk 'NR>3 && NF==4 {print $NF}' | head -1 | awk -F'/' '{print $1}')
            fi

            # 检查是否已解压过
            if [ -n "$top_dir" ] && [ -d "$RAW_DIR/$top_dir" ] && [ "$(ls -A "$RAW_DIR/$top_dir" 2>/dev/null)" ]; then
                file_count=$(find "$RAW_DIR/$top_dir" -type f 2>/dev/null | wc -l)
                echo -e "  ${GREEN}跳过: $fname (已解压到 $RAW_DIR/$top_dir, $file_count 个文件)${NC}"
                continue
            fi

            echo -e "  ${YELLOW}解压: $fname${NC}"

            # 创建临时解压目录
            tmp_dir="$RAW_DIR/_tmp_${fname%.zip}"
            mkdir -p "$tmp_dir"

            # 解压
            unzip -qo "$zip_file" -d "$tmp_dir" && echo "    解压成功" || {
                echo -e "${RED}    解压失败: $zip_file${NC}"
                continue
            }

            # 处理嵌套目录（有些zip包进去还有一层目录）
            inner_count=$(ls -A "$tmp_dir" | wc -l)
            if [ "$inner_count" -eq 1 ]; then
                single_dir=$(ls "$tmp_dir")
                if [ -d "$tmp_dir/$single_dir" ]; then
                    # 有嵌套，移动到目标位置
                    target_name="$single_dir"
                    if [ -d "$RAW_DIR/$target_name" ]; then
                        # 增量合并
                        cp -rn "$tmp_dir/$single_dir"/* "$RAW_DIR/$target_name/" 2>/dev/null || true
                    else
                        mv "$tmp_dir/$single_dir" "$RAW_DIR/$target_name"
                    fi
                    rm -rf "$tmp_dir"
                else
                    mv "$tmp_dir" "$RAW_DIR/$fname"
                fi
            else
                mv "$tmp_dir" "$RAW_DIR/$fname"
            fi

            echo "    解压到: $RAW_DIR/$fname"
        done

        echo -e "${GREEN}  解压完成${NC}"
    fi
else
    echo -e "\n${YELLOW}[1/4] 跳过解压 (--skip-extract)${NC}"
fi

# ---------- Step 2: 检测数据类型 ----------
echo -e "\n${YELLOW}[2/4] 检测数据集结构...${NC}"

# 看目录名判断是 DET 还是 MOT
if [ -d "$RAW_DIR" ]; then
    echo "  原始数据目录:"
    ls -d "$RAW_DIR"/*/ 2>/dev/null | while read d; do
        basename "$d"
    done || echo "  (空目录)"
fi

# ---------- Step 3: 转换 ----------
echo -e "\n${YELLOW}[3/4] 转换为YOLO格式...${NC}"

python tools/prepare_visdrone.py --convert "$RAW_DIR" --out "$OUT_DIR"

# ---------- Step 4: 验证 ----------
echo -e "\n${YELLOW}[4/4] 验证数据集完整性...${NC}"
python tools/prepare_visdrone.py --check "$OUT_DIR"

# ---------- 完成 ----------
echo -e "\n${GREEN}=========================================="
echo -e " 数据准备完成!"
echo -e "==========================================${NC}"

for split in train val test; do
    img_count=$(find "$OUT_DIR/images/$split" -name "*.jpg" -o -name "*.png" 2>/dev/null | wc -l)
    if [ "$img_count" -gt 0 ]; then
        echo -e "  $split: ${img_count} 张图片"
    fi
done

echo ""
echo -e "  下一步: bash scripts/start_train.sh"
echo ""
