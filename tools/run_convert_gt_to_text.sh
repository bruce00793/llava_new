#!/bin/bash
# ============================================
# 将 GT 数据转换为文字格式
# 用于 LLM 文本生成验证实验
# ============================================

set -e

# 配置
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE_DIR="${DATAROOT}/gt_cache"
OUTPUT_DIR="/home/cly/auto/llava_test/LLaVA/data/text_gt"
SAMPLE_RATIO=0.15  # 15%

# 激活环境
source ~/.bashrc
conda activate llava_new

cd /home/cly/auto/llava_test/LLaVA

echo "============================================"
echo "Converting GT to Text Format"
echo "============================================"
echo "Dataset: $DATAROOT"
echo "Version: $VERSION"
echo "GT Cache: $GT_CACHE_DIR"
echo "Output: $OUTPUT_DIR"
echo "Sample Ratio: ${SAMPLE_RATIO}"
echo "============================================"

# 转换训练集
echo ""
echo "[1/2] Converting training set..."
python tools/convert_gt_to_text.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-dir $GT_CACHE_DIR \
    --output-dir $OUTPUT_DIR \
    --split train \
    --sample-ratio $SAMPLE_RATIO \
    --seed 42

# 转换验证集（使用 100% 用于评估）
echo ""
echo "[2/2] Converting validation set (100% for evaluation)..."
python tools/convert_gt_to_text.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-dir $GT_CACHE_DIR \
    --output-dir $OUTPUT_DIR \
    --split val \
    --sample-ratio 1.0 \
    --seed 42

echo ""
echo "============================================"
echo "✅ Conversion completed!"
echo "   Output directory: $OUTPUT_DIR"
echo "============================================"
