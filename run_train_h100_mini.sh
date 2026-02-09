#!/bin/bash
# ============================================
# H100 Mini数据集快速测试脚本
# ============================================
# 用于快速验证训练流程，建议正式训练前先运行此脚本
# 预计运行时间: ~10-15分钟 (3个epoch)
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection - Mini数据集测试 (H100)"
echo "============================================"

# 环境配置
source ~/.bashrc
conda activate llava_new
export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# Mini 数据集配置
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
VERSION="v1.0-mini"

# GT Cache 路径
# Mini 只有 train cache (324个样本)，val 设为 none 跳过验证
GT_CACHE_TRAIN="${DATAROOT}/gt_cache_v1.0-mini_train.pkl"
GT_CACHE_VAL="none"

LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# 快速测试配置 (只训练3个epoch)
BATCH_SIZE=2          # 保守配置
ACCUMULATION_STEPS=4  # 减少累积步数加速测试 (effective=8)
EPOCHS=3

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/h100_mini_test_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

echo ""
echo "配置:"
echo "  数据集: $VERSION (Mini)"
echo "  Batch Size: $BATCH_SIZE × $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS (快速测试)"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查 GT Cache
if [ ! -d "$GT_CACHE_TRAIN" ] && [ ! -f "$GT_CACHE_TRAIN" ]; then
    echo "❌ GT Cache 不存在: $GT_CACHE_TRAIN"
    exit 1
fi
echo "✅ GT Cache 检查通过"

echo ""
echo "开始训练..."
echo ""

python train_map_detection.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --llm-path $LLM_PATH \
    --qformer-pretrained none \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr-qformer-backbone 5e-5 \
    --lr-qformer-decoder 3e-4 \
    --lr-qformer-projector 4e-4 \
    --lr-queries 4e-4 \
    --lr-decoder 4e-4 \
    --weight-decay 0.01 \
    --warmup-steps 50 \
    --grad-clip 15.0 \
    --use-ema \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 10 \
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ Mini数据集测试完成！"
echo "   如果没有错误，可以开始完整数据集训练"
echo "   运行: bash run_train_h100.sh"
echo "============================================"
