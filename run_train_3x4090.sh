#!/bin/bash
# ============================================
# 3×4090 分布式训练脚本
# ============================================
# 使用方法:
#   1. srun --partition=gpu8 --gres=gpu:3 --mem=200G --time=3-00:00:00 --pty bash
#   2. bash run_train_3x4090.sh
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection Training - 3×4090"
echo "============================================"

# 激活 conda 环境
source ~/.bashrc
conda activate llava_new

# 设置 3 张 4090
export CUDA_VISIBLE_DEVICES=0,1,2
NUM_GPUS=3

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# 数据配置
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE_DIR="${DATAROOT}/gt_cache"
GT_CACHE_TRAIN="${GT_CACHE_DIR}"
GT_CACHE_VAL="${GT_CACHE_DIR}"

# 模型配置
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
QFORMER_PRETRAINED="none"

# ============================================
# 训练配置 (3×4090)
# ============================================
# Effective batch size = 3卡 × 1 × 11累积 = 33 ≈ 32
BATCH_SIZE=1
ACCUMULATION_STEPS=11
EPOCHS=24

# 学习率 (与 H100 一致)
LR_BACKBONE=3e-5
LR_DECODER=2e-4
LR_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_MAP_DECODER=2.5e-4

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=10.0

# 恢复训练配置
RESUME="/home/cly/auto/llava_test/LLaVA/outputs/h100_v1.0-trainval_20260115_200512/checkpoint_epoch_19.pth"

# 输出配置
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/3x4090_${VERSION}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# 打印配置
echo ""
echo "配置:"
echo "  GPU: ${NUM_GPUS}×4090"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo "  学习率: backbone=$LR_BACKBONE"
echo "  恢复自: $RESUME"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# 检查
if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "❌ GT Cache 不存在: $GT_CACHE_TRAIN"
    exit 1
fi
echo "✅ GT Cache 检查通过"

if [ ! -f "$RESUME" ]; then
    echo "❌ Checkpoint 不存在: $RESUME"
    exit 1
fi
echo "✅ Checkpoint 检查通过"

# 开始训练
echo ""
echo "============================================"
echo "开始 3×4090 分布式训练..."
echo "============================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_map_detection.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --llm-path $LLM_PATH \
    --qformer-pretrained $QFORMER_PRETRAINED \
    --resume $RESUME \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr-qformer-backbone $LR_BACKBONE \
    --lr-qformer-decoder $LR_DECODER \
    --lr-qformer-projector $LR_PROJECTOR \
    --lr-queries $LR_QUERIES \
    --lr-decoder $LR_MAP_DECODER \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --use-ema \
    --ema-decay 0.9999 \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
