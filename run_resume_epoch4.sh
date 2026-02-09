#!/bin/bash
# ============================================
# 从 Epoch 4 恢复训练脚本
# ============================================
# 使用方法:
#   1. 先申请资源: srun --partition=gpu8 --gres=gpu:6 --mem=180G --cpus-per-task=24 --time=5-00:00:00 --pty bash
#   2. 运行脚本:   cd /home/cly/auto/llava_test/LLaVA && bash run_resume_epoch4.sh
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection - 从 Epoch 4 恢复训练"
echo "============================================"
echo "开始时间: $(date)"

# ========== 环境配置 ==========
source ~/.bashrc
conda activate llava_new

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
NUM_GPUS=6

PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ========== 路径配置 ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
GT_CACHE_DIR="${DATAROOT}/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
OUTPUT_DIR="/home/cly/auto/llava_test/LLaVA/outputs/6x4090_fresh_20260121_095129"
CHECKPOINT="${OUTPUT_DIR}/checkpoint_epoch_4.pth"

# ========== 检查 Checkpoint ==========
if [ ! -f "$CHECKPOINT" ]; then
    echo "❌ Checkpoint 不存在: $CHECKPOINT"
    ls -la ${OUTPUT_DIR}/*.pth 2>/dev/null || echo "没有找到任何 checkpoint"
    exit 1
fi
echo "✅ Checkpoint: $CHECKPOINT"

# ========== 训练配置 ==========
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24

LR_BACKBONE=3e-5
LR_DECODER=2e-4
LR_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_MAP_DECODER=2.5e-4

WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=10.0

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: ${NUM_GPUS}×RTX 4090"
echo "  恢复自: Epoch 4 → 继续训练到 Epoch 24"
echo "  剩余: 20 个 epoch"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 开始训练 ==========
echo "============================================"
echo "开始恢复训练..."
echo "============================================"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_map_detection.py \
    --dataroot $DATAROOT \
    --version v1.0-trainval \
    --gt-cache-train $GT_CACHE_DIR \
    --gt-cache-val $GT_CACHE_DIR \
    --llm-path $LLM_PATH \
    --qformer-pretrained none \
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
    --resume $CHECKPOINT \
    2>&1 | tee -a "${OUTPUT_DIR}/train_resume.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "============================================"
