#!/bin/bash
# ============================================
# 从 Epoch 8 恢复训练 (6x4090)
# 已修复 NaN 导致的 NCCL 死锁问题
# ============================================

set -e

# 环境设置
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export NCCL_DEBUG=WARN
export NCCL_TIMEOUT=3600  # 增加到 1 小时超时

# 路径配置
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
GT_CACHE_DIR="${DATAROOT}/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
OUTPUT_DIR="${PROJECT_DIR}/outputs/6x4090_fresh_20260125_143156"
CHECKPOINT="${OUTPUT_DIR}/checkpoint_epoch_8.pth"

# 训练参数
NUM_GPUS=6
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24

# 学习率 (使用正确的参数名称)
LR_QFORMER_BACKBONE=3e-5
LR_QFORMER_DECODER=2e-4
LR_QFORMER_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_DECODER=2.5e-4  # 这是 map decoder 的学习率

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=10.0

echo "============================================"
echo "从 Epoch 8 恢复训练"
echo "  Checkpoint: $CHECKPOINT"
echo "  GPU 数量: $NUM_GPUS"
echo "  Effective Batch: $((BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))"
echo "============================================"

cd $PROJECT_DIR

torchrun --nproc_per_node=$NUM_GPUS \
    train_map_detection.py \
    --dataroot $DATAROOT \
    --version v1.0-trainval \
    --gt-cache-train $GT_CACHE_DIR \
    --gt-cache-val $GT_CACHE_DIR \
    --llm-path $LLM_PATH \
    --qformer-pretrained none \
    --output-dir $OUTPUT_DIR \
    --resume $CHECKPOINT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --lr-queries $LR_QUERIES \
    --lr-decoder $LR_DECODER \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --use-ema \
    --ema-decay 0.9999 \
    --fp16 \
    --log-interval 20 \
    2>&1 | tee -a "${OUTPUT_DIR}/train_resume_epoch8.log"

echo "============================================"
echo "✅ 训练完成！"
echo "============================================"
