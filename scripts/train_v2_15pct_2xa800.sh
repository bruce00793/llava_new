#!/bin/bash
# ============================================================
# Q-Former V2 快速验证 — 15% 数据子集（2×A800）
# ============================================================

NUM_GPUS=2
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
# ★ 增加 NCCL 超时时间（默认30分钟，改为2小时，避免保存 checkpoint 时超时）
export NCCL_TIMEOUT=7200
# ★ NCCL 调试（可选，遇到问题时开启）
# export NCCL_DEBUG=INFO

DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
SUBSET_FILE="./data/subset_15pct_scenes.txt"

# ===== 恢复训练（从 checkpoint 继续）=====
# 设置为 checkpoint 路径即可恢复，留空则从零开始
RESUME_CKPT="./outputs/v2_15pct_2xa800_20260211_152449/checkpoint_epoch_14.pth"

# ===== 输出目录（恢复时使用原目录，否则创建新目录）=====
if [ -n "$RESUME_CKPT" ]; then
    OUTPUT_DIR="$(dirname $RESUME_CKPT)"
else
    OUTPUT_DIR="./outputs/v2_15pct_2xa800_$(date +%Y%m%d_%H%M%S)"
fi

EPOCHS=24
BATCH_SIZE=1
ACCUMULATION_STEPS=30     # 2卡 × 30累积 = 有效批量60（与6卡×10一致）
NUM_WORKERS=4
WARMUP_STEPS=75

# 学习率（与主训练一致：只有 cls_head 提高）
LR_QFORMER_BACKBONE=2e-6
LR_QFORMER_DECODER=5e-6
LR_QFORMER_PROJECTOR=2e-5
LR_QUERIES=5e-5
LR_CLS_HEAD=2e-4
LR_DECODER=2e-5
LR_LORA=1e-4
LR_SCENE_INTERACTION=2e-5

WEIGHT_DECAY=0.01
GRAD_CLIP=35.0

echo "================================================================="
echo "Q-Former V2 快速验证 — 15% 子集（2×A800）"
echo "================================================================="
echo "Q-Former:       V2（三阶段双流 + 压缩瓶颈）"
echo "GPU:            2× A800"
echo "子集文件:       $SUBSET_FILE"
echo "输出目录:       $OUTPUT_DIR"
echo "恢复训练:       ${RESUME_CKPT:-无（从零开始）}"
echo "Batch × Accum × GPU: $BATCH_SIZE × $ACCUMULATION_STEPS × $NUM_GPUS = 有效批量 $((BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))"
echo "================================================================="

cd /home/cly/auto/llava_test/LLaVA

torchrun --nproc_per_node=$NUM_GPUS \
    train_map_detection.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --gt-cache-train "$GT_CACHE" \
    --gt-cache-val "$GT_CACHE" \
    --subset-scenes "$SUBSET_FILE" \
    --llm-path "$LLM_PATH" \
    --qformer-pretrained none \
    --qformer-version v2 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers $NUM_WORKERS \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --lr-queries $LR_QUERIES \
    --lr-cls-head $LR_CLS_HEAD \
    --lr-decoder $LR_DECODER \
    --lr-lora $LR_LORA \
    --lr-scene-interaction $LR_SCENE_INTERACTION \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --bf16 \
    --use-ema \
    --ema-decay 0.9999 \
    --output-dir "$OUTPUT_DIR" \
    --log-interval 10 \
    --save-interval 1 \
    --eval-interval 2 \
    ${RESUME_CKPT:+--resume "$RESUME_CKPT"}

echo "================================================================="
echo "训练完成！Checkpoint 保存在: $OUTPUT_DIR"
echo "================================================================="
