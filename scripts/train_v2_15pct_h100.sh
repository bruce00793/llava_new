#!/bin/bash
# ============================================================
# Q-Former V2 快速验证 — 15% 数据子集（单卡 H100）
# ============================================================
#
# 与 6×4090 版本的区别：
#   - GPU: 6×4090 → 1×H100
#   - 梯度累积: 10 → 60（保持有效批量 = 60 不变）
#   - 不使用 torchrun 分布式，直接 python 单卡运行
#
# 用法: bash scripts/train_v2_15pct_h100.sh
# ============================================================

export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=4

DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
SUBSET_FILE="./data/subset_15pct_scenes.txt"

OUTPUT_DIR="./outputs/v2_15pct_h100_$(date +%Y%m%d_%H%M%S)"

EPOCHS=24
BATCH_SIZE=1
ACCUMULATION_STEPS=60     # 1卡 × 60累积 = 有效批量60（与6卡×10一致）
NUM_WORKERS=4
WARMUP_STEPS=75

# 学习率（与 6×4090 版本完全一致）
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
echo "Q-Former V2 快速验证 — 15% 子集（单卡 H100）"
echo "================================================================="
echo "Q-Former:       V2（三阶段双流 + 压缩瓶颈）"
echo "GPU:            1× H100 PCIe"
echo "子集文件:       $SUBSET_FILE"
echo "输出目录:       $OUTPUT_DIR"
echo "Batch × Accum:  $BATCH_SIZE × $ACCUMULATION_STEPS = 有效批量 $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "================================================================="

cd /home/cly/auto/llava_test/LLaVA

python train_map_detection.py \
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
    --eval-interval 2

echo "================================================================="
echo "训练完成！Checkpoint 保存在: $OUTPUT_DIR"
echo "================================================================="
