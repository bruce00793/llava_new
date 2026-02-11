#!/bin/bash
# ============================================================
# Q-Former V2 快速验证 — 15% 数据子集
# 6×4090 分布式训练
# ============================================================
#
# 目的：用 15% 数据快速验证 V2 Q-Former 能跑通全流程
# 同样的 15% 数据后续可用 V1 跑对比实验
#
# 子集：data/subset_15pct_scenes.txt（105 个场景，seed=42）
# Q-Former：V2（三阶段双流 + 压缩瓶颈）
# 学习率：与 train_fresh_v2.sh 一致（cls_head 独立高 LR）
#
# 用法: bash scripts/train_v2_15pct.sh
# ============================================================

NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
SUBSET_FILE="./data/subset_15pct_scenes.txt"

OUTPUT_DIR="./outputs/v2_15pct_$(date +%Y%m%d_%H%M%S)"

EPOCHS=24
BATCH_SIZE=1
ACCUMULATION_STEPS=10
NUM_WORKERS=4
WARMUP_STEPS=75           # 15% 数据 → 步数少，warmup 缩短

# 学习率（与 train_fresh_v2.sh 一致）
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
echo "Q-Former V2 快速验证 — 15% 数据子集 (105 scenes)"
echo "================================================================="
echo "Q-Former:    V2（三阶段双流 + 压缩瓶颈）"
echo "子集文件:    $SUBSET_FILE"
echo "输出目录:    $OUTPUT_DIR"
echo "Effective Batch: $((BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))"
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
    --eval-interval 2

echo "================================================================="
echo "训练完成！Checkpoint 保存在: $OUTPUT_DIR"
echo "================================================================="
