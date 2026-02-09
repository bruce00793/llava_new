#!/bin/bash

# Training script for LLaVA Map Detection - Stage 2
# Joint training with BLIP-2 pretrained Q-Former

# Set CUDA devices
export CUDA_VISIBLE_DEVICES=0

# Dataset paths
DATAROOT="/path/to/nuscenes"  # 修改为你的nuScenes路径
VERSION="v1.0-mini"  # 或 v1.0-trainval

# Model paths
LLM_PATH="lmsys/vicuna-7b-v1.5"  # 或本地路径

# Output directory
OUTPUT_DIR="./outputs/map_detection_stage2_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
EPOCHS=24
BATCH_SIZE=4
NUM_WORKERS=4

# Learning rates (Stage 2)
LR_QFORMER=1e-5    # Fine-tune from BLIP-2
LR_QUERIES=1e-4    # Train from scratch
LR_DECODER=1e-4    # Train from scratch

# Other settings
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=0.1

echo "=========================================="
echo "LLaVA Map Detection - Stage 2 Training"
echo "=========================================="
echo "Dataset: $DATAROOT ($VERSION)"
echo "Output: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rates:"
echo "  Q-Former: $LR_QFORMER"
echo "  Queries: $LR_QUERIES"
echo "  Decoder: $LR_DECODER"
echo "=========================================="

python train_map_detection.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --llm-path "$LLM_PATH" \
    --qformer-pretrained blip2 \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --num-workers $NUM_WORKERS \
    --lr-qformer $LR_QFORMER \
    --lr-queries $LR_QUERIES \
    --lr-decoder $LR_DECODER \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --fp16 \
    --output-dir "$OUTPUT_DIR" \
    --log-interval 10 \
    --save-interval 1 \
    --eval-interval 1

echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $OUTPUT_DIR"
echo "=========================================="

