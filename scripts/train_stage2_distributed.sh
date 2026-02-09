#!/bin/bash

# Distributed training script for LLaVA Map Detection - Stage 2
# Multi-GPU training with torchrun

# Set number of GPUs
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Dataset paths
DATAROOT="/path/to/nuscenes"
VERSION="v1.0-mini"

# Model paths
LLM_PATH="lmsys/vicuna-7b-v1.5"

# Output directory
OUTPUT_DIR="./outputs/map_detection_stage2_distributed_$(date +%Y%m%d_%H%M%S)"

# Training hyperparameters
EPOCHS=24
BATCH_SIZE=2  # Per GPU batch size
NUM_WORKERS=4

# Learning rates (Stage 2)
LR_QFORMER=1e-5
LR_QUERIES=1e-4
LR_DECODER=1e-4

# Other settings
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=0.1

echo "=========================================="
echo "LLaVA Map Detection - Stage 2 Training (Distributed)"
echo "=========================================="
echo "GPUs: $NUM_GPUS"
echo "Dataset: $DATAROOT ($VERSION)"
echo "Output: $OUTPUT_DIR"
echo "Batch size per GPU: $BATCH_SIZE"
echo "Total batch size: $((BATCH_SIZE * NUM_GPUS))"
echo "=========================================="

torchrun --nproc_per_node=$NUM_GPUS \
    train_map_detection.py \
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

