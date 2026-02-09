#!/bin/bash
# ============================================================================
# Map Detection Evaluation Script
# 
# Evaluates trained model and computes mAP following MapTR protocol
# ============================================================================

# Checkpoint path (change this to your checkpoint)
CHECKPOINT="/home/cly/auto/llava_test/LLaVA/outputs/6x4090_fresh_20260125_143156/best_model_ema.pth"

# Alternative: use final model
# CHECKPOINT="/home/cly/auto/llava_test/LLaVA/outputs/6x4090_fresh_20260125_143156/final_model_ema.pth"

# Data paths
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"

# Output directory
OUTPUT_DIR="./eval_results_$(date +%Y%m%d_%H%M%S)"

# Evaluation settings
NUM_SAMPLES=0  # 0 = all samples
NUM_VIS=20     # Number of samples to visualize

# Run evaluation
echo "============================================"
echo "Map Detection Evaluation"
echo "============================================"
echo "Checkpoint: $CHECKPOINT"
echo "Output: $OUTPUT_DIR"
echo "============================================"

python evaluate_and_visualize.py \
    --checkpoint $CHECKPOINT \
    --dataroot $DATAROOT \
    --gt-cache $GT_CACHE \
    --output-dir $OUTPUT_DIR \
    --num-samples $NUM_SAMPLES \
    --num-vis $NUM_VIS \
    --visualize

echo "============================================"
echo "Evaluation complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "============================================"
