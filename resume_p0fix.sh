#!/bin/bash
# ============================================================
# 恢复训练脚本 - 从 epoch 2 checkpoint 继续 + 提高学习率
# ============================================================
#
# 修改内容（相比原训练）：
#   ★ 从零训练模块 LR 提高 5×（修复学习过慢问题）：
#     - Map Decoder:        2e-5 → 1e-4
#     - Map Queries:        5e-5 → 2.5e-4
#     - Scene Interaction:  2e-5 → 1e-4
#     - QFormer Projector:  2e-5 → 5e-5
#   ★ 分类头添加 DETR 标准先验偏置初始化（代码层面修复）
#   ★ Resume 时自动用新 LR 覆盖 checkpoint 旧 LR
#
#   预训练模块保持不变：
#     - QFormer Backbone:   2e-6（不变）
#     - QFormer Decoder:    5e-6（不变）
#     - LoRA:               1e-4（不变）
#
# 用法: 在 tmux 中运行: bash resume_p0fix.sh
# ============================================================

cd /home/cly/auto/llava_test/LLaVA

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

echo "================================================================="
echo "恢复训练 - 从 epoch 3 开始 (共 24 epoch)"
echo "Checkpoint: checkpoint_epoch_2.pth"
echo "★ 学习率调整：从零模块提高 5×"
echo "================================================================="
echo "学习率（新）:"
echo "  QFormer backbone:   2e-6  (不变)"
echo "  QFormer decoder:    5e-6  (不变)"
echo "  QFormer projector:  5e-5  (旧: 2e-5, ×2.5)"
echo "  Map Queries:        2.5e-4 (旧: 5e-5, ×5)"
echo "  Map Decoder:        1e-4  (旧: 2e-5, ×5)"
echo "  Scene Interaction:  1e-4  (旧: 2e-5, ×5)"
echo "  LoRA:               1e-4  (不变)"
echo "================================================================="
nvidia-smi --query-gpu=index,name,memory.used --format=csv
echo "================================================================="

torchrun --nproc_per_node=6 \
    train_map_detection.py \
    --dataroot "/home/cly/auto/llava_test/LLaVA/data/nuscenes" \
    --version "v1.0-trainval" \
    --gt-cache-train "/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache" \
    --gt-cache-val "/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache" \
    --llm-path "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5" \
    --qformer-pretrained none \
    --resume "./outputs/6x4090_p0fix_20260210_154231/checkpoint_epoch_2.pth" \
    --epochs 24 \
    --batch-size 1 \
    --accumulation-steps 10 \
    --num-workers 4 \
    --lr-qformer-backbone 2e-6 \
    --lr-qformer-decoder 5e-6 \
    --lr-qformer-projector 5e-5 \
    --lr-queries 2.5e-4 \
    --lr-decoder 1e-4 \
    --lr-lora 1e-4 \
    --lr-scene-interaction 1e-4 \
    --weight-decay 0.01 \
    --warmup-steps 500 \
    --grad-clip 35.0 \
    --bf16 \
    --use-ema \
    --ema-decay 0.9999 \
    --output-dir "./outputs/6x4090_p0fix_20260210_154231" \
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1

echo "================================================================="
echo "训练完成！"
echo "================================================================="
