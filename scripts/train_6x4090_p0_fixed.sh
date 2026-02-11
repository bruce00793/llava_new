#!/bin/bash

# ============================================================
# LLaVA Map Detection - 全修复版训练脚本
# 6×4090 分布式训练
# ============================================================
#
# 修复内容（累计）：
#   P0-1: InitPointsHead 锚点+偏移（防止模式坍塌）
#   P0-2: 辅助损失权重 0.5×6→0.2×5（主损失主导训练）
#   P0-3: 辅助损失 detach pred_logits（分类头梯度 7×→1×）
#   P1:   学习率整体降低 2-5×
#   P3:   梯度累积 5→10，effective batch 30→60
#   ★ Loss归一化: pts÷20, dir÷19（修复20×/19×膨胀）
#   ★ 梯度裁剪: 1.0→35.0（配合loss归一化，恢复MapTR标准）
#   ★ 评估器: score_threshold 0.1→0.3, predict用sigmoid
#
# 用法：
#   在 salloc 会话中: bash scripts/train_6x4090_p0_fixed.sh
# ============================================================

# ===== 环境 =====
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ===== 数据路径（与上次训练一致）=====
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"

# ===== 模型路径 =====
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# ===== 输出目录 =====
OUTPUT_DIR="./outputs/6x4090_p0fix_$(date +%Y%m%d_%H%M%S)"

# ===== 训练超参数（P1+P3 修复）=====
EPOCHS=24
BATCH_SIZE=1              # 每 GPU batch size
ACCUMULATION_STEPS=10     # P3修复：5→10，effective batch = 1×10×6 = 60
NUM_WORKERS=4
WARMUP_STEPS=500

# ===== 学习率（P1修复：整体降低 2-5×）=====
LR_QFORMER_BACKBONE=2e-6    # 旧: 5e-5  → 新: 2e-6
LR_QFORMER_DECODER=5e-6     # 旧: 4e-4  → 新: 5e-6
LR_QFORMER_PROJECTOR=2e-5   # 旧: 5e-4  → 新: 2e-5
LR_QUERIES=5e-5             # 旧: 5e-4  → 新: 5e-5
LR_DECODER=2e-5             # 旧: 5e-4  → 新: 2e-5
LR_LORA=1e-4                # 旧: 2e-4  → 新: 1e-4
LR_SCENE_INTERACTION=2e-5   # 新增参数

# ===== 其他 =====
WEIGHT_DECAY=0.01
GRAD_CLIP=35.0              # Loss归一化修复后恢复MapTR标准值（旧1.0过紧）

echo "================================================================="
echo "LLaVA Map Detection - P0+P1+P3 修复版（6×4090 分布式）"
echo "================================================================="
echo "GPU 数量:          $NUM_GPUS"
echo "数据集:            $DATAROOT ($VERSION)"
echo "输出目录:          $OUTPUT_DIR"
echo "Batch/GPU:         $BATCH_SIZE"
echo "梯度累积:          $ACCUMULATION_STEPS"
echo "Effective Batch:   $((BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))"
echo "学习率:"
echo "  QFormer backbone:  $LR_QFORMER_BACKBONE"
echo "  QFormer decoder:   $LR_QFORMER_DECODER"
echo "  QFormer projector: $LR_QFORMER_PROJECTOR"
echo "  Queries:           $LR_QUERIES"
echo "  Decoder:           $LR_DECODER"
echo "  LoRA:              $LR_LORA"
echo "  Scene Interaction: $LR_SCENE_INTERACTION"
echo "梯度裁剪:          $GRAD_CLIP"
echo "================================================================="

cd /home/cly/auto/llava_test/LLaVA

torchrun --nproc_per_node=$NUM_GPUS \
    train_map_detection.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --gt-cache-train "$GT_CACHE" \
    --gt-cache-val "$GT_CACHE" \
    --llm-path "$LLM_PATH" \
    --qformer-pretrained none \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers $NUM_WORKERS \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --lr-queries $LR_QUERIES \
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
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1

echo "================================================================="
echo "训练完成！"
echo "Checkpoint 保存在: $OUTPUT_DIR"
echo "================================================================="
