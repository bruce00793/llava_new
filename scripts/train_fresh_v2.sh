#!/bin/bash
# ============================================================
# LLaVA Map Detection - V2 从零训练脚本
# 6×4090 分布式训练
# ============================================================
#
# 核心改进（相比 V1）：
#   ★ cls_head 独立参数组 LR=5e-4（MapTR 级别，解决全背景坍塌）
#   ★ DETR 标准 cls_head bias 初始化（sigmoid≈0.01）
#   ★ 从零模块 LR 温和提升 2-2.5×（避免 5× 导致的梯度爆炸）
#   ★ 修复训练日志 JSON 序列化 bug
#
# 学习率设计：
#   - QFormer Backbone:   2e-6  (预训练，不变)
#   - QFormer Decoder:    5e-6  (预训练，不变)
#   - QFormer Projector:  3e-5  (适应新任务, 1.5×)
#   - Map Queries:        1e-4  (从零, 2×)
#   - Cls Head:           5e-4  (★独立高LR, MapTR级别)
#   - Map Decoder(其他):  5e-5  (从零, 2.5×)
#   - Scene Interaction:  5e-5  (从零, 2.5×)
#   - LoRA:               1e-4  (不变)
#
# 用法: bash scripts/train_fresh_v2.sh
# ============================================================

# ===== 环境 =====
NUM_GPUS=6
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
export OMP_NUM_THREADS=8
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0

# ===== 数据路径 =====
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache"

# ===== 模型路径 =====
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# ===== 恢复训练（从 checkpoint 继续）=====
# 设置为 checkpoint 路径即可恢复，留空则从零开始
RESUME_CKPT="./outputs/v2_fresh_20260211_150914/checkpoint_epoch_3.pth"

# ===== 输出目录（恢复时使用原目录，否则创建新目录）=====
if [ -n "$RESUME_CKPT" ]; then
    OUTPUT_DIR="$(dirname $RESUME_CKPT)"
else
    OUTPUT_DIR="./outputs/v2_fresh_$(date +%Y%m%d_%H%M%S)"
fi

# ===== 训练超参数 =====
EPOCHS=24
BATCH_SIZE=1
ACCUMULATION_STEPS=10     # effective batch = 1×10×6 = 60
NUM_WORKERS=4
WARMUP_STEPS=500

# ===== 学习率 =====
LR_QFORMER_BACKBONE=2e-6
LR_QFORMER_DECODER=5e-6
LR_QFORMER_PROJECTOR=2e-5
LR_QUERIES=5e-5
LR_CLS_HEAD=2e-4            # ★ 独立高 LR（唯一提高的模块）
LR_DECODER=2e-5
LR_LORA=1e-4
LR_SCENE_INTERACTION=2e-5

# ===== 其他 =====
WEIGHT_DECAY=0.01
GRAD_CLIP=35.0

echo "================================================================="
echo "LLaVA Map Detection V2 - 从零训练（cls_head 独立 LR）"
echo "================================================================="
echo "GPU 数量:          $NUM_GPUS"
echo "输出目录:          $OUTPUT_DIR"
echo "恢复训练:          ${RESUME_CKPT:-无（从零开始）}"
echo "Effective Batch:   $((BATCH_SIZE * ACCUMULATION_STEPS * NUM_GPUS))"
echo "学习率:"
echo "  QFormer backbone:  $LR_QFORMER_BACKBONE"
echo "  QFormer decoder:   $LR_QFORMER_DECODER"
echo "  QFormer projector: $LR_QFORMER_PROJECTOR"
echo "  Map Queries:       $LR_QUERIES"
echo "  ★ Cls Head:        $LR_CLS_HEAD (独立高LR)"
echo "  Map Decoder:       $LR_DECODER"
echo "  Scene Interaction: $LR_SCENE_INTERACTION"
echo "  LoRA:              $LR_LORA"
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
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1 \
    ${RESUME_CKPT:+--resume "$RESUME_CKPT"}

echo "================================================================="
echo "训练完成！"
echo "Checkpoint 保存在: $OUTPUT_DIR"
echo "================================================================="
