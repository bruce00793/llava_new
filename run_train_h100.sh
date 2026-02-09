#!/bin/bash
# ============================================
# H100 单卡训练脚本 - 对齐 MapTR
# ============================================
# 使用方法:
#   1. 连接到 H100 (3天时间限制): yrun h100_pcie_1 -t 3-00:00:00
#   2. 运行训练: bash run_train_h100.sh
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 环境配置
# ============================================
echo "============================================"
echo "LLaVA Map Detection Training - H100"
echo "============================================"

# 激活 conda 环境
source ~/.bashrc
conda activate llava_new

# 设置 CUDA
export CUDA_VISIBLE_DEVICES=0

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ============================================
# 数据配置
# ============================================
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"

# GT Cache 路径 - Train/Val 共用同一个目录
# annotations/ 包含所有样本，通过 splits/ 区分 train/val
GT_CACHE_DIR="${DATAROOT}/gt_cache"
GT_CACHE_TRAIN="${GT_CACHE_DIR}"
GT_CACHE_VAL="${GT_CACHE_DIR}"

# ============================================
# 模型配置
# ============================================
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# Q-Former 预训练: 'none' 因为我们使用 ResNet50 (与 BLIP-2 的 ViT 不兼容)
QFORMER_PRETRAINED="none"

# ============================================
# 训练配置 (针对 LLaVA-based 架构优化 - 修复OOM版)
# ============================================
# Effective batch size = 1 × 1 × 32 = 32 (与 MapTR 相同)
# 减小 batch_size 以避免 OOM，增加累积步数保持有效 batch size
BATCH_SIZE=1
ACCUMULATION_STEPS=32
EPOCHS=24

# 学习率 (降低 37-40% 以解决 NaN 问题，平衡稳定性和收敛速度)
# 参考 MapTR 配置，适度降低
LR_BACKBONE=3e-5
LR_DECODER=2e-4
LR_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_MAP_DECODER=2.5e-4

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=10.0        # 适度降低 (原15.0)

# ============================================
# 恢复训练配置 (从 checkpoint 继续)
# ============================================
# 设置为 checkpoint 路径以恢复训练，留空则从头开始
# 从 Epoch 7 恢复 (Epoch 8 开始出现大量 NaN，所以从 7 重新开始)
RESUME="/home/cly/auto/llava_test/LLaVA/outputs/h100_v1.0-trainval_20260115_092343/checkpoint_epoch_7.pth"

# ============================================
# 输出配置
# ============================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/h100_${VERSION}_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ============================================
# 打印配置
# ============================================
echo ""
echo "配置:"
echo "  数据集: $VERSION"
echo "  Batch Size: $BATCH_SIZE × $ACCUMULATION_STEPS 累积 = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo "  学习率: backbone=$LR_BACKBONE, others=$LR_DECODER"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ============================================
# 检查 GT Cache
# ============================================
if [ ! -d "$GT_CACHE_TRAIN" ] && [ ! -f "$GT_CACHE_TRAIN" ]; then
    echo "❌ 训练 GT Cache 不存在: $GT_CACHE_TRAIN"
    echo "   请先运行: python tools/generate_gt_cache.py --split train"
    exit 1
fi

if [ ! -d "$GT_CACHE_VAL" ] && [ ! -f "$GT_CACHE_VAL" ]; then
    echo "⚠️  验证 GT Cache 不存在: $GT_CACHE_VAL"
    echo "   将只进行训练，不进行验证"
fi

echo "✅ GT Cache 检查通过"

# ============================================
# 开始训练
# ============================================
echo ""
echo "============================================"
echo "开始训练..."
echo "============================================"

# 构建 resume 参数
RESUME_ARG=""
if [ -n "$RESUME" ] && [ -f "$RESUME" ]; then
    RESUME_ARG="--resume $RESUME"
    echo "📂 从 checkpoint 恢复: $RESUME"
fi

python train_map_detection.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --llm-path $LLM_PATH \
    --qformer-pretrained $QFORMER_PRETRAINED \
    $RESUME_ARG \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 8 \
    --lr-qformer-backbone $LR_BACKBONE \
    --lr-qformer-decoder $LR_DECODER \
    --lr-qformer-projector $LR_PROJECTOR \
    --lr-queries $LR_QUERIES \
    --lr-decoder $LR_MAP_DECODER \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --use-ema \
    --ema-decay 0.9999 \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
