#!/bin/bash
# ============================================
# 6×4090 从头训练脚本 - 基于 H100 配置
# ============================================
# 使用方法 (两步):
#   1. 申请资源 (在登录节点执行):
#      srun --partition=gpu8 --gres=gpu:6 --mem=180G --cpus-per-task=24 --time=5-00:00:00 --pty bash
#   
#   2. 运行训练 (在计算节点执行):
#      cd /home/cly/auto/llava_test/LLaVA
#      bash run_train_6x4090_fresh.sh
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection - 6×4090 从头训练"
echo "============================================"
echo "开始时间: $(date)"

# ========== 环境配置 ==========
source ~/.bashrc
conda activate llava_new

# 设置 6 张 4090
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
NUM_GPUS=6

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ========== 数据配置 ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE_DIR="${DATAROOT}/gt_cache"

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
QFORMER_PRETRAINED="none"

# ========== 训练配置 (6×4090) ==========
# Effective batch size = 6卡 × 1 × 5累积 = 30 ≈ 32 (与 H100/MapTR 一致)
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24

# 学习率 (与 H100 完全一致)
LR_BACKBONE=3e-5
LR_DECODER=2e-4
LR_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_MAP_DECODER=2.5e-4

# 其他 (与 H100 一致)
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=10.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/6x4090_fresh_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: ${NUM_GPUS}×RTX 4090 (48GB)"
echo "  数据集: $VERSION (nuScenes 全量)"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo "  学习率: backbone=$LR_BACKBONE, decoder=$LR_DECODER"
echo "  FP16: 启用 (与 H100 一致)"
echo "  每 epoch 保存 checkpoint"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 检查 ==========
if [ ! -d "$GT_CACHE_DIR" ]; then
    echo "❌ GT Cache 不存在: $GT_CACHE_DIR"
    exit 1
fi
echo "✅ GT Cache 检查通过"

if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过"

# 保存配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLaVA Map Detection - 6×4090 Fresh
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090 (48GB)
Dataset: nuScenes $VERSION
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled (same as H100)

Learning Rates:
  Q-Former Backbone: $LR_BACKBONE
  Q-Former Decoder:  $LR_DECODER
  Q-Former Projector: $LR_PROJECTOR
  Map Queries:       $LR_QUERIES
  Map Decoder:       $LR_MAP_DECODER
========================================
EOF

# ========== 开始训练 ==========
echo ""
echo "============================================"
echo "开始 6×4090 分布式训练 (从头开始)..."
echo "预计时间: ~2-3 天"
echo "============================================"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_map_detection.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_DIR \
    --gt-cache-val $GT_CACHE_DIR \
    --llm-path $LLM_PATH \
    --qformer-pretrained $QFORMER_PRETRAINED \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
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
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
