#!/bin/bash
# ============================================
# 6×4090 Mini 数据集测试脚本
# ============================================
# 基于 run_train_6x4090_fresh.sh，只修改了数据集为 mini
# 用于快速验证 GT 数据和训练流程是否正确
#
# 使用方法:
#   cd /home/cly/auto/llava_test/LLaVA
#   bash run_test_mini.sh
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection - 6×4090 Mini 数据集测试"
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

# ========== 数据配置 (Mini 数据集) ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
VERSION="v1.0-mini"
GT_CACHE_TRAIN="${DATAROOT}/gt_cache_v1.0-mini_train.pkl"  # train GT 缓存
GT_CACHE_VAL="${DATAROOT}/gt_cache_v1.0-mini_val.pkl"      # val GT 缓存

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
QFORMER_PRETRAINED="none"

# ========== 训练配置 (与原始 6×4090 一致) ==========
# Effective batch size = 6卡 × 1 × 5累积 = 30 ≈ 32 (与 H100/MapTR 一致)
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24  # 与原始 6×4090 训练一致

# 学习率 (与原始完全一致)
LR_BACKBONE=3e-5
LR_DECODER=2e-4
LR_PROJECTOR=2.5e-4
LR_QUERIES=2.5e-4
LR_MAP_DECODER=2.5e-4

# 其他 (与原始一致)
WEIGHT_DECAY=0.01
WARMUP_STEPS=500  # 与原始 6×4090 训练一致
GRAD_CLIP=10.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/mini_test_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: ${NUM_GPUS}×RTX 4090 (48GB)"
echo "  数据集: $VERSION (nuScenes Mini)"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS (与原始一致)"
echo "  学习率: backbone=$LR_BACKBONE, decoder=$LR_DECODER"
echo "  FP16: 启用"
echo "  EMA: 启用"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 检查 ==========
if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "❌ Train GT Cache 不存在: $GT_CACHE_TRAIN"
    echo "   请先运行 GT 生成脚本"
    exit 1
fi
echo "✅ Train GT Cache 检查通过: $GT_CACHE_TRAIN"

if [ ! -d "$GT_CACHE_VAL" ]; then
    echo "❌ Val GT Cache 不存在: $GT_CACHE_VAL"
    echo "   请先运行 GT 生成脚本"
    exit 1
fi
echo "✅ Val GT Cache 检查通过: $GT_CACHE_VAL"

if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过 (本地路径)"

# 保存配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLaVA Map Detection - Mini 数据集测试
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090 (48GB)
Dataset: nuScenes $VERSION
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled
EMA: Enabled

Learning Rates:
  Q-Former Backbone: $LR_BACKBONE
  Q-Former Decoder:  $LR_DECODER
  Q-Former Projector: $LR_PROJECTOR
  Map Queries:       $LR_QUERIES
  Map Decoder:       $LR_MAP_DECODER

GT Cache: $GT_CACHE_DIR
========================================
EOF

# ========== 开始训练 ==========
echo ""
echo "============================================"
echo "开始 6×4090 分布式训练 (Mini 数据集测试)..."
echo "============================================"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_map_detection.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
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
    --save-interval 9999 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 测试完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
