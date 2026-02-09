#!/bin/bash
# ============================================
# 3×4090 从头训练脚本 - 基于 H100 配置
# ============================================
# 使用方法 (两步):
#   1. 申请资源 (在登录节点执行):
#      srun --partition=gpu8 --gres=gpu:3 --mem=200G --cpus-per-task=16 --time=5-00:00:00 --pty bash
#   
#   2. 运行训练 (在计算节点执行):
#      cd /home/cly/auto/llava_test/LLaVA
#      bash run_train_3x4090_fresh.sh
# ============================================

set -e  # 遇到错误立即退出

# ============================================
# 环境配置
# ============================================
echo "============================================"
echo "LLaVA Map Detection Training - 3×4090"
echo "============================================"
echo "开始时间: $(date)"

# 激活 conda 环境
source ~/.bashrc
conda activate llava_new

# 设置 3 张 4090
export CUDA_VISIBLE_DEVICES=0,1,2
NUM_GPUS=3

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ============================================
# 数据配置
# ============================================
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"

# GT Cache 路径 - Train/Val 共用同一个目录
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
# 训练配置 (3×4090, 对齐 H100 配置)
# ============================================
# H100: Effective batch size = 1 × 32累积 = 32
# 3×4090: Effective batch size = 3卡 × 1 × 11累积 = 33 ≈ 32
BATCH_SIZE=1
ACCUMULATION_STEPS=11
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

# ============================================
# 输出配置
# ============================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/3x4090_fresh_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ============================================
# 打印配置
# ============================================
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

if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过"

# 保存配置到文件
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLaVA Map Detection Training Config
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090 (48GB)
Dataset: nuScenes $VERSION
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled (same as H100)

Learning Rates (same as H100):
  Q-Former Backbone: $LR_BACKBONE
  Q-Former Decoder:  $LR_DECODER
  Q-Former Projector: $LR_PROJECTOR
  Map Queries:       $LR_QUERIES
  Map Decoder:       $LR_MAP_DECODER

Regularization:
  Weight Decay: $WEIGHT_DECAY
  Warmup Steps: $WARMUP_STEPS
  Gradient Clip: $GRAD_CLIP

Save/Eval:
  Save Interval: 1 epoch
  Eval Interval: 1 epoch
  EMA: Enabled (0.9999)
========================================
EOF

# ============================================
# 开始训练
# ============================================
echo ""
echo "============================================"
echo "开始 3×4090 分布式训练 (从头开始)..."
echo "预计时间: ~4-5 天"
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
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
