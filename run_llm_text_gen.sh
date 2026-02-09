#!/bin/bash
# ============================================
# 单阶段 LLM 文本生成验证实验 (2 卡版本)
# ============================================
#
# 验证目标:
#   与主训练唯一区别: LLM 直接输出文字 (无 Map Decoder)
#   如果结果好 → 问题在 Map Decoder
#   如果结果差 → LLM 对视觉特征理解不够
#
# 与主训练一致的组件:
#   Q-Former (768 queries), Map Queries (1050),
#   MapAttentionMask, LoRA (r=32, alpha=64),
#   BF16, ImageNet 预处理
#
# 硬件配置:
#   2 GPU × batch_size=1 × accumulation=15 = 有效 batch 30
#
# SLURM 启动方法:
#   salloc -p gpu8 --gres=gpu:4090:2 -t 2-00:00:00 --cpus-per-task=16 --mem=56G
#   srun --pty bash
#   cd /home/cly/auto/llava_test/LLaVA
#   bash run_llm_text_gen.sh
# ============================================

set -e

echo "============================================"
echo "单阶段 LLM 文本生成验证"
echo "============================================"
echo "开始时间: $(date)"

# ========== 环境配置 ==========
source ~/.bashrc
conda activate llava_new

# GPU 配置 (2 卡版本，因为 gpu8 只有 2 张空闲)
export CUDA_VISIBLE_DEVICES=0,1
NUM_GPUS=2

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ========== 数据配置 ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE_TRAIN="${DATAROOT}/gt_cache"
GT_CACHE_VAL="${DATAROOT}/gt_cache"
SAMPLE_RATIO=0.15

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# ========== 训练配置 (2 卡调整) ==========
BATCH_SIZE=1
ACCUMULATION_STEPS=15  # 2 卡 × 1 × 15 = 30 (与 6 卡 × 1 × 5 = 30 保持一致)
EPOCHS=20

# 学习率 (与主训练一致!)
LR_QFORMER_BACKBONE=5e-5
LR_QFORMER_DECODER=4e-4
LR_QFORMER_PROJECTOR=5e-4
LR_QUERIES=5e-4
LR_LORA=2e-4

# LoRA (与主训练一致!)
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.1

# 优化器 (与主训练一致!)
WEIGHT_DECAY=0.01
WARMUP_STEPS=500
GRAD_CLIP=35.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/llm_text_gen_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "=========================================="
echo "配置信息 (与主训练一致)"
echo "=========================================="
echo "  GPU: ${NUM_GPUS}x RTX 4090"
echo "  数据: nuScenes $VERSION, ratio=$SAMPLE_RATIO"
echo "  Epochs: $EPOCHS"
echo "  Effective Batch: ${NUM_GPUS} x $BATCH_SIZE x $ACCUMULATION_STEPS = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo ""
echo "  学习率 (与主训练一致):"
echo "    Q-Former Backbone: $LR_QFORMER_BACKBONE"
echo "    Q-Former Decoder:  $LR_QFORMER_DECODER"
echo "    Q-Former Projector: $LR_QFORMER_PROJECTOR"
echo "    Map Queries:       $LR_QUERIES"
echo "    LoRA:              $LR_LORA"
echo ""
echo "  LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT"
echo "  BF16: enabled"
echo "  Grad Clip: $GRAD_CLIP (与主训练 MapTR 一致)"
echo "  Warmup: $WARMUP_STEPS steps"
echo ""
echo "  输出: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ========== 环境检查 ==========
if [ ! -d "$LLM_PATH" ]; then
    echo "LLM not found: $LLM_PATH"
    exit 1
fi
echo "LLM: OK"

if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "GT Cache not found: $GT_CACHE_TRAIN"
    echo "Trying mini dataset..."
    DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
    VERSION="v1.0-mini"
    GT_CACHE_TRAIN="${DATAROOT}/gt_cache"
    GT_CACHE_VAL="${DATAROOT}/gt_cache"
    SAMPLE_RATIO=1.0
fi

if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "GT Cache not found: $GT_CACHE_TRAIN"
    exit 1
fi
echo "GT Cache: OK"

python -c "import peft" 2>/dev/null || pip install peft -q
echo "peft: OK"

# ========== 保存配置 ==========
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
Single-Stage LLM Text Generation Verification
========================================
Start: $(date)
GPU: ${NUM_GPUS}x RTX 4090
Dataset: nuScenes $VERSION, ratio=$SAMPLE_RATIO
Epochs: $EPOCHS
Effective Batch: $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
BF16: enabled
Grad Clip: $GRAD_CLIP

LR (same as main training):
  Q-Former Backbone: $LR_QFORMER_BACKBONE
  Q-Former Decoder:  $LR_QFORMER_DECODER
  Q-Former Projector: $LR_QFORMER_PROJECTOR
  Map Queries:       $LR_QUERIES
  LoRA:              $LR_LORA

LoRA: r=$LORA_R, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT

GT Format: 20 points, [-1,1] normalized, 2 decimal places
Loss: Cross-Entropy on GT text tokens
========================================
EOF

# ========== 开始训练 ==========
echo "============================================"
echo "Starting training..."
echo "============================================"
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29502 \
    train_llm_text_gen.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --sample-ratio $SAMPLE_RATIO \
    --llm-path $LLM_PATH \
    --use-lora \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --lr-queries $LR_QUERIES \
    --lr-lora $LR_LORA \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --bf16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "Training completed!"
echo "  End: $(date)"
echo "  Output: $OUTPUT_DIR"
echo "============================================"
