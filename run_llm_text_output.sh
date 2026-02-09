#!/bin/bash
# ============================================
# LLM Text Output 验证实验 (支持 LoRA 微调)
# ============================================
# 目的：验证Q-Former + LLM能否理解视觉特征并输出地图元素
# 使用 LoRA 微调 LLM，让其学习理解视觉 tokens
#
# 使用方法:
#   cd /home/cly/auto/llava_test/LLaVA
#   bash run_llm_text_output.sh
# ============================================

set -e

echo "============================================"
echo "LLM Text Output - LoRA 微调验证实验"
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
# 完整数据集路径
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE_TRAIN="${DATAROOT}/gt_cache"
GT_CACHE_VAL="${DATAROOT}/gt_cache"

# 使用15%的数据
SAMPLE_RATIO=0.15

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# ========== LoRA 配置 (优化后) ==========
USE_LORA=true           # 是否使用 LoRA
LORA_R=32               # LoRA rank (增大以提高适应能力)
LORA_ALPHA=64           # LoRA scaling factor (= 2 * r)
LORA_DROPOUT=0.05       # LoRA dropout
LR_LORA=2e-4            # LoRA 学习率 (增大，加速学习)

# ========== 训练配置 ==========
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=15  # 增加 epochs，给模型更多学习时间

# Q-Former 学习率 (与检测头训练一致)
LR_QFORMER_BACKBONE=3e-5
LR_QFORMER_DECODER=2e-4
LR_QFORMER_PROJECTOR=2.5e-4

# Map Queries 学习率 (与检测头训练一致)
LR_MAP_QUERIES=1e-4

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
GRAD_CLIP=10.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
if [ "$USE_LORA" = true ]; then
    OUTPUT_DIR="${PROJECT_DIR}/outputs/llm_text_lora_${TIMESTAMP}"
else
    OUTPUT_DIR="${PROJECT_DIR}/outputs/llm_text_frozen_${TIMESTAMP}"
fi
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: ${NUM_GPUS}×RTX 4090"
echo "  数据集: $VERSION"
echo "  数据比例: ${SAMPLE_RATIO} (15%)"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo ""
echo "  Q-Former 学习率:"
echo "    backbone=$LR_QFORMER_BACKBONE"
echo "    decoder=$LR_QFORMER_DECODER"
echo "    projector=$LR_QFORMER_PROJECTOR"
echo ""
echo "  Map Queries (1050):"
echo "    lr=$LR_MAP_QUERIES"
echo ""
if [ "$USE_LORA" = true ]; then
    echo "  LoRA 配置:"
    echo "    rank=$LORA_R, alpha=$LORA_ALPHA"
    echo "    dropout=$LORA_DROPOUT"
    echo "    lr=$LR_LORA"
else
    echo "  LLM: 冻结 (不使用 LoRA)"
fi
echo ""
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  FP16: 启用"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 检查 ==========
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过"

# 检查GT缓存
if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "⚠️  GT Cache (train) 不存在: $GT_CACHE_TRAIN"
    echo "   将尝试使用 mini 数据集进行测试"
    
    # 切换到 mini 数据集
    DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
    VERSION="v1.0-mini"
    GT_CACHE_TRAIN="${DATAROOT}/gt_cache_v1.0-mini_train.pkl"
    GT_CACHE_VAL="${DATAROOT}/gt_cache_v1.0-mini_val.pkl"
    SAMPLE_RATIO=1.0
    
    echo "   使用 mini 数据集: $DATAROOT"
fi

if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "❌ GT Cache (train) 不存在: $GT_CACHE_TRAIN"
    exit 1
fi
echo "✅ GT Cache 检查通过"

# 检查 peft 是否安装
python -c "import peft" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ peft 未安装，正在安装..."
    pip install peft -q
fi
echo "✅ peft 检查通过"

# 保存配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLM Text Output - LoRA 微调验证实验
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090
Dataset: nuScenes $VERSION
Sample Ratio: $SAMPLE_RATIO
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled

LoRA Config:
  Enabled: $USE_LORA
  Rank: $LORA_R
  Alpha: $LORA_ALPHA
  Dropout: $LORA_DROPOUT
  LR: $LR_LORA

Q-Former LR:
  Backbone: $LR_QFORMER_BACKBONE
  Decoder: $LR_QFORMER_DECODER
  Projector: $LR_QFORMER_PROJECTOR

Map Queries LR: $LR_MAP_QUERIES

GT Cache Train: $GT_CACHE_TRAIN
GT Cache Val: $GT_CACHE_VAL
========================================
EOF

# ========== 构建命令 ==========
CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29501 \
    train_llm_text_output.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --sample-ratio $SAMPLE_RATIO \
    --llm-path $LLM_PATH \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --lr-map-queries $LR_MAP_QUERIES \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 20"

# 添加 LoRA 参数
if [ "$USE_LORA" = true ]; then
    CMD="$CMD \
    --use-lora \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --lr-lora $LR_LORA"
else
    CMD="$CMD --no-lora"
fi

# ========== 开始训练 ==========
echo ""
echo "============================================"
echo "开始训练..."
echo "============================================"
echo ""

eval $CMD 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
