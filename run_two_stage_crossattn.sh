#!/bin/bash
# ============================================
# 两阶段验证实验 + Cross-Attention
# ============================================
#
# 设计思路：
# 阶段 1: 与原始检测头完全一致的信息提取
#     [scene_tokens] + [prompt] + [map_queries] → LLM (自定义掩码) → map_features
#
# 【新增】Cross-Attention 增强:
#     map_features + scene_tokens → MapSceneInteractionLayer → enhanced_map_features
#
# 阶段 2: 从 enhanced_map_features 生成文字
#     [enhanced_map_features] + [response_prompt] + [gt_text] → LLM → Loss
#
# 验证目标：
# - 对比原版 two_stage_verification：Cross-Attention 是否有帮助？
# - 如果更好 → 说明直接交互有价值
# - 如果差不多 → 说明 LLM 已完成足够的信息融合
#
# 使用方法:
#   cd /home/cly/auto/llava_test/LLaVA
#   bash run_two_stage_crossattn.sh
# ============================================

set -e

echo "============================================"
echo "两阶段验证实验 + Cross-Attention"
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
GT_CACHE_TRAIN="${DATAROOT}/gt_cache"
GT_CACHE_VAL="${DATAROOT}/gt_cache"

# 使用15%的数据
SAMPLE_RATIO=0.15

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# ========== Cross-Attention 配置 ==========
CROSSATTN_LAYERS=3
CROSSATTN_EMBED_DIM=256
CROSSATTN_HEADS=8
LR_CROSSATTN=2e-4

# ========== LoRA 配置 ==========
USE_LORA=true
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
LR_LORA=2e-4

# ========== 训练配置 ==========
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24

# Q-Former 学习率
LR_QFORMER_BACKBONE=3e-5
LR_QFORMER_DECODER=2e-4
LR_QFORMER_PROJECTOR=2.5e-4

# Map Queries 学习率
LR_MAP_QUERIES=2e-3

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
GRAD_CLIP=10.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/two_stage_crossattn_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "============================================"
echo "实验配置"
echo "============================================"
echo ""
echo "验证目标: Cross-Attention 是否能帮助 map_features 从 scene_tokens 提取更多信息"
echo ""
echo "与原版 two_stage_verification 的区别:"
echo "  【新增】Cross-Attention Layer (${CROSSATTN_LAYERS}层, dim=${CROSSATTN_EMBED_DIM})"
echo "  在阶段1和阶段2之间，让 map_features 直接与 scene_tokens 交互"
echo ""
echo "硬件配置:"
echo "  GPU: ${NUM_GPUS}×RTX 4090"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo ""
echo "数据配置:"
echo "  数据集: nuScenes $VERSION"
echo "  数据比例: ${SAMPLE_RATIO} (15%)"
echo "  Epochs: $EPOCHS"
echo ""
echo "Cross-Attention 配置:"
echo "  Layers: $CROSSATTN_LAYERS"
echo "  Embed Dim: $CROSSATTN_EMBED_DIM"
echo "  Heads: $CROSSATTN_HEADS"
echo "  LR: $LR_CROSSATTN"
echo ""
echo "学习率配置:"
echo "  Q-Former backbone: $LR_QFORMER_BACKBONE"
echo "  Q-Former decoder:  $LR_QFORMER_DECODER"
echo "  Q-Former projector: $LR_QFORMER_PROJECTOR"
echo "  Map Queries:       $LR_MAP_QUERIES"
echo "  Cross-Attention:   $LR_CROSSATTN"
if [ "$USE_LORA" = true ]; then
    echo "  LoRA (r=$LORA_R, alpha=$LORA_ALPHA): $LR_LORA"
fi
echo ""
echo "输出目录: $OUTPUT_DIR"
echo ""
echo "============================================"

# ========== 检查 ==========
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过"

# 检查GT缓存
if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "⚠️  GT Cache 不存在: $GT_CACHE_TRAIN"
    echo "   尝试使用 mini 数据集..."
    
    DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
    VERSION="v1.0-mini"
    GT_CACHE_TRAIN="${DATAROOT}/gt_cache_v1.0-mini_train.pkl"
    GT_CACHE_VAL="${DATAROOT}/gt_cache_v1.0-mini_val.pkl"
    SAMPLE_RATIO=1.0
    
    echo "   使用 mini 数据集: $DATAROOT"
fi

if [ ! -d "$GT_CACHE_TRAIN" ]; then
    echo "❌ GT Cache 不存在: $GT_CACHE_TRAIN"
    exit 1
fi
echo "✅ GT Cache 检查通过"

# 检查 peft
python -c "import peft" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ peft 未安装，正在安装..."
    pip install peft -q
fi
echo "✅ peft 检查通过"

# 保存配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
两阶段验证实验 + Cross-Attention
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090
Dataset: nuScenes $VERSION
Sample Ratio: $SAMPLE_RATIO
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled

验证设计:
  阶段 1: 与原始检测头一致的信息提取
    - 拼接: [scene_tokens] + [prompt] + [map_queries]
    - 使用自定义掩码 MapAttentionMask
    - 输出: map_features [B, 1050, 4096]
  
  【新增】Cross-Attention 增强:
    - map_features + scene_tokens → MapSceneInteractionLayer
    - 让 map_features 直接从 scene_tokens 提取视觉信息
    - 输出: enhanced_map_features [B, 1050, 4096]
  
  阶段 2: 从 enhanced_map_features 生成文字
    - 拼接: [enhanced_map_features] + [response] + [gt_text]
    - 标准 causal attention
    - 关键: gt_text 看不到 scene_tokens！

Cross-Attention Config:
  Layers: $CROSSATTN_LAYERS
  Embed Dim: $CROSSATTN_EMBED_DIM
  Heads: $CROSSATTN_HEADS
  LR: $LR_CROSSATTN

LoRA Config:
  Enabled: $USE_LORA
  Rank: $LORA_R
  Alpha: $LORA_ALPHA
  Dropout: $LORA_DROPOUT
  LR: $LR_LORA

Learning Rates:
  Q-Former backbone: $LR_QFORMER_BACKBONE
  Q-Former decoder: $LR_QFORMER_DECODER
  Q-Former projector: $LR_QFORMER_PROJECTOR
  Map Queries: $LR_MAP_QUERIES
  Cross-Attention: $LR_CROSSATTN

GT Cache:
  Train: $GT_CACHE_TRAIN
  Val: $GT_CACHE_VAL
========================================
EOF

# ========== 构建命令 ==========
CMD="torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29503 \
    train_two_stage_crossattn.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache-train $GT_CACHE_TRAIN \
    --gt-cache-val $GT_CACHE_VAL \
    --sample-ratio $SAMPLE_RATIO \
    --llm-path $LLM_PATH \
    --crossattn-layers $CROSSATTN_LAYERS \
    --crossattn-embed-dim $CROSSATTN_EMBED_DIM \
    --crossattn-heads $CROSSATTN_HEADS \
    --lr-crossattn $LR_CROSSATTN \
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
echo "开始两阶段 + Cross-Attention 验证训练..."
echo "============================================"
echo ""

eval $CMD 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo ""
echo "结果对比:"
echo "  - 与原版 two_stage_verification 对比 Val Loss"
echo "  - 如果 Cross-Attn 版本更低 → 直接交互有帮助"
echo "  - 关注 Cross-Attn 梯度是否正常"
echo "============================================"
