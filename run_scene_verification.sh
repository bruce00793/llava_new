#!/bin/bash
# ============================================
# Scene Verification - 验证 Q-Former 768 queries 的表示能力
# ============================================
# 目的：验证 768 scene queries 能否准确代表 6 张图片的全部场景内容
# 与主训练架构一致，使用 768 个 scene tokens
# 不使用 1050 map queries
# 输出：场景中所有元素（23类3D目标 + 3类地图元素）的类别和位置
#
# 使用方法:
#   1. 申请 1 卡 4090: salloc -p gpu8 --gres=gpu:4090:1 -t 5-00:00:00 --cpus-per-task=8 --mem=32G
#   2. 进入节点: srun --pty bash
#   3. 运行脚本: cd /home/cly/auto/llava_test/LLaVA && bash run_scene_verification.sh
# ============================================

set -e

echo "============================================"
echo "Scene Verification - Q-Former 768 queries"
echo "============================================"
echo "开始时间: $(date)"

# ========== 环境配置 ==========
source ~/.bashrc
conda activate llava_new

# 设置 1 张 4090
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# 项目路径
PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ========== 数据配置 ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="${DATAROOT}/gt_cache"

# 使用15%的数据
SAMPLE_RATIO=0.15

# ========== 模型配置 ==========
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# 预训练 Q-Former 权重（不使用）
# QFORMER_PRETRAINED="/home/cly/auto/llava_test/LLaVA/outputs/h100_v1.0-trainval_20260114_142901/best_model.pth"
QFORMER_PRETRAINED=""

# ========== LoRA 配置 ==========
LORA_R=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
LR_LORA=2e-4

# ========== 训练配置 ==========
BATCH_SIZE=1
ACCUMULATION_STEPS=8  # 有效 batch = 8
EPOCHS=24

# Q-Former 学习率
LR_QFORMER_BACKBONE=3e-5
LR_QFORMER_DECODER=2e-4
LR_QFORMER_PROJECTOR=2.5e-4

# 其他
WEIGHT_DECAY=0.01
WARMUP_STEPS=100
GRAD_CLIP=10.0

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/scene_verification_768q_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: 1×RTX 4090 (48GB)"
echo "  验证目标: 768 scene queries 表示能力 (与主训练架构一致)"
echo "  不使用 1050 map queries"
if [ -n "$QFORMER_PRETRAINED" ] && [ -f "$QFORMER_PRETRAINED" ]; then
    echo "  预训练 Q-Former: $QFORMER_PRETRAINED"
else
    echo "  预训练 Q-Former: 不使用"
fi
echo "  数据集: $VERSION"
echo "  数据比例: ${SAMPLE_RATIO} (15%)"
echo "  Batch Size: $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo ""
echo "  Q-Former 学习率:"
echo "    backbone=$LR_QFORMER_BACKBONE"
echo "    decoder=$LR_QFORMER_DECODER"
echo "    projector=$LR_QFORMER_PROJECTOR"
echo ""
echo "  LoRA 配置:"
echo "    rank=$LORA_R, alpha=$LORA_ALPHA"
echo "    lr=$LR_LORA"
echo ""
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 检查 ==========
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM 检查通过"

if [ ! -d "$GT_CACHE" ]; then
    echo "⚠️  GT Cache 不存在: $GT_CACHE"
    # 切换到 mini 数据集
    DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
    VERSION="v1.0-mini"
    GT_CACHE="${DATAROOT}/gt_cache_v1.0-mini_train.pkl"
    SAMPLE_RATIO=1.0
    echo "   使用 mini 数据集: $DATAROOT"
fi
echo "✅ GT Cache 检查通过"

# 检查 pyquaternion
python -c "from pyquaternion import Quaternion" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ pyquaternion 未安装，正在安装..."
    pip install pyquaternion -q
fi
echo "✅ pyquaternion 检查通过"

# 保存配置
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
Scene Verification - Q-Former 768 queries
========================================
Start Time: $(date)
GPU: 1×RTX 4090
Dataset: nuScenes $VERSION
Sample Ratio: $SAMPLE_RATIO
Epochs: $EPOCHS
Batch Size (effective): $((BATCH_SIZE * ACCUMULATION_STEPS))
FP16: Enabled

验证目标: 768 scene queries 能否代表 6 张图片的全部内容
与主训练架构一致（768 = 6相机 × 128 tokens/相机）
不使用 1050 map queries

LoRA Config:
  Rank: $LORA_R
  Alpha: $LORA_ALPHA
  LR: $LR_LORA

Q-Former LR:
  Backbone: $LR_QFORMER_BACKBONE
  Decoder: $LR_QFORMER_DECODER
  Projector: $LR_QFORMER_PROJECTOR

GT Cache: $GT_CACHE
========================================
EOF

# ========== 开始训练 ==========
echo ""
echo "============================================"
echo "开始训练..."
echo "============================================"
echo ""

CMD="python train_scene_verification.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache $GT_CACHE \
    --sample-ratio $SAMPLE_RATIO \
    --llm-path $LLM_PATH"

# 如果有预训练 Q-Former，添加参数
if [ -n "$QFORMER_PRETRAINED" ] && [ -f "$QFORMER_PRETRAINED" ]; then
    CMD="$CMD --qformer-pretrained $QFORMER_PRETRAINED"
    echo "✅ 使用预训练 Q-Former: $QFORMER_PRETRAINED"
else
    echo "ℹ️ 不使用预训练 Q-Former，从头开始训练"
fi

CMD="$CMD \
    --use-lora \
    --lora-r $LORA_R \
    --lora-alpha $LORA_ALPHA \
    --lora-dropout $LORA_DROPOUT \
    --lr-lora $LR_LORA \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr-qformer-backbone $LR_QFORMER_BACKBONE \
    --lr-qformer-decoder $LR_QFORMER_DECODER \
    --lr-qformer-projector $LR_QFORMER_PROJECTOR \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    --log-interval 20"

echo "执行命令: $CMD"
eval $CMD 2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
