#!/bin/bash
# ============================================
# LLM Verification - 验证 LLM 输出的 Map Features 质量
# ============================================
# 验证目标：
#   假设 Q-Former 输出正确，验证 LLM + Map Queries 能否：
#   1. 理解场景信息
#   2. 输出有意义的 instance_features
#
# 验证方法：Linear Probing
#   - 冻结 Q-Former + LLM
#   - 只训练一个简单的分类头
#   - 如果分类头能预测正确类别 → LLM features 有效
#
# 验证流程：
#   6 张图 → Q-Former → 768 scene tokens
#                         ↓
#          [Text + Scene Tokens + 1050 Map Queries]
#                         ↓
#                     LLM Forward
#                         ↓
#          instance_features [B, 50, 4096]
#                         ↓
#          简单分类头 → 类别预测
#                         ↓
#                与 GT 比较
#
# 验证结论：
#   ✅ 成功 (Acc > 60%) → LLM features 有效，问题在 MapDecoder
#   ❌ 失败 → LLM/LoRA/Map Queries 有问题
#
# 使用方法:
#   1. 申请 1 卡 4090: salloc -p gpu8 --gres=gpu:4090:1 -t 5-00:00:00 --cpus-per-task=8 --mem=64G
#   2. 进入节点: srun --pty bash
#   3. 运行: cd /home/cly/auto/llava_test/LLaVA && bash run_llm_verification.sh
# ============================================

set -e

echo "============================================"
echo "LLM Verification - Linear Probing on Map Features"
echo "============================================"
echo "开始时间: $(date)"

# ========== 环境配置 ==========
source ~/.bashrc
conda activate llava_new

export CUDA_VISIBLE_DEVICES=0

PROJECT_DIR="/home/cly/auto/llava_test/LLaVA"
cd $PROJECT_DIR

# ========== 数据配置 ==========
DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes"
VERSION="v1.0-trainval"
GT_CACHE="${DATAROOT}/gt_cache"
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"

# 使用 10% 数据
SAMPLE_RATIO=0.1

# ========== 训练配置 ==========
BATCH_SIZE=2
ACCUMULATION_STEPS=4
EPOCHS=10
LR=1e-4

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/llm_verification_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: 1×RTX 4090"
echo "  验证方法: Linear Probing on instance_features"
echo "  验证目标: LLM 输出的 50 个实例 features 能否预测类别"
echo "  数据比例: ${SAMPLE_RATIO} (10%)"
echo "  Batch Size: $BATCH_SIZE × $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  LLM: $LLM_PATH"
echo "  成功标准: Map Element Accuracy > 60%"
echo "  输出目录: $OUTPUT_DIR"
echo ""

# ========== 检查 ==========
if [ ! -d "$GT_CACHE" ]; then
    echo "⚠️  GT Cache 不存在: $GT_CACHE"
    DATAROOT="/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
    VERSION="v1.0-mini"
    GT_CACHE="${DATAROOT}/gt_cache"
    SAMPLE_RATIO=1.0
    echo "   切换到 mini 数据集"
fi

# ========== 保存配置 ==========
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLM Verification - Linear Probing
========================================
Start Time: $(date)
GPU: 1×RTX 4090
Dataset: nuScenes $VERSION
Sample Ratio: $SAMPLE_RATIO
LLM: $LLM_PATH

验证方法: Linear Probing on instance_features
  - 冻结 Q-Former + LLM
  - 只训练一个简单的分类头 (4096 → 4)
  - 验证 LLM 输出的 features 是否包含类别信息

验证流程:
  6 张图 → Q-Former → 768 scene tokens
                        ↓
         [Text + Scene Tokens + 1050 Map Queries]
                        ↓
                    LLM Forward
                        ↓
         instance_features [B, 50, 4096]
                        ↓
         Linear Layer → 类别预测 [B, 50, 4]

成功标准:
  - Map Element Accuracy > 60%

验证结论:
  ✅ 成功 → LLM features 有效，问题在 MapDecoder 点预测部分
  ❌ 失败 → LLM/LoRA/Map Queries/Cross-Attention 有问题
========================================
EOF

# ========== 开始验证 ==========
echo "============================================"
echo "开始 LLM 验证..."
echo "============================================"

python train_llm_verification.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache $GT_CACHE \
    --llm-path $LLM_PATH \
    --sample-ratio $SAMPLE_RATIO \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr $LR \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee "${OUTPUT_DIR}/verification.log"

echo ""
echo "============================================"
echo "✅ 验证完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
