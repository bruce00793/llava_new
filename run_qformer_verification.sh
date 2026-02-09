#!/bin/bash
# ============================================
# Q-Former Verification - Linear Probing 验证 (增强版)
# ============================================
# 验证方法：场景级别目标统计预测（业界标准 Linear Probing）
#
# 核心思想：
# - 用最简单的结构（只用线性层）验证特征质量
# - 如果简单的线性层就能预测出场景中各类目标的数量和位置
# - 说明 Q-Former 768 tokens 包含了完整的场景【语义+位置】信息
#
# 架构：
#   6 张图 → Q-Former → 768 tokens → Global Pooling → 3×Linear
#     → 各类数量 [B, 13]
#     → 各类中心位置均值 [B, 13, 2] (新增)
#     → 各类位置分散度 [B, 13, 2] (新增)
#
# 预测内容：
# - 10 类 3D 目标: car, truck, bus, pedestrian, etc.
# - 3 类地图元素: divider, ped_crossing, boundary
#
# 成功标准：
# - 数量 MAE < 2.0: 平均数量误差小于 2 个
# - 存在性准确率 > 80%: 能准确判断某类目标是否存在
# - 中心 MAE < 0.15: 位置预测误差 (归一化空间) (新增)
#
# 使用方法:
#   1. 申请 1 卡 4090: salloc -p gpu8 --gres=gpu:4090:1 -t 5-00:00:00 --cpus-per-task=8 --mem=32G
#   2. 进入节点: srun --pty bash
#   3. 运行: cd /home/cly/auto/llava_test/LLaVA && bash run_qformer_verification.sh
# ============================================

set -e

echo "============================================"
echo "Q-Former Verification - 768 queries"
echo "不使用 LLM，直接验证 Q-Former"
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

# 使用 15% 数据快速验证
SAMPLE_RATIO=0.15

# ========== 训练配置 ==========
BATCH_SIZE=2
ACCUMULATION_STEPS=4  # 有效 batch = 8
EPOCHS=20
LR=1e-4

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/qformer_verification_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "配置:"
echo "  GPU: 1×RTX 4090"
echo "  验证方法: Linear Probing（业界标准）"
echo "  验证目标: Q-Former 768 tokens 能否代表完整场景信息"
echo "  预测任务: 场景中各类目标的数量（13 类）"
echo "  架构: Q-Former → Global Pooling → Linear (只用 1 个线性层)"
echo "  数据比例: ${SAMPLE_RATIO} (15%)"
echo "  Batch Size: $BATCH_SIZE × $ACCUMULATION_STEPS = $((BATCH_SIZE * ACCUMULATION_STEPS))"
echo "  Epochs: $EPOCHS"
echo "  Learning Rate: $LR"
echo "  成功标准: MAE < 2.0, 存在性准确率 > 80%"
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
Q-Former Verification - Linear Probing
========================================
Start Time: $(date)
GPU: 1×RTX 4090
Dataset: nuScenes $VERSION
Sample Ratio: $SAMPLE_RATIO

验证方法: Linear Probing（业界标准特征验证方法）
验证目标: Q-Former 768 tokens 能否代表完整场景信息

架构（极简设计）:
  6 张图 → Q-Former → 768 tokens → Global Pooling → Linear → 各类数量

预测任务: 场景中各类目标的数量（共 13 类）
  - 10 类 3D 目标: car, truck, bus, trailer, construction_vehicle,
                   pedestrian, motorcycle, bicycle, barrier, traffic_cone
  - 3 类地图元素: divider, ped_crossing, boundary

成功标准:
  - MAE < 2.0: 平均数量误差小于 2 个
  - 存在性准确率 > 80%: 能准确判断某类目标是否存在

验证结论:
  ✅ 成功 → Q-Former 768 tokens 能有效表示场景信息
            如果主训练效果不好，问题在 LLM 或 MapDecoder
  ❌ 失败 → Q-Former 设计需要改进

Epochs: $EPOCHS
Batch Size: $((BATCH_SIZE * ACCUMULATION_STEPS))
Learning Rate: $LR
========================================
EOF

# ========== 开始训练 ==========
echo "============================================"
echo "开始 Q-Former 验证训练..."
echo "============================================"

python train_qformer_verification.py \
    --dataroot $DATAROOT \
    --version $VERSION \
    --gt-cache $GT_CACHE \
    --sample-ratio $SAMPLE_RATIO \
    --epochs $EPOCHS \
    --batch-size $BATCH_SIZE \
    --accumulation-steps $ACCUMULATION_STEPS \
    --num-workers 4 \
    --lr $LR \
    --fp16 \
    --output-dir $OUTPUT_DIR \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================"
echo "✅ 验证完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"
