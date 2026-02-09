#!/bin/bash
# ============================================
# 6×4090 训练脚本 - LLaVA Map Detection
# ============================================
# 使用方法:
#   1. 申请资源:
#      salloc -p gpu8 --gres=gpu:4090:6 -t 5-00:00:00 --cpus-per-task=48 --mem=192G
#   
#   2. 进入计算节点:
#      srun --pty bash
#
#   3. 运行训练:
#      cd /home/cly/auto/llava_test/LLaVA
#      bash run_train_4x4090.sh
# ============================================

set -e

echo "============================================"
echo "LLaVA Map Detection - 6×4090 训练"
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
# 本地路径加载，不使用 blip2
LLM_PATH="/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
QFORMER_PRETRAINED="none"  # 从头训练 Q-Former

# ========== 训练配置 (6×4090) ==========
# Effective batch size = 6卡 × 1 × 5累积 = 30 (接近 MapTR 的 32)
BATCH_SIZE=1
ACCUMULATION_STEPS=5
EPOCHS=24

# 学习率配置 (当前训练有效，保持原配置)
LR_BACKBONE=5e-5       # Q-Former backbone
LR_DECODER=4e-4        # Q-Former decoder
LR_PROJECTOR=5e-4      # Q-Former projector
LR_QUERIES=5e-4        # Map Queries
LR_MAP_DECODER=5e-4    # Map Decoder
LR_LORA=2e-4           # LoRA 参数

# 优化器配置
WEIGHT_DECAY=0.01
WARMUP_STEPS=1000      # warmup 步数
GRAD_CLIP=35.0         # 梯度裁剪 (与MapTR一致)

# ========== 输出配置 ==========
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="${PROJECT_DIR}/outputs/6x4090_${TIMESTAMP}"
mkdir -p $OUTPUT_DIR

# ========== 打印配置 ==========
echo ""
echo "=========================================="
echo "配置信息:"
echo "=========================================="
echo "  GPU: ${NUM_GPUS}×RTX 4090"
echo "  数据集: $VERSION (nuScenes 全量)"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: ${NUM_GPUS}卡 × $BATCH_SIZE × $ACCUMULATION_STEPS累积 = $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))"
echo ""
echo "学习率:"
echo "  Q-Former Backbone: $LR_BACKBONE"
echo "  Q-Former Decoder:  $LR_DECODER"
echo "  Q-Former Projector: $LR_PROJECTOR"
echo "  Map Queries:       $LR_QUERIES"
echo "  Map Decoder:       $LR_MAP_DECODER"
echo "  LoRA:              $LR_LORA"
echo ""
echo "其他:"
echo "  Weight Decay: $WEIGHT_DECAY"
echo "  Warmup Steps: $WARMUP_STEPS"
echo "  Grad Clip: $GRAD_CLIP"
echo "  BF16: 启用 (替代FP16，防止梯度溢出)"
echo "  EMA: 启用"
echo ""
echo "输出目录: $OUTPUT_DIR"
echo "=========================================="
echo ""

# ========== 环境检查 ==========
echo "[检查环境...]"

# 检查 GT Cache
if [ ! -d "$GT_CACHE_DIR" ]; then
    echo "❌ GT Cache 不存在: $GT_CACHE_DIR"
    echo "   请先运行: python tools/generate_gt_from_maptr.py"
    exit 1
fi
echo "✅ GT Cache: $GT_CACHE_DIR"

# 检查 LLM
if [ ! -d "$LLM_PATH" ]; then
    echo "❌ LLM 不存在: $LLM_PATH"
    exit 1
fi
echo "✅ LLM: $LLM_PATH"

# 检查 GPU
echo ""
echo "[GPU 信息]"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# ========== 保存配置 ==========
cat > "${OUTPUT_DIR}/config.txt" << EOF
========================================
LLaVA Map Detection - 6×4090 Training
========================================
Start Time: $(date)
GPU: ${NUM_GPUS}×RTX 4090
Dataset: nuScenes $VERSION
Epochs: $EPOCHS
Batch Size (effective): $((NUM_GPUS * BATCH_SIZE * ACCUMULATION_STEPS))

Learning Rates:
  Q-Former Backbone: $LR_BACKBONE
  Q-Former Decoder:  $LR_DECODER
  Q-Former Projector: $LR_PROJECTOR
  Map Queries:       $LR_QUERIES
  Map Decoder:       $LR_MAP_DECODER
  LoRA:              $LR_LORA

Optimizer:
  Weight Decay: $WEIGHT_DECAY
  Warmup Steps: $WARMUP_STEPS
  Grad Clip: $GRAD_CLIP

Loss Weights (from map_loss.py):
  weight_cls: 2.0
  weight_pts: 5.0
  weight_dir: 0.25 (折中方案)

LoRA Config:
  rank: 32
  alpha: 64
  target_modules: q_proj, k_proj, v_proj, o_proj

Q-Former: From scratch (no BLIP2 pretrain)
========================================
EOF

# ========== 开始训练 ==========
echo "============================================"
echo "开始 6×4090 分布式训练..."
echo "预计时间: ~2-3 天"
echo "============================================"
echo ""

# 使用 torchrun 进行分布式训练
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
    --lr-lora $LR_LORA \
    --weight-decay $WEIGHT_DECAY \
    --warmup-steps $WARMUP_STEPS \
    --grad-clip $GRAD_CLIP \
    --use-ema \
    --ema-decay 0.9999 \
    --bf16 \
    --output-dir $OUTPUT_DIR \
    --resume "/home/cly/auto/llava_test/LLaVA/outputs/6x4090_20260206_180145/checkpoint_epoch_12.pth" \
    --log-interval 20 \
    --save-interval 1 \
    --eval-interval 1 \
    2>&1 | tee -a "${OUTPUT_DIR}/train.log"

# ========== 训练完成 ==========
echo ""
echo "============================================"
echo "✅ 训练完成！"
echo "   结束时间: $(date)"
echo "   输出目录: $OUTPUT_DIR"
echo "============================================"

# 显示最终结果
if [ -f "${OUTPUT_DIR}/train.log" ]; then
    echo ""
    echo "[最后几行日志]"
    tail -20 "${OUTPUT_DIR}/train.log"
fi
