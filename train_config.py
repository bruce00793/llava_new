# ============================================
# 训练配置 - 对齐MapTR (1× H100 80GB)
# ============================================

import os

# ============================================
# 路径配置
# ============================================
DATA_ROOT = "/home/cly/auto/llava_test/LLaVA/data/nuscenes"  # 完整nuScenes路径
# DATA_ROOT = "/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"  # Mini版本(调试用)
LLM_PATH = "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
CLIP_PATH = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
OUTPUT_DIR = "/home/cly/auto/llava_test/LLaVA/outputs"

# ============================================
# 数据配置
# ============================================
DATASET_VERSION = "v1.0-trainval"  # 完整版
# DATASET_VERSION = "v1.0-mini"    # Mini版(调试用)
NUM_WORKERS = 8  # H100服务器通常有更多CPU核心

# ============================================
# Batch Size 配置 (1× H100 80GB)
# ============================================
# 保守配置：先用 batch_size=2 确保稳定
# 如果显存充足，可以尝试提升到 batch_size=4
# 显存估算：LLM(14GB) + Q-Former(3GB) + Decoder(1GB) + 激活(~10GB/batch)
BATCH_SIZE_PER_GPU = 2    # 保守起见先用2，稳定后可尝试4
NUM_GPUS = 1              # 1张H100
ACCUMULATION_STEPS = 16   # 梯度累积步数 (2×16=32)

# 等效Batch Size = 2 × 1 × 16 = 32 (与MapTR相同)
EFFECTIVE_BATCH_SIZE = BATCH_SIZE_PER_GPU * NUM_GPUS * ACCUMULATION_STEPS

# ============================================
# 学习率配置 (方案A: 适度提高，加速收敛)
# ============================================
# MapTR 使用 6e-4，但我们的架构包含 LLM
# 由于 LLM 完全冻结，可训练参数量 (~56M) 和 MapTR (~50M) 相当
# 方案A: 适度提高学习率 (比原配置高 2-5 倍，但仍比 MapTR 保守)
BASE_LR = 4e-4  # 提高到 4e-4

# 分组学习率 (方案A调整)
LR_CONFIG = {
    'qformer_backbone': 5e-5,    # 1e-5 → 5e-5 (提高5倍，ImageNet预训练需谨慎)
    'qformer_decoder': 3e-4,     # 1e-4 → 3e-4 (提高3倍)
    'qformer_projector': 4e-4,   # 2e-4 → 4e-4 (提高2倍，关键桥梁)
    'map_queries': 4e-4,         # 2e-4 → 4e-4 (提高2倍，从头训练)
    'map_decoder': 4e-4,         # 2e-4 → 4e-4 (提高2倍，从头训练)
}

# ============================================
# 优化器配置
# ============================================
OPTIMIZER = "AdamW"
WEIGHT_DECAY = 0.01       # MapTR使用0.01，保持一致
GRAD_CLIP_NORM = 15.0     # 5.0 → 15.0 (提高3倍，MapTR用35)

# ============================================
# 学习率调度
# ============================================
LR_SCHEDULER = "cosine"   # 余弦退火
WARMUP_ITERS = 1000       # warmup步数（增加到1000，因为模型更大）
WARMUP_RATIO = 1.0 / 3    # warmup起始比例
MIN_LR_RATIO = 1e-3       # 最小学习率比例

# ============================================
# 训练配置
# ============================================
TOTAL_EPOCHS = 24         # MapTR训练24个epoch
EVAL_INTERVAL = 1         # 每1个epoch评估 (单卡训练时频繁评估)
SAVE_INTERVAL = 2         # 每2个epoch保存checkpoint
LOG_INTERVAL = 20         # 每20步打印日志 (累积步数较多，减少日志间隔)

# ============================================
# 混合精度
# ============================================
USE_FP16 = True           # H100支持FP16/BF16，使用混合精度加速
FP16_LOSS_SCALE = 512.0   # MapTR使用512

# ============================================
# Loss权重 (理性分析后的折中方案)
# ============================================
# Dir Loss 按实例数归一化，量级约 9.5
# 
# 损失贡献分析 (weight_dir=0.25):
#   Pts Loss (~4) × 5.0 = 20.0  (主导，约65%)
#   Cls Loss (~0.5) × 2.0 = 1.0  (辅助，约3%)
#   Dir Loss (~9.5) × 0.25 = 2.4  (约8%，有意义但不主导)
# 
# 对比:
#   - MapTR 用 0.005 → 方向损失几乎无作用 (0.5%)
#   - 之前 2.0~2.5 → 方向损失主导训练 (60%+)
#   - 折中 0.25 → 方向损失有意义但不主导 (8%)
LOSS_WEIGHTS = {
    'cls': 2.0,
    'pts': 5.0,
    'dir': 0.25,   # 折中方案：方向有约束但不主导
}

# ============================================
# 模型配置
# ============================================
FREEZE_LLM = True         # 冻结LLM
NUM_INSTANCES = 50        # 预测实例数
NUM_POINTS = 20           # 每个实例的点数
NUM_CLASSES = 3           # divider, ped_crossing, boundary

# ============================================
# 其他
# ============================================
SEED = 42
DETERMINISTIC = False     # 设为True可复现，但会变慢

# ============================================
# H100 特定优化
# ============================================
# H100 支持更高效的矩阵运算，可以考虑：
# 1. 使用 torch.compile() 加速 (PyTorch 2.0+)
# 2. 使用 BF16 替代 FP16 (更稳定，H100原生支持)
# 3. 使用 Flash Attention (如果模型支持)
USE_TORCH_COMPILE = False  # 实验性功能，可能需要调试
USE_BF16 = False           # BF16更稳定，但需要验证兼容性

# ============================================
# 打印配置
# ============================================
def print_config():
    print("=" * 70)
    print("训练配置 - 1× H100 80GB (对齐MapTR)")
    print("=" * 70)
    print(f"[数据]")
    print(f"  数据集: {DATASET_VERSION}")
    print(f"  数据路径: {DATA_ROOT}")
    print(f"  Workers: {NUM_WORKERS}")
    print()
    print(f"[Batch Size]")
    print(f"  单卡Batch: {BATCH_SIZE_PER_GPU}")
    print(f"  GPU数量: {NUM_GPUS}")
    print(f"  梯度累积: {ACCUMULATION_STEPS}")
    print(f"  等效Batch: {EFFECTIVE_BATCH_SIZE} (与MapTR相同)")
    print()
    print(f"[学习率]")
    print(f"  基础LR: {BASE_LR}")
    print(f"  调度器: {LR_SCHEDULER}")
    print(f"  Warmup: {WARMUP_ITERS} steps")
    print()
    print(f"[训练]")
    print(f"  Epochs: {TOTAL_EPOCHS}")
    print(f"  混合精度: {'FP16' if USE_FP16 else 'FP32'}")
    print(f"  梯度裁剪: {GRAD_CLIP_NORM}")
    print()
    print(f"[Loss权重] (MapTR)")
    for key, value in LOSS_WEIGHTS.items():
        print(f"  {key}: {value}")
    print("=" * 70)
    
    # 估算训练时间
    estimate_training_time()

def estimate_training_time():
    """估算训练时间"""
    # nuScenes trainval: ~28,130 训练样本
    # Mini: ~324 训练样本
    if 'mini' in DATASET_VERSION:
        num_samples = 324
    else:
        num_samples = 28130
    
    # 每个epoch的步数
    steps_per_epoch = num_samples // BATCH_SIZE_PER_GPU
    total_steps = steps_per_epoch * TOTAL_EPOCHS
    
    # H100 估算: ~0.5秒/step (含梯度累积)
    # 这是保守估计，实际可能更快
    seconds_per_step = 0.5
    total_seconds = total_steps * seconds_per_step
    
    hours = total_seconds / 3600
    
    print()
    print(f"[训练时间估算]")
    print(f"  训练样本数: {num_samples:,}")
    print(f"  每Epoch步数: {steps_per_epoch:,}")
    print(f"  总步数: {total_steps:,}")
    print(f"  预估时间: {hours:.1f} 小时 ({hours/24:.1f} 天)")
    print()
    print("  注: 实际时间可能因数据加载、验证等有所不同")
    print("=" * 70)

if __name__ == "__main__":
    print_config()

