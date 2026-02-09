# 🚀 训练快速开始（3分钟上手）

## 核心命令（复制粘贴即可）

### 1️⃣ 生成GT Cache（只需运行一次）

```bash
cd /home/cly/auto/llava_test/LLaVA

# 修改下面的路径为你的nuScenes路径
DATAROOT="/path/to/nuscenes"
VERSION="v1.0-mini"

# 生成训练集GT
python tools/generate_gt_cache.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --split train \
    --output gt_cache_train.pkl

# 生成验证集GT
python tools/generate_gt_cache.py \
    --dataroot "$DATAROOT" \
    --version "$VERSION" \
    --split val \
    --output gt_cache_val.pkl
```

### 2️⃣ 测试环境（可选但推荐）

```bash
conda activate llava_new
python test_training_setup.py
```

应该看到：`✅ All core tests passed!`

### 3️⃣ 修改配置并开始训练

```bash
# 编辑配置
vim scripts/train_stage2.sh
# 只需修改这一行：
# DATAROOT="/path/to/nuscenes"  ← 改为你的实际路径

# 开始训练！
bash scripts/train_stage2.sh
```

---

## 🎯 预期输出

### 训练开始时
```
==========================================
LLaVA Map Detection - Stage 2 Training
==========================================
Building model...
📥 Loading BLIP-2 pretrained Q-Former...
✅ BLIP-2 Q-Former loaded!
✅ LLM loaded successfully!
✅ Map Decoder initialized
✅ LLM frozen, only training:
   - Q-Former
   - Map Queries (1050 learnable queries)
   - Map Decoder

Parameter Groups:
  qformer:     100,234,567 params, lr=1e-05
  queries:       4,300,800 params, lr=1e-04
  decoder:      11,234,567 params, lr=1e-04
==========================================

Starting Training...
```

### 训练过程中
```
Epoch [1/24] Step [10/200] Loss: 3.5421 (Avg: 3.6234)
LR: q=1.00e-05 m=1.00e-04 d=1.00e-04

Detailed losses:
  loss_cls: 1.2345 (Avg: 1.3456)
  loss_pts: 1.8765 (Avg: 1.9012)
  loss_dir: 0.4311 (Avg: 0.3766)
  loss_total: 3.5421 (Avg: 3.6234)
```

### 保存checkpoint时
```
💾 Checkpoint saved to outputs/map_detection_stage2_XXX/checkpoint_epoch_1.pth
```

---

## 📊 显存参考

| GPU型号 | 推荐Batch Size | 显存占用 |
|---------|---------------|---------|
| A100 40GB | 4 | 23GB |
| V100 32GB | 2 | 15GB |
| RTX 3090 24GB | 1 | 10GB |

---

## ❓ 快速问题排查

### ❌ 问题：OOM (显存不足)
```bash
# 解决：降低batch size
# 编辑 scripts/train_stage2.sh，修改：
BATCH_SIZE=2  # 或 1
```

### ❌ 问题：GT Cache生成失败
```bash
# 检查路径是否正确
ls /path/to/nuscenes/v1.0-mini/
# 应该看到：scene.json, sample.json 等文件
```

### ❌ 问题：模块导入错误
```bash
# 确认环境激活
conda activate llava_new

# 确认在正确目录
cd /home/cly/auto/llava_test/LLaVA
```

### ❌ 问题：BLIP-2下载慢/失败
```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或跳过BLIP-2（不推荐，性能下降）
# 编辑 scripts/train_stage2.sh：
--qformer-pretrained none
```

---

## 📂 训练输出位置

```
outputs/map_detection_stage2_YYYYMMDD_HHMMSS/
├── best_model.pth           ← 用这个做推理！
├── checkpoint_epoch_N.pth
└── final_model.pth
```

---

## 📖 更多文档

- **详细教程**: [TRAINING_GUIDE.md](./TRAINING_GUIDE.md)
- **完整说明**: [STAGE2_TRAINING_README.md](./STAGE2_TRAINING_README.md)
- **代码总结**: [TRAINING_SUMMARY.md](./TRAINING_SUMMARY.md)

---

## ✅ 准备好了吗？

```bash
# 复制这些命令，修改路径后运行：

cd /home/cly/auto/llava_test/LLaVA
DATAROOT="/path/to/nuscenes"  # ← 修改这里

# Step 1: 生成GT
python tools/generate_gt_cache.py --dataroot "$DATAROOT" --version v1.0-mini --split train --output gt_cache_train.pkl
python tools/generate_gt_cache.py --dataroot "$DATAROOT" --version v1.0-mini --split val --output gt_cache_val.pkl

# Step 2: 测试
python test_training_setup.py

# Step 3: 修改并开始训练
vim scripts/train_stage2.sh  # 修改DATAROOT
bash scripts/train_stage2.sh
```

**就是这么简单！🎉**

