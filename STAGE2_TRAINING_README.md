# Stage 2 训练 - 快速启动指南

## 📋 概览

**Stage 2: 联合训练（使用BLIP-2预训练Q-Former）**

- ✅ Q-Former: 1e-5 (微调)
- ✅ Map Queries: 1e-4 (从头训练)  
- ✅ Map Decoder: 1e-4 (从头训练)
- ❌ LLM: 冻结

---

## 🚀 快速开始（3步）

### Step 1: 生成GT Cache

```bash
cd /home/cly/auto/llava_test/LLaVA

# Train set
python tools/generate_gt_cache.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --split train \
    --output gt_cache_train.pkl

# Val set
python tools/generate_gt_cache.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --split val \
    --output gt_cache_val.pkl
```

### Step 2: 修改配置

编辑 `scripts/train_stage2.sh`:

```bash
# 修改这一行：
DATAROOT="/path/to/nuscenes"  # ← 改为你的实际路径
```

### Step 3: 开始训练

```bash
# 单卡训练
bash scripts/train_stage2.sh

# 或多卡训练（推荐）
bash scripts/train_stage2_distributed.sh
```

---

## 🔧 训练前测试

验证环境和模型配置：

```bash
cd /home/cly/auto/llava_test/LLaVA
conda activate llava_new
python test_training_setup.py
```

预期输出：
```
✅ All core tests passed!
```

---

## 📊 显存需求

| 配置 | 显存 | GPU推荐 |
|-----|------|---------|
| Batch=4 | 23GB | A100 40GB |
| Batch=2 | 15GB | V100 32GB |
| Batch=1 | 10GB | RTX 3090 |

---

## 📂 文件说明

```
llava_test/LLaVA/
├── train_map_detection.py              # 主训练脚本
├── scripts/
│   ├── train_stage2.sh                 # 单卡启动脚本
│   └── train_stage2_distributed.sh     # 多卡启动脚本
├── test_training_setup.py              # 训练前测试
├── TRAINING_GUIDE.md                   # 详细训练指南
└── tools/
    └── generate_gt_cache.py            # GT生成工具
```

---

## 📈 监控训练

### 日志示例

```
Epoch [1/24] Step [10/200] Loss: 3.5421 (Avg: 3.6234)
LR: q=1.00e-05 m=1.00e-04 d=1.00e-04

Detailed losses:
  loss_cls: 1.2345    # 分类损失
  loss_pts: 1.8765    # 点回归损失
  loss_dir: 0.4311    # 方向损失
  loss_total: 3.5421  # 总损失
```

### 检查点位置

```
outputs/map_detection_stage2_YYYYMMDD_HHMMSS/
├── best_model.pth           # ← 验证集最优模型
├── checkpoint_epoch_N.pth   # 每epoch保存
└── final_model.pth          # 训练结束
```

---

## ❓ 常见问题

### Q: OOM错误

```bash
# 降低batch size
BATCH_SIZE=2  # 或 1
```

### Q: GT Cache生成失败

```bash
# 检查nuScenes路径
ls /path/to/nuscenes/v1.0-mini/
# 应该看到 scene.json, sample.json 等文件
```

### Q: BLIP-2下载慢

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com
```

### Q: 想跳过BLIP-2预训练

```bash
# 修改 train_stage2.sh:
--qformer-pretrained none  # 不推荐，性能会下降
```

---

## 📖 完整文档

详细说明请查看：[TRAINING_GUIDE.md](./TRAINING_GUIDE.md)

---

## ✅ 训练检查清单

- [ ] nuScenes数据集已下载
- [ ] GT Cache已生成（train + val）
- [ ] 修改了 `scripts/train_stage2.sh` 中的 `DATAROOT`
- [ ] 运行了 `test_training_setup.py` 并通过
- [ ] 显存足够（参考上表）
- [ ] 环境激活：`conda activate llava_new`

**准备好了？开始训练！🚀**

```bash
bash scripts/train_stage2.sh
```

---

**预计训练时间：**
- v1.0-mini: ~2小时 (A100)
- v1.0-trainval: ~24小时 (4×A100)

