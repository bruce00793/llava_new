# LLaVA Map Detection Training Guide

## 训练阶段说明

### Stage 2: 联合训练（当前阶段）

**训练策略：**
- ✅ **Q-Former**: 微调（从BLIP-2预训练权重开始，学习率 1e-5）
- ✅ **Map Queries**: 从头训练（学习率 1e-4）
- ✅ **Map Decoder**: 从头训练（学习率 1e-4）
- ❌ **LLM Backbone**: 冻结（不更新参数）

**优势：**
- 保留LLM的预训练能力
- 大幅降低显存需求（~23GB vs ~93GB）
- 训练速度快（只更新1.6%的参数）
- 防止过拟合

---

## 快速开始

### 1. 准备数据

#### 1.1 下载nuScenes数据集

```bash
# 下载数据到 /path/to/nuscenes
# 目录结构：
# /path/to/nuscenes/
#   ├── maps/
#   ├── samples/
#   ├── sweeps/
#   └── v1.0-mini/ (或 v1.0-trainval)
```

#### 1.2 生成GT Cache（首次训练前必须执行）

```bash
cd /home/cly/auto/llava_test/LLaVA

# 生成train和val的GT cache
python tools/generate_gt_cache.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --split train \
    --output gt_cache_train.pkl

python tools/generate_gt_cache.py \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --split val \
    --output gt_cache_val.pkl
```

**GT Cache的作用：**
- 预处理所有样本的GT（坐标转换、采样到20点、计算AABB等）
- 避免训练时重复计算，大幅加速数据加载
- train set: ~700个样本（mini）/ ~28K个样本（trainval）
- val set: ~200个样本（mini）/ ~6K个样本（trainval）

---

### 2. 修改配置

编辑 `scripts/train_stage2.sh`：

```bash
# 修改这些路径
DATAROOT="/path/to/nuscenes"  # ← 改为你的nuScenes路径
VERSION="v1.0-mini"            # ← 或 v1.0-trainval

# 可选：修改LLM路径（如果已下载到本地）
LLM_PATH="lmsys/vicuna-7b-v1.5"  # ← 或本地路径

# 可选：调整显存配置
BATCH_SIZE=4  # 如果显存不足，改为2
```

---

### 3. 开始训练

#### 单卡训练（推荐用于测试）

```bash
cd /home/cly/auto/llava_test/LLaVA
bash scripts/train_stage2.sh
```

#### 多卡训练（推荐用于完整训练）

```bash
cd /home/cly/auto/llava_test/LLaVA

# 修改 scripts/train_stage2_distributed.sh 中的 NUM_GPUS
# 然后运行：
bash scripts/train_stage2_distributed.sh
```

---

## 显存需求

### 单卡配置

| 配置 | 显存占用 | 推荐GPU |
|-----|---------|---------|
| Batch=4, FP16 | ~23GB | A100 40GB |
| Batch=2, FP16 | ~15GB | V100 32GB |
| Batch=1, FP16 | ~10GB | RTX 3090 24GB |

### 多卡配置（推荐）

| GPUs | Per-GPU Batch | Total Batch | 显存/卡 |
|------|---------------|-------------|---------|
| 4×A100 | 4 | 16 | ~23GB |
| 4×V100 | 2 | 8 | ~15GB |
| 8×A100 | 4 | 32 | ~23GB |

---

## 训练监控

### 日志输出

训练过程中会打印：

```
Epoch [1/24] Step [10/200] Loss: 3.5421 (Avg: 3.6234) LR: q=1.00e-05 m=1.00e-04 d=1.00e-04
  Detailed losses:
    loss_cls: 1.2345 (Avg: 1.3456)
    loss_pts: 1.8765 (Avg: 1.9012)
    loss_dir: 0.4311 (Avg: 0.3766)
    loss_total: 3.5421 (Avg: 3.6234)
```

**指标说明：**
- `loss_cls`: 分类损失（Focal Loss）
- `loss_pts`: 点回归损失（L1 Loss）
- `loss_dir`: 方向损失（Cosine Loss）
- `loss_total`: 总损失（加权和）

### 检查点保存

```
outputs/map_detection_stage2_YYYYMMDD_HHMMSS/
├── config.json                  # 训练配置
├── best_model.pth              # 验证集最优模型
├── checkpoint_epoch_1.pth      # 每epoch的检查点
├── checkpoint_epoch_2.pth
├── ...
└── final_model.pth             # 最终模型
```

---

## 验证和推理

### 验证模型性能

```bash
python test_map_model.py \
    --checkpoint outputs/map_detection_stage2_XXX/best_model.pth \
    --dataroot /path/to/nuscenes \
    --version v1.0-mini \
    --split val
```

### 单样本推理

```python
import torch
from llava.model.map_llava_model import build_map_detector

# 加载模型
model = build_map_detector(
    llm_path='lmsys/vicuna-7b-v1.5',
    freeze_llm=True,
    qformer_pretrained='blip2'
)

# 加载训练好的权重
checkpoint = torch.load('outputs/map_detection_stage2_XXX/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model = model.cuda().eval()

# 推理
with torch.no_grad():
    output = model(images, text_ids, return_loss=False)
    pred_logits = output['pred_logits']  # [B, 50, 4]
    pred_points = output['pred_points']  # [B, 50, 20, 2]
```

---

## 超参数调优建议

### 学习率

```python
# 默认配置（推荐）
LR_QFORMER = 1e-5   # BLIP-2预训练，小学习率微调
LR_QUERIES = 1e-4   # 从头训练，标准学习率
LR_DECODER = 1e-4   # 从头训练，标准学习率

# 如果Loss不下降，尝试：
LR_QFORMER = 5e-6   # 降低Q-Former学习率
LR_QUERIES = 5e-5   # 降低Queries学习率

# 如果Loss震荡，尝试：
WARMUP_STEPS = 1000  # 增加warmup步数
GRAD_CLIP = 0.05     # 降低梯度裁剪阈值
```

### Batch Size

```python
# 有效batch size应在8-32之间
Effective_Batch = Batch_per_GPU × Num_GPUs × Gradient_Accumulation

# 示例：
# 4 GPUs × batch=2 × grad_accum=2 = 16 (推荐)
# 1 GPU × batch=4 × grad_accum=4 = 16 (等效)
```

### 训练轮数

```python
# v1.0-mini (小数据集)
EPOCHS = 24   # 默认

# v1.0-trainval (完整数据集)
EPOCHS = 12   # 数据量大，收敛快
```

---

## 常见问题

### Q1: OOM (Out of Memory)

**解决方案：**
```bash
# 1. 降低batch size
BATCH_SIZE=2  # 或 1

# 2. 启用混合精度（默认已启用）
--fp16

# 3. 减少workers
NUM_WORKERS=2

# 4. 使用梯度累积（需要修改代码）
```

### Q2: GT Cache生成失败

**可能原因：**
- nuScenes路径不正确
- 版本不匹配（mini vs trainval）
- 缺少map数据

**检查方法：**
```bash
# 验证nuScenes加载
python -c "
from nuscenes.nuscenes import NuScenes
nusc = NuScenes(version='v1.0-mini', dataroot='/path/to/nuscenes')
print(f'Loaded {len(nusc.sample)} samples')
"
```

### Q3: BLIP-2下载失败

**解决方案：**
```bash
# 手动下载BLIP-2模型
# 或使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 或跳过BLIP-2预训练（不推荐）
--qformer-pretrained none
```

### Q4: Loss不下降

**检查清单：**
1. ✅ GT cache生成正确
2. ✅ 数据预处理正确（坐标归一化）
3. ✅ 学习率设置合理
4. ✅ 梯度正常传播（检查`requires_grad`）

**调试命令：**
```bash
# 打印模型参数统计
python -c "
from llava.model.map_llava_model import build_map_detector
model = build_map_detector(freeze_llm=True, qformer_pretrained='blip2')
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f'{name}: {param.shape}')
"
```

---

## 下一步：Stage 3（可选）

**如果Stage 2的性能已经满足需求，可以跳过Stage 3。**

Stage 3会解冻LLM进行微调，需要：
- 更多显存（~93GB）
- 更小学习率（1e-6）
- 更短训练时间（3 epochs）

---

## 联系与支持

- 代码路径: `/home/cly/auto/llava_test/LLaVA`
- 历史对话: 查看完整pipeline设计
- 问题反馈: 记录到issue或对话中

---

**祝训练顺利！🚀**

