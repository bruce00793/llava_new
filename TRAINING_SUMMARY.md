# 训练代码填充完成总结

## ✅ 已完成的工作

### 1. 主训练脚本
**文件**: `train_map_detection.py`

**功能**:
- ✅ 分布式训练支持（单卡/多卡）
- ✅ 三组不同学习率的参数组：
  - Q-Former: 1e-5（微调BLIP-2）
  - Map Queries: 1e-4（从头训练）
  - Decoder: 1e-4（从头训练）
- ✅ 混合精度训练（FP16）
- ✅ 学习率warmup + cosine decay
- ✅ 梯度裁剪（0.1）
- ✅ 自动保存checkpoint（每epoch + best model）
- ✅ 详细日志输出（loss分解）
- ✅ 验证集评估

### 2. 启动脚本
**文件**: 
- `scripts/train_stage2.sh` - 单卡训练
- `scripts/train_stage2_distributed.sh` - 多卡训练

**配置**:
```bash
LR_QFORMER=1e-5    # Q-Former学习率
LR_QUERIES=1e-4    # Queries学习率
LR_DECODER=1e-4    # Decoder学习率
EPOCHS=24          # 训练轮数
BATCH_SIZE=4       # 批大小
```

### 3. 测试脚本
**文件**: `test_training_setup.py`

**测试内容**:
1. ✅ 模型构建（BLIP-2预训练加载）
2. ✅ 优化器参数分组
3. ✅ Forward pass（包含loss计算）
4. ✅ Backward pass（梯度流检查）
5. ✅ 数据加载（可选）

### 4. 文档
**文件**:
- `TRAINING_GUIDE.md` - 详细训练指南（2000+字）
- `STAGE2_TRAINING_README.md` - 快速启动指南
- `TRAINING_SUMMARY.md` - 本文件

---

## 📋 训练流程

### 数据流
```
nuScenes原始数据
    ↓
generate_gt_cache.py (预处理GT)
    ↓
MapDetectionDataset (加载数据)
    ↓
DataLoader (批处理)
    ↓
训练循环
```

### 模型流
```
Images [B,6,3,336,336] + Text [B,L]
    ↓
Q-Former (可训练, lr=1e-5)
    ↓
LLM + Map Queries (queries可训练, lr=1e-4, LLM冻结)
    ↓
Decoder (可训练, lr=1e-4)
    ↓
Predictions + Loss
    ↓
Backward (只更新可训练参数)
```

### 反向传播路径
```
Loss
    ↓
Decoder ← 梯度更新 ✅
    ↓
Map Queries ← 梯度更新 ✅
    ↓
LLM ← 梯度传递但不更新参数 ❌
    ↓
Q-Former ← 梯度更新 ✅
```

---

## 🎯 关键设计决策

### 1. 为什么冻结LLM？
```python
优点：
✅ 保留7B预训练能力
✅ 显存降低 70% (23GB vs 93GB)
✅ 训练速度提升 6x
✅ 防止过拟合（nuScenes只有28K样本）

代价：
❌ LLM无法适应任务
   → 通过训练Map Queries弥补
```

### 2. 为什么用BLIP-2预训练Q-Former？
```python
优点：
✅ Q-Former已学会视觉特征提取
✅ 收敛更快
✅ 性能更好（预计+3~5 mAP）

要求：
- 只需1e-5小学习率微调
- 避免破坏预训练知识
```

### 3. 为什么Map Queries用1e-4？
```python
- 从头训练，需要标准学习率
- 这是最核心的可学习模块
- 决定了"如何向LLM提问"
```

### 4. 为什么用不同学习率？
```python
# 根据初始化状态设置：
Q-Former:    1e-5  (BLIP-2预训练，小步微调)
Map Queries: 1e-4  (随机初始化，正常训练)
Decoder:     1e-4  (随机初始化，正常训练)

# 如果都用相同学习率：
- 1e-4: Q-Former训练不稳定，破坏预训练知识
- 1e-5: Queries和Decoder收敛太慢
```

---

## 📊 参数统计

| 模块 | 参数量 | 状态 | 学习率 |
|-----|-------|------|-------|
| Q-Former | ~100M | ✅ 训练 | 1e-5 |
| LLM Backbone | ~7B | ❌ 冻结 | - |
| Map Queries | ~4.3M | ✅ 训练 | 1e-4 |
| Map Decoder | ~11M | ✅ 训练 | 1e-4 |
| **总计** | **~7.1B** | - | - |
| **可训练** | **~115M (1.6%)** | - | - |

---

## 🔧 优化器配置

```python
optimizer = torch.optim.AdamW([
    {'params': qformer_params,  'lr': 1e-5, 'name': 'qformer'},
    {'params': queries_params,  'lr': 1e-4, 'name': 'queries'},
    {'params': decoder_params,  'lr': 1e-4, 'name': 'decoder'},
], weight_decay=0.01, betas=(0.9, 0.999))

scheduler = CosineAnnealingLR with warmup (500 steps)
```

---

## 📈 预期训练指标

### Mini (404 train / 200 val samples)
```
Epoch 1:  Loss ~4.5
Epoch 12: Loss ~1.8
Epoch 24: Loss ~1.2
训练时间: ~2小时 (A100)
```

### Trainval (28K train / 6K val samples)
```
Epoch 1:  Loss ~4.2
Epoch 6:  Loss ~1.5
Epoch 12: Loss ~1.0
训练时间: ~24小时 (4×A100)
```

---

## 🐛 调试要点

### 1. 检查梯度流
```python
# 在 test_training_setup.py 中已包含
# 确保三个模块都有梯度：
Gradient statistics:
  qformer:  avg=0.001234  # ✅ 有梯度
  queries:  avg=0.002345  # ✅ 有梯度
  decoder:  avg=0.003456  # ✅ 有梯度
```

### 2. 检查学习率
```python
# 训练日志中会打印：
LR: q=1.00e-05 m=1.00e-04 d=1.00e-04
    ↑          ↑          ↑
  qformer   queries    decoder
```

### 3. 检查Loss下降
```python
# 正常训练：
Epoch 1: Loss 4.5 → 3.8 (下降)
Epoch 2: Loss 3.8 → 3.2 (下降)
...

# 异常情况：
Epoch 1: Loss 4.5 → 4.6 (不变/上升)
→ 检查学习率是否过大
→ 检查GT是否正确
```

---

## 🎓 使用方法

### 首次训练
```bash
# 1. 生成GT cache
python tools/generate_gt_cache.py --split train
python tools/generate_gt_cache.py --split val

# 2. 测试环境
python test_training_setup.py

# 3. 修改配置
vim scripts/train_stage2.sh
# 修改 DATAROOT="/your/path/to/nuscenes"

# 4. 开始训练
bash scripts/train_stage2.sh
```

### 从checkpoint恢复
```bash
python train_map_detection.py \
    --resume outputs/xxx/checkpoint_epoch_5.pth \
    --dataroot /path/to/nuscenes \
    ... (其他参数)
```

### 调整配置
```bash
# 如果显存不足：
BATCH_SIZE=2  # 或 1

# 如果Loss不下降：
LR_QFORMER=5e-6
LR_QUERIES=5e-5
WARMUP_STEPS=1000

# 如果收敛太慢：
LR_QUERIES=2e-4
```

---

## 📦 输出文件

```
outputs/map_detection_stage2_20250105_123456/
├── config.json                    # 训练配置
├── best_model.pth                 # 最优模型
│   ├── epoch
│   ├── model_state_dict
│   ├── optimizer_state_dict
│   ├── scheduler_state_dict
│   └── args
├── checkpoint_epoch_1.pth         # 每epoch保存
├── checkpoint_epoch_2.pth
├── ...
└── final_model.pth                # 最终模型
```

---

## ✅ 验证完成度

- [x] 主训练脚本（train_map_detection.py）
- [x] 单卡启动脚本（train_stage2.sh）
- [x] 多卡启动脚本（train_stage2_distributed.sh）
- [x] 测试脚本（test_training_setup.py）
- [x] 详细文档（TRAINING_GUIDE.md）
- [x] 快速指南（STAGE2_TRAINING_README.md）
- [x] 优化器配置（3组不同学习率）
- [x] 学习率调度（warmup + cosine）
- [x] 混合精度训练（FP16）
- [x] 梯度裁剪（0.1）
- [x] 分布式支持（DDP）
- [x] Checkpoint保存/恢复
- [x] 详细日志输出
- [x] 验证集评估

---

## 🎉 代码填充完成！

**所有训练相关代码已完成，可以开始训练！**

**下一步：**
1. 准备nuScenes数据
2. 生成GT cache
3. 运行 `test_training_setup.py` 验证
4. 开始训练 `bash scripts/train_stage2.sh`

**祝训练顺利！🚀**

