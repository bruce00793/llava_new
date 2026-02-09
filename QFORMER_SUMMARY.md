# Q-Former 实现总结

## ✅ 已完成

### 1. 核心模块：`llava/model/qformer.py`

**组件**：
- ✅ `PositionEmbeddingSine`: 2D位置编码
- ✅ `QFormer`: 主模块
  - Backbone接口（默认ResNet-50）
  - Position Encoding（2D + Camera ID）
  - Learnable Scene Queries（512个）
  - Transformer Decoder（6层）
  - Projector（256 → 4096）
- ✅ `build_qformer()`: 构建函数

**特点**：
- 代码量：~250行
- 参数量：~40M（轻量）
- 设计：参考ORION，简化冗余

---

## 📋 架构设计（参考ORION）

### 核心流程

```
输入：images (B, 6, 3, 336, 336)
    ↓
Backbone (ResNet50)
    ↓
img_feats (B*6, 256, 21, 21)
    ↓
Position Encoding (2D + Camera ID)
    ↓
memory (B, 2646, 256)  # 2646 = 6×21×21
    ↓
Transformer Decoder (512 queries, 6 layers)
    ↓
scene_features (B, 512, 256)
    ↓
Projector (MLP)
    ↓
输出：scene_tokens (B, 512, 4096)
```

### 与ORION对比

| 组件 | ORION | Q-Former（你的） |
|------|-------|-----------------|
| Backbone | EVA-ViT (1B+) | ResNet50 (25M) |
| 位置编码 | 3D深度编码 | 2D + Camera ID |
| Query | det(256) + map(257) | scene(512) |
| 输出 | (513, 4096) | (512, 4096) |
| 参数量 | ~1B | ~40M |

**优势**：简洁、轻量、易调试

---

## 🎯 关键设计点

### 1. 简化的位置编码

**ORION方式**（复杂）：
```python
# 3D位置编码
coords3d = img2lidar @ pixel_coords  # 需要相机参数
pos_embed = mlp(coords3d)  # 复杂计算
```

**你的方式**（简洁）：
```python
# 2D位置编码 + Camera ID
pos_embed = PositionEmbeddingSine(h, w)  # 标准2D编码
cam_embed = CameraEmbedding(cam_id)  # 相机身份
img_feats = img_feats + pos_embed + cam_embed
```

### 2. 统一的Scene Queries

**ORION方式**：
```python
det_query = nn.Embedding(256, 256)  # 物体检测
map_query = nn.Embedding(257, 256)  # 地图检测
# 两个独立的head，分别处理
```

**你的方式**：
```python
scene_query = nn.Embedding(512, 256)  # 统一的场景query
# 一个Decoder处理所有信息，自动分工
```

### 3. Query-based Fusion

**核心思想**（来自DETR/ORION）：
- 512个可学习query向量
- 通过Cross-Attention从图像提取信息
- 训练时自动学习每个query的职责
- 灵活、高效、可解释性强

---

## 📁 文件结构

```
llava/model/
├── qformer.py                 # Q-Former实现（核心）
├── QFORMER_DESIGN.md          # 设计文档
└── map_config.py              # 配置（已有）
```

---

## 🚀 使用方法

### 基本使用

```python
from llava.model.qformer import build_qformer

# 配置
config = {
    'embed_dims': 256,
    'num_queries': 512,
    'num_decoder_layers': 6,
    'llm_hidden_size': 4096,
}

# 构建
qformer = build_qformer(config)

# Forward
imgs = batch['images']  # (B, 6, 3, 336, 336)
scene_tokens = qformer(imgs)  # (B, 512, 4096)
```

### 与Dataset集成

```python
from llava.data.map_dataset import create_dataloader

# 创建dataloader
dataloader = create_dataloader(...)

for batch in dataloader:
    imgs = batch['images']  # (B, 6, 3, 336, 336)
    
    # Q-Former提取场景特征
    scene_tokens = qformer(imgs)  # (B, 512, 4096)
    
    # scene_tokens会替换text_ids中的IMAGE_TOKEN_INDEX (-200)
```

---

## 🔧 配置参数

### 推荐配置（默认）

```python
config = {
    'embed_dims': 256,              # 特征维度
    'num_queries': 512,             # Scene query数量
    'num_decoder_layers': 6,        # Decoder层数
    'num_heads': 8,                 # Attention头数
    'ffn_dims': 2048,               # FFN维度
    'dropout': 0.1,                 # Dropout率
    'llm_hidden_size': 4096,        # LLM维度（固定）
}
```

### 可调参数

| 参数 | 范围 | 影响 |
|------|------|------|
| `embed_dims` | 128-512 | 特征表达能力，越大越强但计算量大 |
| `num_queries` | 256-1024 | 场景token数量，影响下游任务 |
| `num_decoder_layers` | 3-12 | 模型深度，越深表达能力越强 |

---

## 📊 计算复杂度

### 参数量分解

```
Backbone (ResNet50):      25.6M
Position Encoding:        可忽略
Query Embedding:          0.13M  (512×256)
Decoder (6层):            ~12M
Projector:                ~2M    (256→512→4096)
────────────────────────────────
总计:                     ~40M
```

### 内存占用（Batch=4）

```
输入images:               4×6×3×336×336×4 bytes = ~26MB
Backbone features:        4×6×256×21×21×4 bytes = ~5MB
Memory (展平):            4×2646×256×4 bytes = ~11MB
Scene tokens:             4×512×4096×4 bytes = ~34MB
────────────────────────────────
总计:                     ~76MB (forward)
```

---

## 🎓 训练策略

### 阶段1：冻结Backbone（推荐先做）

```python
# 只训练Decoder + Projector
for name, param in qformer.named_parameters():
    if 'img_backbone' in name:
        param.requires_grad = False
    else:
        param.requires_grad = True
```

**优势**：
- 训练快
- 稳定
- 适合初期调试

### 阶段2：端到端微调

```python
# 解冻所有参数
for param in qformer.parameters():
    param.requires_grad = True
```

**学习率建议**：
- Backbone: 1e-5（小学习率）
- Decoder: 1e-4
- Projector: 1e-4

---

## 🔍 调试建议

### 1. 测试Forward

```bash
cd /home/cly/auto/llava_test/LLaVA
python llava/model/qformer.py
```

预期输出：
```
✓ Input shape: (2, 6, 3, 336, 336)
✓ Output shape: (2, 512, 4096)
✓ Q-Former test passed!
```

### 2. 检查梯度流

```python
scene_tokens = qformer(imgs)
loss = scene_tokens.sum()
loss.backward()

# 检查梯度
for name, param in qformer.named_parameters():
    if param.grad is not None:
        print(f"{name}: grad_norm={param.grad.norm().item():.4f}")
```

### 3. 可视化Attention

```python
# 在Decoder中添加return_attention=True
# 可视化哪些query关注哪些图像区域
```

---

## 🚧 后续优化方向

### 1. 升级Backbone

```python
# 从ResNet50升级到更强的backbone
from timm import create_model
backbone = create_model('eva_giant_patch14_336', pretrained=True)
```

### 2. 加入3D位置编码（可选）

```python
# 参考ORION的3D编码
ray_dirs = compute_ray_direction(cam_intrinsics)
pos_embed_3d = mlp(ray_dirs)
```

### 3. Deformable Attention（高效）

```python
# 替换标准Attention
from mmcv.ops import MultiScaleDeformableAttention
# 计算量更小，效果可能更好
```

---

## ✅ 总结

### 已实现
1. ✅ 核心Q-Former模块（~250行代码）
2. ✅ 参考ORION架构，简化冗余
3. ✅ 完整的数据流：images → scene_tokens
4. ✅ 易于理解、调试、扩展

### 特点
- **简洁**：只保留核心组件
- **轻量**：40M参数（vs ORION的1B+）
- **高效**：训练快，易调试
- **灵活**：模块化设计，易扩展

### 下一步
- 集成到完整的训练pipeline
- 连接LLM（替换IMAGE_TOKEN_INDEX）
- 添加Output Queries和Detection Head

**Q-Former实现完成！代码简洁、架构清晰、参考ORION核心思想。** 🎉

