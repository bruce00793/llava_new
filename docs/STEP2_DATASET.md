# Step 2: 数据集实现

## 概述

Step 2 实现了 `MapDetectionDataset`，负责加载nuScenes数据和预处理，为模型训练提供标准输入。

## 架构设计

```
MapDetectionDataset
    │
    ├── __init__: 加载nuScenes + GT cache
    │
    ├── __getitem__:
    │   ├── 加载6张相机图像
    │   ├── CLIP预处理 (336x336)
    │   ├── 加载GT并归一化坐标
    │   └── 返回 {images, gt, sample_token, img_metas}
    │
    └── collate_fn: 处理变长GT的padding
```

## 数据流

```
原始数据                    Dataset处理                   输出
────────────────────────────────────────────────────────────────
6张相机图像              CLIP预处理              images (B,6,3,336,336)
(1600x900 JPG)     →    归一化、resize      →   范围: [-2, 2]

GT Cache                坐标归一化              gt_classes (B, max_N)
(ego坐标)          →    [x,y] → [0,1]      →   gt_points (B, max_N, 20, 2)
[-15,30] ~ [15,30]                               gt_is_closed (B, max_N)
                                                  gt_bbox (B, max_N, 4)
                                                  gt_mask (B, max_N)
```

## 主要模块

### 1. MapDetectionDataset

**文件**: `llava/data/map_dataset.py`

**功能**:
- 加载nuScenes数据集
- 读取Step 1生成的GT cache
- CLIP图像预处理
- 坐标归一化到[0, 1]

**关键方法**:

```python
class MapDetectionDataset(Dataset):
    def __init__(
        dataroot: str,           # nuScenes根目录
        version: str,            # 'v1.0-mini' 或 'v1.0-trainval'
        split: str,              # 'train' 或 'val'
        gt_cache_path: str,      # GT cache文件路径
        image_processor: CLIPImageProcessor,  # 图像处理器
    )
    
    def __getitem__(idx) -> dict:
        返回:
            images: (6, 3, H, W)          # CLIP预处理后的图像
            gt: MapGroundTruth            # GT对象
            sample_token: str             # nuScenes sample token
            img_metas: dict               # 相机参数等元数据
```

### 2. 坐标归一化

**原始坐标系**: Ego坐标系 (车辆中心，车头朝上)
- X: [-15, 15] (左右30米)
- Y: [-30, 30] (前后60米)

**归一化公式**:
```python
x_norm = (x - x_min) / (x_max - x_min)  # [-15, 15] → [0, 1]
y_norm = (y - y_min) / (y_max - y_min)  # [-30, 30] → [0, 1]
```

**反归一化** (用于可视化/评估):
```python
x = x_norm * (x_max - x_min) + x_min
y = y_norm * (y_max - y_min) + y_min
```

### 3. collate_fn

**功能**: 处理变长GT的batch组装

**问题**: 每个样本的GT数量不同 (0 ~ N)

**解决方案**: Padding到batch内最大值，用mask标记有效GT

```python
def collate_fn(batch) -> dict:
    返回:
        images: (B, 6, 3, H, W)
        gt_classes: (B, max_N)        # 类别
        gt_points: (B, max_N, 20, 2)  # 点坐标
        gt_is_closed: (B, max_N)      # 是否闭合
        gt_bbox: (B, max_N, 4)        # 边界框
        gt_mask: (B, max_N)           # 有效GT标记 (True/False)
        sample_tokens: List[str]
        img_metas: List[dict]
```

**示例**:
```python
# Batch中的GT数量: [3, 5, 2, 4]
# → max_N = 5, padding后:
gt_mask = [
    [True, True, True, False, False],   # 3个有效GT
    [True, True, True, True, True],     # 5个有效GT
    [True, True, False, False, False],  # 2个有效GT
    [True, True, True, True, False],    # 4个有效GT
]
```

## 使用方式

### 方式1: 直接创建Dataset

```python
from llava.data import MapDetectionDataset

dataset = MapDetectionDataset(
    dataroot='data/nuscenes_mini',
    version='v1.0-mini',
    split='train',
    gt_cache_path='data/gt_cache_v1.0-mini_train.pkl',
)

# 获取一个样本
sample = dataset[0]
print(sample['images'].shape)      # (6, 3, 336, 336)
print(sample['gt'].classes.shape)  # (N,)
```

### 方式2: 使用create_dataloader

```python
from llava.data import create_dataloader

dataloader = create_dataloader(
    dataroot='data/nuscenes_mini',
    version='v1.0-mini',
    split='train',
    batch_size=4,
    num_workers=4,
    shuffle=True,
)

# 训练循环
for batch in dataloader:
    images = batch['images']        # (4, 6, 3, 336, 336)
    gt_classes = batch['gt_classes']  # (4, max_N)
    gt_mask = batch['gt_mask']      # (4, max_N)
    
    # 只处理有效的GT
    for i in range(len(images)):
        valid_gts = gt_mask[i]
        num_valid = valid_gts.sum()
        print(f"Sample {i}: {num_valid} GTs")
```

### 方式3: 集成到训练脚本

```python
from transformers import CLIPImageProcessor
from llava.data import create_dataloader

# 使用LLaVA的image processor
image_processor = CLIPImageProcessor.from_pretrained(
    "liuhaotian/llava-v1.5-7b"
)

train_loader = create_dataloader(
    dataroot='data/nuscenes_mini',
    version='v1.0-mini',
    split='train',
    image_processor=image_processor,
    batch_size=4,
    num_workers=4,
)

val_loader = create_dataloader(
    dataroot='data/nuscenes_mini',
    version='v1.0-mini',
    split='val',
    image_processor=image_processor,
    batch_size=4,
    num_workers=4,
    shuffle=False,
)
```

## 测试

运行测试脚本验证数据集:

```bash
cd LLaVA
python tests/test_dataset.py
```

测试内容:
1. Dataset基本加载
2. 单个样本检查
3. DataLoader batching
4. 坐标归一化/反归一化

## 数据格式

### Dataset输出 (单个样本)

```python
{
    'images': Tensor(6, 3, 336, 336),  # CLIP预处理后的图像
    'gt': MapGroundTruth {
        'classes': Tensor(N,),          # [0, 1, 2, ...]
        'points': Tensor(N, 20, 2),     # 归一化坐标 [0, 1]
        'is_closed': Tensor(N,),        # [False, False, True, ...]
        'bbox': Tensor(N, 4),           # [x_min, y_min, x_max, y_max]
    },
    'sample_token': str,                # 'ca9a282c9e77460f8360f564131a8af5'
    'img_metas': {
        'sample_token': str,
        'scene_token': str,
        'timestamp': int,
        'ego_pose': dict,
        'cam_intrinsics': List[List[List[float]]],  # (6, 3, 3)
        'cam_extrinsics': List[dict],               # (6,)
        'pc_range': List[float],                    # [-15, -30, -2, 15, 30, 2]
    }
}
```

### DataLoader输出 (batch)

```python
{
    'images': Tensor(B, 6, 3, 336, 336),
    'gt_classes': Tensor(B, max_N),
    'gt_points': Tensor(B, max_N, 20, 2),
    'gt_is_closed': Tensor(B, max_N),
    'gt_bbox': Tensor(B, max_N, 4),
    'gt_mask': Tensor(B, max_N),         # Bool mask
    'sample_tokens': List[str],          # len=B
    'img_metas': List[dict],             # len=B
}
```

## 相机顺序

6个相机的固定顺序:
1. `CAM_FRONT` - 前视
2. `CAM_FRONT_RIGHT` - 右前
3. `CAM_FRONT_LEFT` - 左前
4. `CAM_BACK` - 后视
5. `CAM_BACK_LEFT` - 左后
6. `CAM_BACK_RIGHT` - 右后

## 注意事项

1. **坐标系统**: 所有坐标都在ego坐标系下，归一化到[0, 1]
2. **GT padding**: 使用`gt_mask`区分真实GT和padding
3. **图像处理**: 使用CLIP的预处理，保持与LLaVA一致
4. **内存优化**: 使用`pin_memory=True`加速GPU传输
5. **数据分割**: mini版本简单8:2分割，完整版应使用官方split

## 下一步: Step 3

Step 3 将实现Q-Former模块:
- 输入: `images (B, 6, 3, 336, 336)`
- 输出: `scene_tokens (B, 512, 4096)`

Q-Former架构:
```
6张图像 → CLIP ViT → 6组特征
                ↓
        Q-Former (512 queries)
                ↓
        512个scene tokens
```
