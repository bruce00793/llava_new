# 完整数据流：从nuScenes到Q-Former输出

## 目录
- [阶段零：nuScenes原始数据](#阶段零nuscenes原始数据)
- [阶段一：GT预处理（离线）](#阶段一gt预处理离线)
- [阶段二：Dataset加载](#阶段二dataset加载)
- [阶段三：Batch构建](#阶段三batch构建)
- [阶段四：Q-Former处理](#阶段四q-former处理)
- [完整数据流总览](#完整数据流总览)

---

## 阶段零：nuScenes原始数据

### 数据结构
```
nuScenes/
├── samples/
│   ├── CAM_FRONT/
│   │   └── n015-2018-07-18-11-07-57+0800__CAM_FRONT__1531883530412470.jpg
│   ├── CAM_FRONT_LEFT/
│   ├── CAM_FRONT_RIGHT/
│   ├── CAM_BACK/
│   ├── CAM_BACK_LEFT/
│   └── CAM_BACK_RIGHT/
├── v1.0-trainval/
│   ├── sample.json           # 关键帧索引
│   ├── sample_data.json      # 传感器数据
│   ├── ego_pose.json         # 车辆位姿
│   ├── calibrated_sensor.json # 相机标定
│   └── ...
└── maps/expansion/
    └── map_data.json         # 地图标注
```

### 单个Sample的原始数据

**图像数据**：
```python
6张JPG图像文件：
- 路径：samples/CAM_FRONT/xxx.jpg
- 尺寸：1600 × 900 (width × height)
- 格式：RGB, uint8
- 值域：[0, 255]
```

**相机参数（从JSON读取）**：
```python
camera_intrinsic: List[3×3] (6个相机各一个)
示例：
[[1266.417, 0.0,      816.267],
 [0.0,      1266.417, 491.507],
 [0.0,      0.0,      1.0    ]]

camera_calibration: dict (6个)
{
    "translation": [1.70, 0.016, 1.51],  # 相机→车辆的平移
    "rotation": [0.5, -0.5, 0.5, -0.5]   # 四元数格式
}
```

**车辆位姿（从JSON读取）**：
```python
ego_pose: dict
{
    "translation": [411.3, 1180.9, 0.0],  # 世界坐标
    "rotation": [0.572, -0.002, 0.012, -0.820]  # 四元数
}
```

**地图标注（从map API获取）**：
```python
map_elements: List[dict]
[
    {
        "class": "lane_divider",
        "points": [[x1,y1], [x2,y2], ...],  # 世界坐标
        "is_closed": False
    },
    {
        "class": "ped_crossing",
        "points": [[x1,y1], [x2,y2], ...],  # 世界坐标
        "is_closed": True
    },
    ...
]
```

---

## 阶段一：GT预处理（离线）

### 步骤1.1：提取地图元素（世界坐标）

**代码位置**：`llava/data/map_gt/nuscenes_map_api.py`

**输入**：
```python
sample_token: str
ego_pose: {
    "translation": [411.3, 1180.9, 0.0],
    "rotation": [0.572, -0.002, 0.012, -0.820]
}
```

**处理**：
```python
# 1. 获取车辆位置
ego_xy = ego_pose['translation'][:2]  # [411.3, 1180.9]

# 2. 搜索60米半径内的地图元素
records = map_api.get_records_in_radius(
    x=ego_xy[0], y=ego_xy[1],
    radius=60.0,
    layer_names=['road_divider', 'lane_divider', 'ped_crossing']
)

# 3. 提取几何信息
for record in records:
    points_world = extract_geometry(record)  # [N, 2] 世界坐标
```

**输出**：
```python
map_elements: List[dict]
[
    {
        'class_id': 0,  # road_divider
        'points_world': array([[411.5, 1181.2],
                              [412.0, 1182.5],
                              ...], dtype=float32),  # [N, 2]
        'is_closed': False
    },
    {
        'class_id': 2,  # ped_crossing
        'points_world': array([[410.0, 1200.0],
                              [415.0, 1200.0],
                              ...], dtype=float32),  # [M, 2]
        'is_closed': True
    },
    ...
]
```

---

### 步骤1.2：坐标变换（世界→车辆ego）

**代码位置**：`llava/data/map_gt/geometry.py` - `transform_to_ego()`

**输入**：
```python
points_world: array [N, 2]  # 世界坐标
    [[411.5, 1181.2],
     [412.0, 1182.5],
     ...]

ego_translation: array [3]  # [411.3, 1180.9, 0.0]
ego_rotation: array [3, 3]  # 旋转矩阵
    [[cos(θ), -sin(θ), 0],
     [sin(θ),  cos(θ), 0],
     [0,       0,      1]]
```

**处理**：
```python
# 转换为齐次坐标
points_3d = np.hstack([points_world, np.zeros((N, 1))])  # [N, 3]

# 变换：p_ego = R^T @ (p_world - t)
points_centered = points_3d - ego_translation
points_ego = points_centered @ ego_rotation.T

# 只保留x, y
points_ego_2d = points_ego[:, :2]
```

**输出**：
```python
points_ego: array [N, 2], dtype=float32
    [[0.2, 0.3],    # 车辆前方0.2m右，0.3m前
     [0.7, 1.8],
     ...]
# 坐标系：x轴左(-)/右(+)，y轴后(-)/前(+)
# 单位：米
```

---

### 步骤1.3：ROI裁剪

**代码位置**：`llava/data/map_gt/geometry.py` - `clip_polyline_by_roi()`

**输入**：
```python
points_ego: array [N, 2]  # 可能超出感知范围
PC_RANGE = [-15, -30, -2, 15, 30, 2]  # [x_min, y_min, z_min, x_max, y_max, z_max]
```

**处理**：
```python
from shapely.geometry import LineString, box

# 定义ROI
roi = box(x_min=-15, y_min=-30, x_max=15, y_max=30)

# 裁剪
if is_closed:  # Polygon
    poly = Polygon(points_ego)
    clipped = poly.intersection(roi)
else:  # Polyline
    line = LineString(points_ego)
    clipped = line.intersection(roi)

# 提取坐标
clipped_points = np.array(clipped.coords)
```

**输出**：
```python
clipped_segments: List[array]
[
    array([[−14.5, 10.2],
           [−10.3, 12.5],
           ...], dtype=float32),  # [M1, 2]
    array([[5.0, −20.0],
           [8.0, −18.0],
           ...], dtype=float32),  # [M2, 2]
]
# 注意：一个元素可能被裁剪成多段
# 坐标范围：x∈[-15,15], y∈[-30,30]
```

---

### 步骤1.4：采样20个点

**代码位置**：`llava/data/map_gt/geometry.py` - `sample_polyline_20()`

**输入**：
```python
polyline: array [M, 2]  # M个点，数量不定
```

**处理**：
```python
from shapely.geometry import LineString

line = LineString(polyline)
total_length = line.length

# 按弧长均匀采样20个点
distances = np.linspace(0, total_length, 20)
points_20 = []
for dist in distances:
    point = line.interpolate(dist)
    points_20.append([point.x, point.y])

points_20 = np.array(points_20, dtype=np.float32)
```

**输出**：
```python
points_20: array [20, 2], dtype=float32
    [[−14.5, 10.2],
     [−13.8, 10.5],
     [−13.1, 10.8],
     ...
     [8.0, −18.0]]

# 对于polygon（人行横道）：
points_20[0] ≈ points_20[19]  # 首尾相同（闭合）
# 顺时针排列
```

---

### 步骤1.5：坐标归一化

**代码位置**：`llava/data/map_dataset.py` - `_normalize_coords()`

**输入**：
```python
points_ego: array [20, 2], dtype=float32
    # 值域：x∈[-15,15], y∈[-30,30] 米

PC_RANGE = [-15, -30, -2, 15, 30, 2]
```

**处理**：
```python
x_min, x_max = -15, 15  # 范围30米
y_min, y_max = -30, 30  # 范围60米

# 归一化到[-1, 1]
x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
y_norm = (y - y_min) / (y_max - y_min) * 2 - 1
```

**输出**：
```python
points_normalized: array [20, 2], dtype=float32
    [[0.033, 0.340],   # x=0.033在[-1,1]中，对应ego系中约0.5米
     [0.080, 0.350],
     ...
     [0.067, -0.200]]

# 值域：[-1, 1]
# 中心点(0, 0)对应车辆位置
# (-1, -1)对应左后角，(1, 1)对应右前角
```

---

### 步骤1.6：计算AABB包围盒

**代码位置**：`llava/data/map_gt/geometry.py` - `compute_aabb()`

**输入**：
```python
points_normalized: array [20, 2]
```

**处理**：
```python
x_coords = points[:, 0]
y_coords = points[:, 1]

x_min, x_max = x_coords.min(), x_coords.max()
y_min, y_max = y_coords.min(), y_coords.max()

x_center = (x_min + x_max) / 2
y_center = (y_min + y_max) / 2
width = x_max - x_min
height = y_max - y_min

bbox = [x_center, y_center, width, height]
```

**输出**：
```python
bbox: array [4], dtype=float32
    [0.050, 0.270, 0.040, 0.200]
    # [x_center, y_center, width, height]
    # 归一化后的值域：[-1, 1]
```

---

### 步骤1.7：保存GT缓存

**代码位置**：`llava/data/map_gt/cache.py` - `process_one_sample()`

**输出文件**：`gt_cache/annotations/{sample_token}.pkl`

**内容**：
```python
{
    'sample_token': 'ca9a282c9e77460f8360f564131a8af5',
    'gt_classes': array([0, 0, 1, 2, 0], dtype=int64),
    'gt_points': array([
        [[0.033, 0.340], [0.080, 0.350], ..., [0.067, -0.200]],  # 实例0
        [[−0.500, 0.800], [...], ...],  # 实例1
        ...
    ], shape=[5, 20, 2], dtype=float32),
    'gt_is_closed': array([False, False, False, True, False], dtype=bool),
    'gt_bbox': array([
        [0.050, 0.270, 0.040, 0.200],  # 实例0的bbox
        [−0.450, 0.750, 0.100, 0.300],
        ...
    ], shape=[5, 4], dtype=float32)
}
```

**阶段一总结**：
- 输入：nuScenes原始数据（JSON + 地图）
- 输出：每个sample一个.pkl文件
- 内容：类别、20点、是否闭合、包围盒
- 坐标系：车辆ego，归一化到[-1, 1]

---

## 阶段二：Dataset加载

### 步骤2.1：Dataset初始化

**代码位置**：`llava/data/map_dataset.py` - `MapDetectionDataset.__init__()`

**操作**：
```python
dataset = MapDetectionDataset(
    dataroot='/path/to/nuscenes',
    version='v1.0-mini',
    split='train',
    gt_cache_path='/path/to/gt_cache',
    prompt='请帮我识别图中的车道线、道路边界、人行横道三类物体'
)
```

**内部状态**：
```python
self.nusc = NuScenes(version='v1.0-mini', dataroot=dataroot)
self.sample_tokens = ['token1', 'token2', ...]  # 80个样本（mini）
self.image_processor = CLIPImageProcessor.from_pretrained(...)
self.tokenizer = AutoTokenizer.from_pretrained('vicuna-7b-v1.5')

# Prompt处理
self.prompt_with_images = "<image>\n请帮我识别..."
self.prompt_ids = tokenizer_image_token(self.prompt_with_images)
# 结果：[1, 319, 526, ..., -200, ..., 2]  # -200是IMAGE_TOKEN_INDEX
```

---

### 步骤2.2：__getitem__ - 加载单个样本

**代码位置**：`llava/data/map_dataset.py` - `__getitem__()`

#### 子步骤2.2.1：加载6张图像

**输入**：
```python
sample_token: str
```

**处理**：
```python
sample = self.nusc.get('sample', sample_token)

images = []
for cam_name in ['CAM_FRONT', 'CAM_FRONT_RIGHT', ...]:
    # 1. 获取图像路径
    cam_token = sample['data'][cam_name]
    cam_data = self.nusc.get('sample_data', cam_token)
    img_path = os.path.join(dataroot, cam_data['filename'])
    
    # 2. 读取图像
    img = Image.open(img_path).convert('RGB')
    # PIL Image, 尺寸=(1600, 900), mode='RGB'
    
    # 3. CLIP预处理
    processed = self.image_processor(images=img, return_tensors='pt')
    img_tensor = processed['pixel_values'][0]
    # shape: (3, 336, 336)
    # dtype: torch.float32
    # 值域: 约[-2.5, 2.5]（CLIP归一化）
    
    images.append(img_tensor)

# 4. 堆叠
images = torch.stack(images, dim=0)
```

**输出**：
```python
images: torch.FloatTensor
    shape: (6, 3, 336, 336)
    dtype: torch.float32
    值域: 约[-2.5, 2.5]
    内存: 6×3×336×336×4 bytes ≈ 8.1 MB
```

---

#### 子步骤2.2.2：加载GT

**输入**：
```python
sample_token: str
gt_cache_path: str
```

**处理**：
```python
# 1. 从缓存加载
gt_file = os.path.join(gt_cache_path, f'{sample_token}.pkl')
with open(gt_file, 'rb') as f:
    gt_dict = pickle.load(f)

# 2. 转换为MapGroundTruth对象
gt = MapGroundTruth(
    class_labels=torch.from_numpy(gt_dict['gt_classes']),
    points=torch.from_numpy(gt_dict['gt_points']),
    bbox=torch.from_numpy(gt_dict['gt_bbox'])
)
gt.is_closed = torch.from_numpy(gt_dict['gt_is_closed'])
```

**输出**：
```python
gt: MapGroundTruth对象
    gt.class_labels: torch.LongTensor, shape=(N_gt,)
        例如：tensor([0, 0, 1, 2, 0])  # 5个实例
    
    gt.points: torch.FloatTensor, shape=(N_gt, 20, 2)
        例如：tensor([
            [[0.033, 0.340], [0.080, 0.350], ..., [0.067, -0.200]],
            [[-0.500, 0.800], [...], ...],
            ...
        ])
        值域：[-1, 1]
    
    gt.is_closed: torch.BoolTensor, shape=(N_gt,)
        例如：tensor([False, False, False, True, False])
    
    gt.bbox: torch.FloatTensor, shape=(N_gt, 4)
        例如：tensor([
            [0.050, 0.270, 0.040, 0.200],
            [-0.450, 0.750, 0.100, 0.300],
            ...
        ])
```

---

#### 子步骤2.2.3：加载相机参数

**输入**：
```python
sample_token: str
```

**处理**：
```python
sample = self.nusc.get('sample', sample_token)

# 1. 获取ego_pose
lidar_token = sample['data']['LIDAR_TOP']
lidar_data = self.nusc.get('sample_data', lidar_token)
ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])

# 2. 获取相机参数
cam_intrinsics = []
cam_extrinsics = []
for cam_name in CAMERA_NAMES:
    cam_token = sample['data'][cam_name]
    cam_data = self.nusc.get('sample_data', cam_token)
    cam_calib = self.nusc.get('calibrated_sensor', 
                               cam_data['calibrated_sensor_token'])
    
    # 内参
    intrinsic = cam_calib['camera_intrinsic']  # List[List[float]]
    cam_intrinsics.append(intrinsic)
    
    # 外参
    cam_extrinsics.append({
        'translation': cam_calib['translation'],  # [x, y, z]
        'rotation': cam_calib['rotation']  # [w, x, y, z] 四元数
    })

img_metas = {
    'ego_pose': {
        'translation': ego_pose['translation'],
        'rotation': ego_pose['rotation']
    },
    'cam_intrinsics': cam_intrinsics,
    'cam_extrinsics': cam_extrinsics,
    ...
}
```

**输出**：
```python
img_metas: dict
{
    'sample_token': 'ca9a282c...',
    'ego_pose': {
        'translation': [411.3, 1180.9, 0.0],
        'rotation': [0.572, -0.002, 0.012, -0.820]
    },
    'cam_intrinsics': [
        [[1266.417, 0.0, 816.267],
         [0.0, 1266.417, 491.507],
         [0.0, 0.0, 1.0]],
        [...],  # 6个相机
    ],
    'cam_extrinsics': [
        {
            'translation': [1.70, 0.016, 1.51],
            'rotation': [0.5, -0.5, 0.5, -0.5]
        },
        {...},  # 6个相机
    ],
    'pc_range': [-15, -30, -2, 15, 30, 2]
}
```

---

#### 子步骤2.2.4：返回单个样本

**输出**：
```python
return {
    'images': torch.FloatTensor (6, 3, 336, 336),
    'text_ids': torch.LongTensor (L,),  # 例如L=76
    'gt': MapGroundTruth对象,
    'sample_token': str,
    'img_metas': dict
}
```

---

## 阶段三：Batch构建

### 步骤3.1：collate_fn处理

**代码位置**：`llava/data/map_dataset.py` - `collate_fn()`

**输入**：
```python
batch: List[dict]，长度=B (batch_size)
[
    {'images': (6,3,336,336), 'text_ids': (76,), 'gt': ..., ...},
    {'images': (6,3,336,336), 'text_ids': (76,), 'gt': ..., ...},
    ...
]
```

#### 子步骤3.1.1：堆叠images

**处理**：
```python
images = torch.stack([item['images'] for item in batch], dim=0)
```

**输出**：
```python
images: torch.FloatTensor
    shape: (B, 6, 3, 336, 336)
    dtype: torch.float32
    值域: [-2.5, 2.5]
    
例如B=4:
    shape: (4, 6, 3, 336, 336)
    内存: 4×6×3×336×336×4 bytes ≈ 32.4 MB
```

---

#### 子步骤3.1.2：堆叠text_ids

**处理**：
```python
text_ids = torch.stack([item['text_ids'] for item in batch], dim=0)
```

**输出**：
```python
text_ids: torch.LongTensor
    shape: (B, L)  # 例如(4, 76)
    dtype: torch.int64
    
示例内容：
tensor([[1, 319, 526, ..., -200, ..., 29962, 2],
        [1, 319, 526, ..., -200, ..., 29962, 2],
        [1, 319, 526, ..., -200, ..., 29962, 2],
        [1, 319, 526, ..., -200, ..., 29962, 2]])

关键点：
- 第0列都是1（BOS token）
- 中间某位置都是-200（IMAGE_TOKEN_INDEX）
- 最后一列都是2（EOS token）
- 所有样本使用相同的prompt，所以内容相同
```

---

#### 子步骤3.1.3：处理相机参数

**处理**：
```python
from pyquaternion import Quaternion

cam_intrinsics = []
cam_extrinsics = []
ego_poses = []

for item in batch:
    # 内参：直接转tensor
    intrinsics = torch.tensor(
        item['img_metas']['cam_intrinsics'], 
        dtype=torch.float32
    )  # (6, 3, 3)
    cam_intrinsics.append(intrinsics)
    
    # 外参：四元数→旋转矩阵
    extrinsics_list = []
    for ext in item['img_metas']['cam_extrinsics']:
        mat = torch.eye(4, dtype=torch.float32)
        
        # Rotation
        quat = Quaternion(ext['rotation'])
        mat[:3, :3] = torch.from_numpy(quat.rotation_matrix)
        
        # Translation
        mat[:3, 3] = torch.tensor(ext['translation'])
        
        extrinsics_list.append(mat)
    
    extrinsics = torch.stack(extrinsics_list)  # (6, 4, 4)
    cam_extrinsics.append(extrinsics)
    
    # Ego pose：同样处理
    ego_dict = item['img_metas']['ego_pose']
    ego_mat = torch.eye(4)
    ego_quat = Quaternion(ego_dict['rotation'])
    ego_mat[:3, :3] = torch.from_numpy(ego_quat.rotation_matrix)
    ego_mat[:3, 3] = torch.tensor(ego_dict['translation'])
    ego_poses.append(ego_mat)

cam_intrinsics = torch.stack(cam_intrinsics)  # (B, 6, 3, 3)
cam_extrinsics = torch.stack(cam_extrinsics)  # (B, 6, 4, 4)
ego_poses = torch.stack(ego_poses)  # (B, 4, 4)
```

**输出**：
```python
cam_intrinsics: torch.FloatTensor, shape=(B, 6, 3, 3)
示例（B=1的第0个相机）：
tensor([[[505.668,   0.000, 326.507],
         [  0.000, 505.668, 196.403],
         [  0.000,   0.000,   1.000]]])

cam_extrinsics: torch.FloatTensor, shape=(B, 6, 4, 4)
示例（B=1的第0个相机）：
tensor([[[ 0.0200, -0.9998,  0.0300,  1.7008],
         [ 0.0100,  0.0300,  0.9995,  0.0159],
         [-0.9998, -0.0200,  0.0100,  1.5110],
         [ 0.0000,  0.0000,  0.0000,  1.0000]]])

ego_pose: torch.FloatTensor, shape=(B, 4, 4)
示例（B=1）：
tensor([[[ 0.5720, -0.8201,  0.0120, 411.3039],
         [ 0.8201,  0.5720, -0.0017, 1180.890],
         [ 0.0000,  0.0118,  0.9999,   0.0000],
         [ 0.0000,  0.0000,  0.0000,   1.0000]]])
```

---

#### 子步骤3.1.4：Padding GT

**处理**：
```python
# 找到最大GT数量
max_num_gts = max(len(item['gt'].class_labels) for item in batch)

# 初始化padding tensor
gt_classes = torch.zeros(B, max_num_gts, dtype=torch.long)
gt_points = torch.zeros(B, max_num_gts, 20, 2, dtype=torch.float32)
gt_is_closed = torch.zeros(B, max_num_gts, dtype=torch.bool)
gt_bbox = torch.zeros(B, max_num_gts, 4, dtype=torch.float32)
gt_mask = torch.zeros(B, max_num_gts, dtype=torch.bool)

# 填入真实GT
for i, item in enumerate(batch):
    gt = item['gt']
    num_gts = len(gt.class_labels)
    
    if num_gts > 0:
        gt_classes[i, :num_gts] = gt.class_labels
        gt_points[i, :num_gts] = gt.points
        gt_is_closed[i, :num_gts] = gt.is_closed
        gt_bbox[i, :num_gts] = gt.bbox
        gt_mask[i, :num_gts] = True  # 标记有效GT
```

**输出**：
```python
gt_classes: torch.LongTensor, shape=(B, max_N)
示例（B=2, max_N=8）：
tensor([[0, 0, 1, 2, 0, 0, 0, 0],  # 样本0有5个GT，后3个是padding
        [1, 2, 0, 1, 2, 0, 0, 0]]) # 样本1有6个GT，后2个是padding

gt_points: torch.FloatTensor, shape=(B, max_N, 20, 2)
示例：
tensor([[
    [[0.033, 0.340], [0.080, 0.350], ..., [0.067, -0.200]],  # GT 0
    [[-0.500, 0.800], [...], ...],  # GT 1
    ...
    [[0.0, 0.0], [0.0, 0.0], ..., [0.0, 0.0]]  # padding
]])

gt_mask: torch.BoolTensor, shape=(B, max_N)
示例：
tensor([[True, True, True, True, True, False, False, False],
        [True, True, True, True, True, True, False, False]])
```

---

### 步骤3.2：最终Batch输出

**输出**：
```python
batch = {
    # ========== 图像数据 ==========
    'images': torch.FloatTensor, shape=(B, 6, 3, 336, 336)
        值域: [-2.5, 2.5]
        内存: B×6×3×336×336×4 bytes
    
    # ========== 文本数据 ==========
    'text_ids': torch.LongTensor, shape=(B, L)
        包含IMAGE_TOKEN_INDEX (-200)
        示例：[1, 319, ..., -200, ..., 2]
    
    # ========== 相机参数 ==========
    'cam_intrinsics': torch.FloatTensor, shape=(B, 6, 3, 3)
        6个相机的内参矩阵
    
    'cam_extrinsics': torch.FloatTensor, shape=(B, 6, 4, 4)
        6个相机到车辆的变换矩阵（含rotation）
    
    'ego_pose': torch.FloatTensor, shape=(B, 4, 4)
        车辆到世界的变换矩阵
    
    # ========== GT标注 ==========
    'gt_classes': torch.LongTensor, shape=(B, max_N)
        值域: [0, 1, 2]
    
    'gt_points': torch.FloatTensor, shape=(B, max_N, 20, 2)
        值域: [-1, 1]
    
    'gt_is_closed': torch.BoolTensor, shape=(B, max_N)
    
    'gt_bbox': torch.FloatTensor, shape=(B, max_N, 4)
        值域: [-1, 1]
    
    'gt_mask': torch.BoolTensor, shape=(B, max_N)
        标记有效GT（True）和padding（False）
    
    # ========== 元数据 ==========
    'sample_tokens': List[str], 长度B
    'img_metas': List[dict], 长度B
}
```

---

## 阶段四：Q-Former处理

### 步骤4.1：提取图像特征

**代码位置**：`llava/model/qformer.py` - `extract_img_feat()`

**输入**：
```python
imgs: torch.FloatTensor, shape=(B, 6, 3, 336, 336)
```

**处理**：
```python
B, N, C, H, W = imgs.shape  # B=batch, N=6 cameras
imgs = imgs.reshape(B * N, C, H, W)  # (B*6, 3, 336, 336)

# Backbone (ResNet50)
img_feats = self.img_backbone(imgs)

# 假设使用ResNet50的layer4输出
# stride=16, 336/16=21
```

**输出**：
```python
img_feats: torch.FloatTensor
    shape: (B*6, 256, 21, 21)
    dtype: torch.float32
    
示例B=4:
    shape: (24, 256, 21, 21)
    内存: 24×256×21×21×4 bytes ≈ 5.4 MB
```

---

### 步骤4.2：添加位置编码

**代码位置**：`llava/model/qformer.py` - `add_position_encoding()`

**输入**：
```python
img_feats: (B*N, C, h, w) = (B*6, 256, 21, 21)
```

**处理**：
```python
BN, C, h, w = img_feats.shape

# 1. 2D位置编码
pos_embed = self.position_encoding(h, w)  # (h, w, C)
pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)  # (1, C, h, w)
pos_embed = pos_embed.expand(BN, -1, -1, -1)  # (B*6, 256, 21, 21)

# 2. Camera ID embedding
cam_ids = torch.arange(6).repeat(B)  # [0,1,2,3,4,5, 0,1,2,3,4,5, ...]
cam_embed = self.camera_embed(cam_ids)  # (B*6, 256)
cam_embed = cam_embed[:, :, None, None].expand(-1, -1, h, w)  # (B*6, 256, 21, 21)

# 3. 相加
img_feats = img_feats + pos_embed + cam_embed

# 4. 展平
img_feats = img_feats.flatten(2).permute(0, 2, 1)  # (B*6, h*w, 256)
img_feats = img_feats.reshape(B, 6*h*w, C)  # (B, 2646, 256)
```

**输出**：
```python
memory: torch.FloatTensor
    shape: (B, N*h*w, C) = (B, 2646, 256)
    # 2646 = 6 cameras × 21 × 21
    
示例B=4:
    shape: (4, 2646, 256)
    内存: 4×2646×256×4 bytes ≈ 10.8 MB
```

---

### 步骤4.3：准备Scene Queries

**代码位置**：`llava/model/qformer.py` - `forward()`

**处理**：
```python
# Learnable queries
queries = self.query_embed.weight  # (512, 256)
queries = queries.unsqueeze(0).expand(B, -1, -1)  # (B, 512, 256)
```

**输出**：
```python
queries: torch.FloatTensor
    shape: (B, 512, 256)
    dtype: torch.float32
    
    # 这512个向量是可学习的参数
    # 初始：随机初始化
    # 训练后：每个query学会提取特定类型的场景信息
```

---

### 步骤4.4：Transformer Decoder

**代码位置**：`llava/model/qformer.py` - `forward()`

**输入**：
```python
tgt: queries (B, 512, 256)
memory: (B, 2646, 256)
```

**处理**：
```python
# 6层Transformer Decoder
# 每层包含：Self-Attention + Cross-Attention + FFN

for layer in range(6):
    # Self-Attention: queries之间交互
    queries = SelfAttention(Q=queries, K=queries, V=queries)
    
    # Cross-Attention: queries从图像提取信息
    queries = CrossAttention(Q=queries, K=memory, V=memory)
    
    # FFN
    queries = FFN(queries)

scene_features = queries
```

**输出**：
```python
scene_features: torch.FloatTensor
    shape: (B, 512, 256)
    dtype: torch.float32
    
    # 这512个特征向量已经融合了6张图的信息
    # 每个向量提取了特定的场景信息
```

---

### 步骤4.5：投影到LLM维度

**代码位置**：`llava/model/qformer.py` - `forward()`

**输入**：
```python
scene_features: (B, 512, 256)
```

**处理**：
```python
# MLP projector
scene_tokens = self.projector(scene_features)

# projector结构：
# Linear(256 → 512) + GELU + Linear(512 → 4096)
```

**输出**：
```python
scene_tokens: torch.FloatTensor
    shape: (B, 512, 4096)
    dtype: torch.float32
    
示例B=4:
    shape: (4, 512, 4096)
    内存: 4×512×4096×4 bytes ≈ 33.6 MB
    
值域: 无固定范围（取决于训练）
```

---

### Q-Former最终输出

**总结**：
```python
# Q-Former的forward返回
scene_tokens = qformer(
    imgs=batch['images'],  # (B, 6, 3, 336, 336)
    cam_intrinsics=batch['cam_intrinsics'],  # 可选
    cam_extrinsics=batch['cam_extrinsics'],  # 可选
)

# 输出
scene_tokens: torch.FloatTensor
    shape: (B, 512, 4096)
    含义: 
        - 512个token代表整个场景
        - 每个token是4096维向量
        - 已融合6张相机图像的信息
        - 维度与LLM对齐，准备替换IMAGE_TOKEN_INDEX
```

---

## 完整数据流总览

### 数据维度变化

```
【原始数据】
6张JPG图像: 6 × (1600, 900, 3) uint8
地图标注: List[不定长点序列] 世界坐标

    ↓ GT预处理（离线）

【GT缓存】.pkl文件
gt_classes: (N_gt,) int64, [0,1,2]
gt_points: (N_gt, 20, 2) float32, [-1, 1]
gt_bbox: (N_gt, 4) float32, [-1, 1]

    ↓ Dataset.__getitem__

【单个样本】
images: (6, 3, 336, 336) float32, [-2.5, 2.5]
text_ids: (L,) int64, 包含-200
gt: MapGroundTruth对象

    ↓ collate_fn

【Batch】
images: (B, 6, 3, 336, 336)
text_ids: (B, L)
cam_intrinsics: (B, 6, 3, 3)
cam_extrinsics: (B, 6, 4, 4)
ego_pose: (B, 4, 4)
gt_classes: (B, max_N)
gt_points: (B, max_N, 20, 2)
gt_mask: (B, max_N)

    ↓ Q-Former.extract_img_feat

【Backbone输出】
img_feats: (B*6, 256, 21, 21)

    ↓ Q-Former.add_position_encoding

【Memory】
memory: (B, 2646, 256)  # 2646 = 6×21×21

    ↓ Q-Former.decoder

【Scene Features】
scene_features: (B, 512, 256)

    ↓ Q-Former.projector

【最终输出】
scene_tokens: (B, 512, 4096)
```

---

### 完整Pipeline总结

| 阶段 | 输入 | 输出 | 操作 |
|------|------|------|------|
| **GT预处理** | nuScenes原始数据 | .pkl缓存 | 世界→ego→裁剪→采样→归一化 |
| **Dataset加载** | sample_token | 单个样本dict | 加载图像+GT+相机参数 |
| **Batch构建** | List[样本] | Batch dict | Stack+Padding |
| **Q-Former** | Batch | scene_tokens | Backbone→Decoder→Projector |

**最终得到**：
```python
scene_tokens: (B, 512, 4096)
# 准备替换text_ids中的IMAGE_TOKEN_INDEX (-200)
# 然后送入LLM
```

---

## 关键数值示例（B=4）

```python
# Batch size = 4

images:         (4, 6, 3, 336, 336)    32.4 MB
text_ids:       (4, 76)                 2.4 KB
cam_intrinsics: (4, 6, 3, 3)           1.1 KB
cam_extrinsics: (4, 6, 4, 4)           1.5 KB
ego_pose:       (4, 4, 4)               256 B
gt_classes:     (4, 20)                 320 B   # 假设max_N=20
gt_points:      (4, 20, 20, 2)         25.6 KB
gt_mask:        (4, 20)                 80 B

Q-Former内部：
img_feats:      (24, 256, 21, 21)      5.4 MB
memory:         (4, 2646, 256)        10.8 MB
queries:        (4, 512, 256)          2.1 MB
scene_tokens:   (4, 512, 4096)        33.6 MB  ← 最终输出

总内存峰值：约 85 MB (forward pass, 单次forward)
```

---

**这就是从nuScenes原始数据到Q-Former输出的完整流程！每一步的数据格式和维度都已详细列出。** 📊✨

