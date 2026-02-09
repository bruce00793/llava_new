# Step 1: Ground Truth Generation (MapTR Aligned)

## Overview

This step generates ground truth annotations for map detection from nuScenes dataset, strictly aligned with MapTR's GT format and processing pipeline.

**Goal**: For each sample (frame), extract 3 types of map elements (divider, lane_divider, ped_crossing) and represent each instance as **20 uniformly sampled points** in ego/LiDAR BEV coordinates.

## Coordinate System

Following MapTR:
- **Frame**: ego/LiDAR BEV (origin at ego vehicle center)
- **Axes**: 
  - x: left(-) to right(+)
  - y: back(-) to front(+)
- **Units**: meters
- **pc_range**: `[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]`
- **patch_size**: (60m, 30m) - height × width

## Class Mapping

| nuScenes Layer  | Our Class ID | Class Name      | Geometry Type |
|-----------------|--------------|-----------------|---------------|
| road_divider    | 0            | divider         | Polyline      |
| lane_divider    | 1            | lane_divider    | Polyline      |
| ped_crossing    | 2            | ped_crossing    | Polygon       |

## GT Data Format

For each sample, we generate:

```python
{
    'sample_token': str,                          # nuScenes sample token
    'gt_classes': np.ndarray,                     # [M], int64, values in {0, 1, 2}
    'gt_points': np.ndarray,                      # [M, 20, 2], float32, in meters (ego frame)
    'gt_is_closed': np.ndarray,                   # [M], bool (True for polygons, False for polylines)
    'gt_bbox': np.ndarray,                        # [M, 4], float32, AABB [cx, cy, w, h]
}
```

where `M` is the number of instances (can be 0).

## Processing Pipeline

### 1. Extract Map Elements (World Coordinates)
- Query nuScenes map API with ego position
- Get geometries for 3 classes within search radius
- Extract node coordinates in world frame

### 2. Transform to Ego Frame
- Get ego pose from LIDAR_TOP sensor
- Transform: `p_ego = R.T @ (p_world - t)`
- Return 2D projection (x, y)

### 3. Clip by ROI
- **Polylines**: 
  - Clip with ROI rectangle
  - If result is MultiLineString, split into multiple instances
  - Filter by arc length (min: 1.0m)
- **Polygons**:
  - Clip with ROI rectangle
  - If result is MultiPolygon, split into multiple instances
  - Filter by area (min: 0.5m²)

### 4. Sample to 20 Points
- **Polylines**: uniform arc-length sampling
  ```python
  distances = np.linspace(0, total_length, 20)
  points = [line.interpolate(dist) for dist in distances]
  # pts[0] != pts[19] (open curve)
  ```
- **Polygons**: uniform perimeter sampling (clockwise, **first == last**)
  ```python
  distances = np.linspace(0, perimeter, 20)  # NOT 21!
  points = [boundary.interpolate(dist) for dist in distances]
  points = ensure_clockwise(points)
  # pts[0] == pts[19] (closed curve) - THIS IS HOW MAPTR DETECTS POLYGONS!
  ```
  
**IMPORTANT**: MapTR uses `is_poly = (pts[0] == pts[-1])` to detect polygons. So polygon points must have identical first and last points.

### 5. Compute AABB
- `bbox = [x_center, y_center, width, height]`
- Derived from the 20 points (for visualization/auxiliary)

### 6. Validate and Cache
- Check for NaN/Inf
- Check for degenerate geometries
- Save to pickle file

## Equivalence Classes (For Matching/Loss)

Although not stored in GT, equivalence classes are defined for Hungarian matching:

- **Polylines** (divider, lane_divider): **2 variants**
  - Original order: `[p0, p1, ..., p19]`
  - Reversed order: `[p19, p18, ..., p0]` (using `flip`)
  - Detection: `pts[0] != pts[-1]`
  
- **Polygons** (ped_crossing): **20 variants**
  - 20 cyclic shifts using `roll`: `[pk, pk+1, ..., p19, p0, ..., pk-1]` for k=0..19
  - Clockwise direction is fixed, no reverse
  - Detection: `pts[0] == pts[-1]` (first and last points are identical)

**MapTR判断逻辑**:
```python
is_poly = fixed_num_pts[0].equal(fixed_num_pts[-1])  # 首尾相同 = 多边形
if is_poly:
    for shift_right_i in range(fixed_num):
        shift_pts_list.append(fixed_num_pts.roll(shift_right_i, 0))  # 循环移位
else:
    shift_pts_list.append(fixed_num_pts)         # 原始
    shift_pts_list.append(fixed_num_pts.flip(0)) # 反向
```

This is handled during matching/loss computation, not in GT generation.

## Module Structure

```
llava/data/map_gt/
├── __init__.py                 # Module exports
├── geometry.py                 # Geometric operations
│   ├── transform_to_ego()
│   ├── clip_polyline_by_roi()
│   ├── clip_polygon_by_roi()
│   ├── sample_polyline_20()
│   ├── sample_polygon_20()
│   ├── ensure_clockwise()
│   ├── compute_aabb()
│   ├── filter_by_length()
│   ├── filter_by_area()
│   └── is_valid_instance()
│
├── nuscenes_map_api.py         # nuScenes data interface
│   └── NuScenesMapAPI
│       ├── get_ego_pose()
│       ├── get_sample_location()
│       └── get_map_elements()
│
└── cache.py                    # GT cache management
    └── GTCache
        ├── process_one_sample()
        ├── build()               # Generate cache for a split
        └── load()                # Load cached GT
```

## Usage

### Generate GT Cache

```bash
# For training split
python tools/generate_gt_cache.py --dataroot /path/to/nuscenes --version v1.0-trainval --split train

# For validation split
python tools/generate_gt_cache.py --dataroot /path/to/nuscenes --version v1.0-trainval --split val
```

### Load GT in Code

```python
from llava.data.map_gt import GTCache

cache = GTCache(dataroot='/path/to/nuscenes', version='v1.0-trainval')

# Load GT for a sample
gt_data = cache.load('sample_token_here')

# Access GT
gt_classes = gt_data['gt_classes']      # [M]
gt_points = gt_data['gt_points']        # [M, 20, 2]
gt_is_closed = gt_data['gt_is_closed']  # [M]
gt_bbox = gt_data['gt_bbox']            # [M, 4]
```

## Testing

Run comprehensive tests:
```bash
cd /home/cly/auto/llava_test/LLaVA
python tests/test_gt_generation.py
```

Tests include:
1. Geometry functions (transform, clip, sample, AABB)
2. nuScenes map API (loading maps, extracting elements)
3. GT cache building (processing samples, data format validation)

## Output Files

After running GT generation:

```
{dataroot}/gt_cache/
├── annotations/                     # Pickled GT files
│   ├── {sample_token_1}.pkl
│   ├── {sample_token_2}.pkl
│   └── ...
├── splits/                          # Sample token lists
│   ├── train.txt
│   └── val.txt
└── metadata_{split}.json            # Statistics and config
```

### Metadata Example

```json
{
  "version": "1.0",
  "nuscenes_version": "v1.0-trainval",
  "pc_range": [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
  "patch_size": [60.0, 30.0],
  "num_samples": 28130,
  "class_mapping": {
    "road_divider": 0,
    "lane_divider": 1,
    "ped_crossing": 2
  },
  "class_names": ["divider", "lane_divider", "ped_crossing"],
  "thresholds": {
    "min_arc_length": 1.0,
    "min_area": 0.5
  },
  "statistics": {
    "total_samples": 28130,
    "total_instances": 45230,
    "class_counts": [12340, 18920, 13970],
    "empty_samples": 152
  }
}
```

## Key Differences from Previous Implementation

1. **Coordinate system**: Now strictly follows MapTR's definition (x=left/right, y=back/front)
2. **pc_range**: Changed to MapTR's standard `[-15,-30,-2, 15,30,2]`
3. **Polygon sampling**: Now 20 unique points (NOT including duplicate first/last)
4. **Clockwise order**: Enforced for polygons
5. **Module organization**: Cleaner separation (API / geometry / cache)
6. **AABB included**: For visualization and potential auxiliary supervision
7. **Equivalence classes**: Defined but not stored (handled during matching)

## Next Steps

**Step 2**: Create Dataset class that:
- Loads GT from cache
- Loads images (6 cameras)
- Normalizes coordinates (米制 → [-1, 1])
- Constructs batches for training

See `docs/STEP2_DATASET.md` (to be created)

## Notes

- **nuScenes mini** doesn't include map data, so GT will be empty (M=0) for all samples
- For actual training, use **nuScenes v1.0-trainval** with full map data
- Empty GT (M=0) is valid and handled by the model (unmatched predictions become background)

---

**Status**: ✅ Step 1 Completed

**Validated**:
- Geometry processing functions
- nuScenes API interface
- GT cache generation pipeline
- Data format correctness

**Ready for**: Step 2 (Dataset implementation)

