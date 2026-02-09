#!/usr/bin/env python3
"""
使用 MapTR 的代码生成 GT 缓存

这个脚本直接调用 MapTR 的 GT 生成逻辑，确保与 MapTR 100% 一致。

用法:
    python tools/generate_gt_from_maptr.py \
        --dataroot /path/to/nuscenes \
        --version v1.0-trainval \
        --output-dir /path/to/output/gt_cache
"""

import os
import sys
import argparse
import pickle
import numpy as np
from tqdm import tqdm
import torch
from shapely import affinity, ops
from shapely.geometry import LineString, box, MultiPolygon, MultiLineString, Polygon

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion

# ============================================================================
# 以下代码直接从 MapTR 复制，避免 mmdet/mmcv 依赖
# 来源: MapTR/projects/mmdet3d_plugin/datasets/nuscenes_map_dataset.py
# ============================================================================

#将 NumPy 数组（普通计算）转为PyTorch Tensor
def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`."""
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

#存储"一组线段"，每条线段代表一个地图元素
class LiDARInstanceLines(object):
    """Line instance in LIDAR coordinates (直接从 MapTR 复制)"""
    
    def __init__(self, 
                 instance_line_list, 
                 sample_dist=1,     # 采样间隔（米）
                 num_samples=250,   # 最多采样点数
                 padding=False,
                 fixed_num=-1,
                 padding_value=-10000,  # 填充值
                 patch_size=None):
        assert isinstance(instance_line_list, list)
        assert patch_size is not None
        if len(instance_line_list) != 0:
            assert isinstance(instance_line_list[0], LineString)
        self.patch_size = patch_size
        self.max_x = self.patch_size[1] / 2
        self.max_y = self.patch_size[0] / 2
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_num
        self.padding_value = padding_value
        self.instance_list = instance_line_list

    #将每条线均匀采样为 20 个点
    @property
    def fixed_num_sampled_points(self):
        """
        return torch.Tensor([N,fixed_num,2])
        """
        assert len(self.instance_list) != 0
        instance_points_list = []
        for instance in self.instance_list:
            distances = np.linspace(0, instance.length, self.fixed_num)
            sampled_points = np.array([list(instance.interpolate(distance).coords) for distance in distances]).reshape(-1, 2)
            instance_points_list.append(sampled_points)
        instance_points_array = np.array(instance_points_list)
        instance_points_tensor = to_tensor(instance_points_array)
        instance_points_tensor = instance_points_tensor.to(dtype=torch.float32)
        instance_points_tensor[:,:,0] = torch.clamp(instance_points_tensor[:,:,0], min=-self.max_x, max=self.max_x)
        instance_points_tensor[:,:,1] = torch.clamp(instance_points_tensor[:,:,1], min=-self.max_y, max=self.max_y)
        return instance_points_tensor


class VectorizedLocalMap(object):
    """VectorizedLocalMap (直接从 MapTR 复制，100% 一致)"""
    
    CLASS2LABEL = {
        'road_divider': 0,
        'lane_divider': 0,
        'ped_crossing': 1,
        'contours': 2,
        'others': -1
    }
    
    def __init__(self,
                 dataroot,
                 patch_size,
                 map_classes=['divider', 'ped_crossing', 'boundary'],
                 line_classes=['road_divider', 'lane_divider'],
                 ped_crossing_classes=['ped_crossing'],
                 contour_classes=['road_segment', 'lane'],
                 sample_dist=1,
                 num_samples=250,
                 padding=False,
                 fixed_ptsnum_per_line=-1,
                 padding_value=-10000):
        super().__init__()
        self.data_root = dataroot
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        self.vec_classes = map_classes
        self.line_classes = line_classes
        self.ped_crossing_classes = ped_crossing_classes
        self.polygon_classes = contour_classes
        self.sample_dist = sample_dist
        self.num_samples = num_samples
        self.padding = padding
        self.fixed_num = fixed_ptsnum_per_line
        self.padding_value = padding_value
        self.patch_size = patch_size
        
        # 初始化地图 API
        self.map_explorer = {}
        for loc in self.MAPS:
            self.map_explorer[loc] = NuScenesMapExplorer(
                NuScenesMap(dataroot=self.data_root, map_name=loc)
            )
    
    def gen_vectorized_samples(self, location, lidar2global_translation, lidar2global_rotation):
        """使用 lidar2global 获取 GT 地图图层 (与 MapTR 100% 一致)"""
        map_pose = lidar2global_translation[:2]
        rotation = Quaternion(lidar2global_rotation)
        
        patch_box = (map_pose[0], map_pose[1], self.patch_size[0], self.patch_size[1])
        patch_angle = quaternion_yaw(rotation) / np.pi * 180
        
        vectors = []
        for vec_class in self.vec_classes:
            if vec_class == 'divider':
                line_geom = self.get_map_geom(patch_box, patch_angle, self.line_classes, location)
                line_instances_dict = self.line_geoms_to_instances(line_geom)
                for line_type, instances in line_instances_dict.items():
                    for instance in instances:
                        vectors.append((instance, self.CLASS2LABEL.get(line_type, -1)))
            elif vec_class == 'ped_crossing':
                ped_geom = self.get_map_geom(patch_box, patch_angle, self.ped_crossing_classes, location)
                ped_instance_list = self.ped_poly_geoms_to_instances(ped_geom)
                for instance in ped_instance_list:
                    vectors.append((instance, self.CLASS2LABEL.get('ped_crossing', -1)))
            elif vec_class == 'boundary':
                polygon_geom = self.get_map_geom(patch_box, patch_angle, self.polygon_classes, location)
                poly_bound_list = self.poly_geoms_to_instances(polygon_geom)
                for contour in poly_bound_list:
                    vectors.append((contour, self.CLASS2LABEL.get('contours', -1)))
        
        gt_labels = []
        gt_instance = []
        for instance, label in vectors:
            if label != -1:
                gt_instance.append(instance)
                gt_labels.append(label)
        
        gt_instance = LiDARInstanceLines(
            gt_instance, self.sample_dist, self.num_samples, 
            self.padding, self.fixed_num, self.padding_value, 
            patch_size=self.patch_size
        )
        
        return dict(gt_vecs_pts_loc=gt_instance, gt_vecs_label=gt_labels)
    
    def get_map_geom(self, patch_box, patch_angle, layer_names, location):
        """获取地图几何元素"""
        map_geom = []
        for layer_name in layer_names:
            if layer_name in self.line_classes:
                geoms = self.get_divider_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.polygon_classes:
                geoms = self.get_contour_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
            elif layer_name in self.ped_crossing_classes:
                geoms = self.get_ped_crossing_line(patch_box, patch_angle, layer_name, location)
                map_geom.append((layer_name, geoms))
        return map_geom
    
    def get_divider_line(self, patch_box, patch_angle, layer_name, location):
        """获取分隔线"""
        map_api = self.map_explorer[location].map_api
        
        if layer_name not in map_api.non_geometric_line_layers:
            return []
        
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        
        line_list = []
        records = getattr(map_api, layer_name)
        for record in records:
            line = map_api.extract_line(record['line_token'])
            if line.is_empty:
                continue
            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)
        
        return line_list
    
    def get_contour_line(self, patch_box, patch_angle, layer_name, location):
        """获取轮廓线（道路边界）"""
        map_api = self.map_explorer[location].map_api
        
        if layer_name not in map_api.non_geometric_polygon_layers:
            return []
        
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        records = getattr(map_api, layer_name)
        
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [map_api.extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]
                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        else:
            for record in records:
                polygon = map_api.extract_polygon(record['polygon_token'])
                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type == 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)
        
        return polygon_list
    
    def get_ped_crossing_line(self, patch_box, patch_angle, layer_name, location):
        """获取人行横道"""
        map_api = self.map_explorer[location].map_api
        
        if layer_name not in map_api.non_geometric_polygon_layers:
            return []
        
        patch_x, patch_y = patch_box[0], patch_box[1]
        patch = self.map_explorer[location].get_patch_coord(patch_box, patch_angle)
        
        polygon_list = []
        records = getattr(map_api, layer_name)
        for record in records:
            polygon = map_api.extract_polygon(record['polygon_token'])
            if polygon.is_valid:
                new_polygon = polygon.intersection(patch)
                if not new_polygon.is_empty:
                    new_polygon = affinity.rotate(new_polygon, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                    new_polygon = affinity.affine_transform(new_polygon, [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                    if new_polygon.geom_type == 'Polygon':
                        new_polygon = MultiPolygon([new_polygon])
                    polygon_list.append(new_polygon)
        
        return polygon_list
    
    def line_geoms_to_instances(self, line_geom):
        """将线几何转换为实例"""
        line_instances_dict = dict()
        for line_type, a_type_of_lines in line_geom:
            one_type_instances = self._one_type_line_geom_to_instances(a_type_of_lines)
            line_instances_dict[line_type] = one_type_instances
        return line_instances_dict
    
    def _one_type_line_geom_to_instances(self, line_geom):
        """处理单一类型的线"""
        line_instances = []
        for line in line_geom:
            if not line.is_empty:
                if line.geom_type == 'MultiLineString':
                    for single_line in line.geoms:
                        line_instances.append(single_line)
                elif line.geom_type == 'LineString':
                    line_instances.append(line)
        return line_instances
    
    def ped_poly_geoms_to_instances(self, ped_geom):
        """将人行横道多边形转换为线实例 (与 MapTR 100% 一致)"""
        ped_geom = ped_geom[0][1] if ped_geom else []
        if not ped_geom:
            return []
        
        union_segments = ops.unary_union(ped_geom)
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        # 修复：与 MapTR 一致，使用扩张边界 (-0.2, +0.2)
        local_patch = box(-max_x - 0.2, -max_y - 0.2, max_x + 0.2, max_y + 0.2)
        
        # 收集外轮廓和内轮廓
        exteriors = []
        interiors = []
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)
        
        results = []
        # 处理外轮廓
        for ext in exteriors:
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        # 修复：添加内轮廓处理（与 MapTR 一致）
        for inter in interiors:
            if not inter.is_ccw:
                inter = LineString(list(inter.coords)[::-1])
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        return self._one_type_line_geom_to_instances(results)
    
    def poly_geoms_to_instances(self, polygon_geom):
        """将多边形转换为轮廓线实例（边界）(与 MapTR 100% 一致)"""
        # MapTR 固定假设: polygon_geom[0] = road_segment, polygon_geom[1] = lane
        roads = polygon_geom[0][1]
        lanes = polygon_geom[1][1]
        
        # 分别合并 roads 和 lanes，再统一合并
        union_roads = ops.unary_union(roads)
        union_lanes = ops.unary_union(lanes)
        union_segments = ops.unary_union([union_roads, union_lanes])
        
        max_x = self.patch_size[1] / 2
        max_y = self.patch_size[0] / 2
        local_patch = box(-max_x + 0.2, -max_y + 0.2, max_x - 0.2, max_y - 0.2)
        
        # 收集外轮廓和内轮廓
        exteriors = []
        interiors = []
        if union_segments.geom_type != 'MultiPolygon':
            union_segments = MultiPolygon([union_segments])
        for poly in union_segments.geoms:
            exteriors.append(poly.exterior)
            for inter in poly.interiors:
                interiors.append(inter)
        
        results = []
        # 处理外轮廓
        for ext in exteriors:
            if ext.is_ccw:
                ext = LineString(list(ext.coords)[::-1])
            lines = ext.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        # 处理内轮廓
        for inter in interiors:
            if not inter.is_ccw:
                inter = LineString(list(inter.coords)[::-1])
            lines = inter.intersection(local_patch)
            if isinstance(lines, MultiLineString):
                lines = ops.linemerge(lines)
            results.append(lines)
        
        return self._one_type_line_geom_to_instances(results)


print("✓ MapTR core classes loaded (no mmdet/mmcv dependency)")


def parse_args():
    parser = argparse.ArgumentParser(description='Generate GT cache using MapTR')
    parser.add_argument('--dataroot', type=str, 
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--output-dir', type=str, 
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache',
                        help='Output directory for GT cache')
    parser.add_argument('--split', type=str, default='all',
                        choices=['train', 'val', 'all'],
                        help='Which split to process')
    parser.add_argument('--fixed-num', type=int, default=20,
                        help='Number of points per instance (default: 20)')
    parser.add_argument('--patch-size', type=float, nargs=2, default=[60.0, 30.0],
                        help='Patch size [h, w] in meters (default: 60 30)')
    return parser.parse_args()


def get_sample_tokens(nusc, version, split):
    """获取指定 split 的所有 sample tokens"""
    if version == 'v1.0-mini':
        if split == 'train':
            scenes = nusc.scene[:8]
        else:
            scenes = nusc.scene[8:]
    else:
        splits = create_splits_scenes()
        if split == 'train':
            scene_names = set(splits['train'])
        else:
            scene_names = set(splits['val'])
        scenes = [s for s in nusc.scene if s['name'] in scene_names]
    
    sample_tokens = []
    for scene in scenes:
        sample_token = scene['first_sample_token']
        while sample_token:
            sample_tokens.append(sample_token)
            sample = nusc.get('sample', sample_token)
            sample_token = sample['next']
    
    return sample_tokens


def get_lidar_pose(nusc, sample_token):
    """获取 LIDAR_TOP 的位姿"""
    sample = nusc.get('sample', sample_token)
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    calibrated = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
    
    # Ego pose
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    # Sensor calibration
    sensor_translation = np.array(calibrated['translation'])
    sensor_rotation = Quaternion(calibrated['rotation'])
    
    # LiDAR to global
    lidar2global_translation = ego_translation + ego_rotation.rotate(sensor_translation)
    lidar2global_rotation = ego_rotation * sensor_rotation
    
    return lidar2global_translation, lidar2global_rotation.elements


def get_location(nusc, sample_token):
    """获取样本所在位置"""
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    return log['location']


def process_sample(vector_map, nusc, sample_token, fixed_num):
    """
    使用 MapTR 的逻辑处理单个样本
    
    输出格式与你的代码 _process_gt() 期望的格式完全一致：
    - gt_points: 在自车坐标系中的坐标 (x: [-15, 15], y: [-30, 30])
    - 你的 _normalize_coords() 会将其归一化到 [-1, 1]
    
    Returns:
        dict with keys: sample_token, gt_classes, gt_points, gt_is_closed, gt_bbox
    """
    # 获取位姿
    lidar2global_translation, lidar2global_rotation = get_lidar_pose(nusc, sample_token)
    location = get_location(nusc, sample_token)
    
    # 使用 MapTR 的方法生成 GT
    try:
        anns = vector_map.gen_vectorized_samples(
            location, 
            lidar2global_translation, 
            lidar2global_rotation
        )
    except Exception as e:
        print(f"  Warning: Failed to process {sample_token[:16]}: {e}")
        return None
    
    gt_instance = anns['gt_vecs_pts_loc']
    gt_labels = anns['gt_vecs_label']
    
    if len(gt_labels) == 0:
        return {
            'sample_token': sample_token,
            'gt_classes': np.array([], dtype=np.int64),
            'gt_points': np.zeros((0, fixed_num, 2), dtype=np.float32),
            'gt_is_closed': np.array([], dtype=bool),
            'gt_bbox': np.zeros((0, 4), dtype=np.float32),
        }
    
    # 获取采样后的点
    try:
        # 使用 fixed_num_sampled_points 属性获取采样点
        sampled_points = gt_instance.fixed_num_sampled_points.numpy()  # [N, fixed_num, 2]
    except Exception as e:
        print(f"  Warning: Failed to sample points for {sample_token[:16]}: {e}")
        return None
    
    # 转换标签
    gt_classes = np.array(gt_labels, dtype=np.int64)
    gt_points = sampled_points.astype(np.float32)
    # MapTR 输出的坐标范围是 [-15, 15] x [-30, 30]
    # 与你的 pc_range = [-15, -30, -2, 15, 30, 2] 完全一致
    # 不需要额外转换，你的 _normalize_coords() 会处理归一化
    
    # 判断是否闭合（MapTR 的方式：首尾点相同）
    gt_is_closed = np.array([
        np.allclose(pts[0], pts[-1], atol=1e-3) 
        for pts in gt_points
    ], dtype=bool)
    
    # 计算 AABB（在自车坐标系中）
    gt_bbox = np.zeros((len(gt_classes), 4), dtype=np.float32)
    for i, pts in enumerate(gt_points):
        x_min, y_min = pts.min(axis=0)
        x_max, y_max = pts.max(axis=0)
        gt_bbox[i] = [
            (x_min + x_max) / 2,  # cx
            (y_min + y_max) / 2,  # cy
            x_max - x_min,         # w
            y_max - y_min          # h
        ]
    
    # 直接返回自车坐标系下的坐标（不归一化）
    # 你的 map_dataset.py 中的 _normalize_coords() 会处理归一化
    return {
        'sample_token': sample_token,
        'gt_classes': gt_classes,
        'gt_points': gt_points,  # 自车坐标系：x [-15,15], y [-30,30]
        'gt_is_closed': gt_is_closed,
        'gt_bbox': gt_bbox,  # 自车坐标系
    }


def main():
    args = parse_args()
    
    print("=" * 60)
    print("Generate GT Cache using MapTR")
    print("=" * 60)
    print(f"Data root: {args.dataroot}")
    print(f"Version: {args.version}")
    print(f"Output: {args.output_dir}")
    print(f"Patch size: {args.patch_size}")
    print(f"Fixed num: {args.fixed_num}")
    print()
    
    # 创建输出目录
    ann_dir = os.path.join(args.output_dir, 'annotations')
    os.makedirs(ann_dir, exist_ok=True)
    
    # 加载 nuScenes
    print("Loading nuScenes...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # 初始化 MapTR 的 VectorizedLocalMap
    print("\nInitializing MapTR VectorizedLocalMap...")
    vector_map = VectorizedLocalMap(
        dataroot=args.dataroot,
        patch_size=args.patch_size,
        map_classes=['divider', 'ped_crossing', 'boundary'],
        line_classes=['road_divider', 'lane_divider'],
        ped_crossing_classes=['ped_crossing'],
        contour_classes=['road_segment', 'lane'],
        fixed_ptsnum_per_line=args.fixed_num,
    )
    print("✓ VectorizedLocalMap initialized")
    
    # 处理每个 split
    splits_to_process = ['train', 'val'] if args.split == 'all' else [args.split]
    
    for split in splits_to_process:
        print(f"\n{'='*60}")
        print(f"Processing {split} split...")
        print(f"{'='*60}")
        
        sample_tokens = get_sample_tokens(nusc, args.version, split)
        print(f"Found {len(sample_tokens)} samples")
        
        # 统计
        stats = {
            'total': len(sample_tokens),
            'success': 0,
            'failed': 0,
            'empty': 0,
            'class_counts': [0, 0, 0],  # divider, ped_crossing, boundary
        }
        
        for sample_token in tqdm(sample_tokens, desc=f"Processing {split}"):
            result = process_sample(vector_map, nusc, sample_token, args.fixed_num)
            
            if result is None:
                stats['failed'] += 1
                continue
            
            # 保存
            output_path = os.path.join(ann_dir, f"{sample_token}.pkl")
            with open(output_path, 'wb') as f:
                pickle.dump(result, f)
            
            stats['success'] += 1
            
            if len(result['gt_classes']) == 0:
                stats['empty'] += 1
            else:
                for cls_id in result['gt_classes']:
                    if 0 <= cls_id < 3:
                        stats['class_counts'][cls_id] += 1
        
        # 保存 split 文件
        split_dir = os.path.join(args.output_dir, 'splits')
        os.makedirs(split_dir, exist_ok=True)
        split_file = os.path.join(split_dir, f'{split}.txt')
        with open(split_file, 'w') as f:
            f.write('\n'.join(sample_tokens))
        
        # 打印统计
        print(f"\n{split.upper()} Statistics:")
        print(f"  Total: {stats['total']}")
        print(f"  Success: {stats['success']}")
        print(f"  Failed: {stats['failed']}")
        print(f"  Empty: {stats['empty']}")
        print(f"  Class counts:")
        print(f"    divider: {stats['class_counts'][0]}")
        print(f"    ped_crossing: {stats['class_counts'][1]}")
        print(f"    boundary: {stats['class_counts'][2]}")
    
    print(f"\n{'='*60}")
    print("✓ GT cache generation complete!")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
