"""
NuScenes map data API
Extracts map elements in world coordinates

与 MapTR 保持一致的类别定义：
- divider (0): road_divider + lane_divider (合并为一类)
- ped_crossing (1): 人行横道
- boundary (2): 道路边界 (从 road_segment 和 lane 图层提取轮廓)
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString, LinearRing
from shapely.geometry import box as Box
from shapely.ops import unary_union, linemerge
import warnings

warnings.filterwarnings('ignore')


class NuScenesMapAPI:
    """
    Interface to nuScenes map data
    与 MapTR 保持一致的类别定义
    """
    
    # ==========================================================
    # 类别定义 (与 MapTR 一致)
    # ==========================================================
    
    # divider 包含两种分隔线，都映射到类别 0
    DIVIDER_LAYERS = ['road_divider', 'lane_divider']
    
    # ped_crossing 映射到类别 1
    PED_CROSSING_LAYERS = ['ped_crossing']
    
    # boundary 从这些图层提取轮廓，映射到类别 2
    BOUNDARY_LAYERS = ['road_segment', 'lane']
    
    # 类别名称 (与 MapTR 一致)
    CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']
    
    # 类别 ID
    CLASS_IDS = {
        'divider': 0,
        'ped_crossing': 1,
        'boundary': 2,
    }
    
    # 用于向后兼容的 CLASS_MAPPING (nuScenes layer -> class_id)
    CLASS_MAPPING = {
        'road_divider': 0,      # divider
        'lane_divider': 0,      # divider (合并)
        'ped_crossing': 1,      # ped_crossing
        'road_segment': 2,      # boundary
        'lane': 2,              # boundary
    }
    
    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-trainval',
        verbose: bool = False
    ):
        """
        Args:
            dataroot: Path to nuScenes data
            version: Dataset version
            verbose: Print loading info
        """
        self.dataroot = dataroot
        self.version = version
        
        if verbose:
            print(f"Loading nuScenes {version}...")
        
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=verbose)
        
        # Load all available maps
        self.maps = {}
        for location in ['boston-seaport', 'singapore-onenorth', 
                         'singapore-hollandvillage', 'singapore-queenstown']:
            try:
                self.maps[location] = NuScenesMap(dataroot=dataroot, map_name=location)
            except:
                if verbose:
                    print(f"  Map {location} not found, skipping")
        
        if verbose:
            print(f"Loaded {len(self.maps)} maps: {list(self.maps.keys())}")
    
    def get_ego_pose(self, sample_token: str) -> Dict:
        """
        Get ego pose for a sample (based on LIDAR_TOP)
        
        Args:
            sample_token: nuScenes sample token
            
        Returns:
            {
                'translation': [3] array,
                'rotation': Quaternion,
                'rotation_matrix': [3, 3] array,
            }
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Use LIDAR_TOP as reference (same as MapTR)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose_record = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        translation = np.array(ego_pose_record['translation'], dtype=np.float32)
        rotation_quat = Quaternion(ego_pose_record['rotation'])
        rotation_matrix = rotation_quat.rotation_matrix.astype(np.float32)
        
        return {
            'translation': translation,
            'rotation': rotation_quat,
            'rotation_matrix': rotation_matrix,
        }
    
    def get_sample_location(self, sample_token: str) -> str:
        """Get location name for a sample"""
        sample = self.nusc.get('sample', sample_token)
        scene = self.nusc.get('scene', sample['scene_token'])
        log = self.nusc.get('log', scene['log_token'])
        return log['location']
    
    def get_map_elements(
        self,
        sample_token: str,
        search_radius: float = 60.0
    ) -> List[Dict]:
        """
        Get map elements near ego vehicle (in world coordinates)
        与 MapTR 保持一致的类别定义
        
        Args:
            sample_token: Sample token
            search_radius: Search radius in meters
            
        Returns:
            List of {
                'class_id': int (0=divider, 1=ped_crossing, 2=boundary),
                'points_world': np.ndarray [N, 2],
                'is_closed': bool,
                'token': str,
            }
        """
        location = self.get_sample_location(sample_token)
        
        if location not in self.maps:
            return []
        
        map_api = self.maps[location]
        ego_pose = self.get_ego_pose(sample_token)
        ego_xy = ego_pose['translation'][:2]
        
        elements = []
        
        # ==========================================================
        # 1. Divider (类别 0): road_divider + lane_divider
        # ==========================================================
        for layer_name in self.DIVIDER_LAYERS:
            records = map_api.get_records_in_radius(
                ego_xy[0], ego_xy[1],
                radius=search_radius,
                layer_names=[layer_name]
            ).get(layer_name, [])
            
            for record_token in records:
                record = map_api.get(layer_name, record_token)
                points_world = self._extract_line_geometry(map_api, record)
                
                if points_world is None or len(points_world) < 2:
                    continue
                
                elements.append({
                    'class_id': self.CLASS_IDS['divider'],
                    'points_world': points_world,
                    'is_closed': False,
                    'token': record_token,
                })
        
        # ==========================================================
        # 2. Pedestrian Crossing (类别 1)
        # 与 MapTR 一致：作为闭合多边形处理
        # ==========================================================
        ped_elements = self._extract_ped_crossing_elements(
            map_api, ego_xy, search_radius
        )
        elements.extend(ped_elements)
        
        # ==========================================================
        # 3. Boundary (类别 2): 从 road_segment 和 lane 提取边界
        # 与 MapTR 完全一致：合并多边形 → 提取外/内轮廓 → 作为线段
        # ==========================================================
        boundary_elements = self._extract_boundary_elements_maptr_style(
            map_api, ego_xy, search_radius
        )
        elements.extend(boundary_elements)
        
        return elements
    
    def _extract_line_geometry(
        self,
        map_api: NuScenesMap,
        record: Dict
    ) -> Optional[np.ndarray]:
        """
        Extract line geometry from a map record (for dividers)
        
        Returns:
            [N, 2] points in world coordinates or None
        """
        try:
                if 'line_token' not in record:
                    return None
                
                line_token = record['line_token']
                line_record = map_api.get('line', line_token)
                node_tokens = line_record['node_tokens']
                
                points = []
                for node_token in node_tokens:
                    node = map_api.get('node', node_token)
                    points.append([node['x'], node['y']])
                
                return np.array(points, dtype=np.float32)
        
        except Exception as e:
            return None
    
    def _extract_polygon_from_record(
        self,
        map_api: NuScenesMap,
        record: Dict
    ) -> Optional[Polygon]:
        """
        Extract Shapely Polygon from a map record
        
        Returns:
            Shapely Polygon or None
        """
        try:
            if 'polygon_token' not in record:
                return None
            
            polygon_token = record['polygon_token']
            polygon_record = map_api.get('polygon', polygon_token)
            exterior_tokens = polygon_record['exterior_node_tokens']
            
            if len(exterior_tokens) < 3:
                return None
            
            points = []
            for node_token in exterior_tokens:
                node = map_api.get('node', node_token)
                points.append([node['x'], node['y']])
            
            poly = Polygon(points)
            if not poly.is_valid:
                poly = poly.buffer(0)
            
            return poly if poly.is_valid and not poly.is_empty else None
        
        except Exception as e:
            return None
    
    def _extract_ped_crossing_elements(
        self,
        map_api: NuScenesMap,
        ego_xy: np.ndarray,
        search_radius: float
    ) -> List[Dict]:
        """
        Extract pedestrian crossing elements
        
        ★★★ 与 MapTR 完全一致的处理方式 ★★★
        
        MapTR 对 ped_crossing 的处理与 boundary 完全相同：
        1. 合并所有 ped_crossing 多边形
        2. 提取外/内轮廓
        3. 方向校正
        4. 作为线段返回（不是多边形！）
        5. 保留首尾重复点（让采样后自然判断是否闭合）
        
        Args:
            map_api: NuScenes map API
            ego_xy: Ego vehicle position [x, y]
            search_radius: Search radius
            
        Returns:
            List of ped_crossing elements
        """
        elements = []
        
        # 收集所有 ped_crossing 多边形
        ped_polygons = []
        for layer_name in self.PED_CROSSING_LAYERS:
            records = map_api.get_records_in_radius(
                ego_xy[0], ego_xy[1],
                radius=search_radius,
                layer_names=[layer_name]
            ).get(layer_name, [])
            
            for record_token in records:
                record = map_api.get(layer_name, record_token)
                poly = self._extract_polygon_from_record(map_api, record)
                if poly is not None:
                    ped_polygons.append(poly)
        
        if len(ped_polygons) == 0:
            return elements
        
        # ==========================================================
        # 与 MapTR 一致：合并多边形
        # ==========================================================
        try:
            union_ped = unary_union(ped_polygons)
        except Exception as e:
            return elements
        
        if union_ped.is_empty:
            return elements
        
        # 处理单个多边形或多个多边形
        if union_ped.geom_type == 'Polygon':
            polygons = [union_ped]
        elif union_ped.geom_type == 'MultiPolygon':
            polygons = list(union_ped.geoms)
        else:
            return elements
        
        # ==========================================================
        # 提取外轮廓和内洞轮廓
        # ==========================================================
        exteriors = []
        interiors = []
        
        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue
            exteriors.append(poly.exterior)
            for interior in poly.interiors:
                interiors.append(interior)
        
        # ==========================================================
        # 处理每个轮廓
        # ★ 关键：保留首尾重复点，让采样后自然判断闭合
        # ==========================================================
        ped_idx = 0
        
        # 处理外边界 (应为顺时针)
        for ext in exteriors:
            try:
                coords = np.array(ext.coords, dtype=np.float32)
                
                # MapTR: 如果是逆时针，反转为顺时针
                if ext.is_ccw:
                    coords = coords[::-1]
                
                # ★ 关键：不移除首尾重复点！
                # 这样采样后 pts[0] == pts[-1]，MapTR 会正确判断为闭合
                
                if len(coords) >= 3:
                    elements.append({
                        'class_id': self.CLASS_IDS['ped_crossing'],
                        'points_world': coords,
                        'is_closed': True,  # 闭合多边形边界
                        'token': f"ped_ext_{ped_idx}",
                    })
                    ped_idx += 1
                    
            except Exception as e:
                continue
        
        # 处理内洞边界 (应为逆时针)
        for inter in interiors:
            try:
                coords = np.array(inter.coords, dtype=np.float32)
                
                # MapTR: 如果不是逆时针，反转为逆时针
                if not inter.is_ccw:
                    coords = coords[::-1]
                
                if len(coords) >= 3:
                    elements.append({
                        'class_id': self.CLASS_IDS['ped_crossing'],
                        'points_world': coords,
                        'is_closed': True,
                        'token': f"ped_int_{ped_idx}",
                    })
                    ped_idx += 1
                    
            except Exception as e:
                continue
        
        return elements
    
    def _extract_boundary_elements_maptr_style(
        self,
        map_api: NuScenesMap,
        ego_xy: np.ndarray,
        search_radius: float
    ) -> List[Dict]:
        """
        Extract boundary elements from road_segment and lane layers
        
        ★★★ 与 MapTR 完全一致的处理方式 ★★★
        
        MapTR 的处理流程：
        1. 收集所有 road_segment 多边形
        2. 收集所有 lane 多边形
        3. 分别合并 road_segment 和 lane
        4. 再将两者合并成一个大多边形
        5. 提取外轮廓 (exteriors) 和内洞轮廓 (interiors)
        6. 确保方向：外边界顺时针，内边界逆时针
        7. 每个轮廓作为一个 boundary 实例（线段，非多边形）
        
        Args:
            map_api: NuScenes map API
            ego_xy: Ego vehicle position [x, y]
            search_radius: Search radius
            
        Returns:
            List of boundary elements
        """
        elements = []
        
        # ==========================================================
        # Step 1 & 2: 收集所有 road_segment 和 lane 多边形
        # ==========================================================
        road_polygons = []
        lane_polygons = []
        
        for layer_name in self.BOUNDARY_LAYERS:
            records = map_api.get_records_in_radius(
                ego_xy[0], ego_xy[1],
                radius=search_radius,
                layer_names=[layer_name]
            ).get(layer_name, [])
            
            for record_token in records:
                record = map_api.get(layer_name, record_token)
                poly = self._extract_polygon_from_record(map_api, record)
                
                if poly is not None:
                    if layer_name == 'road_segment':
                        road_polygons.append(poly)
                    elif layer_name == 'lane':
                        lane_polygons.append(poly)
        
        if len(road_polygons) == 0 and len(lane_polygons) == 0:
            return elements
        
        # ==========================================================
        # Step 3 & 4: 合并多边形 (与 MapTR 一致)
        # ==========================================================
        try:
            # 合并 road_segment
            if len(road_polygons) > 0:
                union_roads = unary_union(road_polygons)
            else:
                union_roads = Polygon()
            
            # 合并 lane
            if len(lane_polygons) > 0:
                union_lanes = unary_union(lane_polygons)
            else:
                union_lanes = Polygon()
            
            # 将 road 和 lane 合并成一个大多边形
            union_segments = unary_union([union_roads, union_lanes])
            
        except Exception as e:
            return elements
        
        if union_segments.is_empty:
            return elements
        
        # ==========================================================
        # Step 5: 提取外轮廓和内洞轮廓
        # ==========================================================
        exteriors = []
        interiors = []
        
        # 处理单个多边形或多个多边形
        if union_segments.geom_type == 'Polygon':
            polygons = [union_segments]
        elif union_segments.geom_type == 'MultiPolygon':
            polygons = list(union_segments.geoms)
        else:
            return elements
        
        for poly in polygons:
            if not poly.is_valid or poly.is_empty:
                continue
            
            # 外边界
            exteriors.append(poly.exterior)
            
            # 内洞边界
            for interior in poly.interiors:
                interiors.append(interior)
        
        # ==========================================================
        # Step 6 & 7: 处理每个轮廓，确保方向
        # ★ 关键：保留首尾重复点，让采样后自然判断是否闭合
        # ==========================================================
        boundary_idx = 0
        
        # 处理外边界 (应为顺时针，即 is_ccw = False)
        for ext in exteriors:
            try:
                coords = np.array(ext.coords, dtype=np.float32)
                
                # MapTR: 如果是逆时针 (ccw)，则反转为顺时针
                if ext.is_ccw:
                    coords = coords[::-1]
                
                # ★ 关键：不移除首尾重复点！
                # 这样采样后 pts[0] == pts[-1]，MapTR 会正确判断为闭合
                # 后续裁剪时，如果边界被截断，首尾自然会不同
                
                if len(coords) >= 3:
                    elements.append({
                        'class_id': self.CLASS_IDS['boundary'],
                        'points_world': coords,
                        'is_closed': True,  # 闭合多边形边界
                        'token': f"boundary_ext_{boundary_idx}",
                    })
                    boundary_idx += 1
                    
            except Exception as e:
                continue
        
        # 处理内洞边界 (应为逆时针，即 is_ccw = True)
        for inter in interiors:
            try:
                coords = np.array(inter.coords, dtype=np.float32)
                
                # MapTR: 如果不是逆时针，则反转为逆时针
                if not inter.is_ccw:
                    coords = coords[::-1]
                
                # ★ 关键：不移除首尾重复点！
                
                if len(coords) >= 3:
                    elements.append({
                        'class_id': self.CLASS_IDS['boundary'],
                        'points_world': coords,
                        'is_closed': True,  # 闭合多边形边界
                        'token': f"boundary_int_{boundary_idx}",
                    })
                    boundary_idx += 1
                    
            except Exception as e:
                continue
        
        return elements


# ==========================================================
# 兼容旧代码的类别映射 (用于过渡期)
# ==========================================================
CLASS_MAPPING = {
    'road_divider': 0,      # divider
    'lane_divider': 0,      # divider (合并)
    'ped_crossing': 1,      # ped_crossing
    'road_segment': 2,      # boundary
    'lane': 2,              # boundary
}


if __name__ == "__main__":
    import sys
    
    dataroot = '/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini'
    
    print("=" * 60)
    print("Testing NuScenesMapAPI (MapTR-compatible)")
    print("=" * 60)
    print(f"\nClass definitions (与 MapTR 一致):")
    print(f"  0: divider (road_divider + lane_divider)")
    print(f"  1: ped_crossing (闭合多边形)")
    print(f"  2: boundary (road_segment + lane 轮廓，作为线段)")
    print()
    
    try:
        api = NuScenesMapAPI(dataroot, version='v1.0-mini', verbose=True)
        
        # Test on first sample
        first_scene = api.nusc.scene[0]
        sample_token = first_scene['first_sample_token']
        
        print(f"\nTesting sample: {sample_token[:16]}...")
        
        # Test ego pose
        ego_pose = api.get_ego_pose(sample_token)
        print(f"✓ Ego pose: {ego_pose['translation']}")
        
        # Test map elements
        elements = api.get_map_elements(sample_token)
        print(f"✓ Map elements: {len(elements)}")
        
        # 统计各类别数量
        class_counts = {name: 0 for name in api.CLASS_NAMES}
        for elem in elements:
            class_name = api.CLASS_NAMES[elem['class_id']]
            class_counts[class_name] += 1
        
        print(f"\nClass distribution:")
        for name, count in class_counts.items():
            print(f"  {name}: {count}")
        
        # 显示示例
        print(f"\nSample elements:")
        for i, elem in enumerate(elements[:10]):
            print(f"  [{i}] class={api.CLASS_NAMES[elem['class_id']]}, "
                  f"points={len(elem['points_world'])}, closed={elem['is_closed']}")
        
        print("\n" + "=" * 60)
        print("✓ NuScenesMapAPI test passed!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
