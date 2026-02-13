"""
GT cache management
Generates and caches ground truth for map detection
"""

import os
import pickle
import json
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm

try:
    from .nuscenes_map_api import NuScenesMapAPI
    from .geometry import (
        transform_to_ego,
        clip_polyline_by_roi,
        clip_polygon_by_roi,
        sample_polyline_20,
        sample_polygon_20,
        compute_aabb,
        filter_by_length,
        filter_by_area,
        is_valid_instance,
    )
except ImportError:
    # For standalone execution
    from nuscenes_map_api import NuScenesMapAPI
    from geometry import (
        transform_to_ego,
        clip_polyline_by_roi,
        clip_polygon_by_roi,
        sample_polyline_20,
        sample_polygon_20,
        compute_aabb,
        filter_by_length,
        filter_by_area,
        is_valid_instance,
    )


class GTCache:
    """
    Generate and manage GT cache for map detection
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-trainval',
        pc_range: Optional[List[float]] = None,
        min_arc_length: float = 1.0,
        min_area: float = 0.5,
        output_dir: Optional[str] = None,
        verbose: bool = True
    ):
        """
        Args:
            dataroot: Path to nuScenes data
            version: Dataset version
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max]
            min_arc_length: Minimum arc length for polylines (meters)
            min_area: Minimum area for polygons (square meters)
            output_dir: Where to cache GT files
            verbose: Print progress
        """
        self.dataroot = dataroot
        self.version = version
        self.pc_range = pc_range or [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.min_arc_length = min_arc_length
        self.min_area = min_area
        self.output_dir = output_dir or os.path.join(dataroot, 'gt_cache')
        self.verbose = verbose
        
        # Initialize nuScenes API
        self.map_api = NuScenesMapAPI(dataroot, version, verbose=verbose)
    
    def process_one_sample(self, sample_token: str) -> Dict:
        """
        Generate GT for one sample
        
        Args:
            sample_token: Sample token
            
        Returns:
            GT dict with keys: sample_token, gt_classes, gt_points, gt_is_closed, gt_bbox
        """
        # Get ego pose
        ego_pose = self.map_api.get_ego_pose(sample_token)
        
        # Get map elements (world coordinates)
        map_elements = self.map_api.get_map_elements(sample_token)
        
        instances = []
        
        for elem in map_elements:
            class_id = elem['class_id']
            points_world = elem['points_world']
            is_closed = elem['is_closed']
            
            # Transform to ego frame
            points_ego = transform_to_ego(
                points_world,
                ego_pose['translation'],
                ego_pose['rotation_matrix']
            )
            
            # Clip by ROI
            if is_closed:
                clipped_geoms = clip_polygon_by_roi(points_ego, self.pc_range)
                clipped_geoms = filter_by_area(clipped_geoms, self.min_area)
            else:
                clipped_geoms = clip_polyline_by_roi(points_ego, self.pc_range)
                clipped_geoms = filter_by_length(clipped_geoms, self.min_arc_length)
            
            # Sample each clipped geometry to 20 points
            for geom in clipped_geoms:
                if is_closed:
                    points_20 = sample_polygon_20(geom)
                else:
                    points_20 = sample_polyline_20(geom)
                
                if points_20 is None:
                    continue
                
                if not is_valid_instance(points_20):
                    continue
                
                # Compute AABB
                bbox = compute_aabb(points_20)
                
                instances.append({
                    'class_id': class_id,
                    'points': points_20,
                    'is_closed': is_closed,
                    'bbox': bbox,
                })
        
        # Format as arrays
        if len(instances) == 0:
            # Empty GT
            gt_data = {
                'sample_token': sample_token,
                'gt_classes': np.array([], dtype=np.int64),
                'gt_points': np.zeros((0, 20, 2), dtype=np.float32),
                'gt_is_closed': np.array([], dtype=np.bool_),
                'gt_bbox': np.zeros((0, 4), dtype=np.float32),
            }
        else:
            gt_data = {
                'sample_token': sample_token,
                'gt_classes': np.array([inst['class_id'] for inst in instances], dtype=np.int64),
                'gt_points': np.stack([inst['points'] for inst in instances], axis=0),
                'gt_is_closed': np.array([inst['is_closed'] for inst in instances], dtype=np.bool_),
                'gt_bbox': np.stack([inst['bbox'] for inst in instances], axis=0),
            }
        
        return gt_data
    
    def build(self, split: str = 'train') -> Dict:
        """
        Build GT cache for a split
        
        Args:
            split: 'train' or 'val'
            
        Returns:
            Statistics dict
        """
        # Create output directories
        ann_dir = os.path.join(self.output_dir, 'annotations')
        split_dir = os.path.join(self.output_dir, 'splits')
        os.makedirs(ann_dir, exist_ok=True)
        os.makedirs(split_dir, exist_ok=True)
        
        # Get sample tokens for split
        # For mini: first 8 scenes = train, last 2 = val
        # For trainval: use official nuScenes split
        all_scenes = self.map_api.nusc.scene
        
        if self.version == 'v1.0-mini':
            if split == 'train':
                scenes = all_scenes[:8]
            else:
                scenes = all_scenes[8:]
        else:
            # For full dataset, use official nuScenes train/val split
            try:
                from nuscenes.utils.splits import create_splits_scenes
                splits = create_splits_scenes()
                
                if split == 'train':
                    scene_names = set(splits['train'])
                else:
                    scene_names = set(splits['val'])
                
                scenes = [s for s in all_scenes if s['name'] in scene_names]
                print(f"Using official nuScenes {split} split: {len(scenes)} scenes")
                
            except ImportError:
                print("Warning: nuscenes.utils.splits not available, using fallback split")
                # Fallback: scene-based 80/20 split
                split_idx = int(len(all_scenes) * 0.8)
                if split == 'train':
                    scenes = all_scenes[:split_idx]
                else:
                    scenes = all_scenes[split_idx:]
        
        # Collect all sample tokens
        sample_tokens = []
        for scene in scenes:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = self.map_api.nusc.get('sample', sample_token)
                sample_token = sample['next']
        
        if self.verbose:
            print(f"\nProcessing {len(sample_tokens)} samples for {split} split...")
        
        # Statistics
        stats = {
            'total_samples': len(sample_tokens),
            'total_instances': 0,
            'class_counts': [0, 0, 0],
            'empty_samples': 0,
        }
        
        # Process each sample
        for sample_token in tqdm(sample_tokens, desc=f"Generating {split} GT", disable=not self.verbose):
            try:
                gt_data = self.process_one_sample(sample_token)
                
                # Save to file
                output_path = os.path.join(ann_dir, f'{sample_token}.pkl')
                with open(output_path, 'wb') as f:
                    pickle.dump(gt_data, f)
                
                # Update statistics
                M = len(gt_data['gt_classes'])
                stats['total_instances'] += M
                
                if M == 0:
                    stats['empty_samples'] += 1
                else:
                    for cls_id in gt_data['gt_classes']:
                        stats['class_counts'][cls_id] += 1
            
            except Exception as e:
                print(f"\nError processing {sample_token}: {e}")
                continue
        
        # Save split file
        split_file = os.path.join(split_dir, f'{split}.txt')
        with open(split_file, 'w') as f:
            f.write('\n'.join(sample_tokens))
        
        # Save metadata
        metadata = {
            'version': '1.0',
            'nuscenes_version': self.version,
            'pc_range': self.pc_range,
            'patch_size': [
                self.pc_range[4] - self.pc_range[1],  # patch_h
                self.pc_range[3] - self.pc_range[0],  # patch_w
            ],
            'num_samples': len(sample_tokens),
            'class_mapping': self.map_api.CLASS_MAPPING,
            'class_names': self.map_api.CLASS_NAMES,
            'thresholds': {
                'min_arc_length': self.min_arc_length,
                'min_area': self.min_area,
            },
            'statistics': stats,
        }
        
        metadata_file = os.path.join(self.output_dir, f'metadata_{split}.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        if self.verbose:
            print(f"\n{split.upper()} Statistics:")
            print(f"  Total samples: {stats['total_samples']}")
            print(f"  Total instances: {stats['total_instances']}")
            print(f"  Avg instances/sample: {stats['total_instances']/max(stats['total_samples'],1):.2f}")
            print(f"  Empty samples: {stats['empty_samples']}")
            print(f"  Class distribution: {stats['class_counts']}")
            for i, name in enumerate(self.map_api.CLASS_NAMES):
                print(f"    {name}: {stats['class_counts'][i]}")
        
        return metadata
    
    def load(self, sample_token: str) -> Optional[Dict]:
        """
        Load GT for a sample
        
        Args:
            sample_token: Sample token
            
        Returns:
            GT dict or None if not found
        """
        gt_file = os.path.join(self.output_dir, 'annotations', f'{sample_token}.pkl')
        
        if not os.path.exists(gt_file):
            return None
        
        with open(gt_file, 'rb') as f:
            return pickle.load(f)


if __name__ == "__main__":
    import sys
    
    dataroot = '/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini'
    output_dir = '/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini/gt_cache'
    
    print("Testing GT Cache...")
    
    try:
        cache = GTCache(
            dataroot=dataroot,
            version='v1.0-mini',
            pc_range=[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
            output_dir=output_dir,
            verbose=True
        )
        
        # Test single sample
        first_scene = cache.map_api.nusc.scene[0]
        sample_token = first_scene['first_sample_token']
        
        print(f"\nProcessing single sample: {sample_token[:16]}...")
        gt_data = cache.process_one_sample(sample_token)
        
        print(f"\nGT data:")
        print(f"  gt_classes shape: {gt_data['gt_classes'].shape}")
        print(f"  gt_points shape: {gt_data['gt_points'].shape}")
        print(f"  gt_is_closed shape: {gt_data['gt_is_closed'].shape}")
        print(f"  gt_bbox shape: {gt_data['gt_bbox'].shape}")
        
        print("\n✓ GT Cache test passed!")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

