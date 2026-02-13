"""
Map Detection Dataset for nuScenes

This dataset loads:
1. 6-view camera images from nuScenes
2. GT map elements from pre-generated cache (Step 1)
3. Preprocesses images: Photometric augmentation (train) + Resize to 800×448 + ImageNet normalization
4. Normalizes GT coordinates to [-1, 1]
5. Inserts 1 <image> token in prompt (will be replaced by 768 scene tokens from Q-Former)

Author: Auto-generated for LLaVA Map Detection
Date: 2025-12
"""

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
from pyquaternion import Quaternion

from llava.model.map_structures import MapGroundTruth
from llava.model.map_config import DEFAULT_MAP_CONFIG
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX


class MapDetectionDataset(Dataset):
    """
    Dataset for map detection task.
    
    Returns for each sample:
        - images: (6, 3, H, W) - 6 camera views, CLIP preprocessed
        - gt: MapGroundTruth - ground truth map elements
        - sample_token: str - nuScenes sample token
        - img_metas: dict - metadata (camera params, ego pose, etc.)
    """
    
    CAMERA_NAMES = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT', 
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    
    def __init__(
        self,
        dataroot: str,
        version: str = 'v1.0-mini',
        split: str = 'train',
        gt_cache_path: Optional[str] = None,
        image_processor: Optional[Any] = None,
        map_config: Optional[object] = None,
        tokenizer: Optional[Any] = None,
        prompt: Optional[str] = None,
        use_augmentation: bool = True,
        subset_scenes_file: Optional[str] = None,  # 子集场景列表文件路径
    ):
        """
        Args:
            dataroot: Path to nuScenes dataset root
            version: nuScenes version ('v1.0-mini', 'v1.0-trainval')
            split: 'train' or 'val'
            gt_cache_path: Path to GT cache file (from Step 1)
            image_processor: CLIP image processor (optional, for compatibility)
            map_config: Map detection config (if None, uses DEFAULT_MAP_CONFIG)
            tokenizer: LLM tokenizer (if None, will create default)
            prompt: Text prompt for detection task (if None, uses default)
            use_augmentation: Whether to apply photometric augmentation (only for train)
        """
        super().__init__()
        
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.config = map_config or DEFAULT_MAP_CONFIG
        self.subset_scenes_file = subset_scenes_file
        
        # Photometric augmentation only for training
        self.use_augmentation = use_augmentation and (split == 'train')
        if self.use_augmentation:
            print("Photometric augmentation ENABLED for training")
        
        # Set default prompt if not provided
        # Detailed prompt to help LLM understand the autonomous driving map detection task
        if prompt is None:
            prompt = (
                "You are an autonomous driving perception system. "
                "Given 6 surround-view camera images (front, front-left, front-right, back, back-left, back-right), "
                "detect HD map elements in bird's-eye-view (BEV) coordinates.\n"
                "Map element classes:\n"
                "- class 0: divider (lane dividing lines)\n"
                "- class 1: ped_crossing (pedestrian crossing areas)\n"
                "- class 2: boundary (road boundary lines)\n"
                "Output tensor format:\n"
                "- Classification: [B, N, 3] - N=50 instances, 3 class logits per instance\n"
                "- Points: [B, N, 20, 2] - N=50 instances, 20 points per instance, (x, y) coordinates in [-1, 1]\n"
                "Coordinate system: x-axis lateral (left-right), y-axis longitudinal (front-back), ego vehicle at origin."
            )
        
        # Load nuScenes
        from nuscenes.nuscenes import NuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        # Get sample tokens for split
        self.sample_tokens = self._get_split_samples()
        print(f"Loaded {len(self.sample_tokens)} samples for {split} split")
        
        # Setup GT cache path
        if gt_cache_path is None:
            # Default: look for cache directory in dataroot
            gt_cache_path = os.path.join(dataroot, f'gt_cache_{version}_{split}.pkl')
        
        # Check if cache is directory or file
        if os.path.isdir(gt_cache_path):
            # Directory-based cache (from Step 1)
            self.gt_cache_dir = os.path.join(gt_cache_path, 'annotations')
            self.gt_cache = None
            print(f"Using GT cache directory: {self.gt_cache_dir}")
        else:
            # Single file cache
            print(f"Loading GT cache from {gt_cache_path}...")
            with open(gt_cache_path, 'rb') as f:
                self.gt_cache = pickle.load(f)
            self.gt_cache_dir = None
            print(f"Loaded GT for {len(self.gt_cache)} samples")
        
        # Image preprocessing config (similar to MapTR)
        # ResNet50 is fully convolutional - accepts ANY input size
        # Strategy: Resize to target size, H divisible by 32
        # Original: 1600×900 → 800×448 (直接 resize)
        # scale_x = 0.5, scale_y ≈ 0.498
        self.target_img_size = (800, 448)  # (W, H) - H divisible by 32
        
        # ImageNet normalization parameters (RGB order, for ResNet50 pretrained on ImageNet)
        # 与 MapTR 完全一致：
        #   MapTR: mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
        #   本代码: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] (0-1 范围)
        #   验证: 0.485*255=123.675, 0.456*255=116.28, 0.406*255=103.53 ✓
        # 颜色格式：
        #   - MapTR 用 OpenCV 读取 BGR，然后 to_rgb=True 转换为 RGB
        #   - 本代码用 PIL 直接读取 RGB
        #   - 两者最终都是 RGB 格式，与 ResNet50 预训练权重一致
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # RGB order
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # RGB order
        
        # Store external image_processor if provided (for compatibility)
        self.image_processor = image_processor
        
        print(f"Image preprocessing: Resize to {self.target_img_size[0]}x{self.target_img_size[1]} (keep ratio)")
        
        # Setup tokenizer
        if tokenizer is None:
            # Use default tokenizer (will be replaced by LLaVA's in training)
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    "lmsys/vicuna-7b-v1.5",
                    use_fast=False
                )
            except Exception:
                print("Warning: tokenizer not available, using placeholder")
                self.tokenizer = None
        else:
            self.tokenizer = tokenizer
        
        # Setup prompt with 1 image token (Q-Former will produce 768 scene tokens from 6 cameras)
        # The single <image> token will be replaced by 768 scene tokens from Q-Former
        self.prompt_with_images = f"{DEFAULT_IMAGE_TOKEN}\n{prompt}"
        
        # Tokenize prompt with image token replacement
        if self.tokenizer is not None:
            self.prompt_ids = self._tokenize_with_image_token(self.prompt_with_images)
        else:
            # Placeholder: assume prompt length ~20 tokens + 1 image token
            self.prompt_ids = torch.zeros(21, dtype=torch.long)
        
        # Assign collate_fn as instance attribute for DataLoader
        self.collate_fn = collate_fn
    
    def _tokenize_with_image_token(self, prompt: str) -> torch.Tensor:
        """
        Tokenize prompt and replace <image> with IMAGE_TOKEN_INDEX (-200).
        
        Args:
            prompt: Text with <image> tokens
            
        Returns:
            Token IDs with <image> replaced by IMAGE_TOKEN_INDEX (-200)
        """
        # Split by <image> token
        prompt_chunks = [
            self.tokenizer(chunk, add_special_tokens=(i == 0)).input_ids 
            for i, chunk in enumerate(prompt.split(DEFAULT_IMAGE_TOKEN))
        ]
        
        # Insert IMAGE_TOKEN_INDEX between chunks
        input_ids = []
        for i, chunk in enumerate(prompt_chunks):
            input_ids.extend(chunk)
            if i < len(prompt_chunks) - 1:  # Not the last chunk
                input_ids.append(IMAGE_TOKEN_INDEX)
        
        return torch.tensor(input_ids, dtype=torch.long)
    
    def _get_split_samples(self) -> List[str]:
        """
        Get sample tokens for train/val split.
        Uses official nuScenes splits for reproducibility.
        """
        # For mini dataset, use simple scene-based split
        if 'mini' in self.version:
            # Mini has 10 scenes, use first 8 for train, last 2 for val
            all_scenes = self.nusc.scene
            if self.split == 'train':
                scenes = all_scenes[:8]
            else:
                scenes = all_scenes[8:]
            
            # Collect all sample tokens from selected scenes
            sample_tokens = []
            for scene in scenes:
                sample_token = scene['first_sample_token']
                while sample_token:
                    sample_tokens.append(sample_token)
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
            return sample_tokens
        
        # For full dataset, use official nuScenes train/val split
        try:
            from nuscenes.utils.splits import create_splits_scenes
            splits = create_splits_scenes()
            
            if self.split == 'train':
                scene_names = set(splits['train'])
            else:
                scene_names = set(splits['val'])
            
            # 【子集过滤】如果指定了子集文件，只保留文件中的场景
            if self.subset_scenes_file is not None and self.split == 'train':
                with open(self.subset_scenes_file, 'r') as f:
                    subset_names = set(line.strip() for line in f if line.strip())
                scene_names = scene_names & subset_names  # 取交集
                print(f"Using subset: {len(scene_names)} scenes (from {self.subset_scenes_file})")
            
            # Collect all sample tokens from official split scenes
            sample_tokens = []
            for scene in self.nusc.scene:
                if scene['name'] in scene_names:
                    sample_token = scene['first_sample_token']
                    while sample_token:
                        sample_tokens.append(sample_token)
                        sample = self.nusc.get('sample', sample_token)
                        sample_token = sample['next']
            
            print(f"Using official nuScenes {self.split} split: {len(scene_names)} scenes")
            return sample_tokens
            
        except ImportError:
            print("Warning: nuscenes.utils.splits not available, using fallback split")
            # Fallback: scene-based 80/20 split
            all_scenes = self.nusc.scene
            split_idx = int(len(all_scenes) * 0.8)
            if self.split == 'train':
                scenes = all_scenes[:split_idx]
            else:
                scenes = all_scenes[split_idx:]
            
            sample_tokens = []
            for scene in scenes:
                sample_token = scene['first_sample_token']
                while sample_token:
                    sample_tokens.append(sample_token)
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
            return sample_tokens
    
    def __len__(self) -> int:
        return len(self.sample_tokens)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Returns:
            dict with keys:
                - images: torch.Tensor (6, 3, H, W)
                - gt: MapGroundTruth
                - sample_token: str
                - img_metas: dict
        """
        sample_token = self.sample_tokens[idx]
        
        # Load images
        images = self._load_images(sample_token)
        
        # Load GT from cache
        if self.gt_cache_dir is not None:
            # Load from directory
            gt_file = os.path.join(self.gt_cache_dir, f'{sample_token}.pkl')
            with open(gt_file, 'rb') as f:
                gt_dict = pickle.load(f)
        else:
            # Load from dict
            gt_dict = self.gt_cache[sample_token]
        
        gt = self._process_gt(gt_dict)
        
        # Apply BEV horizontal flip augmentation (training only)
        # Similar to MapTR's RandomFlip3D
        flipped = False
        if self.use_augmentation:
            images, gt.points, gt.bbox, flipped = self._apply_random_flip(
                images, gt.points, gt.bbox
            )
        
        # Get metadata (内参会根据图像缩放和翻转进行调整)
        img_metas = self._get_img_metas(sample_token, flipped=flipped)
        
        return {
            'images': images,                    # (6, 3, H, W)
            'text_ids': self.prompt_ids.clone(), # (L,)
            'gt': gt,                            # MapGroundTruth object
            'sample_token': sample_token,
            'img_metas': img_metas,
        }
    
    def _apply_photometric_augmentation(self, img: Image.Image) -> Image.Image:
        """
        Apply photometric augmentation to a single image.
        与 MapTR 的 PhotoMetricDistortion3D 参数完全一致。
        
        MapTR 参数:
        - brightness_delta: 32 → factor ∈ [0.75, 1.25]
        - contrast_range: (0.5, 1.5)
        - saturation_range: (0.5, 1.5)
        - hue_delta: 18 (通过 HSV 空间近似实现)
        
        Args:
            img: PIL Image in RGB format
            
        Returns:
            Augmented PIL Image
        """
        # Brightness: MapTR brightness_delta=32
        # delta=32 对应像素值变化 ±32/128 ≈ ±0.25
        # 50% 概率应用
        if random.random() < 0.5:
            # factor ∈ [0.75, 1.25] 对应 brightness_delta=32
            factor = random.uniform(0.75, 1.25)
            img = ImageEnhance.Brightness(img).enhance(factor)
        
        # Contrast: MapTR contrast_range=(0.5, 1.5)
        # 50% 概率应用
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            img = ImageEnhance.Contrast(img).enhance(factor)
        
        # Saturation: MapTR saturation_range=(0.5, 1.5)
        # 50% 概率应用
        if random.random() < 0.5:
            factor = random.uniform(0.5, 1.5)
            img = ImageEnhance.Color(img).enhance(factor)
        
        # Hue: MapTR hue_delta=18 (度)
        # PIL 没有直接的色调调整，使用 HSV 空间实现
        # 50% 概率应用
        if random.random() < 0.5:
            img = self._adjust_hue(img, delta=18)
        
        return img
    
    def _adjust_hue(self, img: Image.Image, delta: int = 18) -> Image.Image:
        """
        在 HSV 空间调整色调，与 MapTR 的 hue_delta 参数一致。
        
        使用 PIL 的 HSV 模式进行高效转换。
        
        Args:
            img: PIL Image in RGB format
            delta: 色调调整范围 (度)，MapTR 默认为 18
            
        Returns:
            色调调整后的 PIL Image
        """
        # 转为 HSV 模式
        hsv_img = img.convert('HSV')
        h, s, v = hsv_img.split()
        
        # PIL HSV: H 范围是 0-255 (对应 0-360 度)
        # delta 度 → delta * 255 / 360 ≈ delta * 0.708
        hue_shift = int(random.uniform(-delta, delta) * 255 / 360)
        
        # 调整 H 通道
        h_array = np.array(h, dtype=np.int16)
        h_array = (h_array + hue_shift) % 256
        h = Image.fromarray(h_array.astype(np.uint8), mode='L')
        
        # 合并并转回 RGB
        hsv_img = Image.merge('HSV', (h, s, v))
        return hsv_img.convert('RGB')
    
    def _apply_random_flip(
        self, 
        images: torch.Tensor,
        gt_points: torch.Tensor,
        gt_bbox: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """
        随机 BEV 水平翻转（50% 概率）
        
        与 MapTR 的 RandomFlip3D 功能相同：
        - 同时翻转图像和 GT 坐标
        - 利用道路的对称性，数据量相当于翻倍
        
        操作：
        1. 图像：水平翻转 + 左右相机互换
        2. GT 点坐标：x 取负
        3. GT 边界框：cx 取负
        
        注意：相机内参的 cx 也需要调整（在 _get_img_metas 中处理）
        
        Args:
            images: [6, 3, H, W] 6个相机图像
            gt_points: [N, 20, 2] GT 点坐标（归一化后，[-1, 1]）
            gt_bbox: [N, 4] GT 边界框 (cx, cy, w, h)
        
        Returns:
            翻转后的 (images, gt_points, gt_bbox, flipped)
            flipped: bool, 是否执行了翻转
        """
        flipped = False
        if random.random() < 0.5:
            flipped = True
            
            # 1. 翻转图像
            # 实际相机顺序 (CAMERA_NAMES): 
            #   FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
            #   0      1            2           3     4          5
            # 翻转后左右互换:
            #   FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_RIGHT, BACK_LEFT
            # 即: 原索引 1 和 2 互换，原索引 4 和 5 互换
            flip_order = [0, 2, 1, 3, 5, 4]
            flipped_images = []
            for i in flip_order:
                # 水平翻转每张图像（沿宽度维度，即 dim=2）
                flipped_images.append(torch.flip(images[i], dims=[2]))
            images = torch.stack(flipped_images)
            
            # 2. 翻转 GT 点坐标（x 取负）
            # 坐标是归一化的 [-1, 1]，翻转只需取负
            gt_points = gt_points.clone()
            gt_points[..., 0] = -gt_points[..., 0]
            
            # 3. 翻转 GT 边界框（cx 取负）
            if gt_bbox is not None and len(gt_bbox) > 0:
                gt_bbox = gt_bbox.clone()
                gt_bbox[:, 0] = -gt_bbox[:, 0]
        
        return images, gt_points, gt_bbox, flipped
    
    def _load_images(self, sample_token: str) -> torch.Tensor:
        """
        Load and preprocess 6 camera images.
        
        Pipeline (similar to MapTR):
        1. Load 6 camera images from nuScenes (1600×900)
        2. Apply photometric augmentation (training only)
        3. Resize to target size (800×448)
        4. Normalize with ImageNet mean/std (for ResNet50)
        
        Note: ResNet50 is fully convolutional, accepts any input size.
        
        Returns:
            torch.Tensor (6, 3, H, W) - preprocessed images, H=448, W=800
        """
        sample = self.nusc.get('sample', sample_token)
        
        images = []
        for cam_name in self.CAMERA_NAMES:
            # Get camera sample data
            cam_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            
            # Load image (PIL 读取为 RGB 格式，与 MapTR 的 to_rgb=True 效果一致)
            img_path = os.path.join(self.dataroot, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')  # RGB format
            
            # Apply photometric augmentation (training only)
            if self.use_augmentation:
                img = self._apply_photometric_augmentation(img)
            
            # Preprocess: resize + normalize
            img_tensor = self._preprocess_image(img)
            images.append(img_tensor)
        
        # Stack to (6, 3, H, W)
        images = torch.stack(images, dim=0)
        return images
    
    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """
        Preprocess a single image: resize + normalize (similar to MapTR).
        
        Strategy:
        1. Resize to target size (800×448), approximately 0.5x of original
        2. Normalize with ImageNet mean/std (ResNet50 pretrained on ImageNet)
        
        This ensures:
        - H=448 divisible by 32 (ResNet stride requirement)
        - No information loss (no cropping)
        - No distortion (proportional scaling)
        - Compatible with ResNet50 (fully convolutional, any size OK)
        
        Example for nuScenes 1600×900:
        - target = (800, 448)
        - scale_x = 0.5, scale_y ≈ 0.498
        
        Args:
            img: PIL Image (1600×900 for nuScenes)
            
        Returns:
            torch.Tensor (3, H, W) - preprocessed image, H=448, W=800
        """
        target_w, target_h = self.target_img_size  # (800, 448)
        
        # Step 1: Resize to target size (keeps aspect ratio since 800/450 ≈ 1600/900)
        img_resized = img.resize((target_w, target_h), Image.BILINEAR)
        
        # Step 2: Convert to numpy array and normalize to [0, 1]
        # 输入是 RGB 格式 (PIL)，与 MapTR 的 to_rgb=True 后效果一致
        img_array = np.array(img_resized, dtype=np.float32) / 255.0  # (H, W, 3), RGB
        
        # Step 3: Normalize with ImageNet mean/std (RGB order)
        # 与 MapTR 的 img_norm_cfg 完全一致
        normalized = (img_array - self.img_mean) / self.img_std  # RGB normalized
        
        # Step 4: Convert to (3, H, W) tensor
        img_tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).float()
        
        return img_tensor
    
    def _process_gt(self, gt_dict: Dict) -> MapGroundTruth:
        """
        Process GT from cache, normalize coordinates.
        
        Args:
            gt_dict: dict with keys 'gt_classes', 'gt_points', 'gt_is_closed', 'gt_bbox'
        
        Returns:
            MapGroundTruth object with normalized coordinates
        """
        # Extract arrays
        gt_classes = np.array(gt_dict['gt_classes'], dtype=np.int64)
        gt_points = np.array(gt_dict['gt_points'], dtype=np.float32)  # (N, 20, 2)
        gt_is_closed = np.array(gt_dict['gt_is_closed'], dtype=bool)
        gt_bbox = np.array(gt_dict['gt_bbox'], dtype=np.float32)  # (N, 4)
        
        # ========== GT 点起始位置随机化（仅训练时，针对闭合多边形） ==========
        # 目的：防止模型依赖固定的起始点位置
        # 原理：对于闭合多边形（如人行横道、道路边界），20个点形成环形
        #       随机选择起始点，让模型学习形状而非顺序
        if self.use_augmentation and len(gt_points) > 0:
            for i, is_closed in enumerate(gt_is_closed):
                if is_closed:
                    # 随机选择起始点 (0-19)
                    start_idx = random.randint(0, 19)
                    # np.roll 循环移动数组
                    gt_points[i] = np.roll(gt_points[i], -start_idx, axis=0)
        
        # Normalize coordinates from ego frame to [-1, 1]
        gt_points_norm = self._normalize_coords(gt_points)
        gt_bbox_norm = self._normalize_bbox(gt_bbox)
        
        # Create MapGroundTruth
        # Note: MapGroundTruth expects class_labels, not classes
        # We store is_closed separately in the returned gt object
        gt = MapGroundTruth(
            class_labels=torch.from_numpy(gt_classes),
            points=torch.from_numpy(gt_points_norm),
            bbox=torch.from_numpy(gt_bbox_norm),
        )
        # Attach is_closed as additional attribute
        gt.is_closed = torch.from_numpy(gt_is_closed)
        
        return gt
    
    def _normalize_coords(self, coords: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates from ego frame to [-1, 1].
        
        与 MapTR 一致：对超出 BEV 范围的点进行 clip。
        
        Args:
            coords: (N, 20, 2) in ego frame (单位：米)
        
        Returns:
            (N, 20, 2) normalized to [-1, 1]，超出范围的点被 clip
        """
        # 空数组检查（某些样本可能没有 GT）
        if len(coords) == 0:
            return coords.copy()
        
        pc_range = self.config.PC_RANGE  # [-15, -30, -2, 15, 30, 2]
        
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        
        coords_norm = coords.copy()
        # Normalize to [0, 1] then to [-1, 1]
        coords_norm[..., 0] = (coords[..., 0] - x_min) / (x_max - x_min) * 2 - 1
        coords_norm[..., 1] = (coords[..., 1] - y_min) / (y_max - y_min) * 2 - 1
        
        # 【重要】与 MapTR 一致：clip 到 [-1, 1] 范围
        # 防止超出 BEV 范围的点导致 Loss 计算异常
        coords_norm = np.clip(coords_norm, -1.0, 1.0)
        
        return coords_norm
    
    def _normalize_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """
        Normalize bounding box from ego frame to [-1, 1].
        
        Args:
            bbox: (N, 4) [x_center, y_center, width, height] in ego frame
                  NOTE: This format is from compute_aabb() in geometry.py
        
        Returns:
            (N, 4) normalized: [cx_norm, cy_norm, w_norm, h_norm]
                  - cx_norm, cy_norm: center in [-1, 1]
                  - w_norm, h_norm: relative size (0 to 2, where 2 = full range)
        """
        # 空数组检查（某些样本可能没有 GT）
        if len(bbox) == 0:
            return bbox.copy()
        
        pc_range = self.config.PC_RANGE
        
        x_min, y_min = pc_range[0], pc_range[1]
        x_max, y_max = pc_range[3], pc_range[4]
        x_range = x_max - x_min  # 30m
        y_range = y_max - y_min  # 60m
        
        bbox_norm = bbox.copy()
        
        # Normalize center coordinates to [-1, 1]
        bbox_norm[:, 0] = (bbox[:, 0] - x_min) / x_range * 2 - 1  # cx
        bbox_norm[:, 1] = (bbox[:, 1] - y_min) / y_range * 2 - 1  # cy
        
        # Normalize width/height to relative size (0 to 2)
        # w_norm = w / range * 2, so w_norm=2 means bbox spans full range
        bbox_norm[:, 2] = bbox[:, 2] / x_range * 2  # w
        bbox_norm[:, 3] = bbox[:, 3] / y_range * 2  # h
        
        # Clip center to [-1, 1] (instances may extend beyond ROI)
        bbox_norm[:, 0] = np.clip(bbox_norm[:, 0], -1.0, 1.0)
        bbox_norm[:, 1] = np.clip(bbox_norm[:, 1], -1.0, 1.0)
        # Width/height can exceed 2 if bbox extends beyond ROI, clip to reasonable range
        bbox_norm[:, 2] = np.clip(bbox_norm[:, 2], 0.0, 2.0)
        bbox_norm[:, 3] = np.clip(bbox_norm[:, 3], 0.0, 2.0)
        
        return bbox_norm
    
    def _get_img_metas(self, sample_token: str, flipped: bool = False) -> Dict:
        """
        Get image metadata (camera params, ego pose, etc.).
        
        重要：相机内参会根据图像缩放和翻转进行调整！
        
        nuScenes 原图尺寸：1600×900
        缩放后尺寸：800×448 (self.target_img_size)
        
        内参缩放公式：
        - fx_new = fx * scale_x
        - fy_new = fy * scale_y
        - cx_new = cx * scale_x
        - cy_new = cy * scale_y
        
        翻转时的内参调整：
        - cx_new = image_width - cx (主点 x 坐标翻转)
        - 相机顺序也要相应调整
        
        Args:
            sample_token: nuScenes sample token
            flipped: 是否执行了水平翻转
        
        Returns:
            dict with useful metadata (内参已缩放和调整)
        """
        sample = self.nusc.get('sample', sample_token)
        
        # Get ego pose from LIDAR_TOP
        lidar_token = sample['data']['LIDAR_TOP']
        lidar_data = self.nusc.get('sample_data', lidar_token)
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # ========== 计算内参缩放因子 ==========
        # nuScenes 原图尺寸：1600×900
        orig_w, orig_h = 1600, 900
        # 目标尺寸
        target_w, target_h = self.target_img_size  # (800, 448)
        # 缩放因子
        scale_x = target_w / orig_w  # 800 / 1600 = 0.5
        scale_y = target_h / orig_h  # 448 / 900 ≈ 0.4978
        
        # Get calibration for each camera
        cam_intrinsics = []
        cam_extrinsics = []
        
        for cam_name in self.CAMERA_NAMES:
            cam_token = sample['data'][cam_name]
            cam_data = self.nusc.get('sample_data', cam_token)
            cam_calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # ========== 内参缩放 ==========
            # 原始内参矩阵 (3x3)
            # K = | fx  0  cx |
            #     | 0  fy  cy |
            #     | 0   0   1 |
            intrinsic_orig = np.array(cam_calib['camera_intrinsic'], dtype=np.float32)
            
            # 缩放内参
            intrinsic_scaled = intrinsic_orig.copy()
            intrinsic_scaled[0, 0] *= scale_x  # fx
            intrinsic_scaled[1, 1] *= scale_y  # fy
            intrinsic_scaled[0, 2] *= scale_x  # cx
            intrinsic_scaled[1, 2] *= scale_y  # cy
            
            # ========== 翻转时调整内参 ==========
            if flipped:
                # 翻转时，cx 需要调整为 image_width - cx
                # 因为图像水平翻转后，主点的 x 坐标也要翻转
                intrinsic_scaled[0, 2] = target_w - intrinsic_scaled[0, 2]
            
            cam_intrinsics.append(intrinsic_scaled.tolist())
            
            # ========== 外参处理（翻转时需要调整！） ==========
            # 外参描述：相机相对于车辆的位置和朝向
            # Q-Former 使用外参进行 3D 位置编码，翻转时必须正确调整
            translation = np.array(cam_calib['translation'], dtype=np.float32)
            rotation_quat = Quaternion(cam_calib['rotation'])
            
            if flipped:
                # 翻转时调整外参
                # 1. 平移向量：x 取负（相机相对于车辆的位置镜像）
                #    例如：CAM_FRONT_RIGHT 原本 tx=+1.5m，翻转后变成 tx=-1.5m
                translation[0] = -translation[0]
                
                # 2. 旋转：绕 x 轴镜像
                #    四元数 q = (w, x, y, z) 镜像后 q' = (w, x, -y, -z)
                #    这保持四元数的单位性质，对应绕 y-z 平面的反射
                w, x, y, z = rotation_quat.elements
                rotation_quat = Quaternion(w, x, -y, -z)
            
            cam_extrinsics.append({
                'translation': translation.tolist(),
                'rotation': rotation_quat.elements.tolist(),
            })
        
        # ========== 翻转时调整相机顺序 ==========
        if flipped:
            # 实际相机顺序 (CAMERA_NAMES): 
            #   FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT
            #   0      1            2           3     4          5
            # 翻转后左右互换:
            #   FRONT, FRONT_LEFT, FRONT_RIGHT, BACK, BACK_RIGHT, BACK_LEFT
            # 即: 原索引 1 和 2 互换，原索引 4 和 5 互换
            flip_order = [0, 2, 1, 3, 5, 4]
            cam_intrinsics = [cam_intrinsics[i] for i in flip_order]
            cam_extrinsics = [cam_extrinsics[i] for i in flip_order]
        
        return {
            'sample_token': sample_token,
            'scene_token': sample['scene_token'],
            'timestamp': sample['timestamp'],
            'ego_pose': {
                'translation': ego_pose['translation'],
                'rotation': ego_pose['rotation'],
            },
            'cam_intrinsics': cam_intrinsics,
            'cam_extrinsics': cam_extrinsics,
            'pc_range': self.config.PC_RANGE,
            # 添加图像尺寸信息，方便其他模块使用
            'img_shape': (target_h, target_w),  # (H, W)
            'orig_img_shape': (orig_h, orig_w),  # 原图尺寸
            'scale_factor': (scale_x, scale_y),  # 缩放因子
            'flipped': flipped,  # 是否翻转
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Collate function for DataLoader.
    
    Handles variable number of GT instances by padding.
    
    Args:
        batch: List of dicts from __getitem__
    
    Returns:
        Batched dict with:
            - images: (B, 6, 3, H, W)
            - text_ids: (B, L)
            - cam_intrinsics: (B, 6, 3, 3)
            - cam_extrinsics: (B, 6, 4, 4)
            - gt_classes: (B, max_N)
            - gt_points: (B, max_N, 20, 2)
            - gt_is_closed: (B, max_N)
            - gt_bbox: (B, max_N, 4)
            - gt_mask: (B, max_N) - bool mask for valid GTs
            - sample_tokens: List[str]
            - img_metas: List[dict]
    """
    batch_size = len(batch)
    
    # Stack images (all same size)
    images = torch.stack([item['images'] for item in batch], dim=0)
    
    # Stack text_ids (all same length from same prompt)
    text_ids = torch.stack([item['text_ids'] for item in batch], dim=0)
    
    # Stack camera parameters
    cam_intrinsics = []
    cam_extrinsics = []
    ego_poses = []
    
    for item in batch:
        # Convert intrinsics to tensors
        intrinsics = torch.tensor(item['img_metas']['cam_intrinsics'], dtype=torch.float32)  # (6, 3, 3)
        cam_intrinsics.append(intrinsics)
        
        # Extrinsics: build 4x4 matrix from rotation + translation
        extrinsics_list = []
        for ext in item['img_metas']['cam_extrinsics']:
            # Build 4x4 transformation matrix
            mat = torch.eye(4, dtype=torch.float32)
            
            # Rotation: quaternion to rotation matrix
            quat = Quaternion(ext['rotation'])
            mat[:3, :3] = torch.from_numpy(quat.rotation_matrix.astype(np.float32))
            
            # Translation
            mat[:3, 3] = torch.tensor(ext['translation'], dtype=torch.float32)
            extrinsics_list.append(mat)
        extrinsics = torch.stack(extrinsics_list, dim=0)  # (6, 4, 4)
        cam_extrinsics.append(extrinsics)
        
        # Ego pose: build 4x4 matrix
        ego_dict = item['img_metas']['ego_pose']
        ego_mat = torch.eye(4, dtype=torch.float32)
        ego_quat = Quaternion(ego_dict['rotation'])
        ego_mat[:3, :3] = torch.from_numpy(ego_quat.rotation_matrix.astype(np.float32))
        ego_mat[:3, 3] = torch.tensor(ego_dict['translation'], dtype=torch.float32)
        ego_poses.append(ego_mat)
    
    cam_intrinsics = torch.stack(cam_intrinsics, dim=0)  # (B, 6, 3, 3)
    cam_extrinsics = torch.stack(cam_extrinsics, dim=0)  # (B, 6, 4, 4)
    ego_poses = torch.stack(ego_poses, dim=0)  # (B, 4, 4)
    
    # Find max number of GTs in batch
    max_num_gts = max(len(item['gt'].class_labels) for item in batch)
    
    if max_num_gts == 0:
        # No GTs in batch
        return {
            'images': images,
            'text_ids': text_ids,
            'cam_intrinsics': cam_intrinsics,
            'cam_extrinsics': cam_extrinsics,
            'ego_pose': ego_poses,
            'gt_labels': torch.zeros(batch_size, 0, dtype=torch.long),
            'gt_points': torch.zeros(batch_size, 0, 20, 2, dtype=torch.float32),
            'gt_is_closed': torch.zeros(batch_size, 0, dtype=torch.bool),
            'gt_bbox': torch.zeros(batch_size, 0, 4, dtype=torch.float32),
            'gt_masks': torch.zeros(batch_size, 0, dtype=torch.bool),
            'sample_tokens': [item['sample_token'] for item in batch],
            'img_metas': [item['img_metas'] for item in batch],
        }
    
    # Initialize padded tensors
    gt_classes = torch.zeros(batch_size, max_num_gts, dtype=torch.long)
    gt_points = torch.zeros(batch_size, max_num_gts, 20, 2, dtype=torch.float32)
    gt_is_closed = torch.zeros(batch_size, max_num_gts, dtype=torch.bool)
    gt_bbox = torch.zeros(batch_size, max_num_gts, 4, dtype=torch.float32)
    gt_mask = torch.zeros(batch_size, max_num_gts, dtype=torch.bool)
    
    # Fill in actual GTs
    for i, item in enumerate(batch):
        gt = item['gt']
        num_gts = len(gt.class_labels)
        
        if num_gts > 0:
            gt_classes[i, :num_gts] = gt.class_labels
            gt_points[i, :num_gts] = gt.points
            gt_is_closed[i, :num_gts] = gt.is_closed
            gt_bbox[i, :num_gts] = gt.bbox
            gt_mask[i, :num_gts] = True
    
    return {
        'images': images,                # (B, 6, 3, H, W)
        'text_ids': text_ids,            # (B, L)
        'cam_intrinsics': cam_intrinsics,  # (B, 6, 3, 3)
        'cam_extrinsics': cam_extrinsics,  # (B, 6, 4, 4)
        'ego_pose': ego_poses,           # (B, 4, 4)
        'gt_labels': gt_classes,         # (B, max_N) - renamed for consistency with train script
        'gt_points': gt_points,          # (B, max_N, 20, 2)
        'gt_is_closed': gt_is_closed,    # (B, max_N)
        'gt_bbox': gt_bbox,              # (B, max_N, 4)
        'gt_masks': gt_mask,             # (B, max_N) - renamed for consistency with train script
        'sample_tokens': [item['sample_token'] for item in batch],
        'img_metas': [item['img_metas'] for item in batch],
    }


def create_dataloader(
    dataroot: str,
    version: str = 'v1.0-mini',
    split: str = 'train',
    gt_cache_path: Optional[str] = None,
    image_processor: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    prompt: Optional[str] = None,
    batch_size: int = 4,
    num_workers: int = 4,
    shuffle: bool = True,
    use_augmentation: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for map detection.
    
    Args:
        dataroot: Path to nuScenes dataset
        version: nuScenes version
        split: 'train' or 'val'
        gt_cache_path: Path to GT cache
        image_processor: CLIP image processor (optional)
        tokenizer: LLM tokenizer
        prompt: Text prompt for detection (if None, uses default)
        batch_size: Batch size
        num_workers: Number of workers
        shuffle: Whether to shuffle
        use_augmentation: Whether to apply photometric augmentation (auto-disabled for val)
    
    Returns:
        DataLoader
    """
    dataset = MapDetectionDataset(
        dataroot=dataroot,
        version=version,
        split=split,
        gt_cache_path=gt_cache_path,
        image_processor=image_processor,
        tokenizer=tokenizer,
        prompt=prompt,
        use_augmentation=use_augmentation,
    )
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    return dataloader
