"""
Q-Former Verification - 验证 Q-Former 768 queries 的场景表示能力

============================================
验证方法：Linear Probing（业界标准）
============================================
核心思想：用最简单的线性层验证特征质量
- 如果简单的线性层就能完成任务，说明 Q-Former 提取了足够的场景信息
- 不引入复杂模块，结果 100% 反映 Q-Former 的能力

============================================
任务设计：场景级别目标数量预测
============================================
输入：6 张图像
输出：场景中各类目标的数量 [B, 13]

架构：
6 张图 → Q-Former → 768 tokens [B, 768, 4096]
              ↓
      Global Average Pooling
              ↓
      scene_feature [B, 4096]
              ↓
      Linear Layer（唯一可学习参数）
              ↓
      各类目标数量 [B, 13]
      (car: 5, pedestrian: 2, divider: 1, ...)

============================================
验证的问题
============================================
Q1: 768 tokens 是否包含"场景里有什么"的信息？
    → 如果数量预测准确，答案是 YES

Q2: Q-Former 能否区分不同类别？
    → 如果各类数量都准确，答案是 YES

============================================
成功标准
============================================
- 数量 MAE < 2: 平均每类的数量误差小于 2 个
- 存在性准确率 > 80%: 是否存在某类目标的判断准确率

Author: Auto-generated
Date: 2025-02
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle
from typing import Dict, List, Optional, Tuple

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.qformer import build_qformer


# ============================================
# 配置
# ============================================
NUM_SCENE_QUERIES = 768  # 与主训练架构一致
MAX_INSTANCES = 50       # 每个场景最多预测的实例数（场景中目标更多）

# 所有场景类别（10 类 3D 目标 + 3 类地图元素 = 13 类）
OBJECT_CATEGORIES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',  # 车辆
    'pedestrian', 'motorcycle', 'bicycle',                      # 行人和骑行者
    'barrier', 'traffic_cone',                                  # 障碍物
]
MAP_CATEGORIES = ['divider', 'ped_crossing', 'boundary']        # 地图元素

ALL_CATEGORIES = OBJECT_CATEGORIES + MAP_CATEGORIES  # 共 13 类
NUM_CLASSES = len(ALL_CATEGORIES)  # 13

# nuScenes 类别映射
NUSCENES_CATEGORY_MAP = {
    'vehicle.car': 'car',
    'vehicle.truck': 'truck',
    'vehicle.bus.bendy': 'bus',
    'vehicle.bus.rigid': 'bus',
    'vehicle.trailer': 'trailer',
    'vehicle.construction': 'construction_vehicle',
    'human.pedestrian.adult': 'pedestrian',
    'human.pedestrian.child': 'pedestrian',
    'human.pedestrian.construction_worker': 'pedestrian',
    'human.pedestrian.police_officer': 'pedestrian',
    'vehicle.motorcycle': 'motorcycle',
    'vehicle.bicycle': 'bicycle',
    'movable_object.barrier': 'barrier',
    'movable_object.trafficcone': 'traffic_cone',
}


class SceneCountingHead(nn.Module):
    """
    场景级别预测头 - Linear Probing 验证方法（增强版）
    
    设计理念（参考业界标准 Linear Probing）：
    - 用最简单的结构验证特征质量
    - 只用线性层，结果 100% 反映 Q-Former 的能力
    
    任务：
    1. 预测场景中各类目标的数量 [B, 13]（原有）
    2. 预测各类目标的中心位置均值 [B, 13, 2]（新增）
    3. 预测各类目标的位置分散度 [B, 13, 2]（新增）
    
    输入：768 scene tokens [B, 768, 4096]
    """
    
    def __init__(
        self,
        input_dim: int = 4096,      # Q-Former 输出维度
        num_classes: int = 13,      # 场景类别数（10 类 3D + 3 类地图）
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # Linear Probing: 只用线性层！
        # 任务 1: 数量预测 [B, 13]
        self.count_head = nn.Linear(input_dim, num_classes)
        
        # 任务 2: 位置均值预测 [B, 13, 2] (新增)
        # 预测各类目标的平均中心位置 (x, y)
        self.center_head = nn.Linear(input_dim, num_classes * 2)
        
        # 任务 3: 位置方差预测 [B, 13, 2] (新增)
        # 预测各类目标位置的分散程度
        self.variance_head = nn.Linear(input_dim, num_classes * 2)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in [self.count_head, self.center_head, self.variance_head]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
    
    def forward(self, scene_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        预测场景中各类目标的数量和位置统计信息。
        
        Args:
            scene_tokens: [B, 768, 4096] - Q-Former 输出的 scene tokens
        
        Returns:
            pred_counts: [B, 13] - 各类目标的预测数量
            pred_centers: [B, 13, 2] - 各类目标的预测中心均值
            pred_variances: [B, 13, 2] - 各类目标的预测位置方差
        """
        B = scene_tokens.shape[0]
        
        # 1. Global Average Pooling: 768 tokens → 1 个场景向量
        scene_feature = scene_tokens.mean(dim=1)  # [B, 4096]
        
        # 2. 数量预测
        pred_counts = self.count_head(scene_feature)  # [B, 13]
        pred_counts = F.relu(pred_counts)  # 确保数量非负
        
        # 3. 中心位置预测（新增）
        pred_centers = self.center_head(scene_feature)  # [B, 13*2]
        pred_centers = pred_centers.view(B, self.num_classes, 2)  # [B, 13, 2]
        pred_centers = torch.sigmoid(pred_centers)  # 归一化到 [0, 1]
        
        # 4. 位置方差预测（新增）
        pred_variances = self.variance_head(scene_feature)  # [B, 13*2]
        pred_variances = pred_variances.view(B, self.num_classes, 2)  # [B, 13, 2]
        pred_variances = F.relu(pred_variances)  # 方差非负
        
        return {
            'pred_counts': pred_counts,      # [B, 13]
            'pred_centers': pred_centers,    # [B, 13, 2] 新增
            'pred_variances': pred_variances,  # [B, 13, 2] 新增
        }


class QFormerVerificationModel(nn.Module):
    """
    Q-Former 验证模型。
    
    架构：Q-Former → 简单检测头
    不使用 LLM！
    """
    
    def __init__(self):
        super().__init__()
        
        print("\n" + "="*60)
        print("Q-Former Verification Model")
        print(f"  验证目标: {NUM_SCENE_QUERIES} scene queries 的场景表示能力")
        print("  不使用 LLM，直接验证 Q-Former")
        print("="*60)
        
        # 1. Q-Former（配置与主训练完全一致）
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': NUM_SCENE_QUERIES,  # 768
            'num_decoder_layers': 6,
            'llm_hidden_size': 4096,
            'num_heads': 8,
            'ffn_dims': 2048,
            'dropout': 0.1,
            'num_cams': 6,  # 相机数量
            # 3D Position Encoding Config（与 qformer.py 默认值一致）
            'depth_num': 32,          # 32 个深度假设
            'depth_start': 1.0,
            'depth_max': 60.0,
            'use_lid': True,          # LID 深度分布
            'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],  # BEV 范围
        }
        self.qformer = build_qformer(qformer_config)
        print(f"✅ Q-Former initialized ({NUM_SCENE_QUERIES} queries, 与主训练配置一致)")
        
        # 2. 场景数量预测头（Linear Probing - 只用 1 个线性层）
        self.counting_head = SceneCountingHead(
            input_dim=4096,
            num_classes=NUM_CLASSES,  # 13 类（10 类 3D + 3 类地图）
        )
        print(f"✅ Counting head initialized (Linear Probing: 只用 1 个线性层预测 {NUM_CLASSES} 类数量)")
        
        print("="*60 + "\n")
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            images: [B, 6, 3, H, W] - 6 个相机图像
        
        Returns:
            pred_counts: [B, 13] - 各类目标的预测数量
        """
        # Q-Former: 6 images → 768 scene tokens
        scene_tokens = self.qformer(images)  # [B, 768, 4096]
        
        # Counting head: 768 tokens → 场景数量预测
        outputs = self.counting_head(scene_tokens)
        
        return outputs


class QFormerVerificationDataset(Dataset):
    """
    Q-Former 验证数据集。
    
    加载所有场景目标：
    - 10 类 3D 目标（从 nuScenes annotations）
    - 3 类地图元素（从 GT cache）
    
    验证 768 scene tokens 能否代表整个场景的信息。
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        gt_cache_path: str,
        sample_ratio: float = 1.0,
    ):
        self.dataroot = dataroot
        self.version = version
        self.split = split
        
        # Load nuScenes
        from nuscenes import NuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        # Get sample tokens
        self.sample_tokens = self._get_split_tokens(split)
        
        # Apply sample ratio
        if sample_ratio < 1.0:
            num_samples = int(len(self.sample_tokens) * sample_ratio)
            random.shuffle(self.sample_tokens)
            self.sample_tokens = self.sample_tokens[:num_samples]
        
        # GT cache for map elements
        self.gt_ann_dir = os.path.join(gt_cache_path, 'annotations')
        
        # Camera order (与主训练一致)
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # Image preprocessing (与主训练一致)
        self.target_img_size = (800, 448)
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        print(f"Loaded {len(self.sample_tokens)} samples for {split}")
        print(f"  预测类别: {NUM_CLASSES} 类 ({len(OBJECT_CATEGORIES)} 类 3D 目标 + {len(MAP_CATEGORIES)} 类地图元素)")
    
    def _get_split_tokens(self, split: str) -> List[str]:
        from nuscenes.utils.splits import create_splits_scenes
        
        split_scenes = create_splits_scenes()
        if self.version == 'v1.0-mini':
            scene_names = split_scenes['mini_train'] if split == 'train' else split_scenes['mini_val']
        else:
            scene_names = split_scenes['train'] if split == 'train' else split_scenes['val']
        
        sample_tokens = []
        for scene in self.nusc.scene:
            if scene['name'] in scene_names:
                sample_token = scene['first_sample_token']
                while sample_token:
                    sample_tokens.append(sample_token)
                    sample = self.nusc.get('sample', sample_token)
                    sample_token = sample['next']
        
        return sample_tokens
    
    def __len__(self):
        return len(self.sample_tokens)
    
    def _load_images(self, sample_token: str) -> torch.Tensor:
        """加载并预处理 6 张图像"""
        from PIL import Image
        
        sample = self.nusc.get('sample', sample_token)
        images = []
        
        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(self.dataroot, cam_data['filename'])
            
            img = Image.open(img_path).convert('RGB')
            img = img.resize(self.target_img_size, Image.BILINEAR)
            
            # Normalize
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - self.img_mean) / self.img_std
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            
            images.append(img_tensor)
        
        return torch.stack(images, dim=0)  # [6, 3, H, W]
    
    def _load_3d_objects(self, sample_token: str) -> List[Tuple[int, float, float]]:
        """
        加载 3D 目标（10 类）从 nuScenes annotations。
        
        Returns:
            List of (class_id, x_norm, y_norm)
        """
        from pyquaternion import Quaternion
        
        sample = self.nusc.get('sample', sample_token)
        
        # Get ego pose
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        instances = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get category
            category_name = ann['category_name']
            if category_name not in NUSCENES_CATEGORY_MAP:
                continue
            category = NUSCENES_CATEGORY_MAP[category_name]
            
            if category not in OBJECT_CATEGORIES:
                continue
            
            class_id = ALL_CATEGORIES.index(category)
            
            # Transform to ego frame
            global_pos = np.array(ann['translation'][:2])
            pos_ego = ego_rotation.inverse.rotate(np.append(global_pos - ego_translation[:2], 0))[:2]
            x_ego, y_ego = pos_ego[0], pos_ego[1]
            
            # 范围检查: x in [-15, 15], y in [-30, 30]
            if not (-15 <= x_ego <= 15 and -30 <= y_ego <= 30):
                continue
            
            # 归一化到 [0, 1]
            x_norm = (x_ego + 15) / 30
            y_norm = (y_ego + 30) / 60
            
            instances.append((class_id, x_norm, y_norm))
        
        return instances
    
    def _load_map_elements(self, sample_token: str) -> List[Tuple[int, float, float]]:
        """
        加载地图元素（3 类）从 GT cache。
        
        Returns:
            List of (class_id, x_norm, y_norm)
        """
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        
        if not os.path.exists(gt_file):
            return []
        
        with open(gt_file, 'rb') as f:
            gt_data = pickle.load(f)
        
        instances = []
        gt_classes = gt_data['gt_classes']
        gt_points = gt_data['gt_points']  # [N, 20, 2]
        
        for i, (cls_id, points) in enumerate(zip(gt_classes, gt_points)):
            # 地图元素类别 ID = 10 + cls_id (因为前 10 个是 3D 目标)
            class_id = len(OBJECT_CATEGORIES) + cls_id
            
            # 计算中心点
            center_x = (points[:, 0].mean() + 15) / 30
            center_y = (points[:, 1].mean() + 30) / 60
            
            if 0 <= center_x <= 1 and 0 <= center_y <= 1:
                instances.append((class_id, center_x, center_y))
        
        return instances
    
    def _load_gt_stats(self, sample_token: str) -> Dict[str, torch.Tensor]:
        """
        加载场景中各类目标的统计信息（增强版 Linear Probing）
        
        Returns:
            counts: [13] - 每个类别的目标数量
            centers: [13, 2] - 每个类别的中心位置均值
            variances: [13, 2] - 每个类别的位置方差
            exist_mask: [13] - 每个类别是否存在（用于 Loss 计算）
        """
        counts = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        centers = torch.zeros(NUM_CLASSES, 2, dtype=torch.float32)
        variances = torch.zeros(NUM_CLASSES, 2, dtype=torch.float32)
        exist_mask = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        
        # 加载 3D 目标和地图元素
        obj_instances = self._load_3d_objects(sample_token)
        map_instances = self._load_map_elements(sample_token)
        all_instances = obj_instances + map_instances
        
        # 按类别分组实例
        class_positions = {i: [] for i in range(NUM_CLASSES)}
        for class_id, x, y in all_instances:
            if 0 <= class_id < NUM_CLASSES:
                class_positions[class_id].append([x, y])
        
        # 计算每个类别的统计信息
        for class_id in range(NUM_CLASSES):
            positions = class_positions[class_id]
            counts[class_id] = len(positions)
            
            if len(positions) > 0:
                exist_mask[class_id] = 1.0
                pos_array = np.array(positions)  # [N, 2]
                
                # 中心均值
                centers[class_id, 0] = pos_array[:, 0].mean()
                centers[class_id, 1] = pos_array[:, 1].mean()
                
                # 位置方差（如果只有 1 个实例，方差为 0）
                if len(positions) > 1:
                    variances[class_id, 0] = pos_array[:, 0].var()
                    variances[class_id, 1] = pos_array[:, 1].var()
        
        return {
            'counts': counts,        # [13]
            'centers': centers,      # [13, 2]
            'variances': variances,  # [13, 2]
            'exist_mask': exist_mask,  # [13]
        }
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        
        images = self._load_images(sample_token)
        gt_stats = self._load_gt_stats(sample_token)
        
        return {
            'images': images,
            'counts': gt_stats['counts'],        # [13] 各类目标数量
            'centers': gt_stats['centers'],      # [13, 2] 各类中心均值
            'variances': gt_stats['variances'],  # [13, 2] 各类位置方差
            'exist_mask': gt_stats['exist_mask'],  # [13] 存在性掩码
            'sample_token': sample_token,
        }


def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch])
    counts = torch.stack([item['counts'] for item in batch])        # [B, 13]
    centers = torch.stack([item['centers'] for item in batch])      # [B, 13, 2]
    variances = torch.stack([item['variances'] for item in batch])  # [B, 13, 2]
    exist_mask = torch.stack([item['exist_mask'] for item in batch])  # [B, 13]
    tokens = [item['sample_token'] for item in batch]
    
    return {
        'images': images,
        'counts': counts,        # GT: 各类目标数量
        'centers': centers,      # GT: 各类中心均值
        'variances': variances,  # GT: 各类位置方差
        'exist_mask': exist_mask,  # GT: 存在性掩码
        'sample_tokens': tokens,
    }


class CountingLoss(nn.Module):
    """
    增强版场景预测损失函数
    
    任务 1: 数量预测 - MSE Loss
    任务 2: 中心位置预测 - Masked L1 Loss（仅对存在的类别计算）
    任务 3: 位置方差预测 - Masked L1 Loss（仅对存在的类别计算）
    
    Loss 权重设计：
    - 数量预测是主任务，权重 1.0
    - 位置预测是辅助验证，权重较低 0.5
    """
    
    def __init__(
        self, 
        num_classes: int = 13,
        weight_count: float = 1.0,
        weight_center: float = 0.5,
        weight_variance: float = 0.2,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.weight_count = weight_count
        self.weight_center = weight_center
        self.weight_variance = weight_variance
    
    def forward(
        self,
        pred_counts: torch.Tensor,     # [B, 13]
        pred_centers: torch.Tensor,    # [B, 13, 2]
        pred_variances: torch.Tensor,  # [B, 13, 2]
        gt_counts: torch.Tensor,       # [B, 13]
        gt_centers: torch.Tensor,      # [B, 13, 2]
        gt_variances: torch.Tensor,    # [B, 13, 2]
        exist_mask: torch.Tensor,      # [B, 13]
    ) -> Dict[str, torch.Tensor]:
        """
        计算综合损失
        
        Returns:
            loss: 总损失
            loss_count: 数量预测损失
            loss_center: 中心预测损失
            loss_variance: 方差预测损失
            mae: 数量 MAE（监控）
            center_mae: 中心 MAE（监控）
        """
        # 1. 数量预测损失 (MSE)
        loss_count = F.mse_loss(pred_counts, gt_counts)
        
        # 2. 中心位置预测损失 (Masked L1)
        # 只对存在的类别计算
        if exist_mask.sum() > 0:
            mask_expanded = exist_mask.unsqueeze(-1).expand_as(pred_centers)  # [B, 13, 2]
            center_diff = (pred_centers - gt_centers).abs() * mask_expanded
            loss_center = center_diff.sum() / (exist_mask.sum() * 2 + 1e-6)
        else:
            loss_center = torch.tensor(0.0, device=pred_counts.device)
        
        # 3. 方差预测损失 (Masked L1)
        if exist_mask.sum() > 0:
            var_diff = (pred_variances - gt_variances).abs() * mask_expanded
            loss_variance = var_diff.sum() / (exist_mask.sum() * 2 + 1e-6)
        else:
            loss_variance = torch.tensor(0.0, device=pred_counts.device)
        
        # 总损失
        loss = (
            self.weight_count * loss_count +
            self.weight_center * loss_center +
            self.weight_variance * loss_variance
        )
        
        # 监控指标
        with torch.no_grad():
            mae = F.l1_loss(pred_counts, gt_counts)
            
            if exist_mask.sum() > 0:
                center_mae = center_diff.sum() / (exist_mask.sum() * 2 + 1e-6)
            else:
                center_mae = torch.tensor(0.0, device=pred_counts.device)
        
        return {
            'loss': loss,
            'loss_count': loss_count,
            'loss_center': loss_center,
            'loss_variance': loss_variance,
            'mae': mae,
            'center_mae': center_mae,
        }


def train_epoch(model, dataloader, criterion, optimizer, scaler, epoch, args, scheduler=None):
    """
    训练一个 epoch - 增强版场景预测任务
    """
    model.train()
    total_loss = 0
    total_loss_count = 0
    total_loss_center = 0
    total_mae = 0
    total_center_mae = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()           # [B, 6, 3, H, W]
        gt_counts = batch['counts'].cuda()        # [B, 13]
        gt_centers = batch['centers'].cuda()      # [B, 13, 2]
        gt_variances = batch['variances'].cuda()  # [B, 13, 2]
        exist_mask = batch['exist_mask'].cuda()   # [B, 13]
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images)
            losses = criterion(
                outputs['pred_counts'],      # [B, 13]
                outputs['pred_centers'],     # [B, 13, 2]
                outputs['pred_variances'],   # [B, 13, 2]
                gt_counts,
                gt_centers,
                gt_variances,
                exist_mask,
            )
            loss = losses['loss'] / args.accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            
            # Scheduler step after optimizer step
            if scheduler is not None:
                scheduler.step()
        
        total_loss += losses['loss'].item()
        total_loss_count += losses['loss_count'].item()
        total_loss_center += losses['loss_center'].item()
        total_mae += losses['mae'].item()
        total_center_mae += losses['center_mae'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['loss'].item():.4f}",
            'MAE': f"{losses['mae'].item():.2f}",
            'ctr': f"{losses['center_mae'].item():.3f}",
        })
    
    return {
        'loss': total_loss / num_batches,
        'loss_count': total_loss_count / num_batches,
        'loss_center': total_loss_center / num_batches,
        'mae': total_mae / num_batches,
        'center_mae': total_center_mae / num_batches,
    }


@torch.no_grad()
def validate(model, dataloader, criterion, epoch, args):
    """
    验证 - 增强版场景预测任务
    
    评估指标：
    1. 数量 MAE: 各类目标数量的平均绝对误差
    2. 存在性准确率: 判断某类是否存在的准确率
    3. 中心位置 MAE: 存在类别的中心预测误差（新增）
    4. 分类别详细指标
    """
    model.eval()
    total_loss = 0
    total_mae = 0
    total_center_mae = 0
    num_batches = 0
    
    # 收集所有预测和 GT
    all_pred_counts = []
    all_gt_counts = []
    all_pred_centers = []
    all_gt_centers = []
    all_exist_masks = []
    
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].cuda()
        gt_counts = batch['counts'].cuda()
        gt_centers = batch['centers'].cuda()
        gt_variances = batch['variances'].cuda()
        exist_mask = batch['exist_mask'].cuda()
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(images)
            losses = criterion(
                outputs['pred_counts'],
                outputs['pred_centers'],
                outputs['pred_variances'],
                gt_counts,
                gt_centers,
                gt_variances,
                exist_mask,
            )
        
        total_loss += losses['loss'].item()
        total_mae += losses['mae'].item()
        total_center_mae += losses['center_mae'].item()
        num_batches += 1
        
        all_pred_counts.append(outputs['pred_counts'].cpu())
        all_gt_counts.append(gt_counts.cpu())
        all_pred_centers.append(outputs['pred_centers'].cpu())
        all_gt_centers.append(gt_centers.cpu())
        all_exist_masks.append(exist_mask.cpu())
    
    # 汇总
    all_pred_counts = torch.cat(all_pred_counts, dim=0)    # [N, 13]
    all_gt_counts = torch.cat(all_gt_counts, dim=0)        # [N, 13]
    all_pred_centers = torch.cat(all_pred_centers, dim=0)  # [N, 13, 2]
    all_gt_centers = torch.cat(all_gt_centers, dim=0)      # [N, 13, 2]
    all_exist_masks = torch.cat(all_exist_masks, dim=0)    # [N, 13]
    
    avg_loss = total_loss / num_batches
    avg_mae = total_mae / num_batches
    avg_center_mae = total_center_mae / num_batches
    
    # ===== 数量预测指标 =====
    # 分类别数量 MAE
    per_class_count_mae = (all_pred_counts - all_gt_counts).abs().mean(dim=0)  # [13]
    
    # 存在性准确率
    pred_exist = (all_pred_counts > 0.5).float()
    gt_exist = (all_gt_counts > 0).float()
    exist_acc = (pred_exist == gt_exist).float().mean()
    
    # ===== 位置预测指标（新增）=====
    # 分类别中心 MAE（只对存在的样本计算）
    per_class_center_mae = torch.zeros(NUM_CLASSES, 2)
    for c in range(NUM_CLASSES):
        mask = all_exist_masks[:, c] > 0  # [N]
        if mask.sum() > 0:
            center_diff = (all_pred_centers[mask, c] - all_gt_centers[mask, c]).abs()  # [num_exist, 2]
            per_class_center_mae[c] = center_diff.mean(dim=0)  # [2]
    
    # 中心预测的整体 MAE（归一化空间，转换到米）
    # 坐标范围 x: [-15, 15]m, y: [-30, 30]m
    # center_mae 是 [0,1] 归一化空间的误差
    center_mae_meters_x = avg_center_mae * 30  # 30m 范围
    center_mae_meters_y = avg_center_mae * 60  # 60m 范围
    
    # ===== 详细输出 =====
    print(f"\n{'='*80}")
    print(f"Epoch {epoch+1} Validation Results (增强版)")
    print(f"{'='*80}")
    
    print(f"\n📊 数量预测指标:")
    print(f"  Overall Count MAE: {avg_mae:.2f} (目标: < 2.0)")
    print(f"  Existence Accuracy: {exist_acc*100:.1f}% (目标: > 80%)")
    
    print(f"\n📍 位置预测指标 (新增):")
    print(f"  Overall Center MAE: {avg_center_mae:.4f} (归一化空间)")
    print(f"  Approx Center Error: X ~{center_mae_meters_x:.1f}m, Y ~{center_mae_meters_y:.1f}m")
    
    print(f"\n  Loss: {avg_loss:.4f}")
    
    print(f"\n  Per-class 数量 MAE / 中心 MAE:")
    
    # 3D 目标
    print(f"    3D Objects:")
    for i, name in enumerate(OBJECT_CATEGORIES):
        cx, cy = per_class_center_mae[i]
        print(f"      {name:20s}: count={per_class_count_mae[i]:.2f}, center=({cx:.3f}, {cy:.3f})")
    
    # 地图元素
    print(f"    Map Elements:")
    for i, name in enumerate(MAP_CATEGORIES):
        idx = len(OBJECT_CATEGORIES) + i
        cx, cy = per_class_center_mae[idx]
        print(f"      {name:20s}: count={per_class_count_mae[idx]:.2f}, center=({cx:.3f}, {cy:.3f})")
    
    print(f"{'='*80}")
    
    # ===== 综合判断 =====
    count_ok = avg_mae < 2.0 and exist_acc > 0.8
    center_ok = avg_center_mae < 0.15  # 归一化空间误差 < 0.15 (约 4.5m x, 9m y)
    
    if count_ok and center_ok:
        print(f"  ✅ 验证成功! Q-Former 768 tokens 能够有效表示场景的【语义】和【位置】信息")
    elif count_ok and not center_ok:
        print(f"  ⚠️ 部分成功: 数量预测 OK，但位置信息可能不足")
        print(f"     → 建议检查位置编码或特征提取能力")
    elif not count_ok and center_ok:
        print(f"  ⚠️ 部分成功: 位置预测 OK，但数量预测需要改进")
    else:
        print(f"  ❌ 验证失败，Q-Former 设计可能需要改进")
    print(f"{'='*80}\n")
    
    return {
        'loss': avg_loss,
        'mae': avg_mae,
        'center_mae': avg_center_mae,
        'exist_acc': exist_acc.item(),
    }


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.15)
    
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=5.0)
    parser.add_argument('--fp16', action='store_true')
    
    parser.add_argument('--output-dir', type=str, required=True)
    
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("Q-Former Verification - Linear Probing (增强版)")
    print("="*80)
    print(f"验证方法: 场景级别目标统计预测")
    print(f"验证目标: {NUM_SCENE_QUERIES} scene queries 能否代表场景的语义和位置信息")
    print(f"预测内容:")
    print(f"  1. 各类目标的数量 [13] - 验证【语义】信息")
    print(f"  2. 各类目标的中心位置均值 [13, 2] - 验证【位置】信息 (新增)")
    print(f"  3. 各类目标的位置分散度 [13, 2] - 验证【空间分布】信息 (新增)")
    print(f"类别:")
    print(f"  - 10 类 3D 目标: {', '.join(OBJECT_CATEGORIES)}")
    print(f"  - 3 类地图元素: {', '.join(MAP_CATEGORIES)}")
    print(f"检测头: 只用线性层 (Linear Probing)")
    print(f"成功标准:")
    print(f"  - 数量 MAE < 2.0, 存在性准确率 > 80%")
    print(f"  - 中心 MAE < 0.15 (归一化空间)")
    print("="*80)
    
    # Model
    model = QFormerVerificationModel()
    model = model.cuda()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Dataset
    train_dataset = QFormerVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache,
        sample_ratio=args.sample_ratio,
    )
    
    val_dataset = QFormerVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache,
        sample_ratio=0.1,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Loss & Optimizer
    criterion = CountingLoss(num_classes=NUM_CLASSES)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Scheduler
    from transformers import get_cosine_schedule_with_warmup
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=total_steps)
    
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training
    best_mae = float('inf')
    best_exist_acc = 0
    best_center_mae = float('inf')
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, scaler, epoch, args, scheduler)
        
        # 打印训练指标
        print(f"\n📈 Train Epoch {epoch+1}: "
              f"loss={train_metrics['loss']:.4f}, "
              f"count_MAE={train_metrics['mae']:.2f}, "
              f"center_MAE={train_metrics['center_mae']:.4f}")
        
        val_metrics = validate(model, val_loader, criterion, epoch, args)
        
        # Save best (综合 MAE 和 center_mae)
        # 使用综合得分: mae * 0.7 + center_mae * 30 * 0.3 (归一化后的权重)
        current_score = val_metrics['mae'] * 0.7 + val_metrics['center_mae'] * 30 * 0.3
        best_score = best_mae * 0.7 + best_center_mae * 30 * 0.3
        
        if current_score < best_score:
            best_mae = val_metrics['mae']
            best_exist_acc = val_metrics['exist_acc']
            best_center_mae = val_metrics['center_mae']
            save_path = os.path.join(args.output_dir, 'best_model.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': val_metrics,
            }, save_path)
            print(f"✅ Saved best model (count_MAE={best_mae:.2f}, "
                  f"center_MAE={best_center_mae:.4f}, "
                  f"Exist Acc={best_exist_acc*100:.1f}%)")
    
    print("\n" + "="*80)
    print(f"✅ Training completed!")
    print(f"   Best Count MAE: {best_mae:.2f} (目标: < 2.0)")
    print(f"   Best Center MAE: {best_center_mae:.4f} (目标: < 0.15)")
    print(f"   Best Existence Accuracy: {best_exist_acc*100:.1f}% (目标: > 80%)")
    print("="*80)
    
    # 结论
    count_ok = best_mae < 2.0 and best_exist_acc > 0.8
    center_ok = best_center_mae < 0.15
    
    print("\n" + "="*80)
    print("验证结论:")
    if count_ok and center_ok:
        print("  ✅ Q-Former 768 queries 能够有效提取场景的【语义】和【位置】信息！")
        print("  ✅ 场景中有什么、有多少、在哪里 → Q-Former 都知道")
        print("  ✅ 如果主训练效果不好，问题在 LLM 或 MapDecoder，不是 Q-Former")
    elif count_ok and not center_ok:
        print("  ⚠️ Q-Former 能提取【语义】信息（有什么、有多少）")
        print("  ⚠️ 但【位置】信息提取能力有限")
        print("  → 建议: 检查位置编码是否有效，或增加位置敏感的损失")
    elif not count_ok and center_ok:
        print("  ⚠️ Q-Former 能提取【位置】信息")
        print("  ⚠️ 但【语义】信息提取能力有限")
        print("  → 建议: 检查特征提取或 scene queries 设计")
    else:
        print("  ❌ Q-Former 设计可能需要改进")
        print("  ❌ 768 tokens 可能不足以表示完整的场景信息")
        print("  → 建议: 增加 scene queries 数量或调整 decoder 层数")
    print("="*80)


if __name__ == '__main__':
    main()
