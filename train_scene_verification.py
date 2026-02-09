"""
Scene Verification Training - 验证 Q-Former scene queries 的表示能力

目的：
验证 Q-Former 的 768 scene queries 能否准确代表 6 张图片的全部场景内容
- 与主训练架构一致：768 = 6 相机 × 128 tokens/相机
- 不使用 1050 map queries
- 只用 scene tokens + text prompt
- 让 LLM 输出场景中所有元素（10类3D目标 + 3类地图元素）的类别和位置

验证指标：
- format_correct: 输出格式正确率
- total_count_acc: 目标总数预测正确率
- category_count_acc: 各类别数量预测正确率
- instance_recall: 实例召回率（位置匹配）

Author: Auto-generated
Date: 2025-01
"""

import os
import sys
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import AutoTokenizer, CLIPImageProcessor
from tqdm import tqdm
import json
import pickle
from typing import Dict, List, Optional, Tuple
import re

# LoRA support
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("⚠️ peft not installed. LoRA will not be available.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.qformer import QFormer, build_qformer
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM


# ============================================
# 关键配置：Scene Queries 数量
# ============================================
NUM_SCENE_QUERIES = 768  # 与主训练架构一致（原为 1024，现改为 768）


# 有效的目标类别（简化为常见类别）
OBJECT_CATEGORIES = [
    'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
    'pedestrian', 'motorcycle', 'bicycle',
    'barrier', 'traffic_cone'
]

# 地图元素类别
MAP_CATEGORIES = ['divider', 'ped_crossing', 'boundary']

# 所有类别
ALL_CATEGORIES = OBJECT_CATEGORIES + MAP_CATEGORIES

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
    'static_object.bicycle_rack': 'barrier',  # 归类到 barrier
}


class SceneVerificationDataset(Dataset):
    """
    Dataset for scene verification.
    
    GT 包含场景中所有元素（3D目标 + 地图元素）的类别和位置。
    
    GT format:
    "<scene>
    [car] (0.53, 0.27)
    [car] (0.61, 0.42)
    [pedestrian] (0.23, 0.58)
    [divider] (0.50, 0.15)
    </scene>
    Total: 4 objects (2 car, 1 pedestrian, 1 divider)"
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        gt_cache_path: str,
        image_processor,
        tokenizer,
        max_samples: Optional[int] = None,
        sample_ratio: float = 1.0,
    ):
        self.dataroot = dataroot
        self.version = version
        self.split = split
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Load nuScenes
        from nuscenes import NuScenes
        print(f"Loading nuScenes {version} from {dataroot}...")
        self.nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
        
        # Get sample tokens for split
        self.sample_tokens = self._get_split_tokens(split)
        
        # Apply sample ratio
        if sample_ratio < 1.0:
            num_samples = int(len(self.sample_tokens) * sample_ratio)
            random.shuffle(self.sample_tokens)
            self.sample_tokens = self.sample_tokens[:num_samples]
            print(f"Using {sample_ratio*100:.0f}% of data: {len(self.sample_tokens)} samples")
        
        if max_samples is not None:
            self.sample_tokens = self.sample_tokens[:max_samples]
        
        # GT cache for map elements
        self.gt_cache_path = gt_cache_path
        self.gt_ann_dir = os.path.join(gt_cache_path, 'annotations')
        
        # Camera order (same as MapTR)
        self.cam_names = [
            'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
            'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'
        ]
        
        print(f"Loaded {len(self.sample_tokens)} samples for {split} split")
    
    def _get_split_tokens(self, split: str) -> List[str]:
        """Get sample tokens for the given split."""
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
        """Load and preprocess 6 camera images."""
        sample = self.nusc.get('sample', sample_token)
        
        images = []
        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(self.dataroot, cam_data['filename'])
            
            from PIL import Image
            img = Image.open(img_path).convert('RGB')
            
            # Process with CLIP processor
            processed = self.image_processor(img, return_tensors='pt')
            images.append(processed['pixel_values'].squeeze(0))
        
        return torch.stack(images, dim=0)  # [6, 3, H, W]
    
    def _get_ego_pose(self, sample_token: str):
        """Get ego pose for coordinate transformation."""
        sample = self.nusc.get('sample', sample_token)
        lidar_data = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = self.nusc.get('ego_pose', lidar_data['ego_pose_token'])
        return ego_pose
    
    def _global_to_ego(self, global_pos, ego_pose):
        """Transform global coordinates to ego frame."""
        from pyquaternion import Quaternion
        
        # Ego pose
        ego_translation = np.array(ego_pose['translation'])
        ego_rotation = Quaternion(ego_pose['rotation'])
        
        # Transform to ego frame
        pos = np.array(global_pos[:2])  # Only x, y
        pos_ego = ego_rotation.inverse.rotate(np.append(pos - ego_translation[:2], 0))[:2]
        
        return pos_ego
    
    def _load_3d_objects(self, sample_token: str) -> List[Tuple[str, float, float]]:
        """Load 3D object annotations and convert to BEV coordinates."""
        sample = self.nusc.get('sample', sample_token)
        ego_pose = self._get_ego_pose(sample_token)
        
        instances = []
        for ann_token in sample['anns']:
            ann = self.nusc.get('sample_annotation', ann_token)
            
            # Get category
            category_name = ann['category_name']
            if category_name not in NUSCENES_CATEGORY_MAP:
                continue
            category = NUSCENES_CATEGORY_MAP[category_name]
            
            # Get position in ego frame
            try:
                pos_ego = self._global_to_ego(ann['translation'], ego_pose)
                x_ego, y_ego = pos_ego[0], pos_ego[1]
            except:
                continue
            
            # 范围检查: x in [-15, 15], y in [-30, 30]
            if not (-15 <= x_ego <= 15 and -30 <= y_ego <= 30):
                continue
            
            # 归一化到 [0, 1]
            x_norm = (x_ego + 15) / 30
            y_norm = (y_ego + 30) / 60
            
            instances.append((category, x_norm, y_norm))
        
        return instances
    
    def _load_map_elements(self, sample_token: str) -> List[Tuple[str, float, float]]:
        """Load map element annotations."""
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        if not os.path.exists(gt_file):
            return []
        
        with open(gt_file, 'rb') as f:
            gt_data = pickle.load(f)
        
        instances = []
        gt_classes = gt_data['gt_classes']
        gt_points = gt_data['gt_points']  # [N, 20, 2]
        
        for cls_id, points in zip(gt_classes, gt_points):
            category = MAP_CATEGORIES[cls_id]
            
            # 取中点作为位置
            mid_idx = len(points) // 2
            x_ego, y_ego = points[mid_idx]
            
            # 归一化到 [0, 1]
            x_norm = (x_ego + 15) / 30
            y_norm = (y_ego + 30) / 60
            
            # 范围检查
            if 0 <= x_norm <= 1 and 0 <= y_norm <= 1:
                instances.append((category, x_norm, y_norm))
        
        return instances
    
    def _format_gt_as_text(self, instances: List[Tuple[str, float, float]]) -> str:
        """
        Format instances as text for LLM training.
        
        Output format:
        "<scene>
        [car] (0.53, 0.27)
        [pedestrian] (0.23, 0.58)
        </scene>
        Total: 2 objects (1 car, 1 pedestrian)"
        """
        if not instances:
            return "<scene>\n</scene>\nTotal: 0 objects"
        
        # Sort by category for consistency
        instances = sorted(instances, key=lambda x: (ALL_CATEGORIES.index(x[0]) if x[0] in ALL_CATEGORIES else 999, x[1], x[2]))
        
        # Build output
        lines = ["<scene>"]
        category_counts = {}
        
        for cat, x, y in instances:
            lines.append(f"[{cat}] ({x:.2f}, {y:.2f})")
            category_counts[cat] = category_counts.get(cat, 0) + 1
        
        lines.append("</scene>")
        
        # Add summary
        count_parts = [f"{count} {cat}" for cat, count in sorted(category_counts.items())]
        count_str = ", ".join(count_parts)
        lines.append(f"Total: {len(instances)} objects ({count_str})")
        
        return "\n".join(lines)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        
        # Load images
        images = self._load_images(sample_token)  # [6, 3, H, W]
        
        # Load all instances
        obj_instances = self._load_3d_objects(sample_token)
        map_instances = self._load_map_elements(sample_token)
        all_instances = obj_instances + map_instances
        
        # Format GT
        gt_text = self._format_gt_as_text(all_instances)
        
        # Create prompt
        prompt = """You are a scene perception assistant for autonomous driving. Based on the 6 surround-view camera images, detect ALL visible objects and map elements in the scene.

**Object Categories:**
- Vehicles: car, truck, bus, trailer, construction_vehicle, motorcycle, bicycle
- Road Users: pedestrian, barrier, traffic_cone
- Map Elements: divider, ped_crossing, boundary

**Output Format:**
- List each instance: [category] (x, y)
- Coordinates are normalized BEV positions [0,1] with 2 decimal places
- x: left(0) to right(1), y: rear(0) to front(1)
- Wrap in <scene> and </scene> tags
- End with total count

**Example:**
<scene>
[car] (0.53, 0.27)
[pedestrian] (0.23, 0.58)
[divider] (0.50, 0.15)
</scene>
Total: 3 objects (1 car, 1 pedestrian, 1 divider)

Detect all objects and map elements:"""
        
        # Full conversation format
        conversation = f"USER: <image>\n{prompt}\nASSISTANT: {gt_text}"
        
        return {
            'images': images,
            'conversation': conversation,
            'gt_text': gt_text,
            'sample_token': sample_token,
            'num_objects': len(all_instances),
        }


def collate_fn(batch):
    """Custom collate function."""
    images = torch.stack([item['images'] for item in batch], dim=0)
    conversations = [item['conversation'] for item in batch]
    gt_texts = [item['gt_text'] for item in batch]
    sample_tokens = [item['sample_token'] for item in batch]
    num_objects = [item['num_objects'] for item in batch]
    
    return {
        'images': images,
        'conversations': conversations,
        'gt_texts': gt_texts,
        'sample_tokens': sample_tokens,
        'num_objects': num_objects,
    }


class SceneVerificationModel(nn.Module):
    """
    Model for scene verification.
    
    Architecture:
    - Q-Former: 6 images → NUM_SCENE_QUERIES scene tokens
    - LLM (+ LoRA): scene tokens + prompt → text output
    
    不使用 1050 map queries，只验证 scene tokens 的表示能力。
    """
    
    def __init__(
        self,
        llm_path: str,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        qformer_pretrained: str = None,  # 预训练 Q-Former 权重路径
    ):
        super().__init__()
        
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        print("\n" + "="*60)
        print("Initializing Scene Verification Model")
        print(f"  验证目标: {NUM_SCENE_QUERIES} scene queries 的表示能力")
        print("  不使用 1050 map queries")
        if qformer_pretrained:
            print(f"  使用预训练 Q-Former: {qformer_pretrained}")
        if self.use_lora:
            print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print("="*60)
        
        # 1. Load LLM (使用基础 LLaVA，不带 map queries)
        print(f"\nLoading LLM: {llm_path}")
        self.llm = LlavaLlamaForCausalLM.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("✅ LLM loaded")
        
        # 2. Initialize Q-Former
        print("\nInitializing Q-Former...")
        self.num_scene_queries = NUM_SCENE_QUERIES  # 保存为实例变量
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': NUM_SCENE_QUERIES,  # 使用全局配置
            'num_decoder_layers': 6,
            'llm_hidden_size': 4096,
            'num_heads': 8,
            'ffn_dims': 2048,
            'dropout': 0.1,
            'depth_num': 16,
            'use_lid': True,
            'pc_range': [-30.0, 30.0, -15.0, 15.0, -5.0, 5.0],
        }
        self.qformer = build_qformer(qformer_config)
        self.qformer = self.qformer.cuda()
        print(f"✅ Q-Former initialized ({NUM_SCENE_QUERIES} queries)")
        
        # 2.1 加载预训练 Q-Former 权重
        if qformer_pretrained and os.path.exists(qformer_pretrained):
            print(f"\nLoading pretrained Q-Former from: {qformer_pretrained}")
            try:
                checkpoint = torch.load(qformer_pretrained, map_location='cpu')
                
                # 尝试不同的 checkpoint 格式
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # 提取 Q-Former 相关的权重
                qformer_state = {}
                for k, v in state_dict.items():
                    if 'qformer.' in k:
                        # 去掉 'qformer.' 前缀
                        new_key = k.replace('qformer.', '')
                        qformer_state[new_key] = v
                    elif k.startswith('module.qformer.'):
                        new_key = k.replace('module.qformer.', '')
                        qformer_state[new_key] = v
                
                if qformer_state:
                    # 加载权重，允许部分匹配
                    missing, unexpected = self.qformer.load_state_dict(qformer_state, strict=False)
                    print(f"✅ Loaded pretrained Q-Former weights")
                    print(f"   Missing keys: {len(missing)}")
                    print(f"   Unexpected keys: {len(unexpected)}")
                    if missing and len(missing) < 10:
                        print(f"   Missing: {missing}")
                else:
                    print("⚠️ No Q-Former weights found in checkpoint")
            except Exception as e:
                print(f"⚠️ Failed to load pretrained Q-Former: {e}")
        elif qformer_pretrained:
            print(f"⚠️ Pretrained Q-Former not found: {qformer_pretrained}")
        
        # 3. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        
        # Add special tokens
        special_tokens = ['<scene>', '</scene>']
        special_tokens += [f'[{cat}]' for cat in ALL_CATEGORIES]
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        print(f"✅ Added {len(special_tokens)} special tokens")
        
        # 4. Configure LoRA
        if self.use_lora:
            print("\nConfiguring LoRA...")
            
            for param in self.llm.parameters():
                param.requires_grad = False
            
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            self.llm = get_peft_model(self.llm, lora_config)
            
            trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA configured")
            print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("✅ LLM frozen")
        
        # Q-Former always trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        
        print("="*60 + "\n")
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        架构: scene_tokens + text → LLM → text output
        不使用 1050 map queries
        """
        B = images.shape[0]
        device = images.device
        N = self.num_scene_queries  # scene tokens 数量
        
        # 1. Extract scene tokens from Q-Former
        scene_tokens = self.qformer(images)  # [B, N, 4096]
        
        # 2. Get text embeddings
        text_embeds = self.llm.get_input_embeddings()(input_ids)  # [B, L, 4096]
        
        # 3. Concatenate: [scene_tokens, text_embeds]
        inputs_embeds = torch.cat([scene_tokens, text_embeds], dim=1)
        # Shape: [B, N + L, 4096]
        
        # Adjust attention mask
        scene_attn = torch.ones(B, N, device=device, dtype=attention_mask.dtype)
        full_attention_mask = torch.cat([scene_attn, attention_mask], dim=1)
        
        # Adjust labels (ignore scene token positions)
        scene_labels = torch.full((B, N), -100, device=device, dtype=labels.dtype)
        full_labels = torch.cat([scene_labels, labels], dim=1)
        
        # 4. Forward through LLM
        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=full_attention_mask,
            labels=full_labels,
            return_dict=True,
        )
        
        return {
            'loss': outputs.loss,
            'logits': outputs.logits,
        }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompt: str,
        max_new_tokens: int = 512,
    ) -> str:
        """Generate text output for given images."""
        device = images.device
        
        try:
            # 1. Extract scene tokens
            scene_tokens = self.qformer(images.unsqueeze(0))  # [1, N, 4096]
            scene_tokens = scene_tokens.half()
            
            # 2. Tokenize prompt
            prompt_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)
            
            # 3. Combine: scene_tokens + prompt_embeds
            inputs_embeds = torch.cat([scene_tokens, prompt_embeds], dim=1)
            
            # 4. Manual decoding
            generated_ids = []
            past_key_values = None
            
            for step in range(max_new_tokens):
                if step == 0:
                    outputs = self.llm(
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    new_token_embed = self.llm.get_input_embeddings()(
                        torch.tensor([[generated_ids[-1]]], device=device)
                    )
                    outputs = self.llm(
                        inputs_embeds=new_token_embed,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
                generated_ids.append(next_token)
                
                if next_token == self.tokenizer.eos_token_id:
                    break
                if step % 20 == 0 and step > 0:
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if 'Total:' in decoded and 'objects' in decoded:
                        break
            
            return self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
        except Exception as e:
            return f"[生成失败: {str(e)[:100]}]"


def evaluate_scene_output(pred_text: str, gt_text: str) -> Dict[str, float]:
    """
    Evaluate scene detection output.
    
    Returns:
        format_correct: 格式是否正确
        total_count_acc: 总数量是否正确
        category_count_acc: 各类别数量正确率
        instance_recall: 实例召回率（基于位置匹配）
    """
    metrics = {
        'format_correct': 0.0,
        'total_count_acc': 0.0,
        'category_count_acc': 0.0,
        'instance_recall': 0.0,
    }
    
    # Check format
    if '<scene>' in pred_text and '</scene>' in pred_text:
        metrics['format_correct'] = 1.0
    
    # Parse instances
    def parse_instances(text):
        instances = []
        pattern = r'\[(\w+)\]\s*\(([0-9.]+),\s*([0-9.]+)\)'
        for match in re.finditer(pattern, text):
            cat = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            instances.append((cat, x, y))
        return instances
    
    def parse_total(text):
        match = re.search(r'Total:\s*(\d+)\s*objects', text)
        return int(match.group(1)) if match else 0
    
    pred_instances = parse_instances(pred_text)
    gt_instances = parse_instances(gt_text)
    pred_total = parse_total(pred_text)
    gt_total = parse_total(gt_text)
    
    # Total count accuracy
    if gt_total > 0:
        metrics['total_count_acc'] = 1.0 if pred_total == gt_total else 0.0
    
    # Category count accuracy
    def count_by_category(instances):
        counts = {}
        for cat, _, _ in instances:
            counts[cat] = counts.get(cat, 0) + 1
        return counts
    
    pred_counts = count_by_category(pred_instances)
    gt_counts = count_by_category(gt_instances)
    
    if gt_counts:
        correct = sum(1 for cat, cnt in gt_counts.items() if pred_counts.get(cat, 0) == cnt)
        metrics['category_count_acc'] = correct / len(gt_counts)
    
    # Instance recall (position-based matching)
    if gt_instances:
        matched = 0
        for gt_cat, gt_x, gt_y in gt_instances:
            for pred_cat, pred_x, pred_y in pred_instances:
                if pred_cat == gt_cat:
                    dist = ((pred_x - gt_x)**2 + (pred_y - gt_y)**2)**0.5
                    if dist < 0.15:  # 位置误差阈值
                        matched += 1
                        break
        metrics['instance_recall'] = matched / len(gt_instances)
    
    return metrics


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    return rank, world_size, local_rank


def train_epoch(model, dataloader, optimizer, scheduler, scaler, epoch, args, rank, tokenizer):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 使用传入的 tokenizer（已经包含 special tokens）
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        conversations = batch['conversations']
        
        # Tokenize
        tokenized = tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=2048,  # 更长，因为场景元素更多
            return_tensors='pt',
        )
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        
        # Create labels
        labels = input_ids.clone()
        for i, conv in enumerate(conversations):
            assistant_pos = conv.find("ASSISTANT:")
            if assistant_pos != -1:
                prefix = conv[:assistant_pos + len("ASSISTANT:")]
                prefix_len = len(tokenizer(prefix).input_ids)
                labels[i, :prefix_len] = -100
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss'] / args.accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Update
        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
        
        total_loss += loss.item() * args.accumulation_steps
        num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item() * args.accumulation_steps:.4f}",
                'avg': f"{total_loss / num_batches:.4f}"
            })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, tokenizer, epoch, args, rank):
    """Validate and show sample outputs."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_metrics = {
        'format_correct': 0.0,
        'total_count_acc': 0.0,
        'category_count_acc': 0.0,
        'instance_recall': 0.0,
    }
    num_evaluated = 0
    
    for step, batch in enumerate(dataloader):
        images = batch['images'].cuda()
        conversations = batch['conversations']
        gt_texts = batch['gt_texts']
        num_objects = batch['num_objects']
        
        # Tokenize
        tokenized = tokenizer(
            conversations,
            padding=True,
            truncation=True,
            max_length=2048,
            return_tensors='pt',
        )
        input_ids = tokenized.input_ids.cuda()
        attention_mask = tokenized.attention_mask.cuda()
        
        # Create labels
        labels = input_ids.clone()
        for i, conv in enumerate(conversations):
            assistant_pos = conv.find("ASSISTANT:")
            if assistant_pos != -1:
                prefix = conv[:assistant_pos + len("ASSISTANT:")]
                prefix_len = len(tokenizer(prefix).input_ids)
                labels[i, :prefix_len] = -100
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs['loss']
        
        total_loss += loss.item()
        num_batches += 1
        
        # Generate and evaluate for first few batches
        if step < 3 and rank == 0:
            for i in range(min(len(images), 1)):
                # 使用和训练时一样的完整对话格式（带 USER: 和 ASSISTANT: 标记）
                prompt_content = """You are a scene perception assistant for autonomous driving. Based on the 6 surround-view camera images, detect ALL visible objects and map elements in the scene.

**Object Categories:**
- Vehicles: car, truck, bus, trailer, construction_vehicle, motorcycle, bicycle
- Road Users: pedestrian, barrier, traffic_cone
- Map Elements: divider, ped_crossing, boundary

**Output Format:**
- List each instance: [category] (x, y)
- Coordinates are normalized BEV positions [0,1] with 2 decimal places
- x: left(0) to right(1), y: rear(0) to front(1)
- Wrap in <scene> and </scene> tags
- End with total count

**Example:**
<scene>
[car] (0.53, 0.27)
[pedestrian] (0.23, 0.58)
[divider] (0.50, 0.15)
</scene>
Total: 3 objects (1 car, 1 pedestrian, 1 divider)

Detect all objects and map elements:"""
                # 添加对话格式，与训练时一致
                full_prompt = f"USER: <image>\n{prompt_content}\nASSISTANT:"
                base_model = model.module if hasattr(model, 'module') else model
                generated = base_model.generate(images[i], full_prompt)
                
                metrics = evaluate_scene_output(generated, gt_texts[i])
                for k, v in metrics.items():
                    all_metrics[k] += v
                num_evaluated += 1
                
                print(f"\n📝 Sample Output (GT has {num_objects[i]} objects):")
                print(f"GT:\n{gt_texts[i][:500]}...")
                print(f"\nPredicted:\n{generated[:500]}...")
                print(f"Metrics: {metrics}")
    
    # Average metrics
    if num_evaluated > 0:
        for k in all_metrics:
            all_metrics[k] /= num_evaluated
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1} Validation:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Format Correct: {all_metrics['format_correct']*100:.1f}%")
        print(f"  Total Count Acc: {all_metrics['total_count_acc']*100:.1f}%")
        print(f"  Category Count Acc: {all_metrics['category_count_acc']*100:.1f}%")
        print(f"  Instance Recall: {all_metrics['instance_recall']*100:.1f}%")
        print(f"{'='*60}\n")
    
    return avg_loss, all_metrics


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.15)
    
    # Model
    parser.add_argument('--llm-path', type=str, required=True)
    parser.add_argument('--qformer-pretrained', type=str, default=None,
                        help='Path to pretrained Q-Former checkpoint')
    
    # LoRA
    parser.add_argument('--use-lora', action='store_true', default=True)
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=64)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    
    # Training
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Learning rates
    parser.add_argument('--lr-qformer-backbone', type=float, default=3e-5)
    parser.add_argument('--lr-qformer-decoder', type=float, default=2e-4)
    parser.add_argument('--lr-qformer-projector', type=float, default=2.5e-4)
    
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--fp16', action='store_true')
    
    # Output
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--log-interval', type=int, default=20)
    
    args = parser.parse_args()
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    os.makedirs(args.output_dir, exist_ok=True)
    
    if rank == 0:
        print("\n" + "="*60)
        print("Scene Verification Training")
        print(f"验证 Q-Former {NUM_SCENE_QUERIES} queries 的表示能力")
        print("="*60)
        print(f"Data: {args.dataroot}")
        print(f"Sample Ratio: {args.sample_ratio*100:.0f}%")
        print(f"LLM: {args.llm_path}")
        print(f"Scene Queries: {NUM_SCENE_QUERIES}")
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size} x {args.accumulation_steps}")
        print("="*60 + "\n")
    
    # Model
    model = SceneVerificationModel(
        llm_path=args.llm_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        qformer_pretrained=args.qformer_pretrained,
    )
    model = model.cuda()
    
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,
        )
    
    # Tokenizer and image processor
    # 使用模型的 tokenizer（已经添加了 special tokens）
    base_model = model.module if hasattr(model, 'module') else model
    tokenizer = base_model.tokenizer
    print(f"✅ 使用模型的 tokenizer（包含 {len(ALL_CATEGORIES) + 2} 个 special tokens）")
    
    local_clip_path = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
    if os.path.exists(local_clip_path):
        image_processor = CLIPImageProcessor.from_pretrained(local_clip_path)
    else:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    # Datasets
    train_dataset = SceneVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache,
        image_processor=image_processor,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    val_dataset = SceneVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache,
        image_processor=image_processor,
        tokenizer=tokenizer,
        sample_ratio=0.1,
    )
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
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
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    
    qformer_backbone_params = []
    qformer_decoder_params = []
    qformer_projector_params = []
    lora_params = []
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'qformer' in name:
            if 'img_backbone' in name or 'img_neck' in name:
                qformer_backbone_params.append(param)
            elif 'projector' in name:
                qformer_projector_params.append(param)
            else:
                qformer_decoder_params.append(param)
        elif 'lora' in name.lower():
            lora_params.append(param)
    
    param_groups = []
    if qformer_backbone_params:
        param_groups.append({'params': qformer_backbone_params, 'lr': args.lr_qformer_backbone, 'name': 'qformer_backbone'})
    if qformer_decoder_params:
        param_groups.append({'params': qformer_decoder_params, 'lr': args.lr_qformer_decoder, 'name': 'qformer_decoder'})
    if qformer_projector_params:
        param_groups.append({'params': qformer_projector_params, 'lr': args.lr_qformer_projector, 'name': 'qformer_projector'})
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lr_lora, 'name': 'lora'})
    
    if rank == 0:
        print("\nParameter Groups:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']:20s}: {num_params:,} params, lr={group['lr']:.1e}")
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    best_recall = 0.0
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, args, rank, tokenizer
        )
        
        val_loss, val_metrics = validate(
            model, val_loader, tokenizer, epoch, args, rank
        )
        
        # Save best model
        if rank == 0:
            recall = val_metrics['instance_recall']
            if recall > best_recall:
                best_recall = recall
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'metrics': val_metrics,
                }, save_path)
                print(f"✅ Saved best model (recall={recall*100:.1f}%)")
    
    if rank == 0:
        print("\n" + "="*60)
        print(f"✅ Training completed!")
        print(f"   Best Instance Recall: {best_recall*100:.1f}%")
        print("="*60)


if __name__ == '__main__':
    main()
