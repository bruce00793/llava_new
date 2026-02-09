"""
单阶段 LLM 文本生成验证 - 验证 LLM + Map Queries 能否直接输出地图元素

设计思路（与主训练唯一区别：输出文字而非经过 Map Decoder）:
    Images (6 views) → Q-Former → scene_tokens (768)
                                        ↓
    [scene_tokens] + [prompt] + [map_queries(1050)] + [GT文本]
                                        ↓
                                   LLM Forward (MapAttentionMask)
                                        ↓
                            Cross-Entropy Loss on GT文本 tokens

验证目标:
    如果结果好 → LLM + map_queries 能正确提取地图信息，问题在 Map Decoder
    如果结果差 → LLM 对视觉特征的理解不够，需要优化前端

关键一致性:
    - Q-Former: 768 queries, 与主训练一致
    - Map Queries: 1050 (50 instances × 21), 与主训练一致
    - MapAttentionMask: 与主训练一致
    - LLM: BF16 加载, 与主训练一致
    - LoRA: r=32, alpha=64, FP32 参数, 与主训练一致
    - 图像预处理: ImageNet 归一化, 与主训练一致
    - GT 坐标: 20 个点, [-1, 1] 归一化, 与主训练一致
    - Camera参数: cam_intrinsics, cam_extrinsics, 与主训练一致

Author: Auto-generated for Map Detection Verification
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
from transformers import AutoTokenizer
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
    print("peft not installed. LoRA will not be available.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.qformer import QFormer, build_qformer
from llava.model.language_model.llava_map import LlavaMapDetectionModel
from llava.model.map_queries import MapInstancePointQueries, MapAttentionMask


# ============================================================
# 类别定义
# ============================================================
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


# ============================================================
# Dataset
# ============================================================
class MapTextGenDataset(Dataset):
    """
    数据集：图像预处理 + GT 文本格式 与主训练完全一致
    
    GT 文本格式 (20 个点, [-1,1] 归一化):
    <map>
    [divider] (-0.35,-0.68)(-0.34,-0.65)...(-0.30,-0.50)
    [boundary] (0.70,0.50)(0.71,0.53)...(0.80,0.72)
    </map>
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        gt_cache_path: str,
        tokenizer,
        max_samples: Optional[int] = None,
        sample_ratio: float = 1.0,
    ):
        self.dataroot = dataroot
        self.version = version
        self.split = split
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
        
        # GT cache
        self.gt_cache_path = gt_cache_path
        self.gt_ann_dir = os.path.join(gt_cache_path, 'annotations')
        
        # Camera order (与主训练一致!)
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # 图像预处理参数 (与主训练完全一致!)
        self.target_img_size = (800, 448)  # (W, H)
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # 原始图像尺寸和缩放比例 (与主训练一致，用于camera参数缩放)
        self.orig_img_size = (1600, 900)  # nuScenes原始尺寸 (W, H)
        self.scale_x = self.target_img_size[0] / self.orig_img_size[0]
        self.scale_y = self.target_img_size[1] / self.orig_img_size[1]
        
        print(f"Loaded {len(self.sample_tokens)} samples for {split} split")
        print(f"Image preprocessing: {self.target_img_size[0]}x{self.target_img_size[1]}, ImageNet normalization")
    
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
        """Load and preprocess 6 camera images (与主训练完全一致!)"""
        from PIL import Image
        sample = self.nusc.get('sample', sample_token)
        target_w, target_h = self.target_img_size
        
        images = []
        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(self.dataroot, cam_data['filename'])
            img = Image.open(img_path).convert('RGB')
            img = img.resize((target_w, target_h), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_array = (img_array - self.img_mean) / self.img_std
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            images.append(img_tensor)
        
        return torch.stack(images, dim=0)  # [6, 3, H, W]
    
    def _load_cam_params(self, sample_token: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载并处理相机参数 (与主训练 map_dataset.py 完全一致!)
        
        Returns:
            cam_intrinsics: [6, 3, 3] 缩放后的内参
            cam_extrinsics: [6, 4, 4] 外参矩阵
        """
        sample = self.nusc.get('sample', sample_token)
        
        intrinsics_list = []
        extrinsics_list = []
        
        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            
            # 获取标定数据
            calibrated_sensor = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
            
            # 内参 (3x3)，需要根据图像缩放进行调整
            K = np.array(calibrated_sensor['camera_intrinsic'], dtype=np.float32)
            # 缩放内参以匹配resize后的图像
            K[0, :] *= self.scale_x  # fx, cx
            K[1, :] *= self.scale_y  # fy, cy
            intrinsics_list.append(K)
            
            # 外参：从相机坐标系到ego坐标系的变换
            rotation = np.array(calibrated_sensor['rotation'], dtype=np.float32)
            translation = np.array(calibrated_sensor['translation'], dtype=np.float32)
            
            # 构建4x4变换矩阵 (与主训练一致)
            from pyquaternion import Quaternion
            q = Quaternion(rotation)
            R = q.rotation_matrix
            
            E = np.eye(4, dtype=np.float32)
            E[:3, :3] = R
            E[:3, 3] = translation
            extrinsics_list.append(E)
        
        cam_intrinsics = torch.from_numpy(np.stack(intrinsics_list, axis=0))  # [6, 3, 3]
        cam_extrinsics = torch.from_numpy(np.stack(extrinsics_list, axis=0))  # [6, 4, 4]
        
        return cam_intrinsics, cam_extrinsics
    
    def _load_gt(self, sample_token: str) -> Dict:
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        with open(gt_file, 'rb') as f:
            return pickle.load(f)
    
    def _format_gt_as_text(self, gt_data: Dict) -> str:
        """
        将 GT 转换为文字格式 (与 convert_gt_to_text.py 完全一致!)
        
        坐标归一化到 [-1, 1] (与主训练一致):
            x: [-15, 15] → [-1, 1]
            y: [-30, 30] → [-1, 1]
        精度: 2 位小数
        每个实例 20 个点
        """
        gt_classes = np.array(gt_data['gt_classes'])
        gt_points = np.array(gt_data['gt_points'])  # [N, 20, 2]
        
        if len(gt_classes) == 0:
            return "<map>\n</map>"
        
        # PC_RANGE = [-15, -30, -2, 15, 30, 2]
        x_min, y_min = -15, -30
        x_max, y_max = 15, 30
        
        lines_by_class = {0: [], 1: [], 2: []}
        
        for cls_id, points in zip(gt_classes, gt_points):
            cls_id = int(cls_id)
            # 归一化到 [-1, 1]
            x_norm = (points[:, 0] - x_min) / (x_max - x_min) * 2 - 1
            y_norm = (points[:, 1] - y_min) / (y_max - y_min) * 2 - 1
            x_norm = np.clip(x_norm, -1, 1)
            y_norm = np.clip(y_norm, -1, 1)
            
            # 格式化 20 个点
            pts_str = ""
            for i in range(20):
                pts_str += f"({x_norm[i]:.2f},{y_norm[i]:.2f})"
            
            lines_by_class[cls_id].append(f"[{CLASS_NAMES[cls_id]}] {pts_str}")
        
        output_lines = ["<map>"]
        for cls_id in range(3):
            for line in lines_by_class[cls_id]:
                output_lines.append(line)
        output_lines.append("</map>")
        
        return '\n'.join(output_lines)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        images = self._load_images(sample_token)
        cam_intrinsics, cam_extrinsics = self._load_cam_params(sample_token)
        gt_data = self._load_gt(sample_token)
        gt_text = self._format_gt_as_text(gt_data)
        
        # Prompt (与主训练 prompt 格式一致)
        prompt = (
            "You are an autonomous driving perception system. "
            "Given 6 surround-view camera images (front, front-left, front-right, back, back-left, back-right), "
            "detect HD map elements in bird's-eye-view (BEV) coordinates.\n"
            "Map element classes:\n"
            "- class 0: divider (lane dividing lines)\n"
            "- class 1: ped_crossing (pedestrian crossing areas)\n"
            "- class 2: boundary (road boundary lines)\n"
            "For each detected element, output its class and 20 control points.\n"
            "Coordinates normalized to [-1, 1]. x: lateral, y: longitudinal, ego at origin.\n"
            "Format: [class] (x0,y0)(x1,y1)...(x19,y19)\n"
            "Wrap output in <map> </map> tags.\nOutput:"
        )
        
        return {
            'images': images,
            'cam_intrinsics': cam_intrinsics,
            'cam_extrinsics': cam_extrinsics,
            'prompt': prompt,
            'gt_text': gt_text,
            'sample_token': sample_token,
        }


def collate_fn(batch):
    images = torch.stack([item['images'] for item in batch], dim=0)
    cam_intrinsics = torch.stack([item['cam_intrinsics'] for item in batch], dim=0)
    cam_extrinsics = torch.stack([item['cam_extrinsics'] for item in batch], dim=0)
    prompts = [item['prompt'] for item in batch]
    gt_texts = [item['gt_text'] for item in batch]
    sample_tokens = [item['sample_token'] for item in batch]
    return {
        'images': images,
        'cam_intrinsics': cam_intrinsics,
        'cam_extrinsics': cam_extrinsics,
        'prompts': prompts,
        'gt_texts': gt_texts,
        'sample_tokens': sample_tokens,
    }


# ============================================================
# Model
# ============================================================
class SingleStageLLMTextModel(nn.Module):
    """
    单阶段 LLM 文本生成模型
    
    与主训练的唯一区别:
    - 去掉 Map-Scene Interaction 和 Map Decoder
    - LLM 直接输出文字 (cross-entropy loss)
    
    所有前端组件 (Q-Former, Map Queries, MapAttentionMask, LoRA) 
    与主训练 100% 一致
    """
    
    def __init__(
        self,
        llm_path: str,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        print("\n" + "=" * 60)
        print("Initializing Single-Stage LLM Text Generation Model")
        print("  与主训练一致: Q-Former(768) + MapQueries(1050) + MapAttentionMask + LoRA")
        print("  区别: 输出文字 (无 Map-Scene Interaction, 无 Map Decoder)")
        print("=" * 60)
        
        # ===== 1. Q-Former (与主训练完全一致!) =====
        print("\nInitializing Q-Former (与主训练一致)...")
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': 768,
            'num_decoder_layers': 6,
            'llm_hidden_size': 4096,
            'num_heads': 8,
            'ffn_dims': 2048,
            'dropout': 0.1,
            'depth_num': 32,
            'depth_start': 1.0,
            'depth_max': 60.0,
            'use_lid': True,
            'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        }
        self.qformer = build_qformer(qformer_config)
        self.qformer = self.qformer.cuda()
        print("  Q-Former: 768 queries, ResNet50 backbone")
        
        # ===== 2. LLM (BF16 与主训练一致!) =====
        print(f"\nLoading LLM: {llm_path} (BF16)")
        
        import os
        is_distributed = 'WORLD_SIZE' in os.environ and int(os.environ.get('WORLD_SIZE', 1)) > 1
        
        self.llm_dtype = torch.bfloat16
        self.llm = LlavaMapDetectionModel.from_pretrained(
            llm_path,
            torch_dtype=self.llm_dtype,
            device_map=None,
            low_cpu_mem_usage=True,
        )
        print(f"  LLM dtype: {self.llm_dtype}")
        
        # ===== 3. Map Queries (与主训练完全一致!) =====
        print("\nInitializing Map Queries (1050 = 50 instances x 21)...")
        self.map_queries = MapInstancePointQueries(
            num_instances=50,
            num_points=20,
            embed_dim=4096,
        )
        self.map_queries = self.map_queries.cuda()
        
        # 转为 FP32 (与主训练一致)
        self.map_queries = self.map_queries.float()
        with torch.no_grad():
            device = self.map_queries.instance_content.device
            self.map_queries.instance_content.data = torch.randn(
                self.map_queries.instance_content.shape, device=device, dtype=torch.float32
            ) * 0.02
            self.map_queries.point_content.data = torch.randn(
                self.map_queries.point_content.shape, device=device, dtype=torch.float32
            ) * 0.02
        print(f"  Map Queries: FP32, dtype={self.map_queries.instance_content.dtype}")
        
        # ===== 4. Tokenizer + Special Tokens =====
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        special_tokens = ['<map>', '</map>', '[divider]', '[ped_crossing]', '[boundary]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        print(f"  Added {len(special_tokens)} special tokens")
        
        # ===== 5. Gradient Checkpointing =====
        try:
            if hasattr(self.llm, 'gradient_checkpointing_enable'):
                self.llm.gradient_checkpointing_enable()
            elif hasattr(self.llm.model, 'gradient_checkpointing_enable'):
                self.llm.model.gradient_checkpointing_enable()
            print("  Gradient checkpointing enabled")
        except Exception as e:
            print(f"  Gradient checkpointing failed: {e}")
        
        # ===== 6. LoRA (与主训练完全一致!) =====
        if self.use_lora:
            print("\nConfiguring LoRA (与主训练一致)...")
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
            
            # Re-enable gradient checkpointing after LoRA
            try:
                if hasattr(self.llm, 'gradient_checkpointing_enable'):
                    self.llm.gradient_checkpointing_enable()
                elif hasattr(self.llm, 'base_model'):
                    self.llm.base_model.gradient_checkpointing_enable()
            except:
                pass
            
            # 【关键】将 LoRA 参数转换为 FP32 (与主训练一致!)
            lora_count = 0
            for name, param in self.llm.named_parameters():
                if 'lora_' in name and param.requires_grad:
                    param.data = param.data.float()
                    lora_count += 1
            print(f"  Converted {lora_count} LoRA parameters to FP32")
            
            trainable = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.llm.parameters())
            print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
            print(f"  Trainable: {trainable:,} ({100*trainable/total:.2f}%)")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("  LLM frozen (no LoRA)")
        
        # Q-Former, Map Queries always trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        for param in self.map_queries.parameters():
            param.requires_grad = True
        
        # Print total trainable
        total_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"\n  Total trainable params: {total_trainable:,}")
        print("=" * 60 + "\n")
    
    def _get_embed_tokens(self):
        """获取 embedding layer (处理 LoRA 包装)"""
        if self.use_lora and hasattr(self.llm, 'base_model'):
            return self.llm.base_model.model.model.embed_tokens
        else:
            return self.llm.model.embed_tokens
    
    def forward(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_mask: torch.Tensor,
        gt_ids: torch.Tensor,
        gt_mask: torch.Tensor,
        cam_intrinsics: torch.Tensor = None,
        cam_extrinsics: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        单阶段 Forward:
        [scene_tokens(768)] + [prompt(L_p)] + [map_queries(1050)] + [gt_text(L_g)]
        → LLM (扩展的 MapAttentionMask)
        → Cross-Entropy Loss on gt_text
        """
        B = images.shape[0]
        device = images.device
        
        # ===== 1. Q-Former → scene_tokens (与主训练一致, 传递camera参数) =====
        scene_tokens = self.qformer(
            images,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
        )  # [B, 768, 4096]
        scene_tokens = scene_tokens.to(self.llm_dtype)
        
        # ===== 2. Prompt embeddings =====
        embed_tokens = self._get_embed_tokens()
        prompt_embeds = embed_tokens(prompt_ids)  # [B, L_p, 4096]
        L_p = prompt_embeds.shape[1]
        
        # ===== 3. GT text embeddings =====
        gt_embeds = embed_tokens(gt_ids)  # [B, L_g, 4096]
        L_g = gt_embeds.shape[1]
        
        # ===== 4. Map queries (与主训练一致) =====
        map_query_embeds = self.map_queries(B)  # [B, 1050, 4096]
        map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
        
        # ===== 5. 拼接: [scene(768) + prompt(L_p) + map_queries(1050) + gt_text(L_g)] =====
        text_with_scene = torch.cat([scene_tokens, prompt_embeds], dim=1)  # [B, 768+L_p, 4096]
        text_scene_len = text_with_scene.shape[1]  # 768 + L_p
        
        full_input = torch.cat([text_with_scene, map_query_embeds, gt_embeds], dim=1)
        # [B, 768+L_p+1050+L_g, 4096]
        
        # ===== 6. 扩展 MapAttentionMask (关键!) =====
        # 先创建 [text_with_scene + map_queries] 部分的 MapAttentionMask
        prefix_len = text_scene_len + 1050  # 768 + L_p + 1050
        total_len = prefix_len + L_g
        
        # 创建 MapAttentionMask (与主训练一致)
        map_mask = MapAttentionMask.create_mask(
            batch_size=B,
            text_len=text_scene_len,  # scene 已嵌入 text
            scene_len=0,
            num_instances=50,
            num_points=20,
            device=device,
            dtype=full_input.dtype,
        )  # [B, 1, prefix_len, prefix_len]
        
        # 扩展到 [B, 1, total_len, total_len]
        extended_mask = torch.zeros(B, 1, total_len, total_len, device=device, dtype=full_input.dtype)
        
        # 左上角: MapAttentionMask (text+scene+map_queries)
        extended_mask[:, :, :prefix_len, :prefix_len] = map_mask
        
        # 右上角: 前缀 tokens 看不到后面的 gt_text (设为 0, 即 mask)
        # 已经是 0, 无需操作
        
        # 左下角: gt_text tokens 可以看到 map_queries，但不能看到 scene_tokens
        # scene_tokens 占据 [0:768]，prompt 占据 [768:768+L_p]，map_queries 占据 [768+L_p:prefix_len]
        scene_len = 768
        # gt_text 可以看到 prompt 和 map_queries，但不能看到 scene_tokens
        extended_mask[:, :, prefix_len:, scene_len:prefix_len] = 1.0
        
        # 右下角: gt_text tokens 之间使用 causal attention (下三角)
        causal_gt = torch.tril(torch.ones(L_g, L_g, device=device, dtype=full_input.dtype))
        extended_mask[:, :, prefix_len:, prefix_len:] = causal_gt.unsqueeze(0).unsqueeze(0)
        
        # 【修复】不进行手动转换！transformers 4.37.2 的 SDPA 内部会处理
        # 直接使用 1.0/0.0 格式的 mask
        attn_mask = extended_mask
        
        # ===== 7. Labels =====
        # scene + prompt + map_queries: -100 (不计算 loss)
        # gt_text: 真实 token ids (计算 loss)
        prefix_labels = torch.full((B, prefix_len), -100, device=device, dtype=gt_ids.dtype)
        gt_labels = gt_ids.clone()
        # 【修复】将 padding 位置设为 -100，避免计算 padding 的 loss
        gt_labels[gt_mask == 0] = -100
        full_labels = torch.cat([prefix_labels, gt_labels], dim=1)
        
        # ===== 8. LLM Forward =====
        if self.use_lora and hasattr(self.llm, 'base_model'):
            outputs = self.llm.base_model.model(
                input_ids=None,
                attention_mask=attn_mask,
                inputs_embeds=full_input,
                labels=full_labels,
                use_cache=False,
                return_dict=True,
            )
        else:
            outputs = self.llm(
                input_ids=None,
                attention_mask=attn_mask,
                inputs_embeds=full_input,
                labels=full_labels,
                use_cache=False,
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
        cam_intrinsics: torch.Tensor = None,
        cam_extrinsics: torch.Tensor = None,
        max_new_tokens: int = 512,
    ) -> str:
        """
        生成文本输出 (推理)
        
        1. [scene_tokens + prompt + map_queries] → LLM (MapAttentionMask) → KV cache
        2. 自回归生成文本
        """
        device = images.device
        
        try:
            # 1. Q-Former (传递camera参数)
            if cam_intrinsics is not None:
                cam_intrinsics = cam_intrinsics.unsqueeze(0) if cam_intrinsics.dim() == 3 else cam_intrinsics
            if cam_extrinsics is not None:
                cam_extrinsics = cam_extrinsics.unsqueeze(0) if cam_extrinsics.dim() == 3 else cam_extrinsics
            
            scene_tokens = self.qformer(
                images.unsqueeze(0),
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
            )  # [1, 768, 4096]
            scene_tokens = scene_tokens.to(self.llm_dtype)
            
            # 2. Prompt embeddings
            embed_tokens = self._get_embed_tokens()
            prompt_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            prompt_embeds = embed_tokens(prompt_ids)
            
            # 3. Map queries
            map_query_embeds = self.map_queries(1)
            map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
            
            # 4. 拼接 [scene + prompt + map_queries]
            text_with_scene = torch.cat([scene_tokens, prompt_embeds], dim=1)
            text_scene_len = text_with_scene.shape[1]
            prefix_input = torch.cat([text_with_scene, map_query_embeds], dim=1)
            prefix_len = prefix_input.shape[1]
            
            # 5. MapAttentionMask
            map_mask = MapAttentionMask.create_mask(
                batch_size=1,
                text_len=text_scene_len,
                scene_len=0,
                num_instances=50,
                num_points=20,
                device=device,
                dtype=prefix_input.dtype,
            )
            # 【修复】不进行手动转换，直接使用 1.0/0.0 格式
            attn_mask = map_mask
            
            # 6. 第一步 forward: 处理全部 prefix, 获取 KV cache
            if self.use_lora and hasattr(self.llm, 'base_model'):
                outputs = self.llm.base_model.model(
                    input_ids=None,
                    attention_mask=attn_mask,
                    inputs_embeds=prefix_input,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                outputs = self.llm(
                    input_ids=None,
                    attention_mask=attn_mask,
                    inputs_embeds=prefix_input,
                    use_cache=True,
                    return_dict=True,
                )
            
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
            
            # 7. 自回归生成
            generated_ids = [next_token]
            
            for step in range(max_new_tokens - 1):
                if next_token == self.tokenizer.eos_token_id:
                    break
                
                new_embed = embed_tokens(torch.tensor([[next_token]], device=device))
                
                if self.use_lora and hasattr(self.llm, 'base_model'):
                    outputs = self.llm.base_model.model(
                        input_ids=None,
                        inputs_embeds=new_embed,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    outputs = self.llm(
                        input_ids=None,
                        inputs_embeds=new_embed,
                        past_key_values=past_key_values,
                        use_cache=True,
                        return_dict=True,
                    )
                
                past_key_values = outputs.past_key_values
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
                generated_ids.append(next_token)
                
                # 检查是否生成了 </map>
                if step % 10 == 0 and step > 0:
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if '</map>' in decoded:
                        break
            
            return self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
        except Exception as e:
            return f"[generation failed: {str(e)[:200]}]"


# ============================================================
# Evaluation
# ============================================================
def parse_map_text(text: str) -> List[Tuple[str, np.ndarray]]:
    """
    解析 <map>...</map> 文本, 提取 (class_name, points[N, 2]) 列表
    """
    instances = []
    pattern = r'\[(divider|ped_crossing|boundary)\]\s*((?:\([^)]+\))+)'
    
    for match in re.finditer(pattern, text):
        cls_name = match.group(1)
        coords_str = match.group(2)
        coord_pattern = r'\((-?[\d.]+),(-?[\d.]+)\)'
        coords = re.findall(coord_pattern, coords_str)
        if len(coords) >= 2:
            points = np.array([[float(x), float(y)] for x, y in coords])
            instances.append((cls_name, points))
    
    return instances


def chamfer_distance(pts1: np.ndarray, pts2: np.ndarray) -> float:
    """两条 polyline 的 Chamfer Distance"""
    if len(pts1) == 0 or len(pts2) == 0:
        return float('inf')
    diff = pts1[:, np.newaxis, :] - pts2[np.newaxis, :, :]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))
    d1 = dist_matrix.min(axis=1).mean()
    d2 = dist_matrix.min(axis=0).mean()
    return (d1 + d2) / 2


def evaluate_output(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    评估预测文本质量
    
    指标:
    - format_correct: 格式正确率
    - recall: GT 被匹配到的比例
    - precision: 预测中有效的比例
    - f1_score: F1
    - chamfer_error: 匹配实例的平均 CD
    """
    MATCH_THRESHOLD = 0.15
    
    metrics = {
        'format_correct': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'f1_score': 0.0,
        'chamfer_error': 1.0,
    }
    
    if '<map>' not in pred_text:
        return metrics
    metrics['format_correct'] = 1.0
    
    gt_instances = parse_map_text(gt_text)
    pred_instances = parse_map_text(pred_text)
    
    if not gt_instances:
        if not pred_instances:
            metrics.update({'recall': 1.0, 'precision': 1.0, 'f1_score': 1.0, 'chamfer_error': 0.0})
        return metrics
    
    if not pred_instances:
        return metrics
    
    # 按类别匈牙利匹配
    matched_gt, matched_pred = set(), set()
    total_cd = 0.0
    
    for cls_name in CLASS_NAMES:
        gt_cls = [(i, pts) for i, (c, pts) in enumerate(gt_instances) if c == cls_name]
        pred_cls = [(i, pts) for i, (c, pts) in enumerate(pred_instances) if c == cls_name]
        
        if not gt_cls or not pred_cls:
            continue
        
        # 构建代价矩阵
        cost = np.zeros((len(gt_cls), len(pred_cls)))
        for gi, (g_idx, g_pts) in enumerate(gt_cls):
            for pi, (p_idx, p_pts) in enumerate(pred_cls):
                cost[gi, pi] = chamfer_distance(g_pts, p_pts)
        
        # 匈牙利匹配
        try:
            from scipy.optimize import linear_sum_assignment
            row_ind, col_ind = linear_sum_assignment(cost)
        except ImportError:
            row_ind = list(range(min(len(gt_cls), len(pred_cls))))
            col_ind = list(range(min(len(gt_cls), len(pred_cls))))
        
        for r, c in zip(row_ind, col_ind):
            g_idx = gt_cls[r][0]
            p_idx = pred_cls[c][0]
            cd = cost[r, c]
            matched_gt.add(g_idx)
            matched_pred.add(p_idx)
            total_cd += cd
    
    n_matched = len(matched_gt)
    metrics['recall'] = n_matched / len(gt_instances)
    metrics['precision'] = len(matched_pred) / len(pred_instances)
    
    if metrics['recall'] + metrics['precision'] > 0:
        metrics['f1_score'] = 2 * metrics['recall'] * metrics['precision'] / (metrics['recall'] + metrics['precision'])
    
    if n_matched > 0:
        metrics['chamfer_error'] = total_cd / n_matched
    
    return metrics


# ============================================================
# Training & Validation
# ============================================================
def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    
    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl', init_method='env://',
                                world_size=world_size, rank=rank)
        dist.barrier()
    
    return rank, world_size, local_rank


def train_epoch(
    model, dataloader, optimizer, scheduler, tokenizer,
    epoch, args, rank, scaler=None,
):
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        cam_intrinsics = batch['cam_intrinsics'].cuda()
        cam_extrinsics = batch['cam_extrinsics'].cuda()
        prompts = batch['prompts']
        gt_texts = batch['gt_texts']
        
        # Tokenize
        prompt_tok = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        gt_tok = tokenizer(gt_texts, padding=True, truncation=True, max_length=1024, return_tensors='pt')
        
        prompt_ids = prompt_tok.input_ids.cuda()
        prompt_mask = prompt_tok.attention_mask.cuda()
        gt_ids = gt_tok.input_ids.cuda()
        gt_mask = gt_tok.attention_mask.cuda()
        
        # Forward with BF16 autocast
        use_amp = args.bf16
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                images=images,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                gt_ids=gt_ids,
                gt_mask=gt_mask,
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
            )
            loss = outputs['loss'] / args.accumulation_steps
        
        # Backward (BF16 不需要 GradScaler)
        if loss is not None and not torch.isnan(loss):
            loss.backward()
        else:
            if rank == 0:
                print(f"  [Step {step}] Skipping NaN loss", flush=True)
            optimizer.zero_grad()
            continue
        
        # Update
        if (step + 1) % args.accumulation_steps == 0:
            # 检查梯度 NaN
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                if rank == 0 and step < 200:
                    print(f"  [Step {step}] Skipping NaN/Inf gradient", flush=True)
                optimizer.zero_grad()
            else:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                if rank == 0 and (step + 1) % (args.accumulation_steps * args.log_interval) == 0:
                    print(f"  [Step {step+1}] loss={loss.item()*args.accumulation_steps:.4f}, "
                          f"grad_norm={grad_norm:.2f}, lr={scheduler.get_last_lr()[0]:.2e}", flush=True)
        
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item() * args.accumulation_steps
            num_batches += 1
        
        if rank == 0:
            pbar.set_postfix({
                'loss': f"{loss.item()*args.accumulation_steps:.4f}" if loss is not None else "NaN",
                'avg': f"{total_loss/max(num_batches,1):.4f}",
            })
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, dataloader, tokenizer, epoch, args, rank):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    all_metrics = {'format_correct': 0, 'recall': 0, 'precision': 0, 'f1_score': 0, 'chamfer_error': 0}
    num_eval = 0
    sample_outputs = []
    
    for step, batch in enumerate(dataloader):
        images = batch['images'].cuda()
        cam_intrinsics = batch['cam_intrinsics'].cuda()
        cam_extrinsics = batch['cam_extrinsics'].cuda()
        prompts = batch['prompts']
        gt_texts = batch['gt_texts']
        
        # Tokenize
        prompt_tok = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        gt_tok = tokenizer(gt_texts, padding=True, truncation=True, max_length=1024, return_tensors='pt')
        
        prompt_ids = prompt_tok.input_ids.cuda()
        prompt_mask = prompt_tok.attention_mask.cuda()
        gt_ids = gt_tok.input_ids.cuda()
        gt_mask = gt_tok.attention_mask.cuda()
        
        # Forward loss
        use_amp = args.bf16
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            outputs = model(
                images=images,
                prompt_ids=prompt_ids,
                prompt_mask=prompt_mask,
                gt_ids=gt_ids,
                gt_mask=gt_mask,
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
            )
            loss = outputs['loss']
        
        if loss is not None and not torch.isnan(loss):
            total_loss += loss.item()
            num_batches += 1
        
        # 生成评估 (前 5 个 batch, 每 batch 1 个样本)
        if step < 5 and rank == 0:
            base_model = model.module if hasattr(model, 'module') else model
            generated = base_model.generate(
                images[0],
                prompts[0],
                cam_intrinsics=cam_intrinsics[0],
                cam_extrinsics=cam_extrinsics[0],
            )
            
            m = evaluate_output(gt_texts[0], generated)
            for k in all_metrics:
                all_metrics[k] += m[k]
            num_eval += 1
            
            if step == 0:
                sample_outputs.append({'gt': gt_texts[0][:300], 'pred': generated[:300]})
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"[Epoch {epoch+1}] Validation Loss: {avg_loss:.4f}")
        
        if num_eval > 0:
            print(f"\nMetrics ({num_eval} samples):")
            for k, v in all_metrics.items():
                print(f"  {k}: {v/num_eval:.4f}")
        
        if sample_outputs:
            print(f"\nSample:")
            print(f"  GT:   {sample_outputs[0]['gt']}")
            print(f"  Pred: {sample_outputs[0]['pred']}")
        print(f"{'='*60}\n")
    
    return avg_loss


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Single-Stage LLM Text Generation Verification')
    
    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache-train', type=str, required=True)
    parser.add_argument('--gt-cache-val', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.15)
    
    # Model
    parser.add_argument('--llm-path', type=str, required=True)
    
    # LoRA (与主训练一致)
    parser.add_argument('--use-lora', action='store_true', default=True)
    parser.add_argument('--no-lora', dest='use_lora', action='store_false')
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=64)
    parser.add_argument('--lora-dropout', type=float, default=0.1)
    
    # Training
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Learning rates (与主训练一致)
    parser.add_argument('--lr-qformer-backbone', type=float, default=5e-5)
    parser.add_argument('--lr-qformer-decoder', type=float, default=4e-4)
    parser.add_argument('--lr-qformer-projector', type=float, default=5e-4)
    parser.add_argument('--lr-queries', type=float, default=5e-4)
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    
    # Optimization (与主训练一致)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=500)
    parser.add_argument('--grad-clip', type=float, default=35.0)
    parser.add_argument('--bf16', action='store_true', default=True)
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs/llm_text_gen')
    parser.add_argument('--log-interval', type=int, default=20)
    parser.add_argument('--save-interval', type=int, default=1)
    parser.add_argument('--eval-interval', type=int, default=1)
    
    # Resume (恢复训练)
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("Single-Stage LLM Text Generation Verification")
        print(f"{'='*60}")
        print(f"GPU: {world_size} x RTX 4090")
        print(f"Data: {args.version}, ratio={args.sample_ratio}")
        print(f"Epochs: {args.epochs}")
        print(f"Effective batch: {world_size} x {args.batch_size} x {args.accumulation_steps} = "
              f"{world_size * args.batch_size * args.accumulation_steps}")
        print(f"BF16: {args.bf16}")
        print(f"Grad Clip: {args.grad_clip}")
        print(f"Output: {args.output_dir}")
        if args.resume:
            print(f"Resume from: {args.resume}")
        print(f"{'='*60}\n")
    
    # Seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Model
    model = SingleStageLLMTextModel(
        llm_path=args.llm_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )
    model = model.cuda()
    
    tokenizer = model.tokenizer
    
    # DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True,
        )
        try:
            model._set_static_graph()
            if rank == 0:
                print("DDP static graph enabled")
        except:
            pass
    
    # Datasets
    train_dataset = MapTextGenDataset(
        dataroot=args.dataroot, version=args.version, split='train',
        gt_cache_path=args.gt_cache_train, tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    val_dataset = MapTextGenDataset(
        dataroot=args.dataroot, version=args.version, split='val',
        gt_cache_path=args.gt_cache_val, tokenizer=tokenizer,
        sample_ratio=min(args.sample_ratio, 0.1),
    )
    
    # DataLoaders
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True,
    )
    
    # Optimizer (参数分组与主训练一致)
    base_model = model.module if hasattr(model, 'module') else model
    
    param_groups_dict = {
        'qformer_backbone': ([], args.lr_qformer_backbone),
        'qformer_decoder': ([], args.lr_qformer_decoder),
        'qformer_projector': ([], args.lr_qformer_projector),
        'map_queries': ([], args.lr_queries),
        'lora': ([], args.lr_lora),
    }
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        if 'qformer' in name:
            if 'img_backbone' in name or 'img_neck' in name:
                param_groups_dict['qformer_backbone'][0].append(param)
            elif 'projector' in name:
                param_groups_dict['qformer_projector'][0].append(param)
            else:
                param_groups_dict['qformer_decoder'][0].append(param)
        elif 'map_queries' in name:
            param_groups_dict['map_queries'][0].append(param)
        elif 'lora' in name.lower():
            param_groups_dict['lora'][0].append(param)
    
    param_groups = []
    for group_name, (params, lr) in param_groups_dict.items():
        if params:
            param_groups.append({'params': params, 'lr': lr, 'name': group_name})
    
    if rank == 0:
        print("\nParameter Groups:")
        for g in param_groups:
            n = sum(p.numel() for p in g['params'])
            print(f"  {g['name']:20s}: {n:>10,} params, lr={g['lr']:.1e}")
        print()
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    
    # Scheduler (cosine with warmup, 与主训练一致)
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    if rank == 0:
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {args.warmup_steps}")
    
    # Resume from checkpoint
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            if rank == 0:
                print(f"\nLoading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location='cpu')
            
            # Load model state
            state_dict = checkpoint['model_state_dict']
            if hasattr(model, 'module'):
                model.module.load_state_dict(state_dict)
            else:
                model.load_state_dict(state_dict)
            
            # Load optimizer and scheduler
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            start_epoch = checkpoint['epoch'] + 1
            if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
                best_loss = checkpoint['val_loss']
            
            if rank == 0:
                print(f"  Resumed from epoch {start_epoch}")
                print(f"  Best val loss: {best_loss:.4f}")
        else:
            if rank == 0:
                print(f"WARNING: Checkpoint not found: {args.resume}")
    
    # Training loop
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            tokenizer, epoch, args, rank,
        )
        
        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}")
        
        # Validate
        if (epoch + 1) % args.eval_interval == 0:
            if world_size > 1:
                dist.barrier()
            val_loss = validate(model, val_loader, tokenizer, epoch, args, rank)
            if world_size > 1:
                dist.barrier()
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            save_model = model.module if hasattr(model, 'module') else model
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': save_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss if (epoch + 1) % args.eval_interval == 0 else None,
            }, ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            
            # Save best
            if (epoch + 1) % args.eval_interval == 0 and val_loss < best_loss:
                best_loss = val_loss
                best_path = os.path.join(args.output_dir, 'best_model.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'val_loss': val_loss,
                }, best_path)
                print(f"Best model saved: {best_path} (val_loss={val_loss:.4f})")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best val loss: {best_loss:.4f}")
        print(f"Output: {args.output_dir}")
        print(f"{'='*60}")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
