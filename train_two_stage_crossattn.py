"""
两阶段验证实验 + Cross-Attention - 验证 Map-Scene Interaction 的有效性

设计思路：
阶段 1: 与原始检测头完全一致的信息提取
    [scene_tokens] + [prompt] + [map_queries] → LLM (自定义掩码) → map_features

【新增】Cross-Attention 增强:
    map_features + scene_tokens → MapSceneInteractionLayer → enhanced_map_features
    让 map_features 直接从 scene_tokens 提取视觉信息

阶段 2: 从 enhanced_map_features 生成文字
    [enhanced_map_features] + [response_prompt] + [gt_text] → LLM → Loss
    
    关键: gt_text 只能看到 enhanced_map_features，看不到 scene_tokens！

与原始 two_stage_verification.py 的区别：
- 增加了 MapSceneInteractionLayer (3层 Cross-Attention)
- 在阶段1和阶段2之间插入，让 map_features 直接与 scene_tokens 交互

验证逻辑：
- 对比原版：如果 Cross-Attention 版本更好 → 说明直接交互有帮助
- 如果差不多 → 说明 LLM 已经完成了足够的信息融合

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
    print("⚠️ peft not installed. LoRA will not be available.")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.qformer import QFormer, build_qformer
from llava.model.language_model.llava_map import LlavaMapDetectionModel
from llava.model.map_queries import MapInstancePointQueries, MapAttentionMask
from llava.model.map_scene_interaction import MapSceneInteractionLayer, build_map_scene_interaction


class MapTextDataset(Dataset):
    """
    Dataset for two-stage verification.
    图像预处理与主训练完全一致！
    """
    
    CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']
    
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
        
        # Camera order (与主训练一致)
        self.cam_names = [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]
        
        # 图像预处理参数 (与主训练完全一致！)
        self.target_img_size = (800, 448)  # (W, H)
        self.img_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # ImageNet RGB
        self.img_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # ImageNet RGB
        
        print(f"Loaded {len(self.sample_tokens)} samples for {split} split")
        print(f"Image preprocessing: {self.target_img_size[0]}x{self.target_img_size[1]}, ImageNet normalization")
    
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
        """
        Load and preprocess 6 camera images.
        预处理与主训练完全一致！
        """
        from PIL import Image
        
        sample = self.nusc.get('sample', sample_token)
        target_w, target_h = self.target_img_size  # (800, 448)
        
        images = []
        for cam_name in self.cam_names:
            cam_data = self.nusc.get('sample_data', sample['data'][cam_name])
            img_path = os.path.join(self.dataroot, cam_data['filename'])
            
            img = Image.open(img_path).convert('RGB')
            
            # Resize (与主训练一致)
            img = img.resize((target_w, target_h), Image.BILINEAR)
            
            # Convert to numpy and normalize (与主训练一致)
            img_array = np.array(img, dtype=np.float32) / 255.0  # [H, W, 3]
            img_array = (img_array - self.img_mean) / self.img_std
            
            # Convert to tensor [C, H, W]
            img_tensor = torch.from_numpy(img_array.transpose(2, 0, 1)).float()
            images.append(img_tensor)
        
        return torch.stack(images, dim=0)  # [6, 3, H, W]
    
    def _load_gt(self, sample_token: str) -> Dict:
        """Load GT from cache."""
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        with open(gt_file, 'rb') as f:
            return pickle.load(f)
    
    def _format_gt_as_text(self, gt_data: Dict) -> str:
        """
        Convert GT to text format.
        
        【改进设计】
        - 使用 5 个关键点 (p0, p5, p10, p15, p19) 而非仅起止点
        - 坐标精度提高到 2 位小数
        - 保留 ~25% 的几何信息（vs 原来的 10%）
        
        输出格式:
        <map>
        [divider] (0.33,0.28)(0.35,0.32)(0.38,0.40)(0.42,0.48)(0.45,0.55)
        [boundary] (0.10,0.15)(0.15,0.25)(0.22,0.38)(0.30,0.52)(0.40,0.68)
        </map>
        """
        gt_classes = gt_data['gt_classes']
        gt_points = gt_data['gt_points']  # [N, 20, 2]
        
        # 关键点索引: 0, 5, 10, 15, 19 (共 5 个点)
        KEY_POINT_INDICES = [0, 5, 10, 15, 19]
        
        lines_by_class = {0: [], 1: [], 2: []}
        
        for i, (cls_id, points) in enumerate(zip(gt_classes, gt_points)):
            # 归一化坐标到 [0, 1]
            # 原始范围: x: [-15, 15], y: [-30, 30]
            x_norm = (points[:, 0] + 15) / 30
            y_norm = (points[:, 1] + 30) / 60
            
            # 提取 5 个关键点
            key_points_str = ""
            for idx in KEY_POINT_INDICES:
                x, y = x_norm[idx], y_norm[idx]
                key_points_str += f"({x:.2f},{y:.2f})"
            
            lines_by_class[cls_id].append(key_points_str)
        
        output_lines = ["<map>"]
        for cls_id, cls_name in enumerate(self.CLASS_NAMES):
            if lines_by_class[cls_id]:
                for line_str in lines_by_class[cls_id]:
                    output_lines.append(f"[{cls_name}] {line_str}")
        output_lines.append("</map>")
        
        return '\n'.join(output_lines)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        
        images = self._load_images(sample_token)
        gt_data = self._load_gt(sample_token)
        gt_text = self._format_gt_as_text(gt_data)
        
        # 【与主训练完全一致的 prompt！】
        stage1_prompt = (
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
        
        # 【改进的 Stage 2 Prompt】
        # 明确告知输出格式：5 个关键点，2 位小数
        stage2_prompt = """Detect map elements and output in this format:
<map>
[class] (x0,y0)(x1,y1)(x2,y2)(x3,y3)(x4,y4)
</map>

Rules:
- class: divider, ped_crossing, or boundary
- 5 key points per instance: start, 1/4, middle, 3/4, end positions
- Coordinates: BEV normalized to [0,1], 2 decimal places
- x: lateral (0=left, 1=right), y: longitudinal (0=back, 1=front)

Example:
<map>
[divider] (0.33,0.28)(0.35,0.35)(0.38,0.42)(0.41,0.50)(0.45,0.58)
[boundary] (0.10,0.20)(0.15,0.32)(0.22,0.45)(0.30,0.58)(0.40,0.72)
</map>

Output:"""
        
        return {
            'images': images,
            'stage1_prompt': stage1_prompt,
            'stage2_prompt': stage2_prompt,
            'gt_text': gt_text,
            'sample_token': sample_token,
        }


def collate_fn(batch):
    """Custom collate function."""
    images = torch.stack([item['images'] for item in batch], dim=0)
    stage1_prompts = [item['stage1_prompt'] for item in batch]
    stage2_prompts = [item['stage2_prompt'] for item in batch]
    gt_texts = [item['gt_text'] for item in batch]
    sample_tokens = [item['sample_token'] for item in batch]
    
    return {
        'images': images,
        'stage1_prompts': stage1_prompts,
        'stage2_prompts': stage2_prompts,
        'gt_texts': gt_texts,
        'sample_tokens': sample_tokens,
    }


class TwoStageCrossAttnModel(nn.Module):
    """
    两阶段验证模型 + Cross-Attention
    
    与原版的区别：
    - 在阶段1和阶段2之间加入 MapSceneInteractionLayer
    - 让 map_features 直接和 scene_tokens 做 Cross-Attention
    """
    
    def __init__(
        self,
        llm_path: str,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.1,  # 与主训练一致
        # Cross-Attention 配置
        crossattn_layers: int = 3,
        crossattn_embed_dim: int = 256,
        crossattn_heads: int = 8,
    ):
        super().__init__()
        
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        print("\n" + "="*60)
        print("Initializing Two-Stage + Cross-Attention Model")
        print("  阶段 1: 与原始检测头一致的信息提取")
        print(f"  【新增】Cross-Attention: {crossattn_layers} 层, dim={crossattn_embed_dim}")
        print("  阶段 2: 从 enhanced_map_features 生成文字验证")
        if self.use_lora:
            print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print("="*60)
        
        # 1. Load LLM
        print(f"\nLoading LLM: {llm_path}")
        self.llm = LlavaMapDetectionModel.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("✅ LLM loaded")
        
        # 2. Initialize Q-Former (与主训练完全一致！)
        print("\nInitializing Q-Former...")
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': 768,  # 与主训练一致！
            'num_decoder_layers': 6,
            'llm_hidden_size': 4096,
            'num_heads': 8,
            'ffn_dims': 2048,
            'dropout': 0.1,
            # Enhanced 3D Position Encoding (与主训练完全一致！)
            'depth_num': 32,        # 32个深度假设
            'depth_start': 1.0,     # 最小深度 1米
            'depth_max': 60.0,      # 最大深度 60米
            'use_lid': True,        # LID深度分布
            # pc_range 与 MapConfig 保持一致
            'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
        }
        self.qformer = build_qformer(qformer_config)
        self.qformer = self.qformer.cuda()
        print("✅ Q-Former initialized (768 queries)")
        
        # 3. Initialize Map Queries (与主训练完全一致！)
        print("\nInitializing Map Queries (1050 = 50 instances × 21)...")
        self.map_queries = MapInstancePointQueries(
            num_instances=50,
            num_points=20,
            embed_dim=4096,
        )
        self.map_queries = self.map_queries.cuda()
        
        # 转为 FP32 以保证数值稳定性（与主训练一致！）
        self.map_queries = self.map_queries.float()
        with torch.no_grad():
            device = self.map_queries.instance_content.device
            # 重新初始化
            self.map_queries.instance_content.data = torch.randn(
                self.map_queries.instance_content.shape,
                device=device, dtype=torch.float32
            ) * 0.02
            self.map_queries.point_content.data = torch.randn(
                self.map_queries.point_content.shape,
                device=device, dtype=torch.float32
            ) * 0.02
        print(f"✅ Map Queries initialized (FP32, dtype={self.map_queries.instance_content.dtype})")
        
        # 4. 【新增】Map-Scene Interaction Layer (Cross-Attention)
        print(f"\nInitializing Map-Scene Interaction Layer...")
        print(f"  Layers: {crossattn_layers}")
        print(f"  Embed Dim: {crossattn_embed_dim}")
        print(f"  Heads: {crossattn_heads}")
        self.map_scene_interaction = build_map_scene_interaction(
            input_dim=4096,
            embed_dim=crossattn_embed_dim,
            num_heads=crossattn_heads,
            num_layers=crossattn_layers,
            ffn_dim=crossattn_embed_dim * 4,
            dropout=0.1,
        )
        self.map_scene_interaction = self.map_scene_interaction.cuda()
        
        crossattn_params = sum(p.numel() for p in self.map_scene_interaction.parameters())
        print(f"✅ Map-Scene Interaction initialized ({crossattn_params:,} params)")
        
        # 5. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        
        special_tokens = ['<map>', '</map>', '[divider]', '[ped_crossing]', '[boundary]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        print(f"✅ Added {len(special_tokens)} special tokens")
        
        # 6. Enable Gradient Checkpointing
        print("\nEnabling Gradient Checkpointing...")
        try:
            if hasattr(self.llm, 'gradient_checkpointing_enable'):
                self.llm.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            elif hasattr(self.llm.model, 'gradient_checkpointing_enable'):
                self.llm.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled (via .model)")
        except Exception as e:
            print(f"⚠️ Failed to enable gradient checkpointing: {e}")
        
        # 7. Configure LoRA
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
            
            try:
                if hasattr(self.llm, 'gradient_checkpointing_enable'):
                    self.llm.gradient_checkpointing_enable()
                elif hasattr(self.llm, 'base_model'):
                    self.llm.base_model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing re-enabled after LoRA")
            except:
                pass
            
            trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA configured")
            print(f"   Trainable: {trainable_params:,} ({100*trainable_params/total_params:.2f}%)")
        else:
            for param in self.llm.parameters():
                param.requires_grad = False
            print("✅ LLM frozen")
        
        # Q-Former, Map Queries, and Cross-Attention always trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        for param in self.map_queries.parameters():
            param.requires_grad = True
        for param in self.map_scene_interaction.parameters():
            param.requires_grad = True
        
        print("="*60 + "\n")
    
    def forward(
        self,
        images: torch.Tensor,
        stage1_prompt_ids: torch.Tensor,
        stage1_prompt_mask: torch.Tensor,
        stage2_prompt_ids: torch.Tensor,
        stage2_prompt_mask: torch.Tensor,
        gt_ids: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        两阶段 Forward + Cross-Attention
        
        阶段 1: 与原始检测头完全一致
            [scene_tokens (768)] + [stage1_prompt] + [map_queries (1050)]
            使用自定义掩码 MapAttentionMask
            提取 map_features
        
        【新增】Cross-Attention:
            map_features + scene_tokens → MapSceneInteractionLayer → enhanced_map_features
        
        阶段 2: 从 enhanced_map_features 生成文字
            [enhanced_map_features (1050)] + [stage2_prompt] + [gt_text]
            标准 causal attention
        """
        B = images.shape[0]
        device = images.device
        SCENE_LEN = 768  # 与主训练一致
        
        # ========== 阶段 1: 信息提取（与主训练完全一致！）==========
        # 
        # 主训练流程:
        #   1. Q-Former → scene_tokens [B, 768, 4096]
        #   2. text_embeds 中 IMAGE_TOKEN 位置被替换为 scene_tokens
        #   3. [text_with_scene + map_queries] → LLM → query_outputs
        #   4. query_outputs + 原始scene_tokens → Cross-Attention → enhanced_map_features
        #
        # 这里完全复制该流程！
        
        # 1.1 Extract scene tokens from Q-Former
        scene_tokens_original = self.qformer(images)  # [B, 768, 4096]
        scene_tokens = scene_tokens_original.half()  # 转为 FP16 与 LLM 匹配
        
        # 1.2 Get text embeddings
        # 获取 embedding layer
        if hasattr(self.llm, 'base_model'):
            embed_tokens = self.llm.base_model.model.model.embed_tokens
        else:
            embed_tokens = self.llm.model.model.embed_tokens
        
        stage1_embeds = embed_tokens(stage1_prompt_ids)  # [B, L1, 4096]
        L1 = stage1_embeds.shape[1]
        
        # 1.3 【关键！与主训练一致】将 scene_tokens 嵌入到 text 的开头
        # 主训练是替换 IMAGE_TOKEN，这里简化为：text_with_scene = [scene + text]
        # 这样 text_with_scene 的总长度 = 768 + L1
        text_with_scene = torch.cat([scene_tokens, stage1_embeds], dim=1)  # [B, 768+L1, 4096]
        text_with_scene_len = text_with_scene.shape[1]
        
        # 1.4 Get map queries
        map_query_embeds = self.map_queries(B)
        map_query_embeds = map_query_embeds.to(device=device, dtype=text_with_scene.dtype)
        
        # 1.5 拼接：[text_with_scene + map_queries]（与主训练一致！）
        stage1_input = torch.cat([text_with_scene, map_query_embeds], dim=1)
        
        # 1.6 创建自定义掩码（与主训练一致！）
        # 主训练: scene_len=0（因为 scene 已嵌入 text）
        # 这里: text_with_scene 包含 scene，所以 scene_len=0
        custom_mask = MapAttentionMask.create_mask(
            batch_size=B,
            text_len=text_with_scene_len,  # 包含 scene 的 text 长度
            scene_len=0,  # scene 已嵌入 text，所以为 0（与主训练一致！）
            num_instances=50,
            num_points=20,
            device=device,
            dtype=stage1_input.dtype,
        )
        # 注：transformers 期望 4D 掩码格式为加性掩码
        # create_mask 返回 1.0=attend, 0.0=mask
        # 需要转换为 0.0=attend, -inf=mask
        custom_mask = custom_mask.masked_fill(custom_mask == 0, float('-inf'))
        custom_mask = custom_mask.masked_fill(custom_mask == 1, 0.0)
        
        # 1.7 Forward through LLM (阶段 1)
        # 使用与主训练相同的 forward 方式
        if hasattr(self.llm, 'base_model'):
            # LoRA 模式
            outputs = self.llm.base_model.model(
                input_ids=None,
                attention_mask=custom_mask,
                inputs_embeds=stage1_input,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            outputs = self.llm.model(
                input_ids=None,
                attention_mask=custom_mask,
                inputs_embeds=stage1_input,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # 1.8 提取 map_features（与主训练一致！）
        hidden_states = outputs.hidden_states[-1]  # 使用最后一层的 hidden states
        prefix_len = text_with_scene_len  # scene 已嵌入 text
        map_features = hidden_states[:, prefix_len:prefix_len+1050, :]  # [B, 1050, 4096]
        
        # ========== 【新增】Cross-Attention 增强（与主训练一致！）==========
        # 【关键！】使用 **原始 scene tokens** (Q-Former 直接输出)
        map_features_fp32 = map_features.float()
        scene_tokens_for_interaction = scene_tokens_original.float()  # 使用原始的！
        
        enhanced_map_features = self.map_scene_interaction(
            map_features=map_features_fp32,
            scene_tokens=scene_tokens_for_interaction,
        )  # [B, 1050, 4096]
        
        # 转回 FP16
        enhanced_map_features = enhanced_map_features.half()
        
        # ========== 阶段 2: 从 enhanced_map_features 生成文字 ==========
        
        # 2.1 Get stage2 prompt embeddings（使用与阶段 1 一致的 embedding layer）
        stage2_embeds = embed_tokens(stage2_prompt_ids)
        L2 = stage2_embeds.shape[1]
        
        # 2.2 Get gt embeddings
        gt_embeds = embed_tokens(gt_ids)
        Lg = gt_embeds.shape[1]
        
        # 2.3 拼接 (使用增强后的 map_features)
        stage2_input = torch.cat([enhanced_map_features, stage2_embeds, gt_embeds], dim=1)
        
        # 2.4 标准 causal attention mask
        total_len_stage2 = 1050 + L2 + Lg
        stage2_attn_mask = torch.ones(B, total_len_stage2, device=device)
        
        # 2.5 Labels
        map_labels = torch.full((B, 1050), -100, device=device, dtype=gt_ids.dtype)
        stage2_prompt_labels = torch.full((B, L2), -100, device=device, dtype=gt_ids.dtype)
        gt_labels = gt_ids.clone()
        full_labels = torch.cat([map_labels, stage2_prompt_labels, gt_labels], dim=1)
        
        # 2.6 Forward through LLM (阶段 2)
        stage2_outputs = self.llm(
            inputs_embeds=stage2_input,
            attention_mask=stage2_attn_mask,
            labels=full_labels,
            return_dict=True,
            use_cache=False,
        )
        
        return {
            'loss': stage2_outputs.loss,
            'logits': stage2_outputs.logits,
            'map_features': map_features.detach(),
            'enhanced_map_features': enhanced_map_features.detach(),
        }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        stage1_prompt: str,
        stage2_prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """生成文本输出"""
        device = images.device
        SCENE_LEN = 768  # 与主训练一致
        
        try:
            # ===== 阶段 1: 提取 map_features（与主训练完全一致！）=====
            scene_tokens_original = self.qformer(images.unsqueeze(0))
            scene_tokens = scene_tokens_original.half()
            
            # 获取 embedding layer
            if hasattr(self.llm, 'base_model'):
                embed_tokens = self.llm.base_model.model.model.embed_tokens
            else:
                embed_tokens = self.llm.model.model.embed_tokens
            
            stage1_ids = self.tokenizer(stage1_prompt, return_tensors='pt').input_ids.to(device)
            stage1_embeds = embed_tokens(stage1_ids)
            L1 = stage1_embeds.shape[1]
            
            # 将 scene_tokens 嵌入到 text 开头
            text_with_scene = torch.cat([scene_tokens, stage1_embeds], dim=1)
            text_with_scene_len = text_with_scene.shape[1]
            
            map_query_embeds = self.map_queries(1)
            map_query_embeds = map_query_embeds.to(device=device, dtype=text_with_scene.dtype)
            
            # 拼接：[text_with_scene + map_queries]
            stage1_input = torch.cat([text_with_scene, map_query_embeds], dim=1)
            
            # 创建掩码（与主训练一致）
            custom_mask = MapAttentionMask.create_mask(
                batch_size=1,
                text_len=text_with_scene_len,
                scene_len=0,  # scene 已嵌入 text
                num_instances=50,
                num_points=20,
                device=device,
                dtype=stage1_input.dtype,
            )
            custom_mask = custom_mask.masked_fill(custom_mask == 0, float('-inf'))
            custom_mask = custom_mask.masked_fill(custom_mask == 1, 0.0)
            
            # Forward through LLM
            if hasattr(self.llm, 'base_model'):
                outputs = self.llm.base_model.model(
                    input_ids=None,
                    attention_mask=custom_mask,
                    inputs_embeds=stage1_input,
                    output_hidden_states=True,
                    return_dict=True,
                )
            else:
                outputs = self.llm.model(
                    input_ids=None,
                    attention_mask=custom_mask,
                    inputs_embeds=stage1_input,
                    output_hidden_states=True,
                    return_dict=True,
                )
            
            hidden_states = outputs.hidden_states[-1]
            prefix_len = text_with_scene_len
            map_features = hidden_states[:, prefix_len:prefix_len+1050, :]
            
            # ===== Cross-Attention 增强 =====
            map_features_fp32 = map_features.float()
            scene_tokens_fp32 = scene_tokens_original.float()
            enhanced_map_features = self.map_scene_interaction(map_features_fp32, scene_tokens_fp32)
            enhanced_map_features = enhanced_map_features.half()
            
            # ===== 阶段 2: 生成文字 =====
            stage2_ids = self.tokenizer(stage2_prompt, return_tensors='pt').input_ids.to(device)
            stage2_embeds = embed_tokens(stage2_ids)
            
            inputs_embeds = torch.cat([enhanced_map_features, stage2_embeds], dim=1)
            
            # 手动解码
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
                    new_token_embed = embed_tokens(
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
                if step % 10 == 0 and step > 0:
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if '</map>' in decoded:
                        break
            
            return self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
        except Exception as e:
            return f"[生成失败: {str(e)[:100]}]"


def parse_map_instances(text: str) -> List[Tuple[str, np.ndarray]]:
    """
    解析 map 文本，提取实例列表。
    
    Returns:
        List of (class_name, points_array) where points_array is [5, 2]
    """
    instances = []
    
    # 匹配格式: [class] (x0,y0)(x1,y1)(x2,y2)(x3,y3)(x4,y4)
    # 新格式: 5 个点
    instance_pattern = r'\[(divider|ped_crossing|boundary)\]\s*(\([^)]+\))+'
    
    for match in re.finditer(instance_pattern, text):
        cls_name = match.group(1)
        full_match = match.group(0)
        
        # 提取所有坐标点
        coord_pattern = r'\(([\d.]+),([\d.]+)\)'
        coords = re.findall(coord_pattern, full_match)
        
        if len(coords) >= 2:  # 至少有 2 个点
            points = np.array([[float(x), float(y)] for x, y in coords])
            instances.append((cls_name, points))
    
    # 如果没有匹配到新格式，尝试旧格式 (start/end)
    if not instances:
        old_pattern = r'\[(divider|ped_crossing|boundary)\]\s*start\(([\d.]+),([\d.]+)\)\s*end\(([\d.]+),([\d.]+)\)'
        for match in re.finditer(old_pattern, text):
            cls_name = match.group(1)
            start = [float(match.group(2)), float(match.group(3))]
            end = [float(match.group(4)), float(match.group(5))]
            # 插值生成 5 个点
            points = np.array([
                start,
                [start[0]*0.75 + end[0]*0.25, start[1]*0.75 + end[1]*0.25],
                [start[0]*0.5 + end[0]*0.5, start[1]*0.5 + end[1]*0.5],
                [start[0]*0.25 + end[0]*0.75, start[1]*0.25 + end[1]*0.75],
                end
            ])
            instances.append((cls_name, points))
    
    return instances


def chamfer_distance(points1: np.ndarray, points2: np.ndarray) -> float:
    """
    计算两条 polyline 之间的 Chamfer Distance。
    
    Args:
        points1: [N1, 2] 第一条线的点
        points2: [N2, 2] 第二条线的点
    
    Returns:
        Chamfer distance (越小越好)
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # 计算点到点的距离矩阵
    diff = points1[:, np.newaxis, :] - points2[np.newaxis, :, :]  # [N1, N2, 2]
    dist_matrix = np.sqrt((diff ** 2).sum(axis=2))  # [N1, N2]
    
    # Chamfer: 平均最近邻距离
    dist1 = dist_matrix.min(axis=1).mean()  # points1 到 points2 的平均最近距离
    dist2 = dist_matrix.min(axis=0).mean()  # points2 到 points1 的平均最近距离
    
    return (dist1 + dist2) / 2


def hungarian_match(gt_instances: List, pred_instances: List, cls_name: str) -> List[Tuple[int, int, float]]:
    """
    使用匈牙利算法匹配同类别的 GT 和预测实例。
    
    Returns:
        List of (gt_idx, pred_idx, distance)
    """
    gt_same_cls = [(i, inst) for i, (cls, inst) in enumerate(gt_instances) if cls == cls_name]
    pred_same_cls = [(i, inst) for i, (cls, inst) in enumerate(pred_instances) if cls == cls_name]
    
    if not gt_same_cls or not pred_same_cls:
        return []
    
    # 构建代价矩阵
    cost_matrix = np.zeros((len(gt_same_cls), len(pred_same_cls)))
    for i, (gt_idx, gt_pts) in enumerate(gt_same_cls):
        for j, (pred_idx, pred_pts) in enumerate(pred_same_cls):
            cost_matrix[i, j] = chamfer_distance(gt_pts, pred_pts)
    
    # 匈牙利匹配
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        # 简单贪心匹配作为后备
        row_ind, col_ind = [], []
        used_cols = set()
        for i in range(len(gt_same_cls)):
            best_j, best_cost = -1, float('inf')
            for j in range(len(pred_same_cls)):
                if j not in used_cols and cost_matrix[i, j] < best_cost:
                    best_j, best_cost = j, cost_matrix[i, j]
            if best_j >= 0:
                row_ind.append(i)
                col_ind.append(best_j)
                used_cols.add(best_j)
    
    matches = []
    for i, j in zip(row_ind, col_ind):
        gt_idx = gt_same_cls[i][0]
        pred_idx = pred_same_cls[j][0]
        dist = cost_matrix[i, j]
        matches.append((gt_idx, pred_idx, dist))
    
    return matches


def evaluate_text_output(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    【改进版】评估输出质量
    
    评估指标:
    1. format_correct: 格式正确率 (0/1)
    2. recall: 召回率 = 匹配的 GT 数 / GT 总数
    3. precision: 精确率 = 匹配的预测数 / 预测总数
    4. chamfer_error: 匹配实例的平均 Chamfer Distance
    5. class_recall: 各类别的召回率 (dict)
    6. instance_count_error: 各类别实例数差异
    """
    CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']
    MATCH_THRESHOLD = 0.15  # Chamfer distance 阈值，小于此值认为匹配成功
    
    metrics = {
        'format_correct': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'chamfer_error': 1.0,  # 默认最差
        'f1_score': 0.0,
    }
    
    # 检查格式
    map_pattern = r'<map>(.*?)</map>'
    pred_match = re.search(map_pattern, pred_text, re.DOTALL)
    
    if not pred_match:
        return metrics
    
    metrics['format_correct'] = 1.0
    
    # 解析实例
    gt_instances = parse_map_instances(gt_text)
    pred_instances = parse_map_instances(pred_text)
    
    if not gt_instances:
        if not pred_instances:
            metrics['recall'] = 1.0
            metrics['precision'] = 1.0
            metrics['f1_score'] = 1.0
            metrics['chamfer_error'] = 0.0
        return metrics
    
    if not pred_instances:
        return metrics
    
    # 按类别进行匈牙利匹配
    all_matches = []
    for cls_name in CLASS_NAMES:
        matches = hungarian_match(gt_instances, pred_instances, cls_name)
        all_matches.extend(matches)
    
    # 计算指标
    matched_gt = set()
    matched_pred = set()
    total_chamfer = 0.0
    good_matches = 0
    
    for gt_idx, pred_idx, dist in all_matches:
        matched_gt.add(gt_idx)
        matched_pred.add(pred_idx)
        total_chamfer += dist
        if dist < MATCH_THRESHOLD:
            good_matches += 1
    
    # Recall: 有多少 GT 被匹配到
    metrics['recall'] = len(matched_gt) / len(gt_instances) if gt_instances else 0
    
    # Precision: 有多少预测是有效的
    metrics['precision'] = len(matched_pred) / len(pred_instances) if pred_instances else 0
    
    # F1 Score
    if metrics['recall'] + metrics['precision'] > 0:
        metrics['f1_score'] = 2 * metrics['recall'] * metrics['precision'] / (metrics['recall'] + metrics['precision'])
    
    # 平均 Chamfer Error
    if all_matches:
        metrics['chamfer_error'] = total_chamfer / len(all_matches)
    
    return metrics


def setup_distributed():
    """Setup distributed training."""
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
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    tokenizer,
    epoch: int,
    args,
    rank: int,
    scaler=None,
):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    # 梯度监控变量
    grad_norms = {'map_queries': [], 'qformer': [], 'crossattn': [], 'lora': []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        stage1_prompts = batch['stage1_prompts']
        stage2_prompts = batch['stage2_prompts']
        gt_texts = batch['gt_texts']
        
        # Tokenize
        stage1_tok = tokenizer(stage1_prompts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        stage2_tok = tokenizer(stage2_prompts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        gt_tok = tokenizer(gt_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        stage1_ids = stage1_tok.input_ids.cuda()
        stage1_mask = stage1_tok.attention_mask.cuda()
        stage2_ids = stage2_tok.input_ids.cuda()
        stage2_mask = stage2_tok.attention_mask.cuda()
        gt_ids = gt_tok.input_ids.cuda()
        gt_mask = gt_tok.attention_mask.cuda()
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                stage1_prompt_ids=stage1_ids,
                stage1_prompt_mask=stage1_mask,
                stage2_prompt_ids=stage2_ids,
                stage2_prompt_mask=stage2_mask,
                gt_ids=gt_ids,
                gt_mask=gt_mask,
            )
            loss = outputs['loss'] / args.accumulation_steps
        
        # Backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # ===== 梯度监控 =====
        if (step + 1) % args.accumulation_steps == 0 and rank == 0:
            base_model = model.module if hasattr(model, 'module') else model
            
            # 监控 map_queries 梯度
            map_query_grad_norm = 0.0
            for name, param in base_model.map_queries.named_parameters():
                if param.grad is not None:
                    map_query_grad_norm += param.grad.norm().item() ** 2
            map_query_grad_norm = map_query_grad_norm ** 0.5
            grad_norms['map_queries'].append(map_query_grad_norm)
            
            # 监控 Q-Former 梯度
            qformer_grad_norm = 0.0
            for name, param in base_model.qformer.named_parameters():
                if param.grad is not None:
                    qformer_grad_norm += param.grad.norm().item() ** 2
            qformer_grad_norm = qformer_grad_norm ** 0.5
            grad_norms['qformer'].append(qformer_grad_norm)
            
            # 监控 Cross-Attention 梯度 【新增】
            crossattn_grad_norm = 0.0
            for name, param in base_model.map_scene_interaction.named_parameters():
                if param.grad is not None:
                    crossattn_grad_norm += param.grad.norm().item() ** 2
            crossattn_grad_norm = crossattn_grad_norm ** 0.5
            grad_norms['crossattn'].append(crossattn_grad_norm)
            
            # 监控 LoRA 梯度
            lora_grad_norm = 0.0
            for name, param in base_model.llm.named_parameters():
                if param.grad is not None and 'lora' in name.lower():
                    lora_grad_norm += param.grad.norm().item() ** 2
            lora_grad_norm = lora_grad_norm ** 0.5
            grad_norms['lora'].append(lora_grad_norm)
            
            # 每 50 步打印一次梯度信息
            if (step + 1) % (args.accumulation_steps * 50) == 0:
                print(f"\n[梯度监控] Step {step+1}:")
                print(f"  Map Queries 梯度范数:   {map_query_grad_norm:.6f}")
                print(f"  Q-Former 梯度范数:      {qformer_grad_norm:.6f}")
                print(f"  Cross-Attn 梯度范数:    {crossattn_grad_norm:.6f}")  # 新增
                print(f"  LoRA 梯度范数:          {lora_grad_norm:.6f}")
        
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
    
    # Epoch 结束时打印梯度统计
    if rank == 0 and grad_norms['map_queries']:
        import statistics
        print(f"\n{'='*50}")
        print(f"[Epoch {epoch+1}] 梯度统计:")
        print(f"  Map Queries - 平均: {statistics.mean(grad_norms['map_queries']):.6f}")
        print(f"  Q-Former    - 平均: {statistics.mean(grad_norms['qformer']):.6f}")
        print(f"  Cross-Attn  - 平均: {statistics.mean(grad_norms['crossattn']):.6f}")  # 新增
        print(f"  LoRA        - 平均: {statistics.mean(grad_norms['lora']):.6f}")
        print(f"{'='*50}")
    
    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    epoch: int,
    args,
    rank: int,
):
    """Validate and show sample outputs."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 【改进版】评估指标
    all_metrics = {
        'format_correct': 0.0,
        'recall': 0.0,
        'precision': 0.0,
        'chamfer_error': 0.0,
        'f1_score': 0.0,
    }
    num_evaluated = 0
    sample_outputs = []
    
    for step, batch in enumerate(dataloader):
        images = batch['images'].cuda()
        stage1_prompts = batch['stage1_prompts']
        stage2_prompts = batch['stage2_prompts']
        gt_texts = batch['gt_texts']
        
        # Tokenize
        stage1_tok = tokenizer(stage1_prompts, padding=True, truncation=True, max_length=256, return_tensors='pt')
        stage2_tok = tokenizer(stage2_prompts, padding=True, truncation=True, max_length=64, return_tensors='pt')
        gt_tok = tokenizer(gt_texts, padding=True, truncation=True, max_length=512, return_tensors='pt')
        
        stage1_ids = stage1_tok.input_ids.cuda()
        stage1_mask = stage1_tok.attention_mask.cuda()
        stage2_ids = stage2_tok.input_ids.cuda()
        stage2_mask = stage2_tok.attention_mask.cuda()
        gt_ids = gt_tok.input_ids.cuda()
        gt_mask = gt_tok.attention_mask.cuda()
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                stage1_prompt_ids=stage1_ids,
                stage1_prompt_mask=stage1_mask,
                stage2_prompt_ids=stage2_ids,
                stage2_prompt_mask=stage2_mask,
                gt_ids=gt_ids,
                gt_mask=gt_mask,
            )
            loss = outputs['loss']
        
        total_loss += loss.item()
        num_batches += 1
        
        # 生成评估
        if step < 3 and rank == 0:
            for i in range(min(len(images), 1)):
                base_model = model.module if hasattr(model, 'module') else model
                generated = base_model.generate(
                    images[i], stage1_prompts[i], stage2_prompts[i]
                )
                
                metrics = evaluate_text_output(gt_texts[i], generated)
                for k in all_metrics:
                    all_metrics[k] += metrics[k]
                num_evaluated += 1
                
                if step == 0 and i == 0:
                    sample_outputs.append({
                        'gt': gt_texts[i][:200],
                        'pred': generated[:200],
                    })
    
    avg_loss = total_loss / max(num_batches, 1)
    
    if rank == 0:
        print(f"\n[Epoch {epoch+1}] Validation Loss: {avg_loss:.4f}")
        
        if num_evaluated > 0:
            print(f"\n评估指标 (基于 {num_evaluated} 个样本):")
            for k, v in all_metrics.items():
                print(f"  {k}: {v/num_evaluated:.4f}")
        
        if sample_outputs:
            print(f"\n样本输出:")
            print(f"  GT:   {sample_outputs[0]['gt']}")
            print(f"  Pred: {sample_outputs[0]['pred']}")
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache-train', type=str, required=True)
    parser.add_argument('--gt-cache-val', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.15)
    
    # Model
    parser.add_argument('--llm-path', type=str, required=True)
    
    # Cross-Attention 配置
    parser.add_argument('--crossattn-layers', type=int, default=3)
    parser.add_argument('--crossattn-embed-dim', type=int, default=256)
    parser.add_argument('--crossattn-heads', type=int, default=8)
    
    # LoRA
    parser.add_argument('--use-lora', action='store_true')
    parser.add_argument('--no-lora', dest='use_lora', action='store_false')
    parser.add_argument('--lora-r', type=int, default=32)
    parser.add_argument('--lora-alpha', type=int, default=64)
    parser.add_argument('--lora-dropout', type=float, default=0.05)
    
    # Training
    parser.add_argument('--epochs', type=int, default=24)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Learning rates
    parser.add_argument('--lr-qformer-backbone', type=float, default=3e-5)
    parser.add_argument('--lr-qformer-decoder', type=float, default=2e-4)
    parser.add_argument('--lr-qformer-projector', type=float, default=2.5e-4)
    parser.add_argument('--lr-map-queries', type=float, default=2e-3)
    parser.add_argument('--lr-crossattn', type=float, default=2e-4)  # 新增
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    
    # Optimization
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--fp16', action='store_true')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs/two_stage_crossattn')
    parser.add_argument('--log-interval', type=int, default=20)
    
    parser.set_defaults(use_lora=True)
    args = parser.parse_args()
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("两阶段验证实验 + Cross-Attention")
        print(f"{'='*60}")
        print(f"验证目标: Map-Scene Interaction Layer 是否有帮助")
        print(f"Cross-Attention: {args.crossattn_layers} 层, dim={args.crossattn_embed_dim}")
        print(f"阶段 1: 与原始检测头一致的信息提取")
        print(f"【新增】: map_features + scene_tokens → Cross-Attention")
        print(f"阶段 2: 从 enhanced_map_features 生成文字")
        print(f"{'='*60}\n")
    
    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # 不再需要 CLIP processor，使用与主训练一致的 ImageNet 预处理
    if rank == 0:
        print("✅ 使用 ImageNet 预处理 (与主训练一致)")
    
    # Model
    model = TwoStageCrossAttnModel(
        llm_path=args.llm_path,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        crossattn_layers=args.crossattn_layers,
        crossattn_embed_dim=args.crossattn_embed_dim,
        crossattn_heads=args.crossattn_heads,
    )
    model = model.cuda()
    
    tokenizer = model.tokenizer
    
    # DDP
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        model._set_static_graph()
        if rank == 0:
            print("✅ DDP static graph enabled")
    
    # Dataset (不再需要 image_processor)
    train_dataset = MapTextDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache_train,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    val_dataset = MapTextDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache_val,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    # Sampler and DataLoader
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
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
        sampler=val_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    
    # Optimizer
    base_model = model.module if hasattr(model, 'module') else model
    
    qformer_backbone_params = []
    qformer_decoder_params = []
    qformer_projector_params = []
    map_queries_params = []
    crossattn_params = []  # 新增
    lora_params = []
    
    for name, param in base_model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'qformer' in name:
            if 'img_backbone' in name or 'img_neck' in name:
                qformer_backbone_params.append(param)
            elif 'decoder' in name:
                qformer_decoder_params.append(param)
            elif 'projector' in name:
                qformer_projector_params.append(param)
            else:
                qformer_decoder_params.append(param)
        elif 'map_queries' in name:
            map_queries_params.append(param)
        elif 'map_scene_interaction' in name:  # 新增
            crossattn_params.append(param)
        elif 'lora' in name.lower():
            lora_params.append(param)
    
    param_groups = []
    if qformer_backbone_params:
        param_groups.append({'params': qformer_backbone_params, 'lr': args.lr_qformer_backbone, 'name': 'qformer_backbone'})
    if qformer_decoder_params:
        param_groups.append({'params': qformer_decoder_params, 'lr': args.lr_qformer_decoder, 'name': 'qformer_decoder'})
    if qformer_projector_params:
        param_groups.append({'params': qformer_projector_params, 'lr': args.lr_qformer_projector, 'name': 'qformer_projector'})
    if map_queries_params:
        param_groups.append({'params': map_queries_params, 'lr': args.lr_map_queries, 'name': 'map_queries'})
    if crossattn_params:  # 新增
        param_groups.append({'params': crossattn_params, 'lr': args.lr_crossattn, 'name': 'crossattn'})
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
    
    from transformers import get_linear_schedule_with_warmup
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Scaler
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            tokenizer, epoch, args, rank, scaler
        )
        
        val_loss = validate(model, val_loader, tokenizer, epoch, args, rank)
        
        if rank == 0:
            print(f"\n[Epoch {epoch+1}/{args.epochs}] Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # Save best
            if val_loss < best_loss:
                best_loss = val_loss
                save_path = os.path.join(args.output_dir, 'best_model.pt')
                save_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': save_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"✅ Best model saved: {save_path}")
    
    if rank == 0:
        print(f"\n{'='*60}")
        print("训练完成！")
        print(f"最佳验证 Loss: {best_loss:.4f}")
        print(f"输出目录: {args.output_dir}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
