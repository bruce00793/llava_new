"""
两阶段验证实验 - 验证 Map Queries 的有效性

设计思路：
阶段 1: 与原始检测头完全一致的信息提取
    [scene_tokens] + [prompt] + [map_queries] → LLM (自定义掩码) → map_features

阶段 2: 从 map_features 生成文字（验证 map_queries 是否有效）
    [map_features] + [response_prompt] + [gt_text] → LLM → Loss
    
    关键: gt_text 只能看到 map_features，看不到 scene_tokens！

验证逻辑：
- 如果成功 → map_features 包含了三类元素信息 → map_queries 有效
- 如果失败 → map_queries 没能从 scene 中提取有效信息

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
from llava.model.language_model.llava_map import LlavaMapDetectionModel
from llava.model.map_queries import MapInstancePointQueries, MapAttentionMask


class MapTextDataset(Dataset):
    """
    Dataset for two-stage verification.
    """
    
    CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']
    
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
        
        # GT cache
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
    
    def _load_gt(self, sample_token: str) -> Dict:
        """Load GT from cache."""
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        with open(gt_file, 'rb') as f:
            return pickle.load(f)
    
    def _format_gt_as_text(self, gt_data: Dict) -> str:
        """
        Convert GT to text format.
        
        Output format:
        "<map>
        [divider] start(0.50,0.30) end(0.55,0.70)
        [boundary] start(0.12,0.25) end(0.22,0.55)
        </map>"
        """
        gt_classes = gt_data['gt_classes']
        gt_points = gt_data['gt_points']  # [N, 20, 2]
        
        # Normalize points to [0, 1] range
        # Original range: x in [-15, 15], y in [-30, 30]
        
        lines_by_class = {0: [], 1: [], 2: []}
        
        for i, (cls_id, points) in enumerate(zip(gt_classes, gt_points)):
            # Normalize coordinates
            x_norm = (points[:, 0] + 15) / 30  # [0, 1]
            y_norm = (points[:, 1] + 30) / 60  # [0, 1]
            
            # 取起点和终点
            start_x, start_y = x_norm[0], y_norm[0]
            end_x, end_y = x_norm[-1], y_norm[-1]
            
            # Format (1 decimal place)
            line_str = f"start({start_x:.1f},{start_y:.1f}) end({end_x:.1f},{end_y:.1f})"
            lines_by_class[cls_id].append(line_str)
        
        # Build output text
        output_lines = ["<map>"]
        for cls_id, cls_name in enumerate(self.CLASS_NAMES):
            if lines_by_class[cls_id]:
                for line_str in lines_by_class[cls_id]:
                    output_lines.append(f"[{cls_name}] {line_str}")
        output_lines.append("</map>")
        
        return '\n'.join(output_lines)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        
        # Load images
        images = self._load_images(sample_token)  # [6, 3, H, W]
        
        # Load and format GT
        gt_data = self._load_gt(sample_token)
        gt_text = self._format_gt_as_text(gt_data)
        
        # Stage 1 prompt (适中版本)
        stage1_prompt = """Analyze driving scene images for HD map construction.

Detect THREE types of map elements:
1. DIVIDER: Lane dividing lines (solid/dashed white lines, yellow center lines)
2. PED_CROSSING: Pedestrian crosswalks (zebra stripes, crosswalk markings)
3. BOUNDARY: Road boundaries (curbs, road edges, guardrails)

Extract spatial features for each detected element."""
        
        # Stage 2 response prompt (适中版本)
        stage2_prompt = """Output all detected map elements in this format:
<map>
[category] start(x,y) end(x,y)
</map>

Rules:
- category: divider, ped_crossing, or boundary
- start(x,y): Starting point in BEV coordinates
- end(x,y): Ending point in BEV coordinates
- Coordinates: normalized to [0,1], 1 decimal place

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


class TwoStageVerificationModel(nn.Module):
    """
    两阶段验证模型
    
    阶段 1: 与原始检测头完全一致的信息提取
    阶段 2: 从 map_features 生成文字
    """
    
    def __init__(
        self,
        llm_path: str,
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        print("\n" + "="*60)
        print("Initializing Two-Stage Verification Model")
        print("  阶段 1: 与原始检测头一致的信息提取")
        print("  阶段 2: 从 map_features 生成文字验证")
        if self.use_lora:
            print(f"  LoRA: r={lora_r}, alpha={lora_alpha}")
        print("="*60)
        
        # 1. Load LLM (使用 LlavaMapDetectionModel，与检测头一致)
        print(f"\nLoading LLM: {llm_path}")
        self.llm = LlavaMapDetectionModel.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("✅ LLM loaded")
        
        # 2. Initialize Q-Former
        print("\nInitializing Q-Former...")
        qformer_config = {
            'img_backbone': 'resnet50',
            'embed_dims': 256,
            'num_queries': 512,
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
        print("✅ Q-Former initialized (512 queries)")
        
        # 3. Initialize Map Queries (与检测头一致)
        print("\nInitializing Map Queries (1050 = 50 instances × 21)...")
        self.map_queries = MapInstancePointQueries(
            num_instances=50,
            num_points=20,
            embed_dim=4096,
        )
        self.map_queries = self.map_queries.cuda()
        print("✅ Map Queries initialized")
        
        # 4. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        
        # Add special tokens
        special_tokens = ['<map>', '</map>', '[divider]', '[ped_crossing]', '[boundary]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        print(f"✅ Added {len(special_tokens)} special tokens")
        
        # 5. Enable Gradient Checkpointing (关键：节省显存！)
        print("\nEnabling Gradient Checkpointing to save memory...")
        try:
            if hasattr(self.llm, 'gradient_checkpointing_enable'):
                self.llm.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled")
            elif hasattr(self.llm.model, 'gradient_checkpointing_enable'):
                self.llm.model.gradient_checkpointing_enable()
                print("✅ Gradient checkpointing enabled (via .model)")
        except Exception as e:
            print(f"⚠️ Failed to enable gradient checkpointing: {e}")
        
        # 6. Configure LoRA
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
            
            # LoRA 包装后需要再次启用 gradient checkpointing
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
        
        # Q-Former and Map Queries always trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        for param in self.map_queries.parameters():
            param.requires_grad = True
        
        # Q-Former backbone gradient checkpointing
        try:
            if hasattr(self.qformer, 'img_backbone'):
                from torch.utils.checkpoint import checkpoint
                # ResNet50 gradient checkpointing
                if hasattr(self.qformer.img_backbone, 'set_grad_checkpointing'):
                    self.qformer.img_backbone.set_grad_checkpointing(True)
                    print("✅ Q-Former backbone gradient checkpointing enabled")
        except Exception as e:
            print(f"⚠️ Q-Former checkpoint not available: {e}")
        
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
        两阶段 Forward
        
        阶段 1: 与原始检测头完全一致
            [scene_tokens (512)] + [stage1_prompt] + [map_queries (1050)]
            使用自定义掩码 MapAttentionMask
            提取 map_features
        
        阶段 2: 从 map_features 生成文字
            [map_features (1050)] + [stage2_prompt] + [gt_text]
            标准 causal attention
            关键: gt_text 只能看到 map_features，看不到 scene_tokens！
        """
        B = images.shape[0]
        device = images.device
        
        # ========== 阶段 1: 信息提取（与原始检测头一致）==========
        
        # 1.1 Extract scene tokens from Q-Former
        scene_tokens = self.qformer(images)  # [B, 512, 4096]
        
        # 1.2 Get stage1 prompt embeddings
        stage1_embeds = self.llm.get_input_embeddings()(stage1_prompt_ids)  # [B, L1, 4096]
        L1 = stage1_embeds.shape[1]
        
        # 1.3 Get map queries
        map_query_embeds = self.map_queries(B)  # [B, 1050, 4096]
        map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
        
        # 1.4 拼接 (与原始检测头一致)
        stage1_input = torch.cat([scene_tokens, stage1_embeds, map_query_embeds], dim=1)
        # Shape: [B, 512 + L1 + 1050, 4096]
        
        # 1.5 创建自定义掩码 (与原始检测头一致！)
        custom_mask = MapAttentionMask.create_mask(
            batch_size=B,
            text_len=L1,
            scene_len=512,
            num_instances=50,
            num_points=20,
            device=device,
            dtype=stage1_input.dtype,
        )
        
        # 转换 mask 格式：transformers 期望 additive mask
        # 原始: 1=attend, 0=mask
        # 目标: 0=attend, -inf=mask
        # 转换: mask_additive = (1 - mask) * -10000
        custom_mask = (1.0 - custom_mask) * -10000.0
        
        # 1.6 Forward through LLM (阶段 1)
        # 获取底层模型并确保 gradient checkpointing 生效
        if hasattr(self.llm, 'base_model'):
            base_llm = self.llm.base_model.model.model  # LoRA wrapped
        else:
            base_llm = self.llm.model.model
        
        # 确保使用 gradient checkpointing (use_cache=False 是必需的)
        stage1_outputs = base_llm(
            inputs_embeds=stage1_input,
            attention_mask=custom_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,  # 关键：gradient checkpointing 需要禁用 cache
        )
        
        # 1.7 提取 map_features
        hidden_states = stage1_outputs.last_hidden_state  # [B, 512+L1+1050, 4096]
        prefix_len = 512 + L1
        map_features = hidden_states[:, prefix_len:prefix_len+1050, :]  # [B, 1050, 4096]
        
        # ========== 阶段 2: 从 map_features 生成文字 ==========
        # 关键: gt_text 只能看到 map_features，看不到 scene_tokens！
        
        # 2.1 Get stage2 prompt embeddings
        stage2_embeds = self.llm.get_input_embeddings()(stage2_prompt_ids)  # [B, L2, 4096]
        L2 = stage2_embeds.shape[1]
        
        # 2.2 Get gt embeddings
        gt_embeds = self.llm.get_input_embeddings()(gt_ids)  # [B, Lg, 4096]
        Lg = gt_embeds.shape[1]
        
        # 2.3 拼接 (关键: 没有 scene_tokens!)
        stage2_input = torch.cat([map_features, stage2_embeds, gt_embeds], dim=1)
        # Shape: [B, 1050 + L2 + Lg, 4096]
        
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
            use_cache=False,  # 关键：gradient checkpointing 需要禁用 cache
        )
        
        return {
            'loss': stage2_outputs.loss,
            'logits': stage2_outputs.logits,
            'map_features': map_features.detach(),
        }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        stage1_prompt: str,
        stage2_prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        """
        生成文本输出
        """
        device = images.device
        
        try:
            # ===== 阶段 1: 提取 map_features =====
            scene_tokens = self.qformer(images.unsqueeze(0))  # [1, 512, 4096]
            scene_tokens = scene_tokens.half()
            
            stage1_ids = self.tokenizer(stage1_prompt, return_tensors='pt').input_ids.to(device)
            stage1_embeds = self.llm.get_input_embeddings()(stage1_ids)
            L1 = stage1_embeds.shape[1]
            
            map_query_embeds = self.map_queries(1)
            map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
            
            stage1_input = torch.cat([scene_tokens, stage1_embeds, map_query_embeds], dim=1)
            
            custom_mask = MapAttentionMask.create_mask(
                batch_size=1,
                text_len=L1,
                scene_len=512,
                num_instances=50,
                num_points=20,
                device=device,
                dtype=stage1_input.dtype,
            )
            
            # 转换 mask 格式：transformers 期望 additive mask
            custom_mask = (1.0 - custom_mask) * -10000.0
            
            if hasattr(self.llm, 'base_model'):
                base_llm = self.llm.base_model.model.model
            else:
                base_llm = self.llm.model.model
            
            stage1_outputs = base_llm(
                inputs_embeds=stage1_input,
                attention_mask=custom_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            
            hidden_states = stage1_outputs.last_hidden_state
            prefix_len = 512 + L1
            map_features = hidden_states[:, prefix_len:prefix_len+1050, :]
            
            # ===== 阶段 2: 生成文字 =====
            stage2_ids = self.tokenizer(stage2_prompt, return_tensors='pt').input_ids.to(device)
            stage2_embeds = self.llm.get_input_embeddings()(stage2_ids)
            
            inputs_embeds = torch.cat([map_features, stage2_embeds], dim=1)
            
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
                if step % 10 == 0 and step > 0:
                    decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if '</map>' in decoded:
                        break
            
            return self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            
        except Exception as e:
            return f"[生成失败: {str(e)[:100]}]"


def evaluate_text_output(gt_text: str, pred_text: str) -> Dict[str, float]:
    """评估输出质量"""
    metrics = {
        'format_correct': 0.0,
        'class_accuracy': 0.0,
        'instance_count_diff': 0.0,
        'coord_error': 0.0,
    }
    
    map_pattern = r'<map>(.*?)</map>'
    gt_match = re.search(map_pattern, gt_text, re.DOTALL)
    pred_match = re.search(map_pattern, pred_text, re.DOTALL)
    
    if pred_match:
        metrics['format_correct'] = 1.0
    else:
        return metrics
    
    instance_pattern = r'\[(divider|ped_crossing|boundary)\]\s*start\(([\d.]+),([\d.]+)\)\s*end\(([\d.]+),([\d.]+)\)'
    
    gt_instances = re.findall(instance_pattern, gt_text)
    pred_instances = re.findall(instance_pattern, pred_text)
    
    gt_classes = set(inst[0] for inst in gt_instances)
    pred_classes = set(inst[0] for inst in pred_instances)
    
    if gt_classes:
        correct_classes = gt_classes.intersection(pred_classes)
        metrics['class_accuracy'] = len(correct_classes) / len(gt_classes)
    
    gt_count = len(gt_instances)
    pred_count = len(pred_instances)
    if gt_count > 0:
        metrics['instance_count_diff'] = abs(pred_count - gt_count) / gt_count
    
    if pred_instances and gt_instances:
        total_error = 0.0
        matched = 0
        for gt_inst in gt_instances:
            gt_cls = gt_inst[0]
            gt_coords = [float(gt_inst[i]) for i in range(1, 5)]
            
            same_class_preds = [p for p in pred_instances if p[0] == gt_cls]
            if same_class_preds:
                best_error = float('inf')
                for pred_inst in same_class_preds:
                    pred_coords = [float(pred_inst[i]) for i in range(1, 5)]
                    error = sum(abs(g - p) for g, p in zip(gt_coords, pred_coords)) / 4
                    best_error = min(best_error, error)
                total_error += best_error
                matched += 1
        
        if matched > 0:
            metrics['coord_error'] = total_error / matched
    
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
    grad_norms = {'map_queries': [], 'qformer': [], 'lora': []}
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        stage1_prompts = batch['stage1_prompts']
        stage2_prompts = batch['stage2_prompts']
        gt_texts = batch['gt_texts']
        
        # Tokenize stage1 prompts
        stage1_tok = tokenizer(
            stage1_prompts, padding=True, truncation=True,
            max_length=256, return_tensors='pt',
        )
        stage1_ids = stage1_tok.input_ids.cuda()
        stage1_mask = stage1_tok.attention_mask.cuda()
        
        # Tokenize stage2 prompts
        stage2_tok = tokenizer(
            stage2_prompts, padding=True, truncation=True,
            max_length=64, return_tensors='pt',
        )
        stage2_ids = stage2_tok.input_ids.cuda()
        stage2_mask = stage2_tok.attention_mask.cuda()
        
        # Tokenize gt_texts
        gt_tok = tokenizer(
            gt_texts, padding=True, truncation=True,
            max_length=512, return_tensors='pt',
        )
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
                print(f"  Map Queries 梯度范数: {map_query_grad_norm:.6f}")
                print(f"  Q-Former 梯度范数:    {qformer_grad_norm:.6f}")
                print(f"  LoRA 梯度范数:        {lora_grad_norm:.6f}")
                if map_query_grad_norm < 1e-6:
                    print(f"  ⚠️ 警告: Map Queries 梯度过小，可能存在梯度消失问题！")
        
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
        print(f"  Map Queries - 平均: {statistics.mean(grad_norms['map_queries']):.6f}, "
              f"最大: {max(grad_norms['map_queries']):.6f}, "
              f"最小: {min(grad_norms['map_queries']):.6f}")
        print(f"  Q-Former    - 平均: {statistics.mean(grad_norms['qformer']):.6f}, "
              f"最大: {max(grad_norms['qformer']):.6f}, "
              f"最小: {min(grad_norms['qformer']):.6f}")
        print(f"  LoRA        - 平均: {statistics.mean(grad_norms['lora']):.6f}, "
              f"最大: {max(grad_norms['lora']):.6f}, "
              f"最小: {min(grad_norms['lora']):.6f}")
        
        # 判断梯度是否正常
        avg_map_grad = statistics.mean(grad_norms['map_queries'])
        if avg_map_grad < 1e-5:
            print(f"  ⚠️ 警告: Map Queries 平均梯度很小 ({avg_map_grad:.6f})，可能学习不充分")
        elif avg_map_grad > 0.01:
            print(f"  ✅ Map Queries 梯度正常，模型正在学习")
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
    
    all_metrics = {
        'format_correct': 0.0,
        'class_accuracy': 0.0,
        'instance_count_diff': 0.0,
        'coord_error': 0.0,
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
    parser.add_argument('--lr-map-queries', type=float, default=2e-3)  # 增大以补偿梯度衰减
    parser.add_argument('--lr-lora', type=float, default=2e-4)
    
    # Optimization
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--fp16', action='store_true')
    
    # Output
    parser.add_argument('--output-dir', type=str, default='./outputs/two_stage_verification')
    parser.add_argument('--log-interval', type=int, default=20)
    
    parser.set_defaults(use_lora=True)
    args = parser.parse_args()
    
    # Setup
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        print(f"\n{'='*60}")
        print("两阶段验证实验")
        print(f"{'='*60}")
        print(f"验证目标: map_queries 能否从 scene_tokens 中提取三类元素信息")
        print(f"阶段 1: 与原始检测头一致的信息提取")
        print(f"阶段 2: 从 map_features 生成文字 (gt_text 看不到 scene_tokens)")
        print(f"{'='*60}\n")
    
    # Set seed
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Image processor (使用本地 336 分辨率版本)
    local_clip_path = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
    if os.path.exists(local_clip_path):
        image_processor = CLIPImageProcessor.from_pretrained(local_clip_path)
        if rank == 0:
            print(f"✅ 使用本地 CLIP processor: {local_clip_path}")
    else:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
        if rank == 0:
            print("✅ 使用在线 CLIP processor: openai/clip-vit-large-patch14-336")
    
    # Model
    model = TwoStageVerificationModel(
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
            model, device_ids=[local_rank], find_unused_parameters=True
        )
        # 关键：设置静态图，解决两阶段设计中参数被标记两次的问题
        model._set_static_graph()
        if rank == 0:
            print("✅ DDP static graph enabled for two-stage training")
    
    # Dataset
    train_dataset = MapTextDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache_train,
        image_processor=image_processor,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    val_dataset = MapTextDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache_val,
        image_processor=image_processor,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    # Sampler
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None
    
    # Dataloader
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
        elif 'lora' in name.lower():
            lora_params.append(param)
    
    param_groups = []
    if qformer_backbone_params:
        param_groups.append({'params': qformer_backbone_params, 'lr': args.lr_qformer_backbone})
    if qformer_decoder_params:
        param_groups.append({'params': qformer_decoder_params, 'lr': args.lr_qformer_decoder})
    if qformer_projector_params:
        param_groups.append({'params': qformer_projector_params, 'lr': args.lr_qformer_projector})
    if map_queries_params:
        param_groups.append({'params': map_queries_params, 'lr': args.lr_map_queries})
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lr_lora})
    
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
