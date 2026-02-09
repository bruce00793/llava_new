"""
LLM Text Output Training - 验证实验 (支持 LoRA 微调)
直接让LLM以文本形式输出地图元素坐标，验证视觉特征的有效性

目的：
1. 验证Q-Former提取的场景特征是否包含有效的空间信息
2. 验证LLM是否能理解视觉输入并生成有意义的坐标
3. 使用LoRA微调让LLM学习理解视觉tokens
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
    print("   Install with: pip install peft")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.qformer import QFormer, build_qformer
from llava.model.language_model.llava_map import LlavaMapDetectionModel
from llava.model.map_queries import MapInstancePointQueries


class MapTextDataset(Dataset):
    """
    Dataset that generates text-form GT for LLM training.
    
    简化格式：只输出起点和终点
    GT format example:
    "<map>
    [divider] start(0.5,0.3) end(0.5,0.7)
    [boundary] start(0.1,0.2) end(0.2,0.6)
    </map>"
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
        Convert GT to simplified text format for LLM training.
        
        简化格式：只输出起点和终点，降低任务难度
        
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
        # Normalize: x' = (x + 15) / 30, y' = (y + 30) / 60
        
        lines_by_class = {0: [], 1: [], 2: []}
        
        for i, (cls_id, points) in enumerate(zip(gt_classes, gt_points)):
            # Normalize coordinates
            x_norm = (points[:, 0] + 15) / 30  # [0, 1]
            y_norm = (points[:, 1] + 30) / 60  # [0, 1]
            
            # 只取起点(第0个点)和终点(第19个点)
            start_x, start_y = x_norm[0], y_norm[0]
            end_x, end_y = x_norm[-1], y_norm[-1]
            
            # Format as simplified string (1 decimal place for easier learning)
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
        
        # Create prompt - 简化格式：只输出起点和终点
        prompt = """You are a map element detection assistant. Based on the 6 camera images of a driving scene, detect three types of map elements:

1. **Divider**: Lane dividing lines (dashed/solid lines in road center)
2. **Pedestrian Crossing**: Crosswalk/zebra crossing areas
3. **Boundary**: Road edges (curbs, guardrails)

Output format:
- Each element on a new line: [class] start(x,y) end(x,y)
- Coordinates normalized to [0,1] with 1 decimal place
- Wrap output in <map> and </map> tags

Example:
<map>
[divider] start(0.5,0.3) end(0.5,0.7)
[boundary] start(0.1,0.2) end(0.2,0.6)
[ped_crossing] start(0.4,0.4) end(0.6,0.5)
</map>

Detect map elements:"""
        
        # 分离 prompt 和 gt_text（重要！gt_text 需要放在 map_queries 之后）
        # prompt 不包含 gt_text
        prompt_text = f"USER: <image>\n{prompt}\nASSISTANT:"
        
        return {
            'images': images,
            'prompt_text': prompt_text,  # 只包含 prompt，不含 gt
            'gt_text': gt_text,          # gt 单独返回
            'sample_token': sample_token,
        }


def collate_fn(batch):
    """Custom collate function."""
    images = torch.stack([item['images'] for item in batch], dim=0)
    prompt_texts = [item['prompt_text'] for item in batch]
    gt_texts = [item['gt_text'] for item in batch]
    sample_tokens = [item['sample_token'] for item in batch]
    
    return {
        'images': images,
        'prompt_texts': prompt_texts,  # prompts (不含 gt)
        'gt_texts': gt_texts,          # gt 单独
        'sample_tokens': sample_tokens,
    }


class LLaVAMapTextModel(nn.Module):
    """
    LLaVA model for text-based map output.
    
    Architecture:
    1. Q-Former: Extract scene tokens from 6 camera images
    2. LLM (with optional LoRA): Generate text-form map coordinates
    
    LoRA 微调说明：
    - LoRA 在 LLM 的 attention 层添加低秩适配器
    - 原始 LLM 权重冻结，只训练 LoRA 参数
    - 让 LLM 学习理解 Q-Former 输出的视觉 tokens
    """
    
    def __init__(
        self,
        llm_path: str,
        use_lora: bool = True,      # 是否使用 LoRA 微调
        lora_r: int = 16,           # LoRA rank
        lora_alpha: int = 32,       # LoRA scaling factor
        lora_dropout: float = 0.05, # LoRA dropout
        freeze_llm: bool = True,    # 如果不用 LoRA，是否冻结 LLM
    ):
        super().__init__()
        
        self.use_lora = use_lora and PEFT_AVAILABLE
        
        print("\n" + "="*60)
        print("Initializing LLaVA Map Text Model")
        if self.use_lora:
            print(f"  Mode: LoRA 微调 (r={lora_r}, alpha={lora_alpha})")
        else:
            print(f"  Mode: {'LLM 冻结' if freeze_llm else 'LLM 部分训练'}")
        print("="*60)
        
        # 1. Load LLM
        print(f"\nLoading LLM: {llm_path}")
        self.llm = LlavaMapDetectionModel.from_pretrained(
            llm_path,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        )
        print("✅ LLM loaded")
        
        # 2. Initialize Q-Former (使用与检测头训练相同的配置)
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
        print("✅ Q-Former initialized")
        
        # 3. Initialize 1050 Map Queries (与检测头训练相同)
        print("\nInitializing Map Queries (1050 = 50 instances × 21)...")
        self.map_queries = MapInstancePointQueries(
            num_instances=50,
            num_points=20,
            embed_dim=4096,  # 匹配 LLM hidden size
        )
        self.map_queries = self.map_queries.cuda()
        print("✅ Map Queries initialized (1050 queries)")
        
        # 4. Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False)
        
        # Add special tokens
        special_tokens = ['<map>', '</map>', '[divider]', '[ped_crossing]', '[boundary]']
        self.tokenizer.add_special_tokens({'additional_special_tokens': special_tokens})
        self.llm.resize_token_embeddings(len(self.tokenizer))
        print(f"✅ Added {len(special_tokens)} special tokens")
        
        # 4. 配置 LLM 训练策略
        if self.use_lora:
            # 使用 LoRA 微调
            print("\nConfiguring LoRA...")
            
            # 先冻结所有 LLM 参数
            for param in self.llm.parameters():
                param.requires_grad = False
            
            # 配置 LoRA
            # 目标模块：LLaMA 的 attention 层（完整微调）
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=[
                    "q_proj", "k_proj", "v_proj", "o_proj",  # 完整 Attention 层
                    # "gate_proj", "up_proj", "down_proj",  # 可选：FFN 层（如果需要更强适应能力）
                ],
                lora_dropout=lora_dropout,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            
            # 应用 LoRA
            self.llm = get_peft_model(self.llm, lora_config)
            
            # 打印 LoRA 参数信息
            trainable_params = sum(p.numel() for p in self.llm.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.llm.parameters())
            print(f"✅ LoRA configured")
            print(f"   Trainable params: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
            print(f"   Total LLM params: {total_params:,}")
            
        elif freeze_llm:
            # 完全冻结 LLM
            for param in self.llm.parameters():
                param.requires_grad = False
            print("✅ LLM frozen (no LoRA)")
        else:
            # 部分训练 LLM
            for name, param in self.llm.named_parameters():
                if 'lm_head' in name or 'embed_tokens' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
            print("✅ LLM: only lm_head and embed_tokens trainable")
        
        # Q-Former is always trainable
        for param in self.qformer.parameters():
            param.requires_grad = True
        
        # Map Queries is always trainable
        for param in self.map_queries.parameters():
            param.requires_grad = True
        
        print("="*60 + "\n")
    
    def forward(
        self,
        images: torch.Tensor,
        prompt_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        gt_ids: torch.Tensor,
        gt_attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with 1050 map queries.
        
        关键修改：gt_text 放在 map_queries 之后！
        
        拼接顺序：
        [scene_tokens (512)] + [prompt (L_p)] + [map_queries (1050)] + [gt_text (L_g)]
        
        信息流：
        - scene_tokens: 视觉特征
        - prompt: 任务指令
        - map_queries: 从 scene 中提取目标信息（可以看到 scene + prompt）
        - gt_text: 利用 map_queries 的信息生成输出（可以看到所有前面的内容）
        
        这样 gt_text 生成时可以看到 map_queries！
        
        Args:
            images: [B, 6, 3, H, W] - 6 camera images
            prompt_ids: [B, L_p] - tokenized prompt (不含 gt)
            prompt_attention_mask: [B, L_p]
            gt_ids: [B, L_g] - tokenized gt_text
            gt_attention_mask: [B, L_g]
        
        Returns:
            loss: language modeling loss
        """
        B = images.shape[0]
        device = images.device
        
        # 1. Extract scene tokens from Q-Former
        scene_tokens = self.qformer(images)  # [B, 512, 4096]
        
        # 2. Get prompt embeddings (不含 gt)
        prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # [B, L_p, 4096]
        L_p = prompt_embeds.shape[1]
        
        # 3. Get gt embeddings (单独)
        gt_embeds = self.llm.get_input_embeddings()(gt_ids)  # [B, L_g, 4096]
        L_g = gt_embeds.shape[1]
        
        # 4. Get 1050 map queries
        map_query_embeds = self.map_queries(B)  # [B, 1050, 4096]
        map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
        
        # 5. 拼接顺序（关键！）:
        #    [scene_tokens (512)] + [prompt (L_p)] + [map_queries (1050)] + [gt_text (L_g)]
        #    
        #    这样 gt_text 在 map_queries 之后，可以看到 map_queries！
        inputs_embeds = torch.cat([scene_tokens, prompt_embeds, map_query_embeds, gt_embeds], dim=1)
        # Shape: [B, 512 + L_p + 1050 + L_g, 4096]
        
        # 6. Adjust attention mask
        scene_attn = torch.ones(B, 512, device=device, dtype=prompt_attention_mask.dtype)
        query_attn = torch.ones(B, 1050, device=device, dtype=prompt_attention_mask.dtype)
        full_attention_mask = torch.cat([scene_attn, prompt_attention_mask, query_attn, gt_attention_mask], dim=1)
        
        # 7. Adjust labels
        # scene_tokens: -100 (不计算 loss)
        # prompt: -100 (不计算 loss)
        # map_queries: -100 (不计算 loss)
        # gt_text: 真实 token ids (计算 loss)
        scene_labels = torch.full((B, 512), -100, device=device, dtype=gt_ids.dtype)
        prompt_labels = torch.full((B, L_p), -100, device=device, dtype=gt_ids.dtype)
        query_labels = torch.full((B, 1050), -100, device=device, dtype=gt_ids.dtype)
        gt_labels = gt_ids.clone()  # gt 部分计算 loss
        
        full_labels = torch.cat([scene_labels, prompt_labels, query_labels, gt_labels], dim=1)
        
        # 8. Forward through LLM
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
        max_new_tokens: int = 256,
    ) -> str:
        """
        Generate text output for given images with 1050 map queries.
        使用手动解码循环，因为 LLM 的 generate 不支持 inputs_embeds。
        
        生成时的架构：
        [scene_tokens (512)] + [prompt] + [map_queries (1050)] + [生成的 token...]
        
        生成的 token 在 map_queries 之后，可以看到 map_queries！
        """
        device = images.device
        
        try:
            # 1. Extract scene tokens (Q-Former outputs FP32)
            scene_tokens = self.qformer(images.unsqueeze(0))  # [1, 512, 4096]
            scene_tokens = scene_tokens.half()  # Convert to FP16
            
            # 2. Tokenize prompt (不含 gt)
            prompt_ids = self.tokenizer(prompt, return_tensors='pt').input_ids.to(device)
            prompt_embeds = self.llm.get_input_embeddings()(prompt_ids)  # Already FP16
            
            # 3. Get 1050 map queries
            map_query_embeds = self.map_queries(1)  # [1, 1050, 4096]
            map_query_embeds = map_query_embeds.to(device=device, dtype=scene_tokens.dtype)
            
            # 4. Combine: scene_tokens + prompt + map_queries
            #    生成的 token 会在 map_queries 之后，可以看到 map_queries！
            inputs_embeds = torch.cat([scene_tokens, prompt_embeds, map_query_embeds], dim=1)
            # Shape: [1, 512 + L_p + 1050, 4096]
            
            # 手动解码循环 (greedy decoding)
            generated_ids = []
            past_key_values = None
            
            for step in range(max_new_tokens):
                if step == 0:
                    # 第一步：使用完整的 inputs_embeds
                    outputs = self.llm(
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        return_dict=True,
                    )
                else:
                    # 后续步骤：只输入最新生成的 token，利用 KV cache
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
                
                # Greedy: 取概率最高的 token
                next_token = outputs.logits[:, -1, :].argmax(dim=-1).item()
                generated_ids.append(next_token)
                
                # 停止条件
                if next_token == self.tokenizer.eos_token_id:
                    break
                # 每10步检查一次是否生成了 </map>（避免频繁解码）
                if step % 10 == 0 and step > 0:
                    decoded_so_far = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
                    if '</map>' in decoded_so_far:
                        break
            
            # Decode
            generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
            return generated_text
            
        except Exception as e:
            # 如果生成失败，返回错误信息而不是崩溃
            return f"[生成失败: {str(e)[:100]}]"


def evaluate_text_output(gt_text: str, pred_text: str) -> Dict[str, float]:
    """
    评估LLM输出的文本质量。
    
    评估指标：
    1. 格式正确率：输出是否包含 <map>...</map> 结构
    2. 类别检测率：正确识别的类别比例
    3. 实例数量差异：预测实例数与GT实例数的差异
    4. 坐标误差（如果格式正确）
    """
    metrics = {
        'format_correct': 0.0,
        'class_accuracy': 0.0,
        'instance_count_diff': 0.0,
        'coord_error': 0.0,
    }
    
    # 1. 检查格式
    map_pattern = r'<map>(.*?)</map>'
    gt_match = re.search(map_pattern, gt_text, re.DOTALL)
    pred_match = re.search(map_pattern, pred_text, re.DOTALL)
    
    if pred_match:
        metrics['format_correct'] = 1.0
    else:
        # 格式不正确，返回基础指标
        return metrics
    
    # 2. 解析实例
    instance_pattern = r'\[(divider|ped_crossing|boundary)\]\s*start\(([\d.]+),([\d.]+)\)\s*end\(([\d.]+),([\d.]+)\)'
    
    gt_instances = re.findall(instance_pattern, gt_text)
    pred_instances = re.findall(instance_pattern, pred_text)
    
    # 3. 类别统计
    gt_classes = set(inst[0] for inst in gt_instances)
    pred_classes = set(inst[0] for inst in pred_instances)
    
    if gt_classes:
        correct_classes = gt_classes.intersection(pred_classes)
        metrics['class_accuracy'] = len(correct_classes) / len(gt_classes)
    
    # 4. 实例数量差异
    gt_count = len(gt_instances)
    pred_count = len(pred_instances)
    if gt_count > 0:
        metrics['instance_count_diff'] = abs(pred_count - gt_count) / gt_count
    
    # 5. 坐标误差（简化：按类别匹配最近的实例）
    if pred_instances and gt_instances:
        total_error = 0.0
        matched = 0
        for gt_inst in gt_instances:
            gt_cls = gt_inst[0]
            gt_coords = [float(gt_inst[i]) for i in range(1, 5)]
            
            # 找同类别的预测实例
            same_class_preds = [p for p in pred_instances if p[0] == gt_cls]
            if same_class_preds:
                # 找最近的
                min_error = float('inf')
                for pred_inst in same_class_preds:
                    pred_coords = [float(pred_inst[i]) for i in range(1, 5)]
                    error = sum(abs(g - p) for g, p in zip(gt_coords, pred_coords)) / 4
                    min_error = min(min_error, error)
                total_error += min_error
                matched += 1
        
        if matched > 0:
            metrics['coord_error'] = total_error / matched
    
    return metrics


def setup_distributed():
    """Setup distributed training."""
    if 'RANK' in os.environ:
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
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}", disable=(rank != 0))
    
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        prompt_texts = batch['prompt_texts']  # prompts (不含 gt)
        gt_texts = batch['gt_texts']          # gt 单独
        
        # Tokenize prompts (不含 gt)
        prompt_tokenized = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=512,  # prompt 通常不会太长
            return_tensors='pt',
        )
        prompt_ids = prompt_tokenized.input_ids.cuda()
        prompt_attention_mask = prompt_tokenized.attention_mask.cuda()
        
        # Tokenize gt_texts (单独)
        gt_tokenized = tokenizer(
            gt_texts,
            padding=True,
            truncation=True,
            max_length=512,  # gt 也不会太长
            return_tensors='pt',
        )
        gt_ids = gt_tokenized.input_ids.cuda()
        gt_attention_mask = gt_tokenized.attention_mask.cuda()
        
        # Forward (新的拼接顺序: scene + prompt + map_queries + gt)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                gt_ids=gt_ids,
                gt_attention_mask=gt_attention_mask,
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
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    tokenizer,
    epoch: int,
    args,
    rank: int,
):
    """Validate and show sample outputs with metrics."""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    # 评估指标累积
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
        prompt_texts = batch['prompt_texts']  # prompts (不含 gt)
        gt_texts = batch['gt_texts']          # gt 单独
        
        # Tokenize prompts
        prompt_tokenized = tokenizer(
            prompt_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        prompt_ids = prompt_tokenized.input_ids.cuda()
        prompt_attention_mask = prompt_tokenized.attention_mask.cuda()
        
        # Tokenize gt_texts
        gt_tokenized = tokenizer(
            gt_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors='pt',
        )
        gt_ids = gt_tokenized.input_ids.cuda()
        gt_attention_mask = gt_tokenized.attention_mask.cuda()
        
        # Forward
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                prompt_ids=prompt_ids,
                prompt_attention_mask=prompt_attention_mask,
                gt_ids=gt_ids,
                gt_attention_mask=gt_attention_mask,
            )
            loss = outputs['loss']
        
        total_loss += loss.item()
        num_batches += 1
        
        # 对前几个 batch 进行生成和评估
        if step < 3 and rank == 0:
            for i in range(min(len(images), 1)):  # 每个batch评估1个样本
                # 生成时使用 prompt（不含 gt）
                base_model = model.module if hasattr(model, 'module') else model
                generated = base_model.generate(images[i], prompt_texts[i])
                
                # 计算评估指标
                metrics = evaluate_text_output(gt_texts[i], generated)
                for k in all_metrics:
                    all_metrics[k] += metrics[k]
                num_evaluated += 1
                
                # 保存第一个样本用于展示
                if step == 0 and i == 0:
                    sample_outputs.append({
                        'gt': gt_texts[i],
                        'pred': generated,
                        'metrics': metrics,
                    })
    
    avg_loss = total_loss / max(num_batches, 1)
    
    # 计算平均指标
    if num_evaluated > 0:
        for k in all_metrics:
            all_metrics[k] /= num_evaluated
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Validation Epoch {epoch+1}")
        print(f"{'='*60}")
        print(f"Average Loss: {avg_loss:.4f}")
        
        print(f"\n📊 Evaluation Metrics (based on {num_evaluated} samples):")
        print(f"  Format Correct:     {all_metrics['format_correct']*100:.1f}%")
        print(f"  Class Accuracy:     {all_metrics['class_accuracy']*100:.1f}%")
        print(f"  Instance Count Diff: {all_metrics['instance_count_diff']:.2f}")
        print(f"  Coord Error (avg):  {all_metrics['coord_error']:.3f}")
        
        if sample_outputs:
            print(f"\n📝 Sample Output:")
            print(f"GT:\n{sample_outputs[0]['gt']}")
            print(f"\nPredicted:\n{sample_outputs[0]['pred'][:800]}")
        print(f"{'='*60}\n")
    
    return avg_loss, all_metrics


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache-train', type=str, required=True)
    parser.add_argument('--gt-cache-val', type=str, required=True)
    parser.add_argument('--sample-ratio', type=float, default=0.15,
                        help='Ratio of training data to use')
    
    # Model
    parser.add_argument('--llm-path', type=str, required=True)
    
    # LoRA 配置
    parser.add_argument('--use-lora', action='store_true', default=True,
                        help='Use LoRA to fine-tune LLM (recommended)')
    parser.add_argument('--no-lora', action='store_true',
                        help='Disable LoRA, freeze LLM completely')
    parser.add_argument('--lora-r', type=int, default=32,
                        help='LoRA rank (default: 32, 增大以提高适应能力)')
    parser.add_argument('--lora-alpha', type=int, default=64,
                        help='LoRA alpha/scaling (default: 64, = 2*r)')
    parser.add_argument('--lora-dropout', type=float, default=0.05,
                        help='LoRA dropout (default: 0.05)')
    parser.add_argument('--lr-lora', type=float, default=2e-4,
                        help='Learning rate for LoRA parameters (default: 2e-4, 加速学习)')
    
    # Training
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--accumulation-steps', type=int, default=5)
    parser.add_argument('--num-workers', type=int, default=4)
    
    # Learning rates (Q-Former)
    parser.add_argument('--lr-qformer-backbone', type=float, default=3e-5)
    parser.add_argument('--lr-qformer-decoder', type=float, default=2e-4)
    parser.add_argument('--lr-qformer-projector', type=float, default=2.5e-4)
    
    # Learning rate (Map Queries) - 与检测头训练一致
    parser.add_argument('--lr-map-queries', type=float, default=1e-4,
                        help='Learning rate for 1050 map queries (default: 1e-4)')
    
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
    
    # 处理 LoRA 参数
    use_lora = args.use_lora and not args.no_lora
    
    if rank == 0:
        print("\n" + "="*60)
        print("LLM Text Output Training")
        print("="*60)
        print(f"Data: {args.dataroot}")
        print(f"Sample Ratio: {args.sample_ratio*100:.0f}%")
        print(f"LLM: {args.llm_path}")
        print(f"Epochs: {args.epochs}")
        print(f"Batch Size: {args.batch_size} x {args.accumulation_steps} x {world_size}")
        if use_lora:
            print(f"LoRA: ENABLED (r={args.lora_r}, alpha={args.lora_alpha}, lr={args.lr_lora})")
        else:
            print(f"LoRA: DISABLED (LLM frozen)")
        print("="*60 + "\n")
    
    # Model
    model = LLaVAMapTextModel(
        llm_path=args.llm_path,
        use_lora=use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        freeze_llm=True,  # 如果不用 LoRA，则冻结 LLM
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
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    
    local_clip_path = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
    if os.path.exists(local_clip_path):
        image_processor = CLIPImageProcessor.from_pretrained(local_clip_path)
    else:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    # Datasets
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
        sample_ratio=0.1,  # Use 10% of val for quick validation
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
    
    # Optimizer with different learning rates
    base_model = model.module if hasattr(model, 'module') else model
    
    qformer_backbone_params = []
    qformer_decoder_params = []
    qformer_projector_params = []
    map_queries_params = []  # 1050 Map Queries 参数
    lora_params = []  # LoRA 参数
    
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
        elif 'map_queries' in name:
            # 1050 Map Queries 参数
            map_queries_params.append(param)
        elif 'lora' in name.lower():
            # LoRA 参数
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
    if lora_params:
        param_groups.append({'params': lora_params, 'lr': args.lr_lora, 'name': 'lora'})
    
    if rank == 0:
        print("\n" + "="*60)
        print("Parameter Groups:")
        for group in param_groups:
            num_params = sum(p.numel() for p in group['params'])
            print(f"  {group['name']:20s}: {num_params:,} params, lr={group['lr']:.1e}")
        total_trainable = sum(sum(p.numel() for p in g['params']) for g in param_groups)
        print(f"  {'TOTAL':20s}: {total_trainable:,} params")
        print("="*60 + "\n")
    
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.epochs // args.accumulation_steps
    from transformers import get_cosine_schedule_with_warmup
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_steps,
    )
    
    # Scaler for FP16 (只用于Q-Former，LLM已冻结)
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training loop
    best_val_loss = float('inf')
    best_metrics = None
    
    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        # Train
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler,
            tokenizer, epoch, args, rank, scaler
        )
        
        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}")
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, tokenizer, epoch, args, rank)
        
        # Save best (基于 format_correct 和 class_accuracy 的综合评分)
        combined_score = val_metrics['format_correct'] * 0.5 + val_metrics['class_accuracy'] * 0.5
        if rank == 0:
            if val_loss < best_val_loss or combined_score > 0.3:
                best_val_loss = val_loss
                best_metrics = val_metrics
                base_model = model.module if hasattr(model, 'module') else model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': base_model.state_dict(),
                    'val_loss': val_loss,
                    'metrics': val_metrics,
                }, os.path.join(args.output_dir, 'best_model.pth'))
                print(f"💾 Best model saved (val_loss: {val_loss:.4f}, format: {val_metrics['format_correct']*100:.1f}%)")
    
    # Save final
    if rank == 0:
        base_model = model.module if hasattr(model, 'module') else model
        torch.save({
            'epoch': args.epochs - 1,
            'model_state_dict': base_model.state_dict(),
        }, os.path.join(args.output_dir, 'final_model.pth'))
        print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")
        if best_metrics:
            print(f"   Best metrics: format={best_metrics['format_correct']*100:.1f}%, class_acc={best_metrics['class_accuracy']*100:.1f}%")
    
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
