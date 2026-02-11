"""
Training script for LLaVA Map Detection
Stage 2: Joint training with BLIP-2 pretrained Q-Former

Training strategy (Optimized):
- Q-Former Backbone: 5e-6 (freeze-like, minimal change)
- Q-Former Decoder: 1e-5 (fine-tune from BLIP-2)
- Q-Former Projector: 5e-5 (adapt to new task)
- Map Queries: 1e-4 (train from scratch)
- Map Decoder: 1e-4 (train from scratch)
- LLM Backbone: Frozen

Optimizations:
- Gradient Accumulation: effective batch_size = batch_size * accumulation_steps
- Fine-grained Parameter Groups: 5 groups with layer-wise learning rates
- EMA (Exponential Moving Average): smoother model for evaluation
- Optimized Learning Rates: based on pre-training status

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import os
import sys
import json
import copy
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
from transformers import AutoTokenizer, CLIPImageProcessor

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset
from llava.model.map_config import MapDetectionConfig
from llava.model.map_eval import MapEvaluator


class EMA:
    """
    Exponential Moving Average for model parameters.
    
    Maintains a shadow copy of model parameters that is updated with:
        shadow = decay * shadow + (1 - decay) * param
    
    This provides a smoother version of the model for evaluation.
    """
    
    def __init__(self, model, decay: float = 0.9999):
        """
        Args:
            model: The model to track
            decay: EMA decay rate (default: 0.9999)
        """
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """Update shadow parameters with current model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """Replace model parameters with shadow parameters for evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].clone()
    
    def restore(self):
        """Restore original model parameters after evaluation."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name].clone()
        self.backup = {}
    
    def state_dict(self):
        """Get state dict for checkpointing."""
        return {
            'decay': self.decay,
            'shadow': self.shadow,
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict from checkpoint."""
        self.decay = state_dict['decay']
        # Move shadow parameters to the same device as model parameters
        self.shadow = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in state_dict['shadow']:
                self.shadow[name] = state_dict['shadow'][name].to(param.device)


def parse_args():
    parser = argparse.ArgumentParser(description='Train LLaVA Map Detection')
    
    # Dataset
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset root')
    parser.add_argument('--version', type=str, default='v1.0-mini',
                        choices=['v1.0-mini', 'v1.0-trainval'],
                        help='nuScenes version')
    parser.add_argument('--gt-cache-train', type=str, default=None,
                        help='Path to train GT cache (if None, will auto-generate)')
    parser.add_argument('--gt-cache-val', type=str, default=None,
                        help='Path to val GT cache (if None, will auto-generate)')
    parser.add_argument('--subset-scenes', type=str, default=None,
                        help='Path to scene name list file for subset training (e.g., data/subset_15pct_scenes.txt)')
    
    # Model
    parser.add_argument('--llm-path', type=str, default='lmsys/vicuna-7b-v1.5',
                        help='Path to LLM checkpoint')
    parser.add_argument('--qformer-pretrained', type=str, default='blip2',
                        choices=['blip2', 'none'],
                        help='Q-Former pretrained weights (blip2 or none)')
    parser.add_argument('--qformer-version', type=str, default='v1',
                        choices=['v1', 'v2'],
                        help='Q-Former version: v1=原版(BLIP-2), v2=三阶段双流')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Training
    parser.add_argument('--epochs', type=int, default=24,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size per GPU（P3修复：降低到1以提高梯度累积步数）')
    parser.add_argument('--accumulation-steps', type=int, default=10,
                        help='Gradient accumulation steps（P3修复：10→effective_batch=60）')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Fine-grained learning rates（P1修复：降低学习率，稳定训练）
    # 
    # 【P1修复】学习率策略调整：
    # - 旧方案：lr_decoder=1e-4，loss≈280 → 单步更新过大 → 12000次梯度爆炸
    # - 新方案：lr_decoder=2e-5（降低5×），配合更严格的梯度裁剪
    # - 原因：loss 未归一化（MapTR loss≈1-2，我们≈200-300），需要更小的 lr
    parser.add_argument('--lr-qformer-backbone', type=float, default=2e-6,
                        help='Learning rate for Q-Former backbone (pre-trained, minimal change)')
    parser.add_argument('--lr-qformer-decoder', type=float, default=5e-6,
                        help='Learning rate for Q-Former decoder (pre-trained, fine-tune)')
    parser.add_argument('--lr-qformer-projector', type=float, default=3e-5,
                        help='Learning rate for Q-Former projector (adapt to new task)')
    parser.add_argument('--lr-queries', type=float, default=1e-4,
                        help='Learning rate for Map Queries (train from scratch)')
    parser.add_argument('--lr-decoder', type=float, default=5e-5,
                        help='Learning rate for Map Decoder excluding cls_head (train from scratch)')
    parser.add_argument('--lr-cls-head', type=float, default=5e-4,
                        help='Learning rate for Classification Head (独立高LR, MapTR级别)')
    parser.add_argument('--lr-lora', type=float, default=1e-4,
                        help='Learning rate for LoRA parameters (fine-tuning LLM attention)')
    parser.add_argument('--lr-scene-interaction', type=float, default=5e-5,
                        help='Learning rate for Map-Scene Interaction Layer (train from scratch)')
    
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay')
    parser.add_argument('--warmup-steps', type=int, default=500,
                        help='Number of warmup steps')
    parser.add_argument('--grad-clip', type=float, default=1.0,
                        help='Gradient clipping（P1修复：从5.0降低到1.0，更严格控制）')
    
    # EMA
    parser.add_argument('--use-ema', action='store_true', default=True,
                        help='Use Exponential Moving Average')
    parser.add_argument('--ema-decay', type=float, default=0.9999,
                        help='EMA decay rate')
    
    # Mixed precision
    parser.add_argument('--fp16', action='store_true',
                        help='Use FP16 mixed precision training (NOT recommended, use --bf16)')
    parser.add_argument('--bf16', action='store_true',
                        help='Use BF16 mixed precision training (RECOMMENDED for 4090/A100/H100)')
    
    # Debug
    parser.add_argument('--detect-anomaly', action='store_true',
                        help='Enable torch.autograd.detect_anomaly() to find NaN source (SLOW, debug only)')
    
    # Logging & Checkpointing
    parser.add_argument('--output-dir', type=str, default='./outputs/map_detection',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log every N steps')
    parser.add_argument('--save-interval', type=int, default=1,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='Evaluate every N epochs')
    
    # Distributed
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='Local rank for distributed training')
    
    args = parser.parse_args()
    return args


def setup_distributed():
    """Setup distributed training"""
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


def build_optimizer(model, args):
    """
    Build optimizer with fine-grained learning rates for different modules:
    
    Learning Rate Strategy:
    - Q-Former Backbone (ResNet):   5e-6  (minimal change, preserve features)
    - Q-Former Decoder:             1e-5  (fine-tune from BLIP-2)
    - Q-Former Projector:           5e-5  (adapt to new task)
    - Map Queries:                  1e-4  (train from scratch)
    - Map Decoder:                  1e-4  (train from scratch)
    - LoRA parameters:              2e-4  (LoRA fine-tuning, 新增!)
    """
    # Fine-grained parameter groups
    qformer_backbone_params = []  # img_backbone, img_neck
    qformer_decoder_params = []   # decoder layers
    qformer_projector_params = [] # projector to LLM dim
    qformer_other_params = []     # query_embed, camera_embed, position_encoding
    queries_params = []           # instance_queries, point_queries
    cls_head_params = []          # 【新增】分类头独立参数组（需要高 LR）
    decoder_params = []           # feature_reducer, points_head（不含 cls_head）
    scene_interaction_params = [] # map_scene_interaction module
    lora_params = []              # LoRA parameters
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        # Q-Former parameters
        if 'qformer' in name:
            if 'img_backbone' in name or 'img_neck' in name:
                qformer_backbone_params.append(param)
            elif 'decoder' in name:
                qformer_decoder_params.append(param)
            elif 'projector' in name:
                qformer_projector_params.append(param)
            else:
                # query_embed, camera_embed, position_encoding
                qformer_other_params.append(param)
        
        # Map Queries parameters
        elif 'map_queries' in name:
            queries_params.append(param)
        
        # Map Scene Interaction parameters (train from scratch)
        elif 'map_scene_interaction' in name:
            scene_interaction_params.append(param)
        
        # 【关键】分类头独立分组（需要 MapTR 级别高 LR）
        elif 'decoder' in name and 'cls_head' in name:
            cls_head_params.append(param)
        
        # Map Decoder parameters（不含 cls_head）
        elif 'decoder' in name:
            decoder_params.append(param)
        
        # LoRA parameters（必须在 'llm' 检查之前!）
        elif 'lora_' in name:
            lora_params.append(param)
        
        # Skip frozen LLM parameters (lm_head etc.)
        elif 'llm' in name:
            print(f"Warning: Unexpected trainable LLM parameter (not LoRA): {name}")
            continue
        
        else:
            print(f"Warning: Unexpected trainable parameter: {name}")
    
    # Combine qformer_other with qformer_decoder (similar learning rate)
    qformer_decoder_params.extend(qformer_other_params)
    
    # Create parameter groups with different learning rates
    param_groups = []
    
    if qformer_backbone_params:
        param_groups.append({
            'params': qformer_backbone_params,
            'lr': args.lr_qformer_backbone,
            'name': 'qformer_backbone',
        })
    
    if qformer_decoder_params:
        param_groups.append({
            'params': qformer_decoder_params,
            'lr': args.lr_qformer_decoder,
            'name': 'qformer_decoder',
        })
    
    if qformer_projector_params:
        param_groups.append({
            'params': qformer_projector_params,
            'lr': args.lr_qformer_projector,
            'name': 'qformer_projector',
        })
    
    if queries_params:
        param_groups.append({
            'params': queries_params,
            'lr': args.lr_queries,
            'name': 'map_queries',
        })
    
    if cls_head_params:
        param_groups.append({
            'params': cls_head_params,
            'lr': args.lr_cls_head,
            'name': 'cls_head',
        })
    
    if decoder_params:
        param_groups.append({
            'params': decoder_params,
            'lr': args.lr_decoder,
            'name': 'map_decoder',
        })
    
    if scene_interaction_params:
        param_groups.append({
            'params': scene_interaction_params,
            'lr': args.lr_scene_interaction,  # 与 decoder 对齐（从 scratch 训练）
            'name': 'scene_interaction',
        })
    
    # LoRA parameters (for LLM fine-tuning)
    if lora_params:
        param_groups.append({
            'params': lora_params,
            'lr': args.lr_lora,
            'name': 'lora',
        })
    
    # Print parameter counts
    print("\n" + "="*70)
    print("Parameter Groups (Fine-grained Learning Rates):")
    print("="*70)
    total_trainable = 0
    for group in param_groups:
        num_params = sum(p.numel() for p in group['params'])
        total_trainable += num_params
        print(f"  {group['name']:20s}: {num_params:12,} params, lr={group['lr']:.1e}")
    print("-"*70)
    print(f"  {'Total Trainable':20s}: {total_trainable:12,} params")
    # Effective batch size = batch_size × accumulation × world_size
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    effective_batch = args.batch_size * args.accumulation_steps * world_size
    print(f"  {'Effective Batch Size':20s}: {effective_batch:12,} (={args.batch_size}×{args.accumulation_steps}×{world_size}gpu)")
    print("="*70 + "\n")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    
    return optimizer


def get_lr_scheduler(optimizer, args, steps_per_epoch):
    """
    Learning rate scheduler with warmup and cosine decay
    
    参考 MapTR 的策略:
    - warmup_ratio = 1/3 (从 1/3 基础学习率开始)
    - min_lr_ratio = 1e-3 (最小学习率是基础的 0.1%)
    """
    total_steps = args.epochs * steps_per_epoch
    warmup_ratio = 1.0 / 3.0  # MapTR 使用 1/3 作为起始比例
    min_lr_ratio = 1e-3       # MapTR 使用 1e-3 作为最小比例
    
    def lr_lambda(current_step):
        # Warmup: 从 warmup_ratio 线性增加到 1.0
        if current_step < args.warmup_steps:
            # 从 1/3 开始，线性增长到 1.0
            progress = float(current_step) / float(max(1, args.warmup_steps))
            return warmup_ratio + (1.0 - warmup_ratio) * progress
        
        # Cosine decay: 从 1.0 衰减到 min_lr_ratio
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159265359)))
        # 映射到 [min_lr_ratio, 1.0] 范围
        return max(min_lr_ratio, min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    return scheduler


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, epoch, args, rank, local_rank=0, ema=None):
    """
    Train for one epoch with gradient accumulation and EMA.
    
    Args:
        model: The model to train
        dataloader: Training data loader
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision
        epoch: Current epoch number
        args: Training arguments
        rank: Process rank for distributed training
        local_rank: Local rank (GPU index) for CUDA device
        ema: EMA object (optional)
    
    Returns:
        dict: {
            'avg_loss': float,
            'loss_components': {key: avg_value},
            'loss_curve': [{step, total_loss, cls, pts, dir, grad_norm, lr}],
            'grad_stats': {avg, max, min, clipped_steps, total_steps, clip_ratio},
        }
    """
    model.train()
    
    total_loss = 0.0
    loss_dict_accum = {}
    num_updates = 0
    
    # ===== 训练日志：稀疏损失曲线 + 梯度统计 =====
    SAMPLE_INTERVAL = 200  # 每 200 步采样一次
    loss_curve_samples = []  # 稀疏损失曲线
    grad_norms_all = []      # 所有更新步的梯度范数（裁剪前）
    clipped_steps = 0        # 被裁剪的步数（grad_norm > grad_clip）
    nan_skipped_steps = 0    # NaN 跳过的步数
    oom_skipped_steps = 0    # OOM 跳过的步数
    
    # Gradient accumulation setup
    accumulation_steps = args.accumulation_steps
    optimizer.zero_grad()
    
    for step, batch in enumerate(dataloader):
        # Move to GPU
        images = batch['images'].cuda(non_blocking=True)  # [B, 6, 3, 448, 800]
        text_ids = batch['text_ids'].cuda(non_blocking=True)  # [B, L]
        gt_labels = batch['gt_labels'].cuda(non_blocking=True)  # [B, M]
        gt_points = batch['gt_points'].cuda(non_blocking=True)  # [B, M, 20, 2]
        gt_masks = batch['gt_masks'].cuda(non_blocking=True)  # [B, M]
        
        # Camera parameters for 3D position encoding in Q-Former
        cam_intrinsics = batch['cam_intrinsics'].cuda(non_blocking=True)  # [B, 6, 3, 3]
        cam_extrinsics = batch['cam_extrinsics'].cuda(non_blocking=True)  # [B, 6, 4, 4]
        
        # detect_anomaly 已完成调试任务（已定位 NaN 来源为 CoordinateEncoder 频率过高）
        # 正常训练中不再启用（会显著减慢速度且在 DDP 中可能导致崩溃）
        
        # Forward with mixed precision + OOM 保护
        use_amp = args.bf16 or args.fp16
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        
        oom_flag = False
        try:
            with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
                output = model(
                    images=images,
                    text_ids=text_ids,
                    return_loss=True,
                    gt_labels=gt_labels,
                    gt_points=gt_points,
                    gt_masks=gt_masks,
                    cam_intrinsics=cam_intrinsics,
                    cam_extrinsics=cam_extrinsics,
                )
                
                loss = output['loss'].float() / accumulation_steps
                loss_dict = output['loss_dict']
        except RuntimeError as e:
            if "out of memory" in str(e):
                if rank == 0:
                    print(f"⚠️ CUDA OOM at Epoch {epoch+1} Step {step+1}, clearing cache and skipping batch", flush=True)
                torch.cuda.empty_cache()
                optimizer.zero_grad()
                oom_flag = True
            else:
                raise e  # 非 OOM 错误，正常抛出
        
        # OOM 时同步所有 GPU 跳过此 batch
        if dist.is_initialized() and dist.get_world_size() > 1:
            oom_sync = torch.tensor([1.0 if oom_flag else 0.0], device=f'cuda:{local_rank}')
            dist.all_reduce(oom_sync, op=dist.ReduceOp.MAX)
            oom_flag = oom_sync.item() > 0
        if oom_flag:
            oom_skipped_steps += 1
            continue
        
        # ========== NaN 检测和同步跳过 ==========
        # 检测 loss 是否为 NaN 或 Inf
        # 重要：所有 GPU 必须同步决定是否跳过，否则会导致 NCCL 死锁
        has_nan = torch.isnan(loss).any() or torch.isinf(loss).any()
        
        # 在分布式环境中，同步所有 GPU 的 NaN 状态
        if dist.is_initialized() and dist.get_world_size() > 1:
            nan_flag = torch.tensor([1.0 if has_nan else 0.0], device=loss.device)
            dist.all_reduce(nan_flag, op=dist.ReduceOp.MAX)
            has_nan = nan_flag.item() > 0
        
        if has_nan:
            if rank == 0:
                print(f"⚠️ NaN/Inf detected at Epoch {epoch+1} Step {step+1}, skipping this batch (all GPUs synced)")
            optimizer.zero_grad()
            nan_skipped_steps += 1
            continue
        
        # Backward (accumulate gradients)
        # BF16 不需要 GradScaler（指数范围与 FP32 相同）
        # FP16 才需要 scaler 来防止梯度下溢
        if args.fp16 and scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # Accumulate losses for logging (use unscaled loss)
        loss_value = loss.item() * accumulation_steps
        if not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
            total_loss += loss_value
        else:
            # 不使用 continue，因为已经完成了 backward，需要继续执行后续同步操作
            if rank == 0:
                print(f"⚠️ NaN loss value at step {step+1}, not accumulating but continuing")
        for key, value in loss_dict.items():
            if key not in loss_dict_accum:
                loss_dict_accum[key] = 0.0
            loss_dict_accum[key] += value.item()
        
        # Update weights every accumulation_steps
        if (step + 1) % accumulation_steps == 0:
            skip_optimizer_step = False
            current_grad_norm = 0.0
            
            if args.fp16 and scaler is not None:
                scaler.unscale_(optimizer)
                grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                current_grad_norm = grad_norm_before.item() if torch.isfinite(grad_norm_before) else float('nan')
                
                if not torch.isfinite(grad_norm_before):
                    if rank == 0:
                        print(f"❌ [Step {step+1}] Gradient contains NaN/Inf! Skipping optimizer step.", flush=True)
                    skip_optimizer_step = True
                    optimizer.zero_grad()
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if grad_norm_before.item() > args.grad_clip:
                        clipped_steps += 1
                    if rank == 0 and grad_norm_before > args.grad_clip * 10:
                        print(f"⚠️ [Step {step+1}] Large gradient detected! "
                              f"Norm before clip: {grad_norm_before:.2f}", flush=True)
                    scaler.step(optimizer)
                    scaler.update()
            else:
                # BF16 或 FP32 模式 — 不需要 GradScaler
                grad_norm_before = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
                current_grad_norm = grad_norm_before.item() if torch.isfinite(grad_norm_before) else float('nan')
                
                if not torch.isfinite(grad_norm_before):
                    if rank == 0:
                        print(f"❌ [Step {step+1}] Gradient contains NaN/Inf! Skipping optimizer step.", flush=True)
                        # 【NaN 调试】打印哪些参数组包含 NaN 梯度
                        for group in optimizer.param_groups:
                            group_name = group.get('name', 'unknown')
                            nan_count = 0
                            inf_count = 0
                            total_count = 0
                            for p in group['params']:
                                if p.grad is not None:
                                    total_count += 1
                                    if torch.isnan(p.grad).any():
                                        nan_count += 1
                                    if torch.isinf(p.grad).any():
                                        inf_count += 1
                            if nan_count > 0 or inf_count > 0:
                                print(f"   ⚠️ {group_name}: {nan_count} NaN params, {inf_count} Inf params (of {total_count})", flush=True)
                    skip_optimizer_step = True
                    optimizer.zero_grad()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    if grad_norm_before.item() > args.grad_clip:
                        clipped_steps += 1
                    if rank == 0 and grad_norm_before > args.grad_clip * 10:
                        print(f"⚠️ [Step {step+1}] Large gradient detected! "
                              f"Norm before clip: {grad_norm_before:.2f}", flush=True)
                    optimizer.step()
            
            if not skip_optimizer_step:
                optimizer.zero_grad()
                scheduler.step()
                num_updates += 1
                # 记录有效更新步的梯度范数
                if not (current_grad_norm != current_grad_norm):  # not NaN
                    grad_norms_all.append(current_grad_norm)
            
            # Update EMA
            if ema is not None:
                ema.update()
        
        # Logging
        if rank == 0 and (step + 1) % args.log_interval == 0:
            avg_loss = total_loss / (step + 1)
            
            # Get learning rates from all groups
            lr_info = []
            for group in optimizer.param_groups:
                name = group.get('name', 'unknown')[:8]
                lr_info.append(f"{name}={group['lr']:.1e}")
            lr_str = ' '.join(lr_info[:3])  # Show first 3
            
            print(f"Epoch [{epoch+1}/{args.epochs}] Step [{step+1}/{len(dataloader)}] "
                  f"Loss: {loss.item()*accumulation_steps:.4f} (Avg: {avg_loss:.4f}) "
                  f"LR: {lr_str}")
            
            # Print detailed losses
            if (step + 1) % (args.log_interval * 10) == 0:
                print("  Detailed losses:")
                for key, value in loss_dict.items():
                    avg_value = loss_dict_accum[key] / (step + 1)
                    print(f"    {key}: {value.item():.4f} (Avg: {avg_value:.4f})")
        
        # ===== 稀疏损失曲线采样（每 200 步，仅 rank 0）=====
        if rank == 0 and (step + 1) % SAMPLE_INTERVAL == 0:
            sample_point = {
                "步数": step + 1,
                "总损失": round(loss.item() * accumulation_steps, 4),
            }
            # 记录各项损失分量
            for key, value in loss_dict.items():
                # 只保留核心损失项，跳过辅助损失的逐层细节
                if key in ('loss_cls', 'loss_pts', 'loss_dir', 'loss_main', 'loss_aux_total'):
                    short_key = key.replace('loss_', '')
                    sample_point[short_key] = round(value.item(), 4)
            # 最近一次的梯度范数
            if grad_norms_all:
                sample_point["梯度范数"] = round(grad_norms_all[-1], 2)
            # 当前学习率（取第一个参数组，转为 float 防止 JSON 序列化失败）
            sample_point["学习率"] = float(optimizer.param_groups[0]['lr'])
            loss_curve_samples.append(sample_point)
    
    # Handle remaining gradients (if not divisible by accumulation_steps)
    remaining = len(dataloader) % accumulation_steps
    if remaining > 0:
        skip_final_step = False
        if args.fp16 and scaler is not None:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            if not torch.isfinite(grad_norm):
                if rank == 0:
                    print(f"❌ [Final step] Gradient contains NaN/Inf! Skipping.", flush=True)
                skip_final_step = True
                optimizer.zero_grad()
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))
            if not torch.isfinite(grad_norm):
                if rank == 0:
                    print(f"❌ [Final step] Gradient contains NaN/Inf! Skipping.", flush=True)
                skip_final_step = True
                optimizer.zero_grad()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        
        if not skip_final_step:
            optimizer.zero_grad()
            scheduler.step()
            num_updates += 1
        
        if ema is not None:
            ema.update()
    
    # Epoch summary
    avg_loss = total_loss / max(len(dataloader), 1)  # 所有 rank 都计算
    
    # 构建各项损失平均值
    loss_components = {}
    for key, value in loss_dict_accum.items():
        loss_components[key] = round(value / max(len(dataloader), 1), 4)
    
    if rank == 0:
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Total Updates: {num_updates}")
        print(f"  Effective Samples: {len(dataloader) * args.batch_size}")
        for key, value in loss_components.items():
            print(f"  {key}: {value:.4f}")
        if grad_norms_all:
            print(f"  Grad Norm (avg/max/min): {sum(grad_norms_all)/len(grad_norms_all):.2f} / {max(grad_norms_all):.2f} / {min(grad_norms_all):.2f}")
            print(f"  Clipped steps: {clipped_steps}/{len(grad_norms_all)} ({100*clipped_steps/max(len(grad_norms_all),1):.1f}%)")
        print(f"  NaN skipped: {nan_skipped_steps}, OOM skipped: {oom_skipped_steps}")
        print(f"{'='*70}\n")
    
    # 【重要】训练完成后同步所有 GPU，防止某些 GPU 提前进入 validation
    if dist.is_initialized():
        if rank == 0:
            print(f"[DEBUG] Syncing all GPUs after training epoch {epoch+1}...", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"[DEBUG] All GPUs finished training epoch {epoch+1}!", flush=True)
    
    # ===== 构建训练日志返回值 =====
    # 梯度统计
    grad_stats = {}
    if grad_norms_all:
        grad_stats = {
            "平均": round(sum(grad_norms_all) / len(grad_norms_all), 2),
            "最大": round(max(grad_norms_all), 2),
            "最小": round(min(grad_norms_all), 2),
            "被裁剪步数": clipped_steps,
            "有效更新步数": len(grad_norms_all),
            "裁剪比例": round(clipped_steps / max(len(grad_norms_all), 1), 4),
        }
    
    # 当前各参数组学习率（转 float 防止 JSON 序列化失败）
    current_lrs = {}
    for group in optimizer.param_groups:
        name = group.get('name', 'unknown')
        current_lrs[name] = float(group['lr'])
    
    train_log = {
        "平均总损失": round(avg_loss, 4),
        "各项损失": loss_components,
        "损失下降曲线": loss_curve_samples,
        "梯度统计": grad_stats,
        "当前学习率": current_lrs,
        "有效更新步数": num_updates,
        "NaN跳过步数": nan_skipped_steps,
        "OOM跳过步数": oom_skipped_steps,
    }
    
    return train_log


@torch.no_grad()
def validate(model, dataloader, epoch, args, rank, ema=None, compute_map=True):
    """
    Validation with optional EMA model and mAP computation.
    
    If EMA is provided, validation uses the smoothed EMA weights.
    
    Args:
        model: The model to evaluate
        dataloader: Validation data loader
        epoch: Current epoch number
        args: Training arguments
        rank: Process rank for distributed training
        ema: EMA object (optional)
        compute_map: Whether to compute mAP metrics
    
    Returns:
        (avg_loss, metrics_dict) if compute_map else avg_loss
    """
    import sys
    
    # 【调试】打印进入 validation 的信息
    if rank == 0:
        print(f"\n[DEBUG] Entering validation for epoch {epoch+1}...", flush=True)
        sys.stdout.flush()
    
    # 【重要】同步所有 GPU 进程，确保都到达这里
    if dist.is_initialized():
        if rank == 0:
            print(f"[DEBUG] Waiting for all GPUs to sync before validation...", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"[DEBUG] All GPUs synced!", flush=True)
    
    model.eval()
    
    # Apply EMA weights if available
    if ema is not None:
        if rank == 0:
            print(f"[DEBUG] Applying EMA weights...", flush=True)
        ema.apply_shadow()
        if rank == 0:
            print(f"[DEBUG] EMA weights applied!", flush=True)
    
    total_loss = 0.0
    loss_dict_accum = {}
    
    # Initialize evaluator for mAP computation
    evaluator = MapEvaluator() if compute_map else None
    
    if rank == 0:
        print(f"[DEBUG] Starting validation loop, total batches: {len(dataloader)}", flush=True)
    
    for step, batch in enumerate(dataloader):
        if step == 0 and rank == 0:
            print(f"[DEBUG] First validation batch loaded successfully!", flush=True)
        
        # 进度打印（每 100 步打印一次）
        if rank == 0 and (step + 1) % 100 == 0:
            print(f"  Validating... [{step+1}/{len(dataloader)}] ({100*(step+1)/len(dataloader):.1f}%)", flush=True)
        
        # Move to GPU
        images = batch['images'].cuda(non_blocking=True)
        text_ids = batch['text_ids'].cuda(non_blocking=True)
        gt_labels = batch['gt_labels'].cuda(non_blocking=True)
        gt_points = batch['gt_points'].cuda(non_blocking=True)
        gt_masks = batch['gt_masks'].cuda(non_blocking=True)
        
        # Camera parameters for 3D position encoding
        cam_intrinsics = batch.get('cam_intrinsics')
        cam_extrinsics = batch.get('cam_extrinsics')
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.cuda(non_blocking=True)
        if cam_extrinsics is not None:
            cam_extrinsics = cam_extrinsics.cuda(non_blocking=True)
        
        # Forward
        use_amp = args.bf16 or args.fp16
        amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
        with torch.amp.autocast('cuda', enabled=use_amp, dtype=amp_dtype):
            output = model(
                images=images,
                text_ids=text_ids,
                return_loss=True,
                gt_labels=gt_labels,
                gt_points=gt_points,
                gt_masks=gt_masks,
                cam_intrinsics=cam_intrinsics,
                cam_extrinsics=cam_extrinsics,
            )
            
            loss = output['loss'].float()  # 确保 FP32
            loss_dict = output['loss_dict']
        
        # Accumulate loss
        total_loss += loss.item()
        for key, value in loss_dict.items():
            if key not in loss_dict_accum:
                loss_dict_accum[key] = 0.0
            loss_dict_accum[key] += value.item()
        
        # Add predictions to evaluator for mAP computation
        if evaluator is not None and 'pred_logits' in output and 'pred_points' in output:
            evaluator.add_batch(
                pred_logits=output['pred_logits'],
                pred_points=output['pred_points'],
                gt_labels=gt_labels,
                gt_points=gt_points,
                gt_masks=gt_masks,
            )
    
    # Restore original weights
    if ema is not None:
        ema.restore()
    
    # 【重要】分布式验证：同步所有 GPU
    if dist.is_initialized():
        if rank == 0:
            print(f"[DEBUG] Validation loop finished, syncing all GPUs...", flush=True)
        dist.barrier()
        if rank == 0:
            print(f"[DEBUG] All GPUs finished validation loop!", flush=True)
    
    # ========== 分布式汇总：损失 + mAP 预测 ==========
    local_steps = len(dataloader)
    
    if dist.is_initialized() and dist.get_world_size() > 1:
        world_size_val = dist.get_world_size()
        
        # --- 1. 汇总验证损失（所有 GPU → 求和 → 除以总步数）---
        loss_agg = torch.tensor([total_loss, float(local_steps)], device='cuda')
        dist.all_reduce(loss_agg, op=dist.ReduceOp.SUM)
        total_loss = loss_agg[0].item()
        total_steps_all = int(loss_agg[1].item())
        
        for key in loss_dict_accum:
            val_tensor = torch.tensor(loss_dict_accum[key], device='cuda')
            dist.all_reduce(val_tensor, op=dist.ReduceOp.SUM)
            loss_dict_accum[key] = val_tensor.item()
        
        # --- 2. 汇总 mAP 预测结果到 rank 0 ---
        # 每个 GPU 只看到 1/world_size 的验证集
        # 需要把所有 GPU 的预测和 GT 汇总到 rank 0 计算完整 mAP
        if evaluator is not None:
            if rank == 0:
                print(f"[DEBUG] Gathering mAP predictions from {world_size_val} GPUs...", flush=True)
            
            # Step 1: 为每个 GPU 的 sample_id 添加偏移，防止冲突
            # 各 GPU 的 sample_id 都从 0 开始，加上 rank * 100000 使全局唯一
            offset = rank * 100000
            for cls_id in evaluator.pred_instances:
                for pred in evaluator.pred_instances[cls_id]:
                    pred['sample_id'] += offset
            for cls_id in evaluator.gt_instances:
                for gt in evaluator.gt_instances[cls_id]:
                    gt['sample_id'] += offset
            
            # Step 2: 序列化每个 GPU 的评估数据
            local_eval_data = {
                'pred_instances': dict(evaluator.pred_instances),
                'gt_instances': dict(evaluator.gt_instances),
                'sample_count': evaluator.sample_count,
            }
            
            # Step 3: 汇总到 rank 0（使用 gather_object，内部用 pickle 处理 numpy）
            gathered_eval = [None] * world_size_val if rank == 0 else None
            dist.gather_object(local_eval_data, gathered_eval, dst=0)
            
            # Step 4: rank 0 合并所有数据到新的 evaluator
            if rank == 0:
                merged_evaluator = MapEvaluator()
                total_sample_count = 0
                for gpu_data in gathered_eval:
                    for cls_id, preds in gpu_data['pred_instances'].items():
                        merged_evaluator.pred_instances[cls_id].extend(preds)
                    for cls_id, gts in gpu_data['gt_instances'].items():
                        merged_evaluator.gt_instances[cls_id].extend(gts)
                    total_sample_count += gpu_data['sample_count']
                merged_evaluator.sample_count = total_sample_count
                evaluator = merged_evaluator
                print(f"[DEBUG] Merged evaluator: {total_sample_count} samples "
                      f"from {world_size_val} GPUs", flush=True)
    else:
        total_steps_all = local_steps
    
    # Compute metrics
    avg_loss = total_loss / max(total_steps_all, 1)
    val_loss_components = {}
    for key, value in loss_dict_accum.items():
        val_loss_components[key] = round(value / max(total_steps_all, 1), 4)
    
    metrics = {}
    if evaluator is not None and rank == 0:
        metrics = evaluator.compute_metrics()
    
    # Print results
    if rank == 0:
        ema_str = " (EMA)" if ema is not None else ""
        print(f"\n{'='*70}")
        print(f"Validation Epoch {epoch+1}{ema_str}:")
        print(f"  Average Loss: {avg_loss:.4f}")
        for key, value in val_loss_components.items():
            print(f"  {key}: {value:.4f}")
        
        # Print mAP metrics
        if metrics:
            print(f"\n  📊 mAP Metrics:")
            for thresh in [0.5, 1.0, 1.5]:
                mAP_key = f'mAP@{thresh}m'
                if mAP_key in metrics:
                    print(f"    mAP@{thresh}m: {metrics[mAP_key]*100:.2f}%")
            if 'mAP' in metrics:
                print(f"    ⭐ Overall mAP: {metrics['mAP']*100:.2f}%")
            
            # Per-class AP at 1.0m threshold
            print(f"\n  📈 Per-class AP@1.0m:")
            for cls_name in ['divider', 'ped_crossing', 'boundary']:
                ap_key = f'AP_{cls_name}@1.0m'
                if ap_key in metrics:
                    print(f"    {cls_name}: {metrics[ap_key]*100:.2f}%")
        
        print(f"{'='*70}\n")
    
    # ===== 构建验证日志 =====
    val_log = {
        "平均总损失": round(avg_loss, 4),
        "各项损失": val_loss_components,
    }
    
    if metrics:
        # mAP 汇总
        mAP_log = {
            "综合mAP": round(metrics.get('mAP', 0.0) * 100, 2),
        }
        # 各阈值 mAP
        for thresh in [0.5, 1.0, 1.5]:
            key = f'mAP@{thresh}m'
            if key in metrics:
                mAP_log[f"mAP@{thresh}m"] = round(metrics[key] * 100, 2)
        
        # 各类别各阈值 AP
        per_class = {}
        for cls_name in ['divider', 'ped_crossing', 'boundary']:
            cls_aps = {}
            for thresh in [0.5, 1.0, 1.5]:
                ap_key = f'AP_{cls_name}@{thresh}m'
                if ap_key in metrics:
                    cls_aps[f"AP@{thresh}m"] = round(metrics[ap_key] * 100, 2)
            if cls_aps:
                per_class[cls_name] = cls_aps
        mAP_log["各类别AP"] = per_class
        
        # 预测和GT数量
        counts = {}
        for cls_name in ['divider', 'ped_crossing', 'boundary']:
            pred_key = f'num_pred_{cls_name}'
            gt_key = f'num_gt_{cls_name}'
            if pred_key in metrics:
                counts[f"{cls_name}_预测数"] = int(metrics[pred_key])
            if gt_key in metrics:
                counts[f"{cls_name}_标注数"] = int(metrics[gt_key])
        mAP_log["预测与标注数量"] = counts
        
        val_log["mAP"] = mAP_log
    
    if compute_map:
        return avg_loss, metrics, val_log
    else:
        return avg_loss, {}, val_log


def save_checkpoint(model, optimizer, scheduler, epoch, args, filename, ema=None):
    """
    Save checkpoint with optional EMA state.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
    }
    
    # Save EMA state if available
    if ema is not None:
        checkpoint['ema_state_dict'] = ema.state_dict()
    
    save_path = os.path.join(args.output_dir, filename)
    torch.save(checkpoint, save_path)
    print(f"💾 Checkpoint saved to {save_path}")


def save_ema_model(model, ema, args, filename):
    """
    Save model with EMA weights (for inference).
    """
    if ema is None:
        return
    
    # Apply EMA weights
    ema.apply_shadow()
    
    # Save
    save_path = os.path.join(args.output_dir, filename)
    torch.save(model.state_dict(), save_path)
    print(f"💾 EMA model saved to {save_path}")
    
    # Restore original weights
    ema.restore()


def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed
    rank, world_size, local_rank = setup_distributed()
    
    if rank == 0:
        print("\n" + "="*70)
        print("LLaVA Map Detection Training - Stage 2 (Optimized)")
        print("="*70)
        print(f"Training Configuration:")
        for key, value in vars(args).items():
            print(f"  {key}: {value}")
        print("="*70 + "\n")
    
    # Enable anomaly detection if requested (for debugging NaN)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True)
        if rank == 0:
            print("⚠️  ANOMALY DETECTION ENABLED - Training will be 2-3x slower!")
            print("   This will help find the exact source of NaN/Inf errors.")
            print("   Disable with removing --detect-anomaly flag for normal training.\n")
    
    # Create output directory
    if rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        # Save config
        with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        
        # ===== 初始化 training_log.json =====
        training_log_path = os.path.join(args.output_dir, 'training_log.json')
        training_log = {
            "项目": "LLaVA Map Detection",
            "创建时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "配置": {
                "总轮数": args.epochs,
                "每GPU_batch_size": args.batch_size,
                "梯度累积步数": args.accumulation_steps,
                "有效batch_size": args.batch_size * args.accumulation_steps * world_size,
                "GPU数量": world_size,
                "混合精度": "BF16" if args.bf16 else ("FP16" if args.fp16 else "FP32"),
                "梯度裁剪": args.grad_clip,
                "权重衰减": args.weight_decay,
                "warmup步数": args.warmup_steps,
                "EMA": args.use_ema,
                "EMA衰减": args.ema_decay if args.use_ema else None,
                "学习率": {
                    "qformer_backbone": args.lr_qformer_backbone,
                    "qformer_decoder": args.lr_qformer_decoder,
                    "qformer_projector": args.lr_qformer_projector,
                    "map_queries": args.lr_queries,
                    "cls_head": args.lr_cls_head,
                    "map_decoder": args.lr_decoder,
                    "scene_interaction": args.lr_scene_interaction,
                    "lora": args.lr_lora,
                },
            },
            "epochs": [],
        }
        with open(training_log_path, 'w', encoding='utf-8') as f:
            json.dump(training_log, f, indent=2, ensure_ascii=False)
        print(f"📝 训练日志初始化: {training_log_path}")
    
    # Build model
    if rank == 0:
        print("Building model...")
    
    model = build_map_detector(
        llm_path=args.llm_path,
        freeze_llm=True,  # Stage 2: Freeze LLM
        qformer_pretrained='blip2' if args.qformer_pretrained == 'blip2' else None,
        qformer_version=args.qformer_version,
    )
    
    model = model.cuda()
    
    # Wrap with DDP if needed
    if world_size > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # 允许模型有未使用的参数
        )
    
    # Build optimizer
    base_model = model.module if world_size > 1 else model
    optimizer = build_optimizer(base_model, args)
    
    # Initialize EMA
    ema = None
    if args.use_ema:
        if rank == 0:
            print(f"✅ EMA enabled with decay={args.ema_decay}")
        ema = EMA(base_model, decay=args.ema_decay)
    
    # Build datasets
    if rank == 0:
        print("\nLoading datasets...")
    
    # Tokenizer and image processor
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    
    # Use local CLIP path to avoid network issues
    local_clip_path = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
    if os.path.exists(local_clip_path):
        image_processor = CLIPImageProcessor.from_pretrained(local_clip_path)
        if rank == 0:
            print(f"✅ Using local CLIP: {local_clip_path}")
    else:
        image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    
    train_dataset = MapDetectionDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache_train,
        image_processor=image_processor,
        tokenizer=tokenizer,
        subset_scenes_file=args.subset_scenes,
    )
    
    # Check if val GT cache exists before creating val dataset
    val_dataset = None
    if args.gt_cache_val and (os.path.exists(args.gt_cache_val) or os.path.isdir(args.gt_cache_val)):
        val_dataset = MapDetectionDataset(
            dataroot=args.dataroot,
            version=args.version,
            split='val',
            gt_cache_path=args.gt_cache_val,
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
    else:
        if rank == 0:
            print(f"⚠️  Validation GT cache not found: {args.gt_cache_val}")
            print(f"   Skipping validation.")
    
    if rank == 0:
        print(f"✅ Train samples: {len(train_dataset)}")
        if val_dataset is not None:
            print(f"✅ Val samples: {len(val_dataset)}")
        else:
            print(f"⚠️  Val samples: 0 (skipped)")
        print(f"✅ Effective batch size: {args.batch_size * args.accumulation_steps * world_size}")
    
    # Build dataloaders
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    ) if world_size > 1 else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )
    
    val_loader = None
    val_sampler = None
    if val_dataset is not None:
        # 【重要】验证也需要 DistributedSampler，否则会导致 DDP 死锁
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        ) if world_size > 1 else None
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            sampler=val_sampler,  # 使用分布式采样器
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=val_dataset.collate_fn,
        )
    
    # Learning rate scheduler (account for gradient accumulation)
    steps_per_epoch = len(train_loader) // args.accumulation_steps
    scheduler = get_lr_scheduler(optimizer, args, steps_per_epoch)
    
    # Mixed precision scaler with very conservative settings
    # 
    # 【关键修复】大幅降低 init_scale
    # 原因：总 loss ≈ 280-300，经 accumulation 后 loss_per_step ≈ 56
    # 旧 scale=2048 时: scaled_loss = 56 × 2048 = 114,688
    # LLM 权重是 FP16，梯度也是 FP16（max=65504）
    # → 梯度极易溢出 → 每步都 NaN
    #
    # 新 scale=128 时: scaled_loss = 56 × 128 = 7,168
    # 留有足够裕度，FP16 梯度不会溢出
    # BF16 不需要 GradScaler：BF16 指数范围与 FP32 相同（max ~3.4e38），不存在溢出问题
    # FP16 才需要 GradScaler 来防止梯度下溢（但不推荐，因为 7B LLM 反向传播容易溢出）
    if args.fp16 and not args.bf16:
        scaler = GradScaler(
            init_scale=128.0,
            growth_factor=1.5,
            backoff_factor=0.5,
            growth_interval=2000,
        )
        if rank == 0:
            print("⚠️ 使用 FP16 GradScaler（不推荐，建议使用 --bf16）")
    else:
        scaler = None
        if rank == 0 and args.bf16:
            print("✅ 使用 BF16 混合精度训练（无需 GradScaler，不会梯度溢出）")
    
    # Resume from checkpoint if provided
    start_epoch = 0
    if args.resume is not None:
        if rank == 0:
            print(f"\nResuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        
        # Handle module. prefix mismatch between checkpoint and model
        state_dict = checkpoint['model_state_dict']
        model_state_dict = model.state_dict()
        
        # Check if we need to add or remove 'module.' prefix
        checkpoint_has_module = any(k.startswith('module.') for k in state_dict.keys())
        model_has_module = any(k.startswith('module.') for k in model_state_dict.keys())
        
        if checkpoint_has_module and not model_has_module:
            # Remove 'module.' prefix from checkpoint
            state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
            if rank == 0:
                print("  Removed 'module.' prefix from checkpoint keys")
        elif not checkpoint_has_module and model_has_module:
            # Add 'module.' prefix to checkpoint
            state_dict = {'module.' + k: v for k, v in state_dict.items()}
            if rank == 0:
                print("  Added 'module.' prefix to checkpoint keys")
        
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        
        # 【关键】用命令行参数的 LR 覆盖 checkpoint 中保存的 LR
        # 原因：resume 时可能需要调整学习率（如从保守 LR 切换到更高 LR）
        # optimizer.load_state_dict() 会恢复 checkpoint 的 LR，这里用 args 覆盖
        lr_map = {
            'qformer_backbone': args.lr_qformer_backbone,
            'qformer_decoder': args.lr_qformer_decoder,
            'qformer_projector': args.lr_qformer_projector,
            'map_queries': args.lr_queries,
            'cls_head': args.lr_cls_head,
            'map_decoder': args.lr_decoder,
            'scene_interaction': args.lr_scene_interaction,
            'lora': args.lr_lora,
        }
        lr_changed = False
        for group in optimizer.param_groups:
            group_name = group.get('name', '')
            if group_name in lr_map:
                old_lr = group['lr']
                new_lr = lr_map[group_name]
                if abs(old_lr - new_lr) / max(old_lr, 1e-12) > 0.01:  # >1% 变化才算
                    group['lr'] = new_lr
                    group['initial_lr'] = new_lr  # scheduler 也需要这个
                    if rank == 0:
                        print(f"  📝 LR override: {group_name}: {old_lr:.1e} → {new_lr:.1e}")
                    lr_changed = True
                else:
                    group['lr'] = new_lr
                    group['initial_lr'] = new_lr
        
        if lr_changed:
            # LR 发生变化时，重建 scheduler 以使用新的 base LR
            # 保留 cosine decay 的进度（从当前 epoch 继续衰减）
            steps_per_epoch = len(train_loader) // args.accumulation_steps
            scheduler = get_lr_scheduler(optimizer, args, steps_per_epoch)
            # 快进 scheduler 到当前步数
            completed_steps = start_epoch * steps_per_epoch
            for _ in range(completed_steps):
                scheduler.step()
            if rank == 0:
                print(f"  📝 Scheduler rebuilt for new LRs, fast-forwarded {completed_steps} steps")
        
        # Load EMA state if available
        if ema is not None and 'ema_state_dict' in checkpoint:
            ema.load_state_dict(checkpoint['ema_state_dict'])
            if rank == 0:
                print(f"✅ EMA state loaded")
        
        if rank == 0:
            print(f"✅ Resumed from epoch {start_epoch}")
    
    # Training loop
    if rank == 0:
        print("\n" + "="*70)
        print("Starting Training...")
        print(f"  Gradient Accumulation: {args.accumulation_steps} steps")
        print(f"  EMA: {'Enabled' if ema else 'Disabled'}")
        print(f"  Mixed Precision: {'BF16' if args.bf16 else 'FP16' if args.fp16 else 'FP32'}")
        print("="*70 + "\n")
    
    best_val_loss = float('inf')
    best_mAP = 0.0
    training_log_path = os.path.join(args.output_dir, 'training_log.json')
    
    for epoch in range(start_epoch, args.epochs):
        if world_size > 1:
            train_sampler.set_epoch(epoch)
        
        # Train（记录训练耗时）
        import time as _time
        t_train_start = _time.time()
        train_log = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            epoch, args, rank, local_rank=local_rank, ema=ema
        )
        t_train_end = _time.time()
        train_log["训练时长_分钟"] = round((t_train_end - t_train_start) / 60.0, 1)
        
        # Validate (skip if val_loader is None)
        val_log = None
        if val_loader is not None and (epoch + 1) % args.eval_interval == 0:
            # 每个 epoch 都计算 mAP，方便观察修复效果
            should_compute_map = True
            t_val_start = _time.time()
            val_loss, val_metrics, val_log = validate(
                model, val_loader, epoch, args, rank, ema=ema, 
                compute_map=should_compute_map
            )
            t_val_end = _time.time()
            val_log["验证时长_分钟"] = round((t_val_end - t_val_start) / 60.0, 1)
            
            current_mAP = val_metrics.get('mAP', 0.0)
            
            # Save best model (based on mAP if available, otherwise loss)
            if rank == 0:
                # Use mAP as primary metric if available
                if current_mAP > best_mAP:
                    best_mAP = current_mAP
                    best_val_loss = val_loss
                    print(f"🎯 New best mAP: {best_mAP*100:.2f}%")
                    save_checkpoint(
                        base_model,
                        optimizer, scheduler, epoch, args,
                        'best_model.pth', ema=ema
                    )
                    # Also save EMA-only model for inference
                    if ema is not None:
                        save_ema_model(base_model, ema, args, 'best_model_ema.pth')
                elif current_mAP == 0 and val_loss < best_val_loss:
                    # Fallback to loss if mAP is 0
                    best_val_loss = val_loss
                    save_checkpoint(
                        base_model,
                        optimizer, scheduler, epoch, args,
                        'best_model.pth', ema=ema
                    )
                    if ema is not None:
                        save_ema_model(base_model, ema, args, 'best_model_ema.pth')
        
        # Save checkpoint
        if rank == 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                base_model,
                optimizer, scheduler, epoch, args,
                f'checkpoint_epoch_{epoch+1}.pth', ema=ema
            )
        
        # ===== 追加写入 training_log.json（仅 rank 0）=====
        if rank == 0:
            epoch_entry = {
                "epoch": epoch + 1,
                "训练": train_log,
            }
            if val_log is not None:
                epoch_entry["验证"] = val_log
            
            # 读取已有日志，追加当前 epoch，重新写入
            try:
                with open(training_log_path, 'r', encoding='utf-8') as f:
                    full_log = json.load(f)
                full_log['epochs'].append(epoch_entry)
                with open(training_log_path, 'w', encoding='utf-8') as f:
                    json.dump(full_log, f, indent=2, ensure_ascii=False)
                print(f"📝 Epoch {epoch+1} 日志已写入 {training_log_path}")
            except Exception as e:
                print(f"⚠️ 写入训练日志失败: {e}")
    
    # Save final model
    if rank == 0:
        save_checkpoint(
            base_model,
            optimizer, scheduler, args.epochs - 1, args,
            'final_model.pth', ema=ema
        )
        
        # Save EMA model for inference
        if ema is not None:
            save_ema_model(base_model, ema, args, 'final_model_ema.pth')
        
        print("\n" + "="*70)
        print("✅ Training completed!")
        print(f"  Best validation loss: {best_val_loss:.4f}")
        print(f"  Best mAP: {best_mAP*100:.2f}%")
        print(f"  Checkpoints saved to: {args.output_dir}")
        print(f"  训练日志: {training_log_path}")
        print("="*70)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()

