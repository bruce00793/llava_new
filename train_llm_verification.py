"""
LLM Verification - 验证 LLM 部分是否能正确理解场景并输出有效的 Map Features

============================================
验证目标
============================================
假设 Q-Former 输出正确，验证 LLM + Map Queries 能否：
1. 理解场景信息（通过 768 scene tokens）
2. 输出有意义的 Map Features（通过 1050 map queries）

============================================
验证流程
============================================
6 张图 → Q-Former → 768 scene tokens
                        ↓
              [Text Embeds + Scene Tokens + 1050 Map Queries]
                        ↓
                    LLM Forward
                        ↓
              Map-Scene Interaction (Cross-Attention)
                        ↓
              获取 instance_features [B, 50, 4096]
                        ↓
              简单分类头 → 预测 50 个实例的类别
                        ↓
              与 GT 比较

============================================
设计理念
============================================
- 如果 LLM 输出的 map features 包含有效信息
- 那么简单的分类头就能预测正确的类别
- 这是 Linear Probing 的思想，验证 features 的质量

============================================
验证结论
============================================
如果成功（分类准确率 > 60%）：
  → LLM + Map Queries 能理解场景
  → 如果主训练失败，问题在 MapDecoder（点预测部分）

如果失败：
  → LLM/LoRA/Map Queries/Cross-Attention 有问题

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

from llava.model.map_llava_model import LLaVAMapDetector, create_map_detector
from transformers import AutoTokenizer


# ============================================
# 配置
# ============================================
MAP_CATEGORIES = ['divider', 'ped_crossing', 'boundary']
NUM_MAP_CLASSES = len(MAP_CATEGORIES)  # 3
NUM_INSTANCES = 50  # Map queries 输出 50 个实例


class SimpleClassificationHead(nn.Module):
    """
    简单的分类头 - Linear Probing
    
    验证 LLM 输出的 instance_features 是否包含类别信息
    """
    
    def __init__(self, input_dim: int = 4096, num_classes: int = 4):
        """
        Args:
            input_dim: LLM hidden size
            num_classes: 3 个地图类别 + 1 个 no-object
        """
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, instance_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            instance_features: [B, 50, 4096]
        
        Returns:
            logits: [B, 50, 4] - 每个实例的类别预测
        """
        return self.classifier(instance_features)


class LLMVerificationDataset(Dataset):
    """
    LLM 验证数据集 - 加载图像和地图元素 GT
    """
    
    def __init__(
        self,
        dataroot: str,
        version: str,
        split: str,
        gt_cache_path: str,
        tokenizer,
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
        
        # Prompt (简单的任务描述)
        self.prompt = "Detect HD map elements from surround-view images."
        
        print(f"Loaded {len(self.sample_tokens)} samples for {split}")
    
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
    
    def _load_map_gt(self, sample_token: str) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        加载地图元素 GT，转换为实例级别的标签
        
        Returns:
            labels: [50] - 每个实例的类别 (0-2 地图类别，3 表示 no-object)
            mask: [50] - 有效标记
        """
        gt_file = os.path.join(self.gt_ann_dir, f'{sample_token}.pkl')
        
        labels = torch.full((NUM_INSTANCES,), NUM_MAP_CLASSES, dtype=torch.long)  # 默认 no-object
        mask = torch.zeros(NUM_INSTANCES, dtype=torch.bool)
        
        if not os.path.exists(gt_file):
            return labels, mask
        
        with open(gt_file, 'rb') as f:
            gt_data = pickle.load(f)
        
        gt_classes = gt_data['gt_classes']
        num_gt = min(len(gt_classes), NUM_INSTANCES)
        
        for i in range(num_gt):
            labels[i] = gt_classes[i]  # 0, 1, 2
            mask[i] = True
        
        return labels, mask
    
    def _tokenize_prompt(self) -> torch.Tensor:
        """Tokenize prompt"""
        tokens = self.tokenizer(
            self.prompt,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=128,
        )
        return tokens.input_ids.squeeze(0)
    
    def __getitem__(self, idx: int) -> Dict:
        sample_token = self.sample_tokens[idx]
        
        images = self._load_images(sample_token)
        labels, mask = self._load_map_gt(sample_token)
        input_ids = self._tokenize_prompt()
        
        return {
            'images': images,
            'input_ids': input_ids,
            'labels': labels,
            'mask': mask,
            'sample_token': sample_token,
        }


def collate_fn(batch):
    """Collate function with padding"""
    images = torch.stack([item['images'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    masks = torch.stack([item['mask'] for item in batch])
    
    # Pad input_ids to same length
    max_len = max(item['input_ids'].shape[0] for item in batch)
    input_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)
    
    for i, item in enumerate(batch):
        seq_len = item['input_ids'].shape[0]
        input_ids[i, :seq_len] = item['input_ids']
        attention_mask[i, :seq_len] = 1
    
    return {
        'images': images,
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'masks': masks,
    }


def train_epoch(model, cls_head, dataloader, criterion, optimizer, scaler, epoch, args):
    """训练一个 epoch"""
    model.train()
    cls_head.train()
    
    total_loss = 0
    total_correct = 0
    total_samples = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
    optimizer.zero_grad()
    
    for step, batch in enumerate(pbar):
        images = batch['images'].cuda()
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        masks = batch['masks'].cuda()
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            # Forward through model to get map features
            outputs = model(
                images=images,
                input_ids=input_ids,
                return_map_features=True,
            )
            
            # Get instance features
            instance_features = outputs['instance_features']  # [B, 50, 4096]
            
            # Classification
            logits = cls_head(instance_features)  # [B, 50, 4]
            
            # Loss (only on valid instances)
            loss = criterion(logits.view(-1, NUM_MAP_CLASSES + 1), labels.view(-1))
            loss = loss / args.accumulation_steps
        
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (step + 1) % args.accumulation_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cls_head.parameters()), 
                    args.grad_clip
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(cls_head.parameters()),
                    args.grad_clip
                )
                optimizer.step()
            optimizer.zero_grad()
        
        # Metrics
        with torch.no_grad():
            pred = logits.argmax(dim=-1)  # [B, 50]
            correct = ((pred == labels) & masks).sum().item()
            total_correct += correct
            total_samples += masks.sum().item()
        
        total_loss += loss.item() * args.accumulation_steps
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item() * args.accumulation_steps:.4f}",
            'acc': f"{100*total_correct/max(total_samples,1):.1f}%",
        })
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': total_correct / max(total_samples, 1),
    }


@torch.no_grad()
def validate(model, cls_head, dataloader, criterion, epoch, args):
    """验证"""
    model.eval()
    cls_head.eval()
    
    total_loss = 0
    num_batches = 0
    
    # 统计
    class_correct = {i: 0 for i in range(NUM_MAP_CLASSES + 1)}
    class_total = {i: 0 for i in range(NUM_MAP_CLASSES + 1)}
    
    for batch in tqdm(dataloader, desc="Validating"):
        images = batch['images'].cuda()
        input_ids = batch['input_ids'].cuda()
        labels = batch['labels'].cuda()
        masks = batch['masks'].cuda()
        
        with torch.cuda.amp.autocast(enabled=args.fp16):
            outputs = model(
                images=images,
                input_ids=input_ids,
                return_map_features=True,
            )
            
            instance_features = outputs['instance_features']
            logits = cls_head(instance_features)
            loss = criterion(logits.view(-1, NUM_MAP_CLASSES + 1), labels.view(-1))
        
        total_loss += loss.item()
        num_batches += 1
        
        # 分类别统计
        pred = logits.argmax(dim=-1)
        for b in range(images.shape[0]):
            for i in range(NUM_INSTANCES):
                if masks[b, i]:
                    gt_cls = labels[b, i].item()
                    pred_cls = pred[b, i].item()
                    class_total[gt_cls] += 1
                    if pred_cls == gt_cls:
                        class_correct[gt_cls] += 1
    
    # 计算指标
    avg_loss = total_loss / num_batches
    
    total_correct = sum(class_correct.values())
    total_samples = sum(class_total.values())
    overall_acc = total_correct / max(total_samples, 1)
    
    # 地图元素准确率（不含 no-object）
    map_correct = sum(class_correct[i] for i in range(NUM_MAP_CLASSES))
    map_total = sum(class_total[i] for i in range(NUM_MAP_CLASSES))
    map_acc = map_correct / max(map_total, 1)
    
    print(f"\n{'='*70}")
    print(f"Epoch {epoch+1} Validation Results:")
    print(f"{'='*70}")
    print(f"  Overall Accuracy: {overall_acc*100:.1f}%")
    print(f"  Map Element Accuracy: {map_acc*100:.1f}% (目标: > 60%)")
    print(f"  Loss: {avg_loss:.4f}")
    
    print(f"\n  Per-class Accuracy:")
    for i, name in enumerate(MAP_CATEGORIES):
        acc = class_correct[i] / max(class_total[i], 1)
        print(f"    {name:15s}: {acc*100:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    no_obj_acc = class_correct[NUM_MAP_CLASSES] / max(class_total[NUM_MAP_CLASSES], 1)
    print(f"    {'no-object':15s}: {no_obj_acc*100:.1f}% ({class_correct[NUM_MAP_CLASSES]}/{class_total[NUM_MAP_CLASSES]})")
    
    print(f"{'='*70}")
    
    # 判断
    if map_acc > 0.6:
        print(f"  ✅ 验证成功! LLM 输出的 features 包含有效的类别信息")
        print(f"  ✅ 如果主训练效果不好，问题在 MapDecoder（点预测部分）")
    elif map_acc > 0.4:
        print(f"  ⚠️ 部分成功，LLM 有一定理解能力，但可以改进")
    else:
        print(f"  ❌ 验证失败，LLM/LoRA/Map Queries 可能有问题")
    print(f"{'='*70}\n")
    
    return {
        'loss': avg_loss,
        'overall_acc': overall_acc,
        'map_acc': map_acc,
    }


def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataroot', type=str, required=True)
    parser.add_argument('--version', type=str, default='v1.0-trainval')
    parser.add_argument('--gt-cache', type=str, required=True)
    parser.add_argument('--llm-path', type=str, default='/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5')
    parser.add_argument('--sample-ratio', type=float, default=0.1)
    
    parser.add_argument('--epochs', type=int, default=10)
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
    
    print("\n" + "="*70)
    print("LLM Verification - 验证 LLM 是否输出有效的 Map Features")
    print("="*70)
    print(f"验证方法: Linear Probing on instance_features")
    print(f"验证目标: LLM 输出的 50 个实例 features 能否预测类别")
    print(f"成功标准: Map Element Accuracy > 60%")
    print("="*70)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.llm_path, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model (Q-Former + LLM, 不加载 MapDecoder)
    print("\nLoading model...")
    model = create_map_detector(
        llm_path=args.llm_path,
        qformer_pretrained_path=None,
        use_lora=True,
        freeze_llm=False,
    )
    model = model.cuda()
    
    # 冻结大部分参数，只训练分类头
    # （这样可以纯粹验证 LLM 输出的 features 质量）
    for param in model.parameters():
        param.requires_grad = False
    
    # 分类头（可训练）
    cls_head = SimpleClassificationHead(
        input_dim=4096,  # LLM hidden size
        num_classes=NUM_MAP_CLASSES + 1,  # 3 地图类别 + 1 no-object
    ).cuda()
    
    print(f"\nModel loaded. Classification head trainable parameters: {sum(p.numel() for p in cls_head.parameters()):,}")
    
    # Dataset
    train_dataset = LLMVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='train',
        gt_cache_path=args.gt_cache,
        tokenizer=tokenizer,
        sample_ratio=args.sample_ratio,
    )
    
    val_dataset = LLMVerificationDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache,
        tokenizer=tokenizer,
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
    
    # Loss & Optimizer (只优化分类头)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(cls_head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
    
    # Training
    best_map_acc = 0
    
    for epoch in range(args.epochs):
        train_metrics = train_epoch(model, cls_head, train_loader, criterion, optimizer, scaler, epoch, args)
        val_metrics = validate(model, cls_head, val_loader, criterion, epoch, args)
        
        if val_metrics['map_acc'] > best_map_acc:
            best_map_acc = val_metrics['map_acc']
            save_path = os.path.join(args.output_dir, 'best_cls_head.pt')
            torch.save({
                'epoch': epoch,
                'cls_head_state_dict': cls_head.state_dict(),
                'metrics': val_metrics,
            }, save_path)
            print(f"✅ Saved best model (Map Acc={best_map_acc*100:.1f}%)")
    
    print("\n" + "="*70)
    print(f"✅ Training completed!")
    print(f"   Best Map Element Accuracy: {best_map_acc*100:.1f}% (目标: > 60%)")
    print("="*70)
    
    # 结论
    print("\n" + "="*70)
    print("验证结论:")
    if best_map_acc > 0.6:
        print("  ✅ LLM + Map Queries 输出的 features 包含有效的类别信息！")
        print("  ✅ 如果主训练效果不好，问题在 MapDecoder（点预测部分）")
    elif best_map_acc > 0.4:
        print("  ⚠️ LLM 有一定理解能力，但 features 质量可以改进")
        print("  ⚠️ 考虑优化 LoRA 配置或 Map-Scene Interaction")
    else:
        print("  ❌ LLM 输出的 features 质量不足")
        print("  ❌ 问题可能在 LoRA、Map Queries 或 Cross-Attention 设计")
    print("="*70)


if __name__ == '__main__':
    main()
