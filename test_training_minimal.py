"""
最小数据集训练测试
目的：
1. 验证训练流程能跑通
2. 检查是否出现 NaN
3. 验证 Loss 能下降

使用方法：
    cd /home/cly/auto/llava_test/LLaVA
    conda activate llava_new
    python test_training_minimal.py
"""

import os
import sys

# ========== 强制单 GPU ==========
# device_map="auto" 会自动分布 LLM 到多个 GPU，导致设备不一致
# 测试时强制只用一个 GPU，正式训练用 torchrun 分布式
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset


def check_for_nan(tensor, name):
    """检查张量是否包含 NaN"""
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}!")
        return True
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}!")
        return True
    return False


def main():
    print("=" * 70)
    print("最小数据集训练测试")
    print("=" * 70)
    
    # ========== 配置 ==========
    DATAROOT = "/home/cly/auto/llava_test/LLaVA/data/nuscenes"
    VERSION = "v1.0-trainval"
    GT_CACHE = f"{DATAROOT}/gt_cache"
    LLM_PATH = "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
    
    NUM_SAMPLES = 10  # 只用 10 个样本
    NUM_STEPS = 20    # 只训练 20 步
    BATCH_SIZE = 1
    LR = 1e-5         # 降低学习率，避免梯度爆炸
    USE_FP16 = False  # 禁用 FP16，ResNet backbone 在 FP16 下不稳定
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n设备: {device}")
    print(f"GPU 数量: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            mem = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"         显存: {mem:.1f} GB")
    
    # ========== 加载 Tokenizer ==========
    print(f"\n[1/5] 加载数据集 (只用 {NUM_SAMPLES} 个样本)...")
    
    # 从本地加载 tokenizer（避免网络请求）
    print("  加载本地 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        LLM_PATH,
        use_fast=False,
        local_files_only=True,  # 只用本地文件
    )
    
    dataset = MapDetectionDataset(
        dataroot=DATAROOT,
        version=VERSION,
        split='train',
        gt_cache_path=GT_CACHE,
        tokenizer=tokenizer,  # 传入本地 tokenizer
        use_augmentation=False,  # 测试时不用增强
    )
    
    # 只取前 NUM_SAMPLES 个样本
    subset = Subset(dataset, range(min(NUM_SAMPLES, len(dataset))))
    
    dataloader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # 单线程方便调试
        collate_fn=dataset.collate_fn,
    )
    
    print(f"  数据集大小: {len(subset)}")
    
    # ========== 构建模型 ==========
    print(f"\n[2/5] 构建模型...")
    
    model = build_map_detector(
        llm_path=LLM_PATH,
        qformer_pretrained=None,
        freeze_llm=True,
    )
    model = model.to(device)
    model.train()
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params / 1e6:.1f}M")
    print(f"  可训练: {trainable_params / 1e6:.1f}M")
    
    # ========== 优化器 ==========
    print(f"\n[3/5] 设置优化器 (LR={LR})...")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR,
        weight_decay=0.01,
    )
    
    scaler = GradScaler() if USE_FP16 else None
    
    # ========== 训练循环 ==========
    print(f"\n[4/5] 开始训练 ({NUM_STEPS} 步)...")
    print("-" * 70)
    
    losses = []
    step = 0
    nan_detected = False
    
    while step < NUM_STEPS:
        for batch in dataloader:
            if step >= NUM_STEPS:
                break
            
            # 准备数据
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            gt_labels = batch['gt_labels'].to(device)
            gt_points = batch['gt_points'].to(device)
            gt_masks = batch['gt_masks'].to(device)
            cam_intrinsics = batch['cam_intrinsics'].to(device)
            cam_extrinsics = batch['cam_extrinsics'].to(device)
            
            # 检查输入是否有 NaN
            if check_for_nan(images, "input images"):
                nan_detected = True
                break
            
            # 前向传播
            optimizer.zero_grad()
            
            with autocast(enabled=USE_FP16):
                outputs = model(
                    images=images,
                    text_ids=text_ids,
                    return_loss=True,
                    gt_labels=gt_labels,
                    gt_points=gt_points,
                    gt_masks=gt_masks,
                    cam_intrinsics=cam_intrinsics,
                    cam_extrinsics=cam_extrinsics,
                )
                loss = outputs['loss']
            
            # 检查 loss 是否有 NaN
            if check_for_nan(loss, "loss"):
                nan_detected = True
                print(f"  Loss 各项:")
                for k, v in outputs.items():
                    if 'loss' in k.lower() and isinstance(v, torch.Tensor):
                        print(f"    {k}: {v.item():.4f}")
                break
            
            # 反向传播
            if USE_FP16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # 检查梯度是否有 NaN
            grad_nan = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if check_for_nan(param.grad, f"grad of {name}"):
                        grad_nan = True
                        break
            
            if grad_nan:
                nan_detected = True
                break
            
            # 梯度裁剪和更新参数
            if USE_FP16:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()
            
            # 记录 loss
            loss_val = loss.item()
            losses.append(loss_val)
            
            # 打印
            if step % 2 == 0 or step == NUM_STEPS - 1:
                cls_loss = outputs.get('loss_cls', torch.tensor(0)).item()
                pts_loss = outputs.get('loss_pts', torch.tensor(0)).item()
                bbox_loss = outputs.get('loss_bbox', torch.tensor(0)).item()
                print(f"  Step {step:3d}: loss={loss_val:.4f} (cls={cls_loss:.4f}, pts={pts_loss:.4f}, bbox={bbox_loss:.4f})")
            
            step += 1
        
        if nan_detected:
            break
    
    print("-" * 70)
    
    # ========== 结果分析 ==========
    print(f"\n[5/5] 结果分析")
    print("=" * 70)
    
    if nan_detected:
        print("❌ 测试失败: 检测到 NaN/Inf!")
        return False
    
    if len(losses) < 2:
        print("❌ 测试失败: 训练步数不足!")
        return False
    
    # 检查 loss 是否下降
    first_loss = sum(losses[:3]) / 3  # 前 3 步平均
    last_loss = sum(losses[-3:]) / 3   # 后 3 步平均
    
    print(f"\n  Loss 统计:")
    print(f"    初始 (前3步平均): {first_loss:.4f}")
    print(f"    最终 (后3步平均): {last_loss:.4f}")
    print(f"    变化: {last_loss - first_loss:+.4f} ({(last_loss/first_loss - 1)*100:+.1f}%)")
    
    # 判断是否成功
    all_passed = True
    
    print(f"\n  检查项:")
    
    # 1. 无 NaN
    print(f"    [✓] 无 NaN/Inf")
    
    # 2. 训练跑通
    print(f"    [✓] 训练流程跑通 ({step} 步)")
    
    # 3. Loss 下降 (允许小幅波动)
    if last_loss < first_loss * 1.1:  # 允许 10% 的波动
        print(f"    [✓] Loss 趋势正常")
    else:
        print(f"    [⚠️] Loss 上升较多，可能需要调整学习率")
        all_passed = False
    
    print("\n" + "=" * 70)
    if all_passed:
        print("✅ 最小测试通过！可以开始正式训练")
    else:
        print("⚠️ 测试有警告，建议检查后再正式训练")
    print("=" * 70)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
