"""
Minimal training test - 验证训练pipeline能否正常运行
只训练3个steps，用于快速验证代码
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset

print("\n" + "="*70)
print("最小训练测试 - 验证代码能否正常运行")
print("="*70)

# 配置
DATAROOT = "/home/cly/auto/llava_test/LLaVA/data/nuscenes_mini"
VERSION = "v1.0-mini"
BATCH_SIZE = 1  # 最小batch size
NUM_STEPS = 3   # 只训练3步
USE_FP16 = True  # Enable FP16 to reduce memory usage

print(f"\n配置:")
print(f"  数据路径: {DATAROOT}")
print(f"  版本: {VERSION}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  测试步数: {NUM_STEPS}")
print(f"  混合精度: {USE_FP16}")

# Step 1: 构建模型
print("\n" + "="*70)
print("Step 1: 构建模型")
print("="*70)

try:
    # 使用本地模型权重
    LOCAL_LLM_PATH = "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
    print(f"使用本地模型: {LOCAL_LLM_PATH}")
    
    # Q-Former预训练配置
    # 不使用BLIP-2，原因：
    # 1. BLIP-2使用ViT，我们使用ResNet50，架构不匹配（只有39%权重能匹配）
    # 2. ResNet50已有ImageNet预训练，足够强大
    # 3. 随机初始化更稳定，所有参数从同一起点开始
    qformer_pretrained = None
    print(f"Q-Former: 随机初始化 (ResNet50已有ImageNet预训练)")
    
    model = build_map_detector(
        llm_path=LOCAL_LLM_PATH,
        freeze_llm=True,
        qformer_pretrained=qformer_pretrained,
    )
    # 注意：不要调用model.cuda()，因为LLM已经通过device_map="auto"分配好设备了
    model.train()
    print("✅ 模型构建成功")
    print(f"   模型已自动分配到GPU（device_map='auto'）")
except Exception as e:
    print(f"❌ 模型构建失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 2: 创建优化器
print("\n" + "="*70)
print("Step 2: 创建优化器")
print("="*70)

try:
    # 分组参数
    qformer_params = []
    queries_params = []
    decoder_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'qformer' in name:
            qformer_params.append(param)
        elif 'map_queries' in name:
            queries_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': qformer_params, 'lr': 1e-5},
        {'params': queries_params, 'lr': 1e-4},
        {'params': decoder_params, 'lr': 1e-4},
    ], weight_decay=0.01)
    
    print(f"✅ 优化器创建成功")
    print(f"   Q-Former参数: {sum(p.numel() for p in qformer_params):,}")
    print(f"   Queries参数: {sum(p.numel() for p in queries_params):,}")
    print(f"   Decoder参数: {sum(p.numel() for p in decoder_params):,}")
except Exception as e:
    print(f"❌ 优化器创建失败: {e}")
    sys.exit(1)

# Step 3: 加载数据集
print("\n" + "="*70)
print("Step 3: 加载数据集")
print("="*70)

try:
    # 使用本地tokenizer和image processor
    LOCAL_LLM_PATH = "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
    LOCAL_CLIP_PATH = "/home/cly/auto/llava_test/LLaVA/clip-vit-large-patch14-336"
    
    print(f"加载tokenizer: {LOCAL_LLM_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_PATH, use_fast=False)
    
    print(f"加载image processor: {LOCAL_CLIP_PATH}")
    image_processor = CLIPImageProcessor.from_pretrained(LOCAL_CLIP_PATH)
    
    # 检查GT cache
    gt_cache_dir = os.path.join(DATAROOT, f'gt_cache_{VERSION}_train.pkl')
    if not os.path.exists(gt_cache_dir):
        print(f"❌ GT cache不存在: {gt_cache_dir}")
        print("   请先运行: python tools/generate_gt_cache.py --split train")
        sys.exit(1)
    
    dataset = MapDetectionDataset(
        dataroot=DATAROOT,
        version=VERSION,
        split='train',
        gt_cache_path=gt_cache_dir,
        image_processor=image_processor,
        tokenizer=tokenizer,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 不shuffle，方便调试
        num_workers=0,  # 单进程，方便调试
        collate_fn=dataset.collate_fn,
    )
    
    print(f"✅ 数据集加载成功")
    print(f"   样本数量: {len(dataset)}")
    print(f"   Batch数量: {len(dataloader)}")
except Exception as e:
    print(f"❌ 数据集加载失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: 训练测试
print("\n" + "="*70)
print("Step 4: 训练测试 ({}步)".format(NUM_STEPS))
print("="*70)

# Note: Not using GradScaler because we have mixed FP16/FP32 trainable params
# (map_queries in LLM is FP16, Q-Former and Decoder are FP32)

try:
    for step, batch in enumerate(dataloader):
        if step >= NUM_STEPS:
            break
        
        print(f"\n--- Step {step+1}/{NUM_STEPS} ---")
        
        # 移动到GPU（使用GPU 0，因为Q-Former和Decoder在GPU 0）
        device = torch.device('cuda:0')
        images = batch['images'].to(device)
        text_ids = batch['text_ids'].to(device)
        gt_labels = batch['gt_labels'].to(device)
        gt_points = batch['gt_points'].to(device)
        gt_masks = batch['gt_masks'].to(device)
        
        # 相机参数（用于3D位置编码）
        cam_intrinsics = batch.get('cam_intrinsics')
        cam_extrinsics = batch.get('cam_extrinsics')
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.to(device)
        if cam_extrinsics is not None:
            cam_extrinsics = cam_extrinsics.to(device)
        
        print(f"输入形状:")
        print(f"  images: {list(images.shape)}")
        print(f"  text_ids: {list(text_ids.shape)}")
        print(f"  gt_labels: {list(gt_labels.shape)}")
        print(f"  gt_points: {list(gt_points.shape)}")
        print(f"  gt_masks: {list(gt_masks.shape)}")
        print(f"  有效GT数量: {gt_masks.sum().item()}")
        
        # Forward with loss calculation
        with autocast(enabled=USE_FP16):
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
            
            # Check decoder outputs
            pred_logits = output['pred_logits']
            pred_points = output['pred_points']
            print(f"\nDecoder outputs:")
            print(f"  pred_logits: {list(pred_logits.shape)}, has_nan={torch.isnan(pred_logits).any()}")
            if not torch.isnan(pred_logits).any():
                print(f"    range: [{pred_logits.min():.4f}, {pred_logits.max():.4f}]")
            print(f"  pred_points: {list(pred_points.shape)}, has_nan={torch.isnan(pred_points).any()}")
            if not torch.isnan(pred_points).any():
                print(f"    range: [{pred_points.min():.4f}, {pred_points.max():.4f}]")
            
            loss = output['loss']
            loss_dict = output['loss_dict']
        
        print(f"输出:")
        print(f"  pred_logits: {list(output['pred_logits'].shape)}")
        print(f"  pred_points: {list(output['pred_points'].shape)}")
        print(f"  loss_total: {loss.item():.4f}")
        for key, value in loss_dict.items():
            if key != 'loss_total':
                print(f"  {key}: {value.item():.4f}")
        
        # Backward - don't use GradScaler since we have mixed FP16/FP32 trainable params
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        trainable_params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
        if trainable_params:
            torch.nn.utils.clip_grad_norm_(trainable_params, 0.1)
        optimizer.step()
        
        print(f"✅ Step {step+1} 完成")
    
    print("\n" + "="*70)
    print("✅ 训练测试成功！所有步骤正常运行！")
    print("="*70)
    print("\n测试总结:")
    print(f"  ✓ 模型构建成功")
    print(f"  ✓ 数据加载成功")
    print(f"  ✓ Forward pass成功")
    print(f"  ✓ Loss计算成功")
    print(f"  ✓ Backward pass成功")
    print(f"  ✓ 优化器更新成功")
    print("\n🎉 代码可以正常训练！可以使用完整数据集进行训练了！")
    print("="*70)

except Exception as e:
    print(f"\n❌ 训练测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

