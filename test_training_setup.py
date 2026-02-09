"""
Quick test script to verify training setup
Tests model building, optimizer configuration, and data loading
"""

import os
import sys
import torch
from transformers import AutoTokenizer, CLIPImageProcessor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset


def test_model_building():
    """Test 1: Model building with BLIP-2 pretrained"""
    print("\n" + "="*60)
    print("Test 1: Building Model")
    print("="*60)
    
    try:
        model = build_map_detector(
            llm_path='lmsys/vicuna-7b-v1.5',
            freeze_llm=True,
            qformer_pretrained='blip2',
        )
        print("✅ Model built successfully!")
        
        # Check trainable parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"\nParameter Statistics:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Frozen: {total_params - trainable_params:,}")
        print(f"  Trainable %: {100 * trainable_params / total_params:.2f}%")
        
        return model
    
    except Exception as e:
        print(f"❌ Model building failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_optimizer_groups(model):
    """Test 2: Optimizer parameter groups"""
    print("\n" + "="*60)
    print("Test 2: Optimizer Parameter Groups")
    print("="*60)
    
    # Separate parameters
    qformer_params = []
    queries_params = []
    decoder_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        
        if 'qformer' in name:
            qformer_params.append(param)
        elif 'map_queries' in name:
            queries_params.append(param)
        elif 'decoder' in name:
            decoder_params.append(param)
        else:
            other_params.append(param)
    
    # Print counts
    print("\nParameter Groups:")
    print(f"  Q-Former:    {sum(p.numel() for p in qformer_params):12,} params")
    print(f"  Map Queries: {sum(p.numel() for p in queries_params):12,} params")
    print(f"  Decoder:     {sum(p.numel() for p in decoder_params):12,} params")
    
    if other_params:
        print(f"  Other:       {sum(p.numel() for p in other_params):12,} params (unexpected!)")
    
    # Create optimizer
    param_groups = [
        {'params': qformer_params, 'lr': 1e-5, 'name': 'qformer'},
        {'params': queries_params, 'lr': 1e-4, 'name': 'queries'},
        {'params': decoder_params, 'lr': 1e-4, 'name': 'decoder'},
    ]
    
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.01)
    
    print("\n✅ Optimizer created with 3 parameter groups")
    return optimizer


def test_forward_pass(model):
    """Test 3: Forward pass"""
    print("\n" + "="*60)
    print("Test 3: Forward Pass")
    print("="*60)
    
    try:
        model = model.cuda()
        model.train()
        
        # Dummy data
        batch_size = 2
        images = torch.randn(batch_size, 6, 3, 336, 336).cuda()
        text_ids = torch.randint(0, 32000, (batch_size, 100)).cuda()
        gt_labels = torch.randint(0, 3, (batch_size, 10)).cuda()
        gt_points = torch.randn(batch_size, 10, 20, 2).cuda()
        gt_masks = torch.ones(batch_size, 10, dtype=torch.bool).cuda()
        
        print(f"\nInput shapes:")
        print(f"  images: {list(images.shape)}")
        print(f"  text_ids: {list(text_ids.shape)}")
        print(f"  gt_labels: {list(gt_labels.shape)}")
        print(f"  gt_points: {list(gt_points.shape)}")
        
        # Forward
        output = model(
            images=images,
            text_ids=text_ids,
            return_loss=True,
            gt_labels=gt_labels,
            gt_points=gt_points,
            gt_masks=gt_masks,
        )
        
        print(f"\nOutput keys: {list(output.keys())}")
        print(f"  pred_logits: {list(output['pred_logits'].shape)}")
        print(f"  pred_points: {list(output['pred_points'].shape)}")
        print(f"  loss: {output['loss'].item():.4f}")
        
        if 'loss_dict' in output:
            print(f"\nLoss breakdown:")
            for key, value in output['loss_dict'].items():
                print(f"    {key}: {value.item():.4f}")
        
        print("\n✅ Forward pass successful!")
        return output
    
    except Exception as e:
        print(f"❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_backward_pass(model, optimizer, output):
    """Test 4: Backward pass"""
    print("\n" + "="*60)
    print("Test 4: Backward Pass")
    print("="*60)
    
    try:
        # Backward
        loss = output['loss']
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                
                if 'qformer' in name:
                    group = 'qformer'
                elif 'map_queries' in name:
                    group = 'queries'
                elif 'decoder' in name:
                    group = 'decoder'
                else:
                    group = 'other'
                
                if group not in grad_norms:
                    grad_norms[group] = []
                grad_norms[group].append(grad_norm)
        
        print("\nGradient statistics:")
        for group, norms in grad_norms.items():
            avg_norm = sum(norms) / len(norms)
            max_norm = max(norms)
            print(f"  {group:12s}: avg={avg_norm:.6f}, max={max_norm:.6f}, count={len(norms)}")
        
        # Optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()
        
        print("\n✅ Backward pass and optimizer step successful!")
        return True
    
    except Exception as e:
        print(f"❌ Backward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_loading():
    """Test 5: Data loading (if dataset available)"""
    print("\n" + "="*60)
    print("Test 5: Data Loading (Optional)")
    print("="*60)
    
    # This test is optional - only runs if dataset is configured
    print("⚠️  Skipped - Configure dataset paths in test script to enable")
    print("   Modify DATAROOT and VERSION below to test data loading")
    
    # Uncomment and modify these to test data loading:
    # DATAROOT = "/path/to/nuscenes"
    # VERSION = "v1.0-mini"
    # 
    # tokenizer = AutoTokenizer.from_pretrained('lmsys/vicuna-7b-v1.5', use_fast=False)
    # image_processor = CLIPImageProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
    # 
    # dataset = MapDetectionDataset(
    #     dataroot=DATAROOT,
    #     version=VERSION,
    #     split='train',
    #     image_processor=image_processor,
    #     tokenizer=tokenizer,
    # )
    # 
    # print(f"✅ Dataset loaded: {len(dataset)} samples")


def main():
    print("\n" + "="*70)
    print("Training Setup Verification")
    print("="*70)
    print("This script tests:")
    print("  1. Model building with BLIP-2 pretrained Q-Former")
    print("  2. Optimizer parameter group configuration")
    print("  3. Forward pass with loss computation")
    print("  4. Backward pass and gradient flow")
    print("  5. Data loading (optional)")
    print("="*70)
    
    # Test 1: Model building
    model = test_model_building()
    if model is None:
        print("\n❌ Setup verification failed at model building")
        return
    
    # Test 2: Optimizer
    optimizer = test_optimizer_groups(model)
    
    # Test 3: Forward
    output = test_forward_pass(model)
    if output is None:
        print("\n❌ Setup verification failed at forward pass")
        return
    
    # Test 4: Backward
    success = test_backward_pass(model, optimizer, output)
    if not success:
        print("\n❌ Setup verification failed at backward pass")
        return
    
    # Test 5: Data loading
    test_data_loading()
    
    # Summary
    print("\n" + "="*70)
    print("✅ All core tests passed!")
    print("="*70)
    print("\nYou are ready to start training:")
    print("  1. Generate GT cache: python tools/generate_gt_cache.py")
    print("  2. Configure paths in: scripts/train_stage2.sh")
    print("  3. Start training: bash scripts/train_stage2.sh")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()

