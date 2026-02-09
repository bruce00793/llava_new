"""
Test the complete LLaVA Map Detection Model

This script tests:
1. Q-Former: images → scene tokens
2. LLM: text + scene + queries → features
3. Decoder: features → predictions
4. Loss: predictions + GT → loss values

Usage:
    python test_map_model.py
"""

import sys
import torch
from llava.model.map_llava_model import build_map_detector
from llava.constants import IMAGE_TOKEN_INDEX

print("="*80)
print("Testing Complete LLaVA Map Detection Pipeline")
print("="*80)

# ===== Step 1: Create mock data =====
print("\n[Step 1] Creating mock data...")
batch_size = 2

# Images: 6 camera views
images = torch.randn(batch_size, 6, 3, 336, 336)
print(f"  Images: {images.shape}")

# Text IDs with IMAGE_TOKEN_INDEX
text_ids = torch.randint(0, 32000, (batch_size, 100))
text_ids[:, 1] = IMAGE_TOKEN_INDEX  # Insert <image> token at position 1
print(f"  Text IDs: {text_ids.shape}")
print(f"  IMAGE_TOKEN_INDEX at position: {(text_ids == IMAGE_TOKEN_INDEX).nonzero()[:,1].tolist()}")

# GT data
num_gt = 8
gt_labels = torch.randint(0, 3, (batch_size, num_gt))
gt_points = torch.randn(batch_size, num_gt, 20, 2) * 0.5  # Normalized to ~[-1, 1]
gt_masks = torch.ones(batch_size, num_gt, dtype=torch.bool)
print(f"  GT labels: {gt_labels.shape}")
print(f"  GT points: {gt_points.shape}")
print(f"  GT masks: {gt_masks.shape}")

# ===== Step 2: Build model =====
print("\n[Step 2] Building model...")
print("  This may take a few minutes to download weights...")
print("\n  Options:")
print("    1. With BLIP-2 pretrained Q-Former (recommended, ~5min)")
print("    2. From scratch (fast, but worse performance)")
print("\n  Using: Option 2 (from scratch) for quick testing")
print("  Tip: Use qformer_pretrained='blip2' for better results")

try:
    model = build_map_detector(
        llm_path="lmsys/vicuna-7b-v1.5",
        freeze_llm=True,
        qformer_pretrained=None,  # Change to 'blip2' for pretrained
    )
    print("\n  ✅ Model built successfully!")
except Exception as e:
    print(f"  ❌ Error building model: {e}")
    print("  Note: This requires ~13GB GPU memory")
    sys.exit(1)

# ===== Step 3: Forward pass (inference mode) =====
print("\n[Step 3] Forward pass (inference)...")
model.eval()

with torch.no_grad():
    try:
        output = model(images=images, text_ids=text_ids, return_loss=False)
        print("  ✅ Forward pass successful!")
        print(f"\n  Output shapes:")
        print(f"    pred_logits: {output['pred_logits'].shape}")  # (2, 50, 3)
        print(f"    pred_points: {output['pred_points'].shape}")  # (2, 50, 20, 2)
        print(f"    instance_features: {output['instance_features'].shape}")  # (2, 50, 4096)
        print(f"    point_features: {output['point_features'].shape}")  # (2, 50, 20, 4096)
    except Exception as e:
        print(f"  ❌ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ===== Step 4: Forward pass with loss =====
print("\n[Step 4] Forward pass with loss computation...")
model.train()

try:
    output = model(
        images=images,
        text_ids=text_ids,
        return_loss=True,
        gt_labels=gt_labels,
        gt_points=gt_points,
        gt_masks=gt_masks,
    )
    print("  ✅ Loss computation successful!")
    print(f"\n  Loss values:")
    loss_dict = output['loss_dict']
    for key, value in loss_dict.items():
        if torch.is_tensor(value):
            print(f"    {key}: {value.item():.4f}")
        else:
            print(f"    {key}: {value}")
    print(f"\n  Total loss: {output['loss'].item():.4f}")
except Exception as e:
    print(f"  ❌ Loss computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== Step 5: Test prediction with filtering =====
print("\n[Step 5] Prediction with score filtering...")
model.eval()

with torch.no_grad():
    try:
        predictions = model.predict(
            images=images,
            text_ids=text_ids,
            score_threshold=0.3,
        )
        print("  ✅ Prediction successful!")
        for b, pred in enumerate(predictions):
            num_detected = len(pred['labels'])
            print(f"\n  Sample {b}: {num_detected} detections")
            if num_detected > 0:
                print(f"    Labels: {pred['labels'][:5].tolist()}...")  # First 5
                print(f"    Scores: {pred['scores'][:5].tolist()}...")
                print(f"    Points shape: {pred['points'].shape}")
    except Exception as e:
        print(f"  ❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# ===== Summary =====
print("\n" + "="*80)
print("✅ All tests passed!")
print("="*80)
print("\nNext steps:")
print("  1. Prepare real nuScenes data using DataLoader")
print("  2. Write training script with optimizer")
print("  3. Train on GPU with mixed precision")
print("  4. Evaluate on validation set")
print("="*80)

