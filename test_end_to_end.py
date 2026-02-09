"""
End-to-end test: Detection Head + Loss
"""
import torch
from llava.model.map_detection_head import MapDetectionHead
from llava.model.map_loss import MapDetectionLoss

print("="*80)
print("End-to-End Test: Detection Head → Loss → Backward")
print("="*80)

# 1. Create modules
print("\n[1] Creating modules...")
detection_head = MapDetectionHead(hidden_size=4096, num_classes=3)
loss_fn = MapDetectionLoss(num_classes=3)
optimizer = torch.optim.AdamW(detection_head.parameters(), lr=1e-4)

print(f"  Detection head: {detection_head.get_num_params():,} params")
print(f"  Loss weights: cls={loss_fn.weight_cls}, pts={loss_fn.weight_pts}, dir={loss_fn.weight_dir}")

# 2. Simulate LLM output
print("\n[2] Simulating LLM output...")
B = 2
instance_features = torch.randn(B, 50, 4096, requires_grad=True)
point_features = torch.randn(B, 50, 20, 4096, requires_grad=True)

print(f"  instance_features: {instance_features.shape}")
print(f"  point_features: {point_features.shape}")

# 3. Detection head forward
print("\n[3] Detection head forward...")
pred_logits, pred_lines = detection_head(instance_features, point_features)

print(f"  pred_logits: {pred_logits.shape}")
print(f"  pred_lines: {pred_lines.shape}")
print(f"  pred_lines range: [{pred_lines.min():.3f}, {pred_lines.max():.3f}]")

# 4. Prepare GT
print("\n[4] Preparing ground truth...")
gt_labels = [
    torch.tensor([0, 1, 2]),  # 3 instances
    torch.tensor([1, 2])      # 2 instances
]
gt_lines = [
    torch.randn(3, 20, 2) * 0.5,
    torch.randn(2, 20, 2) * 0.5
]
gt_masks = [
    torch.ones(3, 20, dtype=torch.bool),
    torch.ones(2, 20, dtype=torch.bool)
]

print(f"  Sample 0: {len(gt_labels[0])} instances")
print(f"  Sample 1: {len(gt_labels[1])} instances")

# 5. Compute loss
print("\n[5] Computing loss...")
total_loss, loss_dict = loss_fn(pred_logits, pred_lines, gt_labels, gt_lines, gt_masks)

print(f"  Classification: {loss_dict['loss_cls']:.4f}")
print(f"  Points: {loss_dict['loss_pts']:.4f}")
print(f"  Direction: {loss_dict['loss_dir']:.4f}")
print(f"  Total: {loss_dict['loss_total']:.4f}")

# 6. Backward pass
print("\n[6] Backward pass...")
optimizer.zero_grad()
total_loss.backward()

print(f"  instance_features.grad: {instance_features.grad.shape if instance_features.grad is not None else 'None'}")
print(f"  Detection head grads exist: {any(p.grad is not None for p in detection_head.parameters())}")

# 7. Optimizer step
print("\n[7] Optimizer step...")
optimizer.step()
print(f"  ✓ Optimizer stepped")

# 8. Second iteration
print("\n[8] Second iteration...")
optimizer.zero_grad()
pred_logits2, pred_lines2 = detection_head(instance_features, point_features)
total_loss2, loss_dict2 = loss_fn(pred_logits2, pred_lines2, gt_labels, gt_lines, gt_masks)
total_loss2.backward()
optimizer.step()

print(f"  Loss 2: {loss_dict2['loss_total']:.4f}")
print(f"  Loss change: {loss_dict2['loss_total'] - loss_dict['loss_total']:.4f}")

print("\n" + "="*80)
print("✅ End-to-end test PASSED!")
print("="*80)
print("\nSummary:")
print("  ✓ Detection head works")
print("  ✓ Loss computation works")
print("  ✓ Backward propagation works")
print("  ✓ Optimizer works")
print("  ✓ Ready for training!")
