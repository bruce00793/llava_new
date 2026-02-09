"""Example: Using Map Detection Loss"""
import torch
from llava.model.map_detection_head import MapDetectionHead
from llava.model.map_loss import MapDetectionLoss

# Create modules
detection_head = MapDetectionHead()
loss_fn = MapDetectionLoss()

# Simulate data
batch_size = 2
instance_features = torch.randn(batch_size, 50, 4096)
point_features = torch.randn(batch_size, 50, 20, 4096)

gt_labels = [torch.tensor([0, 1]), torch.tensor([2])]
gt_lines = [torch.randn(2, 20, 2)*0.5, torch.randn(1, 20, 2)*0.5]
gt_masks = [torch.ones(2, 20, dtype=torch.bool), torch.ones(1, 20, dtype=torch.bool)]

# Forward
pred_logits, pred_lines = detection_head(instance_features, point_features)
total_loss, loss_dict = loss_fn(pred_logits, pred_lines, gt_labels, gt_lines, gt_masks)

# Backward
total_loss.backward()

print("Loss computed:")
print(f"  Classification: {loss_dict['loss_cls']:.4f}")
print(f"  Points: {loss_dict['loss_pts']:.4f}")
print(f"  Direction: {loss_dict['loss_dir']:.4f}")
print(f"  Total: {loss_dict['loss_total']:.4f}")
print("✓ Training step completed!")
