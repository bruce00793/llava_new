"""
Map Detection Loss with Hungarian Matching

Includes:
1. Hungarian matcher with equivalence class handling
2. Three types of losses: classification, points, direction

Following MapTR design philosophy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from typing import Dict, List, Tuple


class HungarianMatcher(nn.Module):
    """
    Hungarian matcher for map detection with equivalence class handling.
    
    Equivalence classes:
    - Polyline (dividers): 2 classes (forward/backward)
    - Polygon (crossing): 20 classes (each point as start)
    """
    
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_points: float = 5.0,
    ):
        """
        Args:
            cost_class: Weight for classification cost
            cost_points: Weight for points L1 cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_points = cost_points
    
    @torch.no_grad()
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_lines: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_lines: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Perform Hungarian matching for each sample in batch.
        
        Args:
            pred_logits: [B, N, num_classes] classification logits
            pred_lines: [B, N, P, 2] predicted line coordinates
            gt_labels: List of [num_gt] tensors
            gt_lines: List of [num_gt, P, 2] tensors
            gt_masks: List of [num_gt, P] tensors (valid point masks)
        
        Returns:
            List of (pred_indices, gt_indices, best_gt_lines) for each sample
        """
        B, N, num_classes = pred_logits.shape
        _, _, P, _ = pred_lines.shape
        
        # Convert to probabilities
        pred_probs = pred_logits.softmax(dim=-1)  # [B, N, num_classes]
        
        indices = []
        
        for b in range(B):
            # Get data for this sample
            pred_prob = pred_probs[b]  # [N, num_classes]
            pred_line = pred_lines[b]  # [N, P, 2]
            gt_label = gt_labels[b]    # [num_gt]
            gt_line = gt_lines[b]      # [num_gt, P, 2]
            gt_mask = gt_masks[b]      # [num_gt, P]
            
            num_gt = len(gt_label)
            if num_gt == 0:
                # No GT for this sample
                indices.append((
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, P, 2)
                ))
                continue
            
            # 1. Classification cost: [N, num_gt]
            cost_class = -pred_prob[:, gt_label]  # Negative log likelihood
            
            # 2. Points cost with equivalence class handling: [N, num_gt]
            cost_points = self._compute_points_cost(
                pred_line, gt_line, gt_label, gt_mask
            )
            
            # 3. Total cost matrix: [N, num_gt]
            cost_matrix = (
                self.cost_class * cost_class +
                self.cost_points * cost_points
            )
            
            # 4. Hungarian algorithm
            # Handle NaN/Inf in cost matrix (safety check)
            if torch.isnan(cost_matrix).any() or torch.isinf(cost_matrix).any():
                cost_matrix = torch.nan_to_num(cost_matrix, nan=1e6, posinf=1e6, neginf=-1e6)
            
            pred_idx, gt_idx = linear_sum_assignment(cost_matrix.cpu().numpy())
            
            # 5. Find best transformations for matched pairs
            best_gt = self._find_best_transformations(
                pred_line[pred_idx],
                gt_line[gt_idx],
                gt_label[gt_idx],
                gt_mask[gt_idx]
            )
            
            indices.append((
                torch.as_tensor(pred_idx, dtype=torch.long),
                torch.as_tensor(gt_idx, dtype=torch.long),
                best_gt
            ))
        
        return indices
    
    def _compute_points_cost(
        self,
        pred_lines: torch.Tensor,
        gt_lines: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute points L1 cost with equivalence class handling.
        
        Args:
            pred_lines: [N, P, 2]
            gt_lines: [num_gt, P, 2]
            gt_labels: [num_gt]
            gt_masks: [num_gt, P]
        
        Returns:
            cost: [N, num_gt]
        """
        N, P, _ = pred_lines.shape
        num_gt = len(gt_labels)
        
        cost = torch.zeros(N, num_gt, device=pred_lines.device)
        
        for i in range(num_gt):
            gt_line = gt_lines[i]      # [P, 2]
            gt_label = gt_labels[i]
            gt_mask = gt_masks[i]      # [P]
            
            if gt_label in [0, 2]:  # Polyline: divider (0) 和 boundary (2) 是非闭合线
                # 2 equivalence classes: forward and backward
                cost[:, i] = self._polyline_cost(pred_lines, gt_line, gt_mask)
            
            elif gt_label == 1:  # Polygon: ped_crossing (1) 是闭合多边形
                # 20 equivalence classes: each point as start
                cost[:, i] = self._polygon_cost(pred_lines, gt_line, gt_mask)
        
        return cost
    
    def _polyline_cost(
        self,
        pred_lines: torch.Tensor,
        gt_line: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute polyline cost (min over 2 directions).
        
        Args:
            pred_lines: [N, P, 2]
            gt_line: [P, 2]
            gt_mask: [P]
        
        Returns:
            cost: [N]
        """
        # Forward
        diff_forward = torch.abs(pred_lines - gt_line)  # [N, P, 2]
        cost_forward = (diff_forward * gt_mask.unsqueeze(-1)).sum(dim=[1, 2])
        
        # Backward
        gt_reversed = torch.flip(gt_line, dims=[0])
        mask_reversed = torch.flip(gt_mask, dims=[0])
        diff_backward = torch.abs(pred_lines - gt_reversed)
        cost_backward = (diff_backward * mask_reversed.unsqueeze(-1)).sum(dim=[1, 2])
        
        # Take minimum
        cost = torch.min(cost_forward, cost_backward)
        
        # Normalize by number of valid points
        num_valid = gt_mask.sum()
        if num_valid > 0:
            cost = cost / num_valid
        
        return cost
    
    def _polygon_cost(
        self,
        pred_lines: torch.Tensor,
        gt_line: torch.Tensor,
        gt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute polygon cost (min over 20 rotations).
        
        Args:
            pred_lines: [N, P, 2]
            gt_line: [P, 2]
            gt_mask: [P]
        
        Returns:
            cost: [N]
        """
        N, P, _ = pred_lines.shape
        
        costs = []
        for shift in range(P):
            # Rotate GT
            gt_shifted = torch.roll(gt_line, shifts=-shift, dims=0)
            mask_shifted = torch.roll(gt_mask, shifts=-shift, dims=0)
            
            # Compute cost
            diff = torch.abs(pred_lines - gt_shifted)  # [N, P, 2]
            cost = (diff * mask_shifted.unsqueeze(-1)).sum(dim=[1, 2])
            costs.append(cost)
        
        # Take minimum over all rotations
        costs = torch.stack(costs, dim=0)  # [P, N]
        cost = costs.min(dim=0)[0]  # [N]
        
        # Normalize by number of valid points
        num_valid = gt_mask.sum()
        if num_valid > 0:
            cost = cost / num_valid
        
        return cost
    
    def _find_best_transformations(
        self,
        pred_lines: torch.Tensor,
        gt_lines: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_masks: torch.Tensor,
    ) -> torch.Tensor:
        """
        Find best equivalence class transformation for each matched pair.
        
        Args:
            pred_lines: [num_matched, P, 2]
            gt_lines: [num_matched, P, 2]
            gt_labels: [num_matched]
            gt_masks: [num_matched, P]
        
        Returns:
            best_gt_lines: [num_matched, P, 2]
        """
        num_matched = len(pred_lines)
        P = pred_lines.shape[1]
        
        best_gt_lines = []
        
        for i in range(num_matched):
            pred = pred_lines[i]     # [P, 2]
            gt = gt_lines[i]         # [P, 2]
            label = gt_labels[i]
            mask = gt_masks[i]       # [P]
            
            if label in [0, 2]:  # Polyline: divider (0) 和 boundary (2) 是非闭合线
                # Try forward and backward
                diff_forward = torch.abs(pred - gt)
                cost_forward = (diff_forward * mask.unsqueeze(-1)).sum()
                
                gt_reversed = torch.flip(gt, dims=[0])
                diff_backward = torch.abs(pred - gt_reversed)
                cost_backward = (diff_backward * mask.unsqueeze(-1)).sum()
                
                best_gt = gt if cost_forward < cost_backward else gt_reversed
            
            elif label == 1:  # Polygon: ped_crossing (1) 是闭合多边形
                # Try all 20 rotations
                best_cost = float('inf')
                best_gt = gt
                
                for shift in range(P):
                    gt_shifted = torch.roll(gt, shifts=-shift, dims=0)
                    diff = torch.abs(pred - gt_shifted)
                    cost = (diff * mask.unsqueeze(-1)).sum()
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_gt = gt_shifted
            else:
                best_gt = gt
            
            best_gt_lines.append(best_gt)
        
        return torch.stack(best_gt_lines, dim=0) if best_gt_lines else torch.empty(0, P, 2)


class FocalLoss(nn.Module):
    """
    Focal Loss with sigmoid (binary classification for each class).
    
    Following MapTR/DETR standard: use_sigmoid=True
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, num_classes: int = 3):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
    
    def forward(
        self, 
        inputs: torch.Tensor, 
        targets: torch.Tensor,
        avg_factor: float = None
    ) -> torch.Tensor:
        """
        Sigmoid Focal Loss (binary classification for each class).
        
        Args:
            inputs: [N, num_classes] logits
            targets: [N] class indices (can include background class = num_classes)
            avg_factor: normalization factor
        
        Returns:
            loss: scalar
        """
        N, C = inputs.shape
        
        # Create one-hot targets (including background as all zeros)
        target_onehot = torch.zeros(N, C, device=inputs.device)
        
        # Only set 1 for foreground classes
        fg_mask = targets < self.num_classes
        if fg_mask.sum() > 0:
            target_onehot[fg_mask, targets[fg_mask]] = 1
        
        # Sigmoid focal loss (binary classification for each class)
        pred_sigmoid = inputs.sigmoid()
        
        # Focal weight
        pt = (1 - pred_sigmoid) * target_onehot + pred_sigmoid * (1 - target_onehot)
        focal_weight = (self.alpha * target_onehot + (1 - self.alpha) * (1 - target_onehot)) * pt.pow(self.gamma)
        
        # Binary cross entropy
        bce = F.binary_cross_entropy_with_logits(inputs, target_onehot, reduction='none')
        
        loss = focal_weight * bce
        
        # Normalization
        if avg_factor is None:
            return loss.sum() / max(N, 1)
        else:
            return loss.sum() / max(avg_factor, 1)



class MapDetectionLoss(nn.Module):
    """
    Complete loss for map detection.
    
    损失组成：
    - Classification loss: Focal Loss with sigmoid
    - Points loss: L1 Loss
    - Direction loss: Cosine Similarity Loss
    
    损失公式：
    L = w_cls * L_cls + w_pts * L_pts + w_dir * L_dir
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        weight_cls: float = 2.0,
        weight_pts: float = 5.0,
        weight_dir: float = 0.25,         # 折中方案：方向损失贡献约5-8%，有意义但不主导
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
    ):
        """
        Args:
            num_classes: Number of classes (default: 3)
            weight_cls: Weight for classification loss (default: 2.0)
            weight_pts: Weight for points loss (default: 5.0)
            weight_dir: Weight for direction loss (default: 0.25, 折中方案)
                        - MapTR 用 0.005（几乎无作用）
                        - 0.25 让方向损失贡献约 5-8%，有意义但不主导训练
            focal_alpha: Focal loss alpha (default: 0.25)
            focal_gamma: Focal loss gamma (default: 2.0)
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.weight_cls = weight_cls
        self.weight_pts = weight_pts
        self.weight_dir = weight_dir
        
        # Matcher and losses
        self.matcher = HungarianMatcher(cost_class=weight_cls, cost_points=weight_pts)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, num_classes=num_classes)
    
    def forward(
        self,
        pred_logits: torch.Tensor,
        pred_lines: torch.Tensor,
        gt_labels: List[torch.Tensor],
        gt_lines: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute total loss.
        
        Args:
            pred_logits: [B, N, num_classes] classification logits
            pred_lines: [B, N, P, 2] predicted line coordinates
            gt_labels: List of [num_gt] tensors
            gt_lines: List of [num_gt, P, 2] tensors
            gt_masks: List of [num_gt, P] tensors
        
        Returns:
            total_loss: scalar
            loss_dict: dictionary of individual losses
        """
        # ========== 数据验证：检测异常输入 ==========
        # 检查预测值
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            print(f"⚠️ [Loss] pred_logits contains NaN/Inf!")
        if torch.isnan(pred_lines).any() or torch.isinf(pred_lines).any():
            print(f"⚠️ [Loss] pred_lines contains NaN/Inf!")
        
        # 检查 GT 数据
        for b, (gt_l, gt_ln, gt_m) in enumerate(zip(gt_labels, gt_lines, gt_masks)):
            if torch.isnan(gt_ln).any() or torch.isinf(gt_ln).any():
                print(f"⚠️ [Loss] Batch {b}: gt_lines contains NaN/Inf!")
            # 检查 GT 坐标范围（应该在 [-1, 1] 内）
            if gt_ln.numel() > 0:
                gt_min, gt_max = gt_ln.min().item(), gt_ln.max().item()
                if gt_min < -1.5 or gt_max > 1.5:  # 允许一点误差
                    print(f"⚠️ [Loss] Batch {b}: gt_lines out of range! min={gt_min:.3f}, max={gt_max:.3f}")
        
        # Hungarian matching
        indices = self.matcher(pred_logits, pred_lines, gt_labels, gt_lines, gt_masks)
        
        # Compute losses
        loss_cls = self._classification_loss(pred_logits, gt_labels, indices)
        loss_pts = self._points_loss(pred_lines, gt_lines, gt_masks, indices)
        loss_dir = self._direction_loss(pred_lines, gt_lines, gt_masks, indices)
        
        # ========== 数值稳定性：检测并替换 NaN/Inf ==========
        # 如果某个 loss 是 NaN 或 Inf，用 0 替换以保持训练稳定
        def safe_loss(loss, name):
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ {name} is NaN/Inf, replacing with 0")
                return torch.zeros_like(loss)
            return loss
        
        loss_cls = safe_loss(loss_cls, "loss_cls")
        loss_pts = safe_loss(loss_pts, "loss_pts")
        loss_dir = safe_loss(loss_dir, "loss_dir")
        
        # Total loss
        total_loss = (
            self.weight_cls * loss_cls +
            self.weight_pts * loss_pts +
            self.weight_dir * loss_dir
        )
        
        # 最终检查：如果 total_loss 还是 NaN，返回 0
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"⚠️ total_loss is NaN/Inf after individual checks, returning 0")
            total_loss = torch.zeros_like(total_loss)
        
        loss_dict = {
            'loss_cls': loss_cls.detach(),
            'loss_pts': loss_pts.detach(),
            'loss_dir': loss_dir.detach(),
            'loss_total': total_loss.detach(),
        }
        
        return total_loss, loss_dict
    
    def _classification_loss(
        self,
        pred_logits: torch.Tensor,
        gt_labels: List[torch.Tensor],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Compute classification loss.
        
        All predictions are classified (matched → GT class, unmatched → background).
        Following MapTR: use avg_factor for normalization.
        """
        B, N, num_classes = pred_logits.shape
        device = pred_logits.device
        
        # Build target classes
        target_classes = torch.full(
            (B, N), self.num_classes, dtype=torch.long, device=device
        )
        
        num_total_pos = 0
        num_total_neg = 0
        
        for b, (pred_idx, gt_idx, _) in enumerate(indices):
            if len(pred_idx) > 0:
                target_classes[b, pred_idx] = gt_labels[b][gt_idx]
                num_total_pos += len(pred_idx)
            num_total_neg += (N - len(pred_idx))
        
        # Compute avg_factor (following MapTR/DETR)
        # Background class weight = 0, so only positive samples contribute
        avg_factor = num_total_pos * 1.0
        avg_factor = max(avg_factor, 1.0)
        
        # Compute focal loss
        pred_logits_flat = pred_logits.reshape(-1, num_classes)
        target_classes_flat = target_classes.reshape(-1)
        
        loss = self.focal_loss(pred_logits_flat, target_classes_flat, avg_factor=avg_factor)
        
        return loss
    
    def _points_loss(
        self,
        pred_lines: torch.Tensor,
        gt_lines: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Compute points L1 loss (only on matched predictions).
        
        归一化方式：除以实例数（Following MapTR）
        这样长线比短线有更大的 loss，符合直觉（长线更重要）
        
        注意：loss 值会比较大（~10-20），需要配合较小的学习率
        """
        all_diff = []
        num_total_pos = 0
        
        for b, (pred_idx, gt_idx, best_gt) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            # Get matched predictions and transformed GT
            matched_pred = pred_lines[b, pred_idx]  # [num_matched, P, 2]
            # 修复：确保 GT 和 mask 在正确的设备和数据类型上
            matched_gt = best_gt.to(device=matched_pred.device, dtype=matched_pred.dtype)
            matched_mask = gt_masks[b][gt_idx].to(device=matched_pred.device)
            
            # L1 distance
            diff = torch.abs(matched_pred - matched_gt)  # [num_matched, P, 2]
            
            # Apply mask
            diff_masked = diff * matched_mask.unsqueeze(-1)
            
            all_diff.append(diff_masked.sum())
            num_total_pos += len(pred_idx)
        
        if num_total_pos == 0:
            return pred_lines.sum() * 0.0  # Return zero with gradient
        
        # 归一化：除以实例数（Following MapTR）
        total_loss = sum(all_diff) / max(num_total_pos, 1.0)
        
        return total_loss
    
    def _denormalize_pts(self, pts: torch.Tensor) -> torch.Tensor:
        """
        Denormalize points from [-1, 1] to physical coordinates.
        
        Args:
            pts: [..., 2] normalized coordinates in [-1, 1]
        
        Returns:
            [..., 2] physical coordinates in meters
        
        Note: pc_range = [-15, -30, -2, 15, 30, 2]
              x: [-1, 1] → [-15, 15] (30m range)
              y: [-1, 1] → [-30, 30] (60m range)
        """
        denorm_pts = pts.clone()
        # x: [-1, 1] → [-15, 15]
        denorm_pts[..., 0] = pts[..., 0] * 15.0  # half of x range
        # y: [-1, 1] → [-30, 30]
        denorm_pts[..., 1] = pts[..., 1] * 30.0  # half of y range
        return denorm_pts
    
    def _direction_loss(
        self,
        pred_lines: torch.Tensor,
        gt_lines: List[torch.Tensor],
        gt_masks: List[torch.Tensor],
        indices: List[Tuple],
    ) -> torch.Tensor:
        """
        Compute direction consistency loss (only on matched predictions).
        
        归一化方式：除以实例数（Following MapTR）
        这样长线比短线有更大的 loss，与 points_loss 一致
        """
        all_dir_loss = []
        num_total_pos = 0  # 统计实例数
        
        for b, (pred_idx, gt_idx, best_gt) in enumerate(indices):
            if len(pred_idx) == 0:
                continue
            
            # Get matched predictions and transformed GT
            matched_pred = pred_lines[b, pred_idx]  # [num_matched, P, 2]
            matched_gt = best_gt.to(matched_pred.device)  # [num_matched, P, 2]
            matched_mask = gt_masks[b][gt_idx]       # [num_matched, P]
            
            # Denormalize to physical coordinates (following MapTR)
            # This ensures correct direction computation for non-square BEV
            matched_pred_denorm = self._denormalize_pts(matched_pred)
            matched_gt_denorm = self._denormalize_pts(matched_gt)
            
            # Compute direction vectors in physical space
            pred_dirs = matched_pred_denorm[:, 1:] - matched_pred_denorm[:, :-1]  # [num_matched, P-1, 2]
            gt_dirs = matched_gt_denorm[:, 1:] - matched_gt_denorm[:, :-1]        # [num_matched, P-1, 2]
            
            # ========== 数值安全的方向损失 ==========
            # 【根本修复】使用 sqrt(x^2 + eps) 代替 .norm()
            # .norm() = sqrt(sum(x^2)) 在 x=0 时梯度为 0/0 = NaN
            # sqrt(sum(x^2) + eps) 在 x=0 时梯度为 0/sqrt(eps) = 0（安全）
            # 注意：torch.where(cond, a, b) 中即使 cond=False，a 的梯度仍会计算，
            # 而 NaN * 0 = NaN（IEEE 754），所以 .norm() 的 NaN 无法被掩码消除！
            eps_sq = 1e-6  # 加在 sum(x^2) 上，等效于 norm >= 1e-3
            pred_len = torch.sqrt((pred_dirs ** 2).sum(dim=-1, keepdim=True) + eps_sq)
            gt_len = torch.sqrt((gt_dirs ** 2).sum(dim=-1, keepdim=True) + eps_sq)
            
            # 安全归一化（pred_len 永远 > 0，不会除零）
            pred_dirs_norm = pred_dirs / pred_len
            gt_dirs_norm = gt_dirs / gt_len
            
            # 余弦相似度
            cosine_sim = (pred_dirs_norm * gt_dirs_norm).sum(dim=-1).clamp(-1.0, 1.0)
            
            # Direction loss: 1 - similarity
            dir_loss = 1.0 - cosine_sim  # [num_matched, P-1]
            
            # 边掩码：两个端点都有效 + 方向向量长度足够（用 raw squared length 判断，避免 .norm()）
            point_mask = matched_mask[:, :-1] & matched_mask[:, 1:]  # [num_matched, P-1]
            raw_pred_len_sq = (pred_dirs ** 2).sum(dim=-1)  # 梯度安全：d(x^2)/dx = 2x，零点梯度为 0
            raw_gt_len_sq = (gt_dirs ** 2).sum(dim=-1)
            valid_edge = (raw_pred_len_sq > 1e-4) & (raw_gt_len_sq > 1e-4)  # ~0.01m 阈值
            edge_mask = point_mask & valid_edge
            
            # Apply mask and sum
            dir_loss_masked = dir_loss * edge_mask.to(dir_loss.device)
            
            all_dir_loss.append(dir_loss_masked.sum())
            num_total_pos += len(pred_idx)  # 统计实例数
        
        if num_total_pos == 0:
            return pred_lines.sum() * 0.0  # Return zero with gradient
        
        # 归一化：除以实例数（Following MapTR）
        total_loss = sum(all_dir_loss) / max(num_total_pos, 1.0)
        
        return total_loss


def build_loss(config: dict = None) -> MapDetectionLoss:
    """Build loss from config.
    
    Loss 权重说明（Following MapTR 风格）：
    - weight_cls: 2.0 (分类 loss)
    - weight_pts: 5.0 (点位置 loss，主导项)
    - weight_dir: 0.25 (方向 loss，辅助项)
    
    注意：由于 loss 值较大（~280），需要配合较小的学习率使用
    """
    if config is None:
        config = {}
    
    return MapDetectionLoss(
        num_classes=config.get('num_classes', 3),
        weight_cls=config.get('weight_cls', 2.0),
        weight_pts=config.get('weight_pts', 5.0),
        weight_dir=config.get('weight_dir', 0.25),  # 方向 loss 权重（辅助项）
        focal_alpha=config.get('focal_alpha', 0.25),
        focal_gamma=config.get('focal_gamma', 2.0),
    )


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("Testing Map Detection Loss")
    print("="*80)
    
    # Create loss module
    loss_fn = MapDetectionLoss(
        num_classes=3,
        weight_cls=2.0,
        weight_pts=5.0,
        weight_dir=0.25,  # 折中方案
    )
    
    print("\n✓ Loss module created")
    print(f"  Classification weight: {loss_fn.weight_cls}")
    print(f"  Points weight: {loss_fn.weight_pts}")
    print(f"  Direction weight: {loss_fn.weight_dir}")
    
    # Test case 1: Simple matching
    print("\n" + "-"*80)
    print("Test 1: Simple matching")
    print("-"*80)
    
    B, N, P = 2, 50, 20
    
    # Predictions
    pred_logits = torch.randn(B, N, 3)
    pred_lines = torch.randn(B, N, P, 2) * 0.5  # [-0.5, 0.5]
    
    # Ground truth (2 instances per sample)
    gt_labels = [
        torch.tensor([0, 1]),  # Sample 0: road_divider, lane_divider
        torch.tensor([1, 2]),  # Sample 1: lane_divider, ped_crossing
    ]
    
    gt_lines = [
        torch.randn(2, P, 2) * 0.5,  # Sample 0
        torch.randn(2, P, 2) * 0.5,  # Sample 1
    ]
    
    gt_masks = [
        torch.ones(2, P, dtype=torch.bool),  # All valid
        torch.ones(2, P, dtype=torch.bool),  # All valid
    ]
    
    # Compute loss
    total_loss, loss_dict = loss_fn(pred_logits, pred_lines, gt_labels, gt_lines, gt_masks)
    
    print(f"\nPredictions:")
    print(f"  pred_logits: {pred_logits.shape}")
    print(f"  pred_lines: {pred_lines.shape}")
    
    print(f"\nGround Truth:")
    print(f"  Sample 0: {len(gt_labels[0])} instances (classes: {gt_labels[0].tolist()})")
    print(f"  Sample 1: {len(gt_labels[1])} instances (classes: {gt_labels[1].tolist()})")
    
    print(f"\nLosses:")
    print(f"  Classification: {loss_dict['loss_cls']:.4f}")
    print(f"  Points: {loss_dict['loss_pts']:.4f}")
    print(f"  Direction: {loss_dict['loss_dir']:.4f}")
    print(f"  Total: {loss_dict['loss_total']:.4f}")
    
    assert not torch.isnan(total_loss), "Loss is NaN!"
    assert total_loss > 0, "Loss should be positive!"
    
    print("\n✓ Test 1 passed!")
    
    # Test case 2: With padding
    print("\n" + "-"*80)
    print("Test 2: With padding (partial valid points)")
    print("-"*80)
    
    # GT with only 10 valid points
    gt_masks_partial = [
        torch.cat([torch.ones(2, 10), torch.zeros(2, 10)], dim=1).bool(),
        torch.cat([torch.ones(2, 15), torch.zeros(2, 5)], dim=1).bool(),
    ]
    
    total_loss2, loss_dict2 = loss_fn(
        pred_logits, pred_lines, gt_labels, gt_lines, gt_masks_partial
    )
    
    print(f"\nGround Truth:")
    print(f"  Sample 0: valid points = {gt_masks_partial[0].sum(dim=1).tolist()}")
    print(f"  Sample 1: valid points = {gt_masks_partial[1].sum(dim=1).tolist()}")
    
    print(f"\nLosses:")
    print(f"  Classification: {loss_dict2['loss_cls']:.4f}")
    print(f"  Points: {loss_dict2['loss_pts']:.4f}")
    print(f"  Direction: {loss_dict2['loss_dir']:.4f}")
    print(f"  Total: {loss_dict2['loss_total']:.4f}")
    
    assert not torch.isnan(total_loss2), "Loss is NaN!"
    
    print("\n✓ Test 2 passed!")
    
    # Test case 3: Equivalence class (polyline backward)
    print("\n" + "-"*80)
    print("Test 3: Equivalence class - polyline (backward matching)")
    print("-"*80)
    
    # Create a prediction that matches GT reversed
    gt_test = torch.linspace(-0.5, 0.5, P).unsqueeze(-1).repeat(1, 2)  # [P, 2]
    pred_test = torch.flip(gt_test, dims=[0])  # Reversed
    
    pred_logits_test = torch.zeros(1, 10, 3)
    pred_logits_test[0, 0, 0] = 10.0  # Strongly predict class 0
    
    pred_lines_test = torch.zeros(1, 10, P, 2)
    pred_lines_test[0, 0] = pred_test
    
    gt_labels_test = [torch.tensor([0])]  # Polyline (road_divider)
    gt_lines_test = [gt_test.unsqueeze(0)]
    gt_masks_test = [torch.ones(1, P, dtype=torch.bool)]
    
    total_loss3, loss_dict3 = loss_fn(
        pred_logits_test, pred_lines_test,
        gt_labels_test, gt_lines_test, gt_masks_test
    )
    
    print(f"\nSetup:")
    print(f"  GT: forward direction")
    print(f"  Pred: backward direction (reversed)")
    print(f"  Should match via equivalence class!")
    
    print(f"\nLosses:")
    print(f"  Points: {loss_dict3['loss_pts']:.6f} (should be ~0)")
    print(f"  Direction: {loss_dict3['loss_dir']:.6f} (should be ~0)")
    
    print("\n✓ Test 3 passed!")
    
    # Test case 4: Equivalence class (polygon rotation)
    print("\n" + "-"*80)
    print("Test 4: Equivalence class - polygon (rotation matching)")
    print("-"*80)
    
    # Create a prediction that matches GT rotated by 5
    shift = 5
    gt_poly = torch.randn(1, P, 2) * 0.5
    pred_poly = torch.roll(gt_poly, shifts=shift, dims=1)  # Rotated
    
    pred_logits_poly = torch.zeros(1, 10, 3)
    pred_logits_poly[0, 0, 1] = 10.0  # Strongly predict class 1 (ped_crossing = polygon)
    
    pred_lines_poly = torch.zeros(1, 10, P, 2)
    pred_lines_poly[0, 0] = pred_poly[0]
    
    gt_labels_poly = [torch.tensor([1])]  # Polygon: ped_crossing (class 1)
    gt_lines_poly = [gt_poly]
    gt_masks_poly = [torch.ones(1, P, dtype=torch.bool)]
    
    total_loss4, loss_dict4 = loss_fn(
        pred_logits_poly, pred_lines_poly,
        gt_labels_poly, gt_lines_poly, gt_masks_poly
    )
    
    print(f"\nSetup:")
    print(f"  GT: original orientation")
    print(f"  Pred: rotated by {shift} positions")
    print(f"  Should match via equivalence class!")
    
    print(f"\nLosses:")
    print(f"  Points: {loss_dict4['loss_pts']:.6f} (should be ~0)")
    print(f"  Direction: {loss_dict4['loss_dir']:.6f} (should be ~0)")
    
    print("\n✓ Test 4 passed!")
    
    # Test backward pass
    print("\n" + "-"*80)
    print("Test 5: Backward pass")
    print("-"*80)
    
    pred_logits.requires_grad = True
    pred_lines.requires_grad = True
    
    total_loss_grad, _ = loss_fn(pred_logits, pred_lines, gt_labels, gt_lines, gt_masks)
    total_loss_grad.backward()
    
    print(f"\nGradients:")
    print(f"  pred_logits.grad: {pred_logits.grad.shape}, mean={pred_logits.grad.mean():.6f}")
    print(f"  pred_lines.grad: {pred_lines.grad.shape}, mean={pred_lines.grad.mean():.6f}")
    
    assert pred_logits.grad is not None
    assert pred_lines.grad is not None
    assert not torch.isnan(pred_logits.grad).any()
    assert not torch.isnan(pred_lines.grad).any()
    
    print("\n✓ Test 5 passed!")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

