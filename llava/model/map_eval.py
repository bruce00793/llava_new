"""
Map Detection Evaluation Module

Implements mAP (Mean Average Precision) evaluation following MapTR protocol:
- Uses Chamfer Distance (CD) as the matching criterion
- CD thresholds: 0.5m, 1.0m, 1.5m
- Computes AP for each class and then averages

Author: Auto-generated for Map Detection
Date: 2026-01
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment


class MapEvaluator:
    """
    Map Detection Evaluator following MapTR protocol.
    
    Computes mAP using Chamfer Distance as the matching criterion.
    """
    
    def __init__(
        self,
        num_classes: int = 3,
        cd_thresholds: List[float] = [0.5, 1.0, 1.5],
        score_threshold: float = 0.3,  # sigmoid(0)=0.5>0.1 导致全部通过，提高到 0.3 过滤低质预测
        pc_range: List[float] = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],
    ):
        """
        Args:
            num_classes: Number of map element classes (default: 3)
            cd_thresholds: Chamfer Distance thresholds in meters
            score_threshold: Minimum score threshold for predictions
            pc_range: Point cloud range [x_min, y_min, z_min, x_max, y_max, z_max]
        """
        self.num_classes = num_classes
        self.cd_thresholds = cd_thresholds
        self.score_threshold = score_threshold
        self.pc_range = pc_range
        
        # Calculate scale factors for coordinate conversion
        # Normalized coords [-1, 1] -> Real world coords
        self.x_scale = (pc_range[3] - pc_range[0]) / 2.0  # 15.0
        self.y_scale = (pc_range[4] - pc_range[1]) / 2.0  # 30.0
        
        self.class_names = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}
        
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions and ground truths."""
        # Predictions: {class_id: [{'score': float, 'points': np.array}, ...]}
        self.pred_instances = defaultdict(list)
        # Ground truths: {class_id: [{'points': np.array, 'sample_id': int}, ...]}
        self.gt_instances = defaultdict(list)
        self.sample_count = 0
    
    def _normalize_to_world(self, points: np.ndarray) -> np.ndarray:
        """
        Convert normalized coordinates [-1, 1] to world coordinates.
        
        Args:
            points: [N, 2] or [P, 2] array of normalized coordinates
        
        Returns:
            [N, 2] or [P, 2] array of world coordinates in meters
        """
        pts = points.copy()
        pts[..., 0] = pts[..., 0] * self.x_scale  # x: [-1,1] -> [-15, 15]
        pts[..., 1] = pts[..., 1] * self.y_scale  # y: [-1,1] -> [-30, 30]
        return pts
    
    def _compute_chamfer_distance(
        self,
        pred_pts: np.ndarray,
        gt_pts: np.ndarray,
    ) -> float:
        """
        Compute bidirectional Chamfer Distance between two point sets.
        
        Args:
            pred_pts: [P, 2] predicted points
            gt_pts: [P, 2] ground truth points
        
        Returns:
            Chamfer Distance (average of both directions)
        """
        if len(pred_pts) == 0 or len(gt_pts) == 0:
            return float('inf')
        
        # Pred -> GT: for each pred point, find nearest GT point
        diff_pred_to_gt = pred_pts[:, None, :] - gt_pts[None, :, :]  # [P, Q, 2]
        dist_pred_to_gt = np.linalg.norm(diff_pred_to_gt, axis=-1)  # [P, Q]
        min_dist_pred_to_gt = np.min(dist_pred_to_gt, axis=1)  # [P]
        
        # GT -> Pred: for each GT point, find nearest pred point
        min_dist_gt_to_pred = np.min(dist_pred_to_gt, axis=0)  # [Q]
        
        # Bidirectional average
        cd = (np.mean(min_dist_pred_to_gt) + np.mean(min_dist_gt_to_pred)) / 2.0
        return cd
    
    def add_batch(
        self,
        pred_logits: torch.Tensor,
        pred_points: torch.Tensor,
        gt_labels: torch.Tensor,
        gt_points: torch.Tensor,
        gt_masks: torch.Tensor,
    ):
        """
        Add a batch of predictions and ground truths for evaluation.
        
        Args:
            pred_logits: [B, N, num_classes] prediction logits
            pred_points: [B, N, P, 2] predicted points (normalized)
            gt_labels: [B, M] ground truth labels
            gt_points: [B, M, P, 2] ground truth points (normalized)
            gt_masks: [B, M] ground truth valid masks
        """
        B = pred_logits.shape[0]
        
        # Convert to numpy
        pred_logits = pred_logits.detach().cpu()
        pred_points = pred_points.detach().cpu().numpy()
        gt_labels = gt_labels.detach().cpu().numpy()
        gt_points = gt_points.detach().cpu().numpy()
        gt_masks = gt_masks.detach().cpu().numpy()
        
        # Get predictions
        # IMPORTANT: training uses sigmoid focal (multi-label style, no explicit background class).
        # Using softmax here forces every query to pick a class and inflates false positives.
        # For evaluation we use sigmoid scores and take the best class per query.
        pred_probs = pred_logits.float().sigmoid()  # [B, N, C]
        pred_scores, pred_labels = pred_probs.max(dim=-1)  # [B, N]
        pred_scores = pred_scores.numpy()
        pred_labels = pred_labels.numpy()
        
        for b in range(B):
            sample_id = self.sample_count + b
            
            # Filter predictions by score and class
            valid_mask = (pred_scores[b] > self.score_threshold) & (pred_labels[b] < self.num_classes)
            valid_indices = np.where(valid_mask)[0]
            
            for idx in valid_indices:
                cls_id = pred_labels[b, idx]
                score = pred_scores[b, idx]
                pts = self._normalize_to_world(pred_points[b, idx])  # [P, 2]
                
                self.pred_instances[cls_id].append({
                    'score': float(score),
                    'points': pts,
                    'sample_id': sample_id,
                })
            
            # Add ground truths
            for m in range(gt_labels.shape[1]):
                if gt_masks[b, m] > 0.5:  # Valid GT
                    cls_id = int(gt_labels[b, m])
                    if cls_id < self.num_classes:
                        pts = self._normalize_to_world(gt_points[b, m])  # [P, 2]
                        self.gt_instances[cls_id].append({
                            'points': pts,
                            'sample_id': sample_id,
                        })
        
        self.sample_count += B
    
    def _compute_chamfer_distance_batch(
        self,
        pred_pts: np.ndarray,
        gt_pts_batch: np.ndarray,
    ) -> np.ndarray:
        """
        批量计算一个 pred 与多个 GT 的 Chamfer Distance（向量化加速）
        
        Args:
            pred_pts: [P, 2] 单个预测的点
            gt_pts_batch: [M, P, 2] M 个 GT 的点
        
        Returns:
            [M] 每个 GT 对应的 CD
        """
        # pred_pts: [P, 2] → [1, P, 1, 2]
        # gt_pts_batch: [M, Q, 2] → [M, 1, Q, 2]
        pred_expanded = pred_pts[None, :, None, :]    # [1, P, 1, 2]
        gt_expanded = gt_pts_batch[:, None, :, :]      # [M, 1, Q, 2]
        
        # 计算距离矩阵: [M, P, Q]
        diff = pred_expanded - gt_expanded              # [M, P, Q, 2]
        dist_matrix = np.linalg.norm(diff, axis=-1)     # [M, P, Q]
        
        # Pred → GT 最小距离: [M, P]
        min_pred_to_gt = np.min(dist_matrix, axis=2)
        # GT → Pred 最小距离: [M, Q]
        min_gt_to_pred = np.min(dist_matrix, axis=1)
        
        # 双向平均 CD: [M]
        cd = (np.mean(min_pred_to_gt, axis=1) + np.mean(min_gt_to_pred, axis=1)) / 2.0
        return cd
    
    def _compute_ap_for_class(
        self,
        pred_list: List[Dict],
        gt_list: List[Dict],
        cd_threshold: float,
    ) -> Tuple[float, int, int]:
        """
        Compute AP for a single class at a given CD threshold.
        
        【优化】使用向量化批量 CD 计算，大幅加速 mAP 评估
        
        Args:
            pred_list: List of predictions with 'score' and 'points'
            gt_list: List of ground truths with 'points'
            cd_threshold: Chamfer Distance threshold in meters
        
        Returns:
            (AP, num_gt, num_pred)
        """
        num_gt = len(gt_list)
        num_pred = len(pred_list)
        
        if num_gt == 0:
            return 0.0, 0, num_pred
        if num_pred == 0:
            return 0.0, num_gt, 0
        
        # 【优化】限制最大预测数量，防止计算量爆炸
        MAX_PREDS = 5000
        
        # Sort predictions by score (descending)
        sorted_preds = sorted(pred_list, key=lambda x: x['score'], reverse=True)
        if len(sorted_preds) > MAX_PREDS:
            sorted_preds = sorted_preds[:MAX_PREDS]
            num_pred = MAX_PREDS
        
        # Build per-sample GT pools.
        # MapTR protocol matches predictions to GTs within the SAME sample only.
        # Matching across different samples will severely distort AP/mAP.
        gt_pts_by_sample = defaultdict(list)
        for gt in gt_list:
            gt_pts_by_sample[gt['sample_id']].append(gt['points'])
        gt_pts_by_sample = {k: np.stack(v, axis=0) for k, v in gt_pts_by_sample.items()}  # sid -> [Mi, P, 2]

        # Track matched GT per sample
        gt_matched_by_sample = {sid: np.zeros(pts.shape[0], dtype=bool) for sid, pts in gt_pts_by_sample.items()}
        
        tp = np.zeros(num_pred)
        fp = np.zeros(num_pred)
        
        for i, pred in enumerate(sorted_preds):
            pred_pts = pred['points']  # [P, 2]

            sid = pred['sample_id']
            if sid not in gt_pts_by_sample:
                fp[i] = 1
                continue

            gt_pts_all = gt_pts_by_sample[sid]  # [Mi, P, 2]
            gt_matched = gt_matched_by_sample[sid]  # [Mi]

            unmatched_indices = np.where(~gt_matched)[0]
            if len(unmatched_indices) == 0:
                fp[i] = 1
                continue

            unmatched_gt_pts = gt_pts_all[unmatched_indices]  # [M, P, 2]
            cds = self._compute_chamfer_distance_batch(pred_pts, unmatched_gt_pts)  # [M]

            best_local_idx = int(np.argmin(cds))
            min_cd = float(cds[best_local_idx])

            if min_cd < cd_threshold:
                best_gt_local = unmatched_indices[best_local_idx]
                gt_matched[best_gt_local] = True
                tp[i] = 1
            else:
                fp[i] = 1
        
        # Compute cumulative TP/FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Precision and Recall
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        recall = tp_cumsum / (num_gt + 1e-8)
        
        # Compute AP using all-point interpolation (area under curve)
        mrec = np.concatenate(([0.0], recall, [1.0]))
        mpre = np.concatenate(([0.0], precision, [0.0]))
        
        # Make precision monotonically decreasing
        for k in range(len(mpre) - 1, 0, -1):
            mpre[k - 1] = max(mpre[k - 1], mpre[k])
        
        # Compute area under PR curve
        i = np.where(mrec[1:] != mrec[:-1])[0]
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        
        return float(ap), num_gt, num_pred
    
    def compute_metrics(self) -> Dict[str, float]:
        """
        Compute all metrics: mAP at different thresholds and per-class AP.
        
        Returns:
            Dictionary with all metrics
        """
        import time
        
        results = {}
        
        # Statistics
        total_preds = 0
        total_gts = 0
        for cls_id, cls_name in self.class_names.items():
            n_pred = len(self.pred_instances[cls_id])
            n_gt = len(self.gt_instances[cls_id])
            results[f'num_pred_{cls_name}'] = n_pred
            results[f'num_gt_{cls_name}'] = n_gt
            total_preds += n_pred
            total_gts += n_gt
        
        print(f"  [mAP] Computing metrics: {total_preds} predictions, {total_gts} GTs, "
              f"{self.sample_count} samples", flush=True)
        
        # Compute AP at each threshold
        ap_per_threshold = {}
        
        for thresh in self.cd_thresholds:
            aps = []
            for cls_id in range(self.num_classes):
                cls_name = self.class_names.get(cls_id, f'class_{cls_id}')
                pred_list = self.pred_instances[cls_id]
                gt_list = self.gt_instances[cls_id]
                
                t0 = time.time()
                ap, num_gt, num_pred = self._compute_ap_for_class(
                    pred_list, gt_list, thresh
                )
                elapsed = time.time() - t0
                print(f"  [mAP] AP_{cls_name}@{thresh}m: {ap*100:.2f}% "
                      f"({num_pred} preds vs {num_gt} GTs, took {elapsed:.1f}s)", flush=True)
                
                aps.append(ap)
                results[f'AP_{cls_name}@{thresh}m'] = ap
            
            # mAP at this threshold
            mAP_at_thresh = np.mean(aps) if aps else 0.0
            results[f'mAP@{thresh}m'] = mAP_at_thresh
            ap_per_threshold[thresh] = mAP_at_thresh
        
        # Overall mAP (average across all thresholds)
        overall_mAP = np.mean(list(ap_per_threshold.values())) if ap_per_threshold else 0.0
        results['mAP'] = overall_mAP
        
        print(f"  [mAP] Done! Overall mAP: {overall_mAP*100:.2f}%", flush=True)
        
        return results
    
    def print_results(self, results: Optional[Dict[str, float]] = None):
        """
        Print evaluation results in a formatted table.
        
        Args:
            results: Metrics dictionary (if None, will compute)
        """
        if results is None:
            results = self.compute_metrics()
        
        print("\n" + "=" * 70)
        print("Map Detection Evaluation Results (MapTR Protocol)")
        print("=" * 70)
        
        # Statistics
        print("\n📊 Instance Statistics:")
        print("-" * 50)
        print(f"{'Class':<15} {'Predictions':>12} {'Ground Truth':>12}")
        print("-" * 50)
        for cls_id, cls_name in self.class_names.items():
            num_pred = results.get(f'num_pred_{cls_name}', 0)
            num_gt = results.get(f'num_gt_{cls_name}', 0)
            print(f"{cls_name:<15} {num_pred:>12} {num_gt:>12}")
        print("-" * 50)
        
        # Per-class AP at each threshold
        print("\n📈 Average Precision (AP) per Class:")
        print("-" * 70)
        header = f"{'Class':<15}"
        for thresh in self.cd_thresholds:
            header += f" {'AP@'+str(thresh)+'m':>12}"
        print(header)
        print("-" * 70)
        
        for cls_id, cls_name in self.class_names.items():
            row = f"{cls_name:<15}"
            for thresh in self.cd_thresholds:
                ap = results.get(f'AP_{cls_name}@{thresh}m', 0.0)
                row += f" {ap*100:>11.2f}%"
            print(row)
        print("-" * 70)
        
        # mAP at each threshold
        print("\n🎯 Mean Average Precision (mAP):")
        print("-" * 50)
        for thresh in self.cd_thresholds:
            mAP = results.get(f'mAP@{thresh}m', 0.0)
            print(f"  mAP@{thresh}m: {mAP*100:.2f}%")
        print("-" * 50)
        print(f"  ⭐ Overall mAP: {results.get('mAP', 0.0)*100:.2f}%")
        print("=" * 70 + "\n")
        
        return results


def evaluate_model(
    model,
    dataloader,
    device: torch.device,
    evaluator: Optional[MapEvaluator] = None,
    max_samples: Optional[int] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Evaluate a model on a dataset.
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader for the evaluation dataset
        device: Device to run evaluation on
        evaluator: MapEvaluator instance (creates new one if None)
        max_samples: Maximum number of samples to evaluate (None = all)
        verbose: Whether to print progress
    
    Returns:
        Dictionary with all metrics
    """
    # Use BF16 for evaluation (consistent with training)
    
    if evaluator is None:
        evaluator = MapEvaluator()
    else:
        evaluator.reset()
    
    model.eval()
    
    num_batches = len(dataloader)
    if max_samples is not None:
        num_batches = min(num_batches, max_samples // dataloader.batch_size + 1)
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if max_samples is not None and evaluator.sample_count >= max_samples:
                break
            
            # Move to device
            images = batch['images'].to(device)
            text_ids = batch['text_ids'].to(device)
            gt_labels = batch['gt_labels'].to(device)
            gt_points = batch['gt_points'].to(device)
            gt_masks = batch['gt_masks'].to(device)
            
            # Optional camera parameters
            cam_intrinsics = batch.get('cam_intrinsics')
            cam_extrinsics = batch.get('cam_extrinsics')
            if cam_intrinsics is not None:
                cam_intrinsics = cam_intrinsics.to(device)
            if cam_extrinsics is not None:
                cam_extrinsics = cam_extrinsics.to(device)
            
            # Forward pass
            with torch.amp.autocast('cuda', enabled=True, dtype=torch.bfloat16):
                outputs = model(
                    images=images,
                    text_ids=text_ids,
                    cam_intrinsics=cam_intrinsics,
                    cam_extrinsics=cam_extrinsics,
                )
            
            # Add to evaluator
            evaluator.add_batch(
                pred_logits=outputs['pred_logits'],
                pred_points=outputs['pred_points'],
                gt_labels=gt_labels,
                gt_points=gt_points,
                gt_masks=gt_masks,
            )
            
            if verbose and (batch_idx + 1) % 50 == 0:
                print(f"  Evaluated {evaluator.sample_count} samples...")
    
    # Compute and return metrics
    results = evaluator.compute_metrics()
    
    if verbose:
        evaluator.print_results(results)
    
    return results


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    print("Testing MapEvaluator...")
    
    # Create evaluator
    evaluator = MapEvaluator()
    
    # Create dummy predictions and ground truths
    B, N, P = 2, 50, 20
    num_classes = 3
    
    # Dummy predictions
    pred_logits = torch.randn(B, N, num_classes)
    pred_points = torch.rand(B, N, P, 2) * 2 - 1  # Normalized [-1, 1]
    
    # Dummy ground truths
    gt_labels = torch.randint(0, num_classes, (B, 10))
    gt_points = torch.rand(B, 10, P, 2) * 2 - 1
    gt_masks = torch.ones(B, 10)
    
    # Add batch
    evaluator.add_batch(pred_logits, pred_points, gt_labels, gt_points, gt_masks)
    
    # Compute metrics
    results = evaluator.compute_metrics()
    evaluator.print_results(results)
    
    print("✅ MapEvaluator test passed!")
