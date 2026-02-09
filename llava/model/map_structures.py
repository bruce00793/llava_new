"""
Data structures for map detection output
Defines prediction format and post-processing utilities
"""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import numpy as np

from .map_config import MapDetectionConfig, DEFAULT_MAP_CONFIG


@dataclass
class MapPrediction:
    """
    Output structure for map detection
    
    Attributes:
        class_logits: [N, C+1] - Classification logits (C classes + 1 no-object)
        points: [N, 20, 2] - BEV coordinates for 20 points per instance
        bbox: [N, 4] - Optional bounding boxes (x, y, w, h)
        scores: [N] - Confidence scores (after softmax)
        labels: [N] - Predicted class labels
    """
    class_logits: torch.Tensor  # [N, 4]
    points: torch.Tensor  # [N, 20, 2]
    bbox: Optional[torch.Tensor] = None  # [N, 4]
    scores: Optional[torch.Tensor] = None  # [N]
    labels: Optional[torch.Tensor] = None  # [N]
    
    def __post_init__(self):
        """Compute scores and labels from logits"""
        if self.scores is None:
            probs = torch.softmax(self.class_logits, dim=-1)
            self.scores, self.labels = probs.max(dim=-1)
    
    @property
    def num_instances(self) -> int:
        """Number of predicted instances"""
        return self.class_logits.shape[0]
    
    def filter_no_object(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        """
        Filter out predictions classified as 'no-object'
        
        Returns:
            MapPrediction with only valid instances
        """
        valid_mask = self.labels < config.NUM_CLASSES  # Exclude no-object class
        
        return MapPrediction(
            class_logits=self.class_logits[valid_mask],
            points=self.points[valid_mask],
            bbox=self.bbox[valid_mask] if self.bbox is not None else None,
            scores=self.scores[valid_mask] if self.scores is not None else None,
            labels=self.labels[valid_mask] if self.labels is not None else None,
        )
    
    def filter_by_score(self, threshold: float = 0.3):
        """
        Filter predictions by confidence score
        
        Args:
            threshold: Minimum confidence score
            
        Returns:
            Filtered MapPrediction
        """
        if self.scores is None:
            return self
        
        valid_mask = self.scores >= threshold
        
        return MapPrediction(
            class_logits=self.class_logits[valid_mask],
            points=self.points[valid_mask],
            bbox=self.bbox[valid_mask] if self.bbox is not None else None,
            scores=self.scores[valid_mask],
            labels=self.labels[valid_mask],
        )
    
    def denormalize_coords(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        """
        Convert normalized coordinates [-1, 1] back to BEV coordinates [meters]
        
        Returns:
            MapPrediction with denormalized coordinates
        """
        if not config.NORMALIZE_COORDS:
            return self
        
        # Denormalize points
        x_range = config.BEV_X_MAX - config.BEV_X_MIN
        y_range = config.BEV_Y_MAX - config.BEV_Y_MIN
        
        denorm_points = self.points.clone()
        denorm_points[..., 0] = (denorm_points[..., 0] + 1) / 2 * x_range + config.BEV_X_MIN
        denorm_points[..., 1] = (denorm_points[..., 1] + 1) / 2 * y_range + config.BEV_Y_MIN
        
        # Denormalize bbox if present
        # BBox format: [cx_norm, cy_norm, w_norm, h_norm]
        # - cx_norm, cy_norm: center in [-1, 1]
        # - w_norm, h_norm: relative size (0 to 2)
        denorm_bbox = None
        if self.bbox is not None:
            denorm_bbox = self.bbox.clone()
            # Denormalize center: [-1, 1] -> [min, max]
            denorm_bbox[:, 0] = (denorm_bbox[:, 0] + 1) / 2 * x_range + config.BEV_X_MIN  # cx
            denorm_bbox[:, 1] = (denorm_bbox[:, 1] + 1) / 2 * y_range + config.BEV_Y_MIN  # cy
            # Denormalize size: [0, 2] -> [0, range]
            denorm_bbox[:, 2] = denorm_bbox[:, 2] / 2 * x_range  # w
            denorm_bbox[:, 3] = denorm_bbox[:, 3] / 2 * y_range  # h
        
        return MapPrediction(
            class_logits=self.class_logits,
            points=denorm_points,
            bbox=denorm_bbox,
            scores=self.scores,
            labels=self.labels,
        )
    
    def to_dict(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG) -> Dict:
        """
        Convert to dictionary format for visualization/evaluation
        
        Returns:
            Dict with keys: 'class_names', 'points', 'bboxes', 'scores'
        """
        result = {
            'class_names': [config.CLASS_NAMES[label.item()] for label in self.labels],
            'labels': self.labels.cpu().numpy(),
            'points': self.points.cpu().numpy(),  # [N, 20, 2]
            'scores': self.scores.cpu().numpy() if self.scores is not None else None,
        }
        
        if self.bbox is not None:
            result['bboxes'] = self.bbox.cpu().numpy()
        
        return result
    
    def to(self, device):
        """Move all tensors to device"""
        return MapPrediction(
            class_logits=self.class_logits.to(device),
            points=self.points.to(device),
            bbox=self.bbox.to(device) if self.bbox is not None else None,
            scores=self.scores.to(device) if self.scores is not None else None,
            labels=self.labels.to(device) if self.labels is not None else None,
        )


class MapGroundTruth:
    """
    Ground truth structure for map detection
    Used for training and evaluation
    """
    def __init__(
        self,
        class_labels: torch.Tensor,  # [M] - GT class indices
        points: torch.Tensor,  # [M, 20, 2] - GT points
        bbox: Optional[torch.Tensor] = None,  # [M, 4] - GT bboxes
    ):
        self.class_labels = class_labels
        self.points = points
        self.bbox = bbox
    
    @property
    def num_instances(self) -> int:
        """Number of ground truth instances"""
        return self.class_labels.shape[0]
    
    def to(self, device):
        """Move all tensors to device"""
        return MapGroundTruth(
            class_labels=self.class_labels.to(device),
            points=self.points.to(device),
            bbox=self.bbox.to(device) if self.bbox is not None else None,
        )
    
    @staticmethod
    def normalize_coords(
        points: torch.Tensor,
        bbox: Optional[torch.Tensor] = None,
        config: MapDetectionConfig = DEFAULT_MAP_CONFIG
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Normalize BEV coordinates to [-1, 1]
        
        Args:
            points: [M, 20, 2] - Points in BEV coordinates
            bbox: [M, 4] - Bounding boxes (x, y, w, h)
            config: Configuration
            
        Returns:
            Normalized points and bboxes
        """
        if not config.NORMALIZE_COORDS:
            return points, bbox
        
        # Normalize points
        x_range = config.BEV_X_MAX - config.BEV_X_MIN
        y_range = config.BEV_Y_MAX - config.BEV_Y_MIN
        
        norm_points = points.clone()
        norm_points[..., 0] = (norm_points[..., 0] - config.BEV_X_MIN) / x_range * 2 - 1
        norm_points[..., 1] = (norm_points[..., 1] - config.BEV_Y_MIN) / y_range * 2 - 1
        
        # 【重要】与 MapTR 一致：clip 到 [-1, 1] 范围
        # 防止超出 BEV 范围的点导致 Loss 计算异常
        norm_points = norm_points.clamp(-1.0, 1.0)
        
        # Normalize bbox if present
        # BBox format: [x_center, y_center, width, height]
        norm_bbox = None
        if bbox is not None:
            norm_bbox = bbox.clone()
            # Normalize center: [min, max] -> [-1, 1]
            norm_bbox[:, 0] = (norm_bbox[:, 0] - config.BEV_X_MIN) / x_range * 2 - 1  # cx
            norm_bbox[:, 1] = (norm_bbox[:, 1] - config.BEV_Y_MIN) / y_range * 2 - 1  # cy
            # Normalize size: [0, range] -> [0, 2]
            norm_bbox[:, 2] = norm_bbox[:, 2] / x_range * 2  # w
            norm_bbox[:, 3] = norm_bbox[:, 3] / y_range * 2  # h
            
            # Clip center to [-1, 1]
            norm_bbox[:, 0:2] = norm_bbox[:, 0:2].clamp(-1.0, 1.0)
            # Clip size to [0, 2]
            norm_bbox[:, 2:4] = norm_bbox[:, 2:4].clamp(0.0, 2.0)
        
        return norm_points, norm_bbox


def parse_model_output(
    output_tensor: torch.Tensor,
    config: MapDetectionConfig = DEFAULT_MAP_CONFIG
) -> MapPrediction:
    """
    Parse raw model output tensor into MapPrediction structure
    
    Args:
        output_tensor: [N, OUTPUT_DIM] - Raw output from MLP head
        config: Configuration
        
    Returns:
        MapPrediction object
    """
    # Split output tensor
    class_logits = output_tensor[:, :config.NUM_CLASSES_WITH_BG]  # [N, 4]
    points = output_tensor[:, config.NUM_CLASSES_WITH_BG:config.NUM_CLASSES_WITH_BG + config.NUM_POINTS_PER_INSTANCE * 2]
    points = points.reshape(-1, config.NUM_POINTS_PER_INSTANCE, 2)  # [N, 20, 2]
    
    bbox = None
    if config.USE_BBOX:
        bbox = output_tensor[:, -config.BBOX_DIM:]  # [N, 4]
    
    return MapPrediction(
        class_logits=class_logits,
        points=points,
        bbox=bbox,
    )


if __name__ == "__main__":
    # Test data structures
    config = MapDetectionConfig()
    
    # Create dummy prediction
    N = 50
    dummy_output = torch.randn(N, config.OUTPUT_DIM)
    pred = parse_model_output(dummy_output, config)
    
    print(f"Parsed prediction:")
    print(f"  Class logits shape: {pred.class_logits.shape}")
    print(f"  Points shape: {pred.points.shape}")
    print(f"  Bbox shape: {pred.bbox.shape if pred.bbox is not None else None}")
    print(f"  Num instances: {pred.num_instances}")
    
    # Filter predictions
    filtered = pred.filter_no_object(config)
    print(f"\nAfter filtering no-object: {filtered.num_instances} instances")

