"""
Map Detection Head for Map Element Detection

Simple and effective design following MapTR philosophy:
- Classification head: predict element class
- Regression head: predict point coordinates
- No extra fusion needed (Transformer attention already did the job)

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MapDetectionHead(nn.Module):
    """
    Detection head for map element detection.
    
    Components:
    - Classification head: instance features → class logits
    - Regression head: point features → point coordinates
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_classes: int = 3,
        intermediate_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ):
        """
        Args:
            hidden_size: Input feature dimension (default: 4096 for Vicuna-7B)
            num_classes: Number of map element classes (default: 3)
            intermediate_dim: First hidden layer dimension (default: 1024)
            bottleneck_dim: Second hidden layer dimension (default: 256)
            dropout: Dropout rate (default: 0.1)
        """
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Classification head
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, num_classes)
        )
        
        # Regression head
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 2),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        instance_features: torch.Tensor,
        point_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            instance_features: [B, num_instances, hidden_size]
            point_features: [B, num_instances, num_points, hidden_size]
        
        Returns:
            pred_classes: [B, num_instances, num_classes]
            pred_points: [B, num_instances, num_points, 2]
        """
        B, N, H = instance_features.shape
        _, _, P, _ = point_features.shape
        
        # Classification: [B, N, H] → [B, N, num_classes]
        pred_classes = self.cls_head(instance_features)
        
        # Regression: [B, N, P, H] → [B, N, P, 2]
        # Reshape for batch processing
        point_features_flat = point_features.reshape(B * N * P, H)
        pred_points_flat = self.reg_head(point_features_flat)  # [B*N*P, 2]
        pred_points = pred_points_flat.reshape(B, N, P, 2)
        
        return pred_classes, pred_points
    
    def get_num_params(self):
        """Get number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class ClassificationHead(nn.Module):
    """
    Standalone classification head.
    
    For use cases where only classification is needed.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        num_classes: int = 3,
        intermediate_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, hidden_size]
        Returns:
            logits: [B, N, num_classes]
        """
        return self.head(features)


class RegressionHead(nn.Module):
    """
    Standalone regression head.
    
    For use cases where only coordinate regression is needed.
    """
    
    def __init__(
        self,
        hidden_size: int = 4096,
        intermediate_dim: int = 1024,
        bottleneck_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.head = nn.Sequential(
            nn.Linear(hidden_size, intermediate_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck_dim, 2),
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N, P, hidden_size]
        Returns:
            coords: [B, N, P, 2]
        """
        B, N, P, H = features.shape
        
        # Reshape for batch processing
        features_flat = features.reshape(B * N * P, H)
        coords_flat = self.head(features_flat)  # [B*N*P, 2]
        coords = coords_flat.reshape(B, N, P, 2)
        
        return coords


def build_detection_head(config: dict = None) -> MapDetectionHead:
    """
    Build detection head from config.
    
    Args:
        config: dict with keys:
            - hidden_size: int (default: 4096)
            - num_classes: int (default: 3)
            - intermediate_dim: int (default: 1024)
            - bottleneck_dim: int (default: 256)
            - dropout: float (default: 0.1)
    
    Returns:
        MapDetectionHead instance
    """
    if config is None:
        config = {}
    
    return MapDetectionHead(
        hidden_size=config.get('hidden_size', 4096),
        num_classes=config.get('num_classes', 3),
        intermediate_dim=config.get('intermediate_dim', 1024),
        bottleneck_dim=config.get('bottleneck_dim', 256),
        dropout=config.get('dropout', 0.1),
    )


# Test code
if __name__ == "__main__":
    print("="*80)
    print("Testing MapDetectionHead")
    print("="*80)
    
    # Create detection head
    head = MapDetectionHead(
        hidden_size=4096,
        num_classes=3,
        intermediate_dim=1024,
        bottleneck_dim=256,
        dropout=0.1
    )
    
    print(f"\nModel created:")
    print(f"  Hidden size: 4096")
    print(f"  Num classes: 3")
    print(f"  Intermediate dim: 1024")
    print(f"  Bottleneck dim: 256")
    print(f"  Total parameters: {head.get_num_params():,}")
    
    # Test forward pass
    batch_size = 2
    num_instances = 50
    num_points = 20
    hidden_size = 4096
    
    print(f"\n" + "-"*80)
    print("Test 1: Forward pass")
    print("-"*80)
    
    instance_features = torch.randn(batch_size, num_instances, hidden_size)
    point_features = torch.randn(batch_size, num_instances, num_points, hidden_size)
    
    print(f"Input:")
    print(f"  instance_features: {instance_features.shape}")
    print(f"  point_features: {point_features.shape}")
    
    pred_classes, pred_points = head(instance_features, point_features)
    
    print(f"\nOutput:")
    print(f"  pred_classes: {pred_classes.shape}")
    print(f"  pred_points: {pred_points.shape}")
    
    # Check output shapes
    assert pred_classes.shape == (batch_size, num_instances, 3), "Classification output shape mismatch!"
    assert pred_points.shape == (batch_size, num_instances, num_points, 2), "Regression output shape mismatch!"
    
    # Check value ranges
    assert pred_points.min() >= -1.0 and pred_points.max() <= 1.0, "Regression output not in [-1, 1]!"
    
    print("\n✓ Forward pass test passed!")
    
    # Test standalone heads
    print(f"\n" + "-"*80)
    print("Test 2: Standalone heads")
    print("-"*80)
    
    cls_head = ClassificationHead(hidden_size=4096, num_classes=3)
    reg_head = RegressionHead(hidden_size=4096)
    
    pred_cls = cls_head(instance_features)
    pred_reg = reg_head(point_features)
    
    print(f"Classification head output: {pred_cls.shape}")
    print(f"Regression head output: {pred_reg.shape}")
    
    assert pred_cls.shape == (batch_size, num_instances, 3)
    assert pred_reg.shape == (batch_size, num_instances, num_points, 2)
    
    print("\n✓ Standalone heads test passed!")
    
    # Test build function
    print(f"\n" + "-"*80)
    print("Test 3: Build function")
    print("-"*80)
    
    config = {
        'hidden_size': 4096,
        'num_classes': 3,
        'intermediate_dim': 1024,
        'bottleneck_dim': 256,
        'dropout': 0.1
    }
    
    head = build_detection_head(config)
    pred_classes, pred_points = head(instance_features, point_features)
    
    print(f"Built head from config")
    print(f"  Parameters: {head.get_num_params():,}")
    print(f"  Output shapes: {pred_classes.shape}, {pred_points.shape}")
    
    print("\n✓ Build function test passed!")
    
    # Parameter breakdown
    print(f"\n" + "-"*80)
    print("Parameter Breakdown")
    print("-"*80)
    
    cls_params = sum(p.numel() for p in head.cls_head.parameters())
    reg_params = sum(p.numel() for p in head.reg_head.parameters())
    
    print(f"\nClassification head: {cls_params:,} parameters")
    print(f"  Linear(4096 → 1024): {4096*1024 + 1024:,}")
    print(f"  Linear(1024 → 256):  {1024*256 + 256:,}")
    print(f"  Linear(256 → 3):     {256*3 + 3:,}")
    
    print(f"\nRegression head: {reg_params:,} parameters")
    print(f"  Linear(4096 → 1024): {4096*1024 + 1024:,}")
    print(f"  Linear(1024 → 256):  {1024*256 + 256:,}")
    print(f"  Linear(256 → 2):     {256*2 + 2:,}")
    
    print(f"\nTotal: {head.get_num_params():,} parameters")
    
    # Performance test
    print(f"\n" + "-"*80)
    print("Performance Test")
    print("-"*80)
    
    import time
    
    # Warm up
    for _ in range(10):
        _ = head(instance_features, point_features)
    
    # Time
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()
    
    num_iters = 100
    for _ in range(num_iters):
        _ = head(instance_features, point_features)
    
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = time.time() - start
    
    print(f"Average forward time: {elapsed/num_iters*1000:.2f} ms")
    print(f"Throughput: {num_iters/elapsed:.2f} samples/sec")
    
    print("\n" + "="*80)
    print("✅ All tests passed!")
    print("="*80)

