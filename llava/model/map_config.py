"""
Configuration file for Map Detection Task
Defines all hyperparameters and tensor dimensions
"""

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class MapDetectionConfig:
    """Configuration for map detection model"""
    
    # ============ Camera & Image ============
    NUM_CAMERAS: int = 6  # 6 surrounding cameras
    
    # ============ Query Settings ============
    NUM_QUERIES: int = 50  # Number of learnable instance queries (can adjust to 100 if needed)
    NUM_SCENE_TOKENS: int = 768  # Scene tokens from Q-Former
    
    # ============ Class Definition ============
    # 与 MapTR 一致的类别定义:
    # 0: divider (road_divider + lane_divider 合并)
    # 1: ped_crossing (人行横道)
    # 2: boundary (道路边界，从 road_segment 和 lane 提取)
    NUM_CLASSES: int = 3  # divider, ped_crossing, boundary
    NUM_CLASSES_WITH_BG: int = 4  # +1 for no-object class
    
    CLASS_NAMES: List[str] = None
    
    # ============ Point Settings ============
    NUM_POINTS_PER_INSTANCE: int = 20  # Fixed 20 points per instance
    POINT_DIM: int = 2  # (x, y) in BEV coordinates
    
    # ============ BBox Settings ============
    USE_BBOX: bool = True  # Whether to predict bounding box
    BBOX_DIM: int = 4  # (x, y, w, h)
    
    # ============ MLP Head Architecture ============
    # Separate heads for different tasks (more stable training)
    USE_SEPARATE_HEADS: bool = True  # Use separate heads for cls/points/bbox
    
    # Shared feature dimension after dimension reduction
    SHARED_FEATURE_DIM: int = 256  # Reduced from LLM_HIDDEN_SIZE to this
    
    # MLP reduction path: 4096 → 2048 → 1024 → 512 → 256
    MLP_REDUCTION_DIMS: List[int] = None
    
    def __post_init__(self):
        if self.CLASS_NAMES is None:
            # 与 MapTR 一致的类别定义
            self.CLASS_NAMES = [
                'divider',           # 分割线 (road_divider + lane_divider 合并)
                'ped_crossing',      # 人行横道
                'boundary',          # 道路边界 (road_segment + lane 轮廓)
                'no-object'          # 背景类
            ]
        if self.MLP_REDUCTION_DIMS is None:
            self.MLP_REDUCTION_DIMS = [2048, 1024, 512, 256]
        if self.PC_RANGE is None:
            self.PC_RANGE = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    
    # Individual head output dimensions
    @property
    def CLS_OUTPUT_DIM(self) -> int:
        """Classification output dimension"""
        # Using sigmoid focal loss: each class is independent binary classification
        # No need for explicit background class
        return self.NUM_CLASSES  # 3 (not 4, following MapTR)
    
    @property
    def POINTS_OUTPUT_DIM(self) -> int:
        """Points regression output dimension"""
        return self.NUM_POINTS_PER_INSTANCE * self.POINT_DIM  # 40
    
    @property
    def BBOX_OUTPUT_DIM(self) -> int:
        """Bbox regression output dimension"""
        return self.BBOX_DIM if self.USE_BBOX else 0  # 4 or 0
    
    @property
    def OUTPUT_DIM(self) -> int:
        """Total output dimension per query (for legacy support)"""
        dim = self.CLS_OUTPUT_DIM + self.POINTS_OUTPUT_DIM + self.BBOX_OUTPUT_DIM
        return dim  # 4 + 40 + 4 = 48
    
    # ============ LLM Settings ============
    LLM_HIDDEN_SIZE: int = 4096  # LLaVA hidden dimension (depends on model)
    QUERY_EXTRACT_LAYER: int = 32  # Extract query features from layer 32
    
    # ============ BEV Coordinate Range (align with MapTR) ============
    # IMPORTANT: Following MapTR's coordinate system
    # - x-axis: left(-) to right(+)
    # - y-axis: back(-) to front(+)
    # - origin: ego/LiDAR center
    
    # pc_range = [x_min, y_min, z_min, x_max, y_max, z_max]
    # MapTR uses: [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
    PC_RANGE: List[float] = None
    
    @property
    def BEV_X_MIN(self) -> float:
        return self.PC_RANGE[0]
    
    @property
    def BEV_X_MAX(self) -> float:
        return self.PC_RANGE[3]
    
    @property
    def BEV_Y_MIN(self) -> float:
        return self.PC_RANGE[1]
    
    @property
    def BEV_Y_MAX(self) -> float:
        return self.PC_RANGE[4]
    
    @property
    def PATCH_SIZE(self) -> Tuple[float, float]:
        """Patch size (height, width) in meters"""
        patch_h = self.PC_RANGE[4] - self.PC_RANGE[1]  # y_max - y_min = 60
        patch_w = self.PC_RANGE[3] - self.PC_RANGE[0]  # x_max - x_min = 30
        return (patch_h, patch_w)
    
    @property
    def BEV_RANGE(self) -> List[float]:
        """BEV detection range [x_min, y_min, x_max, y_max] (2D only)"""
        return [self.PC_RANGE[0], self.PC_RANGE[1], self.PC_RANGE[3], self.PC_RANGE[4]]
    
    # ============ Coordinate Normalization ============
    NORMALIZE_COORDS: bool = True  # Whether to normalize coordinates to [-1, 1]
    
    # ============ GT Generation Settings ============
    MIN_ARC_LENGTH: float = 1.0    # Minimum arc length for polylines (meters)
    MIN_AREA: float = 0.5          # Minimum area for polygons (square meters)
    
    # ============ Training Settings ============
    # Which modules to train
    TRAIN_QT_FORMER_LAYERS: int = 4  # Train last 4 layers of QT-Former
    TRAIN_LLM_LAYERS: int = 0  # Freeze LLM by default
    TRAIN_QUERY_INIT: bool = True  # Train query initialization
    TRAIN_MLP_HEAD: bool = True  # Train MLP decoder head
    
    # ============ Loss Weights ============
    LOSS_WEIGHT_CLASS: float = 2.0
    LOSS_WEIGHT_POINTS: float = 5.0
    LOSS_WEIGHT_DIR: float = 0.25     # 折中方案：方向损失贡献约5-8%，有意义但不主导
    LOSS_WEIGHT_BBOX: float = 2.0
    
    # ============ Matching Settings ============
    MATCH_COST_CLASS: float = 2.0
    MATCH_COST_POINTS: float = 5.0
    MATCH_COST_BBOX: float = 2.0


# Default configuration instance
DEFAULT_MAP_CONFIG = MapDetectionConfig()


if __name__ == "__main__":
    # Test configuration
    config = MapDetectionConfig()
    print(f"Class names: {config.CLASS_NAMES}")
    print(f"Use separate heads: {config.USE_SEPARATE_HEADS}")
    print(f"Shared feature dim: {config.SHARED_FEATURE_DIM}")
    print(f"MLP reduction path: {config.LLM_HIDDEN_SIZE} → {' → '.join(map(str, config.MLP_REDUCTION_DIMS))}")
    print(f"\nOutput dimensions:")
    print(f"  Classification: {config.CLS_OUTPUT_DIM}")
    print(f"  Points: {config.POINTS_OUTPUT_DIM}")
    print(f"  Bbox: {config.BBOX_OUTPUT_DIM}")
    print(f"  Total: {config.OUTPUT_DIM}")
    print(f"\nOther settings:")
    print(f"  BEV range: {config.BEV_RANGE}")
    print(f"  Total queries: {config.NUM_QUERIES}")
    print(f"  Scene tokens: {config.NUM_SCENE_TOKENS}")

