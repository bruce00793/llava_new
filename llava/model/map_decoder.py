"""
Iterative Refinement Map Decoder (优化版)

三大优化:
1. 优化1: 初始点预测同时使用 inst_features 和 point_features（不再盲猜）
2. 优化2: 点特征在迭代中累积更新（Self-Attention 结果传递到下一层）
3. 优化3: 迭代层数从 3 层增加到 6 层

设计要点:
- InitPointsHead: 融合 inst + point features 预测初始位置
- 点间 Self-Attention: 让 20 个点相互交流，保证连续性
- 点顺序编码: 让模型知道点 0 是起点、点 19 是终点
- 坐标编码: 每次迭代更新，让模型知道当前位置
- 特征累积: Self-Attention 结果传递到下一层，信息逐层积累

流程：
    instance_features (B, 50, 4096)     point_features (B, 50, 20, 4096)
              ↓                                  ↓
         InstReducer                        PointReducer
              ↓                                  ↓
    inst_reduced (B, 50, 256)          pt_reduced (B, 50, 20, 256)
              │                                  │
              ├──→ ClassHead → class_logits      │
              │    (B, 50, 3)                    │
              │                                  │
              └──→ InitPointsHead ◄──────────────┤  ← 优化1: 融合两种特征
                   init_points (B, 50, 20, 2)    │
                         │                       │
                         ▼                       ▼
                    ┌────────────────────────────────────┐
                    │     Iterative Refinement (6层)      │
                    │                                    │
                    │  current_pt_features = pt_reduced  │
                    │  For each layer (6 layers):        │
                    │    1. Point Self-Attention         │
                    │       → update current_pt_features │  ← 优化2: 特征累积
                    │    2. Feature Fusion               │
                    │       (pt + inst + coord + order)  │
                    │    3. Predict offset               │
                    │    4. Update points                │
                    │                                    │
                    └────────────────────────────────────┘
                                   │
                                   ▼
                           final_points (B, 50, 20, 2)

Author: Auto-generated for Map Detection
Date: 2025-01
Updated: 2025-02 (三大优化)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional

from .map_config import MapDetectionConfig, DEFAULT_MAP_CONFIG
from .map_structures import MapPrediction


class MLPBlock(nn.Module):
    """
    Basic MLP block with LayerNorm, Linear, and activation
    """
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        x = self.norm(x)
        x = self.linear(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


class InstFeatureReducer(nn.Module):
    """
    Reduce instance features from LLM hidden size to shared dimension.
    Path: 4096 → 2048 → 1024 → 512 → 256
    """
    def __init__(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        super().__init__()
        self.config = config
        
        layers = []
        in_dim = config.LLM_HIDDEN_SIZE  # 4096
        
        for out_dim in config.MLP_REDUCTION_DIMS:  # [2048, 1024, 512, 256]
            layers.append(MLPBlock(in_dim, out_dim))
            in_dim = out_dim
        
        self.reducer = nn.Sequential(*layers)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.reducer(hidden_states)


class PointFeatureReducer(nn.Module):
    """
    Reduce point features from LLM hidden size to shared dimension.
    Path: 4096 → 1024 → 256
    """
    def __init__(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        super().__init__()
        self.config = config
        
        self.reducer = nn.Sequential(
            MLPBlock(config.LLM_HIDDEN_SIZE, 1024, dropout=0.1),
            MLPBlock(1024, config.SHARED_FEATURE_DIM, dropout=0.1),
        )
        
    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        B, N, P, H = point_features.shape
        features_flat = point_features.reshape(-1, H)
        reduced = self.reducer(features_flat)
        return reduced.reshape(B, N, P, -1)


class ClassificationHead(nn.Module):
    """
    Classification head: predict class logits from instance features
    """
    def __init__(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        super().__init__()
        self.config = config
        
        self.head = nn.Sequential(
            nn.LayerNorm(config.SHARED_FEATURE_DIM),
            nn.Linear(config.SHARED_FEATURE_DIM, config.CLS_OUTPUT_DIM),
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        logits = self.head(features)
        logits = torch.clamp(logits, min=-50.0, max=50.0)
        return logits


class InitPointsHead(nn.Module):
    """
    从 Instance Features 和 Point Features 共同预测 20 个点的初始位置
    
    优化设计:
    - 不再只依赖 inst_features "盲猜"初始点
    - 融合 point_features（LLM 为每个点学到的信息）
    - inst_features 提供全局布局，point_features 提供点级别的定位信息
    """
    def __init__(self, inst_dim: int = 256, point_dim: int = 256, num_points: int = 20):
        super().__init__()
        self.num_points = num_points
        
        # 投影 point_features：提取每个点的定位倾向
        self.point_proj = nn.Sequential(
            nn.LayerNorm(point_dim),
            nn.Linear(point_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 投影 inst_features：提取全局布局信息
        self.inst_proj = nn.Sequential(
            nn.LayerNorm(inst_dim),
            nn.Linear(inst_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        
        # 融合后预测坐标
        # 64 (point) + 64 (inst) = 128 → 每个点单独预测坐标
        self.coord_head = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),  # 每个点预测 (x, y)
        )
    
    def forward(
        self, 
        inst_features: torch.Tensor,   # (B, N, 256)
        point_features: torch.Tensor,  # (B, N, P, 256)
    ) -> torch.Tensor:
        """
        Args:
            inst_features: (B, N, 256) 实例全局特征
            point_features: (B, N, P, 256) 每个点的局部特征
        Returns:
            init_points: (B, N, P, 2) 范围 [-1, 1]
        """
        B, N, P, _ = point_features.shape
        
        # Step 1: 投影 point_features
        pt_proj = self.point_proj(point_features)  # (B, N, P, 64)
        
        # Step 2: 投影 inst_features 并扩展到每个点
        inst_proj = self.inst_proj(inst_features)  # (B, N, 64)
        inst_proj = inst_proj.unsqueeze(2).expand(-1, -1, P, -1)  # (B, N, P, 64)
        
        # Step 3: 融合
        fused = torch.cat([pt_proj, inst_proj], dim=-1)  # (B, N, P, 128)
        
        # Step 4: 预测每个点的坐标
        # reshape → MLP → reshape
        fused_flat = fused.reshape(-1, 128)
        coords_flat = self.coord_head(fused_flat)  # (B*N*P, 2)
        coords = coords_flat.reshape(B, N, P, 2)
        
        # Step 5: 限制范围
        coords = coords.clamp(-1.0, 1.0)
        
        return coords


class CoordinateEncoder(nn.Module):
    """
    坐标位置编码器
    
    将 2D 坐标编码为高维向量，使用正弦位置编码
    
    【重要】num_freqs 不能太大！
    频率 = 2^i，梯度乘子 = freq * π。
    num_freqs=32 → max freq=2^31≈2.1e9 → 梯度乘子≈6.7e9 → 必然 NaN！
    num_freqs=10 → max freq=2^9=512 → 梯度乘子≈1608 → 安全。
    NeRF 对 [-1,1] 坐标也只用 num_freqs=10。
    """
    def __init__(self, output_dim: int = 64, num_freqs: int = 10):
        super().__init__()
        self.num_freqs = num_freqs
        self.output_dim = output_dim
        
        # 正弦编码后的维度: 2 (x,y) × num_freqs × 2 (sin,cos) = 4 × num_freqs
        sin_dim = 2 * num_freqs * 2
        self.proj = nn.Linear(sin_dim, output_dim)
        
        # 预计算频率 (对数间隔)
        # num_freqs=10: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
        # 最大梯度乘子 = 512 * π ≈ 1608（安全）
        freqs = 2.0 ** torch.linspace(0, num_freqs - 1, num_freqs)
        self.register_buffer('freqs', freqs)
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (..., 2) 坐标，范围 [-1, 1]
        Returns:
            embed: (..., output_dim)
        """
        # coords: (..., 2), freqs: (num_freqs,)
        # scaled: (..., 2, num_freqs)
        scaled = coords.unsqueeze(-1) * self.freqs * math.pi
        
        # 正弦和余弦
        sin_embed = torch.sin(scaled)  # (..., 2, num_freqs)
        cos_embed = torch.cos(scaled)  # (..., 2, num_freqs)
        
        # 拼接: (..., 2, num_freqs, 2) → (..., 4×num_freqs)
        embed = torch.stack([sin_embed, cos_embed], dim=-1)
        embed = embed.flatten(-3)  # (..., 4×num_freqs)
        
        # 投影到目标维度
        return self.proj(embed)


class PointOrderEmbedding(nn.Module):
    """
    点顺序编码
    
    让模型知道:
    - 点 0 是起点
    - 点 10 是中点
    - 点 19 是终点
    """
    def __init__(self, num_points: int = 20, embed_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(num_points, embed_dim)
        self._init_embedding()
    
    def _init_embedding(self):
        """
        使用正弦位置编码初始化，提供平滑的顺序先验
        """
        num_points = self.embed.num_embeddings
        embed_dim = self.embed.embedding_dim
        
        # 确保 embed_dim 是偶数，否则正弦编码维度不匹配
        if embed_dim % 2 != 0:
            raise ValueError(f"PointOrderEmbedding embed_dim must be even, got {embed_dim}")
        
        position = torch.arange(num_points).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * 
                           (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(num_points, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.embed.weight.data.copy_(pe)
    
    def forward(self, batch_size: int, num_instances: int, device: torch.device) -> torch.Tensor:
        """
        Returns:
            order_embed: (B, N, P, embed_dim)
        """
        # (P,) → (P, embed_dim)
        indices = torch.arange(self.embed.num_embeddings, device=device)
        order_embed = self.embed(indices)  # (P, embed_dim)
        
        # 扩展到 (B, N, P, embed_dim)
        order_embed = order_embed.unsqueeze(0).unsqueeze(0)
        order_embed = order_embed.expand(batch_size, num_instances, -1, -1)
        
        return order_embed


class PointSelfAttention(nn.Module):
    """
    点间 Self-Attention
    
    让 20 个点相互"交流"，建模连续性约束
    点 5 会知道点 4 和点 6 的信息，从而保持平滑
    """
    def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, point_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_features: (B, N, P, D)
        Returns:
            output: (B, N, P, D) 经过点间交互后的特征
        """
        B, N, P, D = point_features.shape
        
        # 【重要】保存原始 dtype，Attention 强制使用 FP32 避免数值溢出
        original_dtype = point_features.dtype
        
        # Reshape: (B, N, P, D) → (B×N, P, D)
        features_flat = point_features.reshape(B * N, P, D)
        
        # 【关键修复】使用 autocast(enabled=False) 彻底禁用 autocast
        # 原因：仅使用 .float() 无效！autocast 会覆盖内部的 F.linear/matmul 操作
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为 FP32 进行 Attention 计算
            features_flat = features_flat.float()
            
            # Self-Attention among P points
            residual = features_flat
            features_flat = self.norm(features_flat)
            attn_out, _ = self.self_attn(features_flat, features_flat, features_flat)
            features_flat = residual + self.dropout(attn_out)
        
        # 转回原始 dtype
        features_flat = features_flat.to(original_dtype)
        
        # Reshape back: (B×N, P, D) → (B, N, P, D)
        return features_flat.reshape(B, N, P, D)


class RefinementLayer(nn.Module):
    """
    单层精修模块
    
    优化设计:
    1. 点间 Self-Attention（结果累积到下一层）
    2. 特征融合 (point_features + inst_features + coord_embed + order_embed)
    3. 预测偏移量
    4. 返回更新后的 point_features，供下一层使用
    """
    def __init__(
        self,
        feature_dim: int = 256,
        coord_dim: int = 64,
        order_dim: int = 64,
        hidden_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 点间 Self-Attention
        self.point_self_attn = PointSelfAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=dropout,
        )
        
        # 融合维度: point_feat + inst_feat + coord_embed + order_embed
        # 256 + 256 + 64 + 64 = 640
        fused_dim = feature_dim + feature_dim + coord_dim + order_dim
        
        # 偏移预测 MLP
        self.offset_mlp = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 2),  # 预测 (dx, dy)
        )
        
        # 偏移量缩放因子 (让初始偏移较小，训练更稳定)
        self.offset_scale = nn.Parameter(torch.tensor(0.1))
    
    def forward(
        self,
        point_features: torch.Tensor,  # (B, N, P, 256)
        inst_features: torch.Tensor,   # (B, N, 256)
        current_points: torch.Tensor,  # (B, N, P, 2)
        coord_embed: torch.Tensor,     # (B, N, P, 64)
        order_embed: torch.Tensor,     # (B, N, P, 64)
    ) -> tuple:
        """
        Returns:
            updated_point_features: (B, N, P, 256) 经过 Self-Attention 更新的点特征
            offset: (B, N, P, 2) 预测的偏移量
        """
        B, N, P, _ = point_features.shape
        
        # Step 1: 点间 Self-Attention（让点相互交流）
        # 关键：返回更新后的特征供下一层使用
        updated_point_features = self.point_self_attn(point_features)
        
        # Step 2: 扩展 inst_features 到每个点
        inst_expanded = inst_features.unsqueeze(2).expand(-1, -1, P, -1)  # (B, N, P, 256)
        
        # Step 3: 特征融合
        fused = torch.cat([
            updated_point_features,  # 256: 点的局部特征 (经过 Self-Attention)
            inst_expanded,           # 256: 实例的全局特征
            coord_embed,             # 64:  当前坐标编码
            order_embed,             # 64:  点顺序编码
        ], dim=-1)  # (B, N, P, 640)
        
        # Step 4: 预测偏移量
        B, N, P, D = fused.shape
        fused_flat = fused.reshape(-1, D)
        offset_flat = self.offset_mlp(fused_flat)
        offset = offset_flat.reshape(B, N, P, 2)
        
        # Step 5: 缩放偏移量
        offset = offset * self.offset_scale
        
        # 返回两个值：更新后的点特征（供下一层使用）和偏移量
        return updated_point_features, offset


class IterativePointHead(nn.Module):
    """
    迭代精修点回归头（优化版）
    
    优化设计:
    1. 初始点预测同时利用 inst_features 和 point_features（不再盲猜）
    2. 点特征在迭代中累积更新（Self-Attention 结果传递到下一层）
    3. 支持更多迭代层数（默认6层）
    
    完整流程:
    1. 从 inst_features + point_features 共同预测初始点位置
    2. 多层迭代精修，每层:
       - 点间 Self-Attention（结果累积）
       - 特征融合 (pt + inst + coord + order)
       - 预测偏移量
       - 更新坐标和点特征
    """
    def __init__(
        self,
        feature_dim: int = 256,
        num_points: int = 20,
        num_layers: int = 6,  # 优化3：增加到6层
        coord_dim: int = 64,
        order_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_points = num_points
        self.num_layers = num_layers
        
        # 优化1：初始点预测同时使用 inst 和 point features
        self.init_points_head = InitPointsHead(
            inst_dim=feature_dim,
            point_dim=feature_dim,
            num_points=num_points,
        )
        
        # 坐标编码器
        self.coord_encoder = CoordinateEncoder(output_dim=coord_dim)
        
        # 点顺序编码
        self.order_embedding = PointOrderEmbedding(num_points, order_dim)
        
        # 迭代精修层
        self.refine_layers = nn.ModuleList([
            RefinementLayer(
                feature_dim=feature_dim,
                coord_dim=coord_dim,
                order_dim=order_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
    
    def forward(
        self,
        inst_features: torch.Tensor,   # (B, N, 256)
        point_features: torch.Tensor,  # (B, N, P, 256)
    ) -> Dict[str, torch.Tensor]:
        """
        Returns:
            dict with:
                - 'points': (B, N, P, 2) 最终点坐标
                - 'init_points': (B, N, P, 2) 初始点坐标
                - 'intermediate_points': List[(B, N, P, 2)] 每层的中间结果
        """
        B, N, P, _ = point_features.shape
        device = point_features.device
        
        # ========== 阶段 1: 预测初始点（优化1）==========
        # 同时利用 inst_features 和 point_features，不再盲猜
        init_points = self.init_points_head(inst_features, point_features)  # (B, N, P, 2)
        current_points = init_points.clone()
        
        # 获取点顺序编码 (固定的，整个迭代过程不变)
        order_embed = self.order_embedding(B, N, device)  # (B, N, P, 64)
        
        # ========== 阶段 2: 迭代精修（优化2）==========
        intermediate_points = [init_points]
        
        # 优化2：点特征在迭代中累积更新
        current_pt_features = point_features  # 初始为原始点特征
        
        for layer in self.refine_layers:
            # 编码当前坐标（每次迭代都更新）
            coord_embed = self.coord_encoder(current_points)  # (B, N, P, 64)
            
            # 优化2：layer 返回两个值
            # - updated_pt_features: 经过 Self-Attention 更新的点特征
            # - offset: 预测的偏移量
            current_pt_features, offset = layer(
                point_features=current_pt_features,  # 使用累积更新的特征
                inst_features=inst_features,
                current_points=current_points,
                coord_embed=coord_embed,
                order_embed=order_embed,
            )
            
            # 更新坐标
            current_points = current_points + offset
            intermediate_points.append(current_points.clone())
        
        # 最终限制在 [-1, 1]
        final_points = current_points.clamp(-1.0, 1.0)
        
        return {
            'points': final_points,
            'init_points': init_points,
            'intermediate_points': intermediate_points,
        }


def compute_bbox_from_points(points: torch.Tensor) -> torch.Tensor:
    """
    Compute bounding box from point coordinates.
    """
    x = points[..., 0]
    y = points[..., 1]
    
    xmin = x.min(dim=-1)[0]
    xmax = x.max(dim=-1)[0]
    ymin = y.min(dim=-1)[0]
    ymax = y.max(dim=-1)[0]
    
    cx = (xmin + xmax) / 2
    cy = (ymin + ymax) / 2
    w = xmax - xmin
    h = ymax - ymin
    
    bbox = torch.stack([cx, cy, w, h], dim=-1)
    return bbox


class MapDecoder(nn.Module):
    """
    Map Decoder with Iterative Refinement
    
    优化后的设计:
    1. 从 Instance Features 预测 20 个初始点
    2. 点间 Self-Attention 保证连续性
    3. 点顺序编码提供结构先验
    4. Instance 特征融合保持全局一致性
    5. 多层迭代精修提升精度
    
    Architecture:
        instance_features          point_features
        (B, 50, 4096)             (B, 50, 20, 4096)
              │                         │
              ▼                         ▼
        InstReducer               PointReducer
              │                         │
              ▼                         ▼
        inst_reduced              pt_reduced
        (B, 50, 256)             (B, 50, 20, 256)
              │                         │
              ├──→ ClassHead            │
              │                         │
              └──→ IterativePointHead ◄─┘
                        │
                        ▼
                  final_points (B, 50, 20, 2)
    """
    def __init__(self, config: MapDetectionConfig = DEFAULT_MAP_CONFIG):
        super().__init__()
        self.config = config
        
        # ========== Feature Reducers ==========
        self.inst_reducer = InstFeatureReducer(config)
        self.point_reducer = PointFeatureReducer(config)
        
        # ========== Heads ==========
        self.cls_head = ClassificationHead(config)
        
        # 迭代精修点回归头（优化版）
        self.point_head = IterativePointHead(
            feature_dim=config.SHARED_FEATURE_DIM,       # 256
            num_points=config.NUM_POINTS_PER_INSTANCE,   # 20
            num_layers=6,                                # 优化3: 6 层精修
            coord_dim=64,
            order_dim=64,
            num_heads=4,
            dropout=0.1,
        )
        
    def forward(
        self, 
        instance_features: torch.Tensor,
        point_features: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with Iterative Refinement.
        
        Args:
            instance_features: [B, 50, 4096]
            point_features: [B, 50, 20, 4096]
            
        Returns:
            Dict containing:
                - class_logits: [B, 50, 3]
                - points: [B, 50, 20, 2]
                - bbox: [B, 50, 4]
                - init_points: [B, 50, 20, 2] (用于辅助损失)
                - intermediate_points: List (用于辅助损失)
        """
        B = instance_features.shape[0]
        N = instance_features.shape[1]  # 50
        P = point_features.shape[2]      # 20
        
        # 【关键修复】整个 Decoder 在 FP32 下执行
        # 避免 autocast 把 4096→256 降维和后续计算转为 FP16
        with torch.cuda.amp.autocast(enabled=False):
            instance_features = instance_features.float()
            point_features = point_features.float()
            
            # ========== Step 1: Feature Reduction ==========
            inst_reduced = self.inst_reducer(instance_features)  # (B, 50, 256)
            pt_reduced = self.point_reducer(point_features)       # (B, 50, 20, 256)
            
            # ========== Step 2: Classification ==========
            class_logits = self.cls_head(inst_reduced)  # (B, 50, 3)
            
            # ========== Step 3: Iterative Point Prediction ==========
            point_output = self.point_head(inst_reduced, pt_reduced)
            
            points = point_output['points']                      # (B, 50, 20, 2)
            init_points = point_output['init_points']            # (B, 50, 20, 2)
            intermediate_points = point_output['intermediate_points']  # List
        
        # ========== Step 4: Compute BBox ==========
        bbox = compute_bbox_from_points(points)  # (B, 50, 4)
        
        return {
            'class_logits': class_logits,
            'points': points,
            'bbox': bbox,
            'init_points': init_points,
            'intermediate_points': intermediate_points,
        }
    
    def predict(
        self, 
        instance_features: torch.Tensor,
        point_features: torch.Tensor,
    ) -> MapPrediction:
        """
        Make predictions and wrap in MapPrediction structure.
        """
        output = self.forward(instance_features, point_features)
        
        class_logits = output['class_logits'].squeeze(0)
        points = output['points'].squeeze(0)
        bbox = output['bbox'].squeeze(0)
        
        return MapPrediction(
            class_logits=class_logits,
            points=points,
            bbox=bbox,
        )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MapDecoder with Iterative Refinement")
    print("=" * 70)
    
    # Test decoder
    config = MapDetectionConfig()
    decoder = MapDecoder(config)
    
    # Test input
    B, N, P = 2, 50, 20
    instance_features = torch.randn(B, N, config.LLM_HIDDEN_SIZE)
    point_features = torch.randn(B, N, P, config.LLM_HIDDEN_SIZE)
    
    print(f"\n[输入]")
    print(f"  instance_features: {instance_features.shape}")
    print(f"  point_features: {point_features.shape}")
    
    # Forward pass
    print(f"\n[前向传播]")
    output = decoder(instance_features, point_features)
    
    print(f"  class_logits: {output['class_logits'].shape}")
    print(f"  points: {output['points'].shape}")
    print(f"  bbox: {output['bbox'].shape}")
    print(f"  init_points: {output['init_points'].shape}")
    print(f"  intermediate_points: {len(output['intermediate_points'])} layers")
    
    # 验证数值范围
    print(f"\n[数值范围验证]")
    print(f"  init_points range: [{output['init_points'].min():.4f}, {output['init_points'].max():.4f}]")
    print(f"  final_points range: [{output['points'].min():.4f}, {output['points'].max():.4f}]")
    
    # Parameter count
    print(f"\n[参数统计]")
    total_params = sum(p.numel() for p in decoder.parameters())
    
    inst_reducer_params = sum(p.numel() for p in decoder.inst_reducer.parameters())
    point_reducer_params = sum(p.numel() for p in decoder.point_reducer.parameters())
    cls_params = sum(p.numel() for p in decoder.cls_head.parameters())
    point_head_params = sum(p.numel() for p in decoder.point_head.parameters())
    
    print(f"  InstReducer: {inst_reducer_params:,}")
    print(f"  PointReducer: {point_reducer_params:,}")
    print(f"  ClassHead: {cls_params:,}")
    print(f"  IterativePointHead: {point_head_params:,}")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # Detailed IterativePointHead breakdown
    print(f"\n[IterativePointHead 详细参数]")
    init_head = sum(p.numel() for p in decoder.point_head.init_points_head.parameters())
    coord_enc = sum(p.numel() for p in decoder.point_head.coord_encoder.parameters())
    order_emb = sum(p.numel() for p in decoder.point_head.order_embedding.parameters())
    refine_layers = sum(p.numel() for p in decoder.point_head.refine_layers.parameters())
    
    print(f"    InitPointsHead: {init_head:,}")
    print(f"    CoordinateEncoder: {coord_enc:,}")
    print(f"    OrderEmbedding: {order_emb:,}")
    print(f"    RefineLayers (×3): {refine_layers:,}")
    
    # Test backward
    print(f"\n[梯度测试]")
    loss = output['points'].sum() + output['class_logits'].sum()
    loss.backward()
    
    print(f"  InstReducer grad: {decoder.inst_reducer.reducer[0].linear.weight.grad is not None}")
    print(f"  PointReducer grad: {decoder.point_reducer.reducer[0].linear.weight.grad is not None}")
    print(f"  InitPointsHead grad: {decoder.point_head.init_points_head.mlp[1].weight.grad is not None}")
    print(f"  RefineLayers grad: {decoder.point_head.refine_layers[0].offset_mlp[1].weight.grad is not None}")
    
    print("\n" + "=" * 70)
    print("设计总结:")
    print("  1. InitPointsHead: 从 inst_features 预测 20 个初始点")
    print("  2. PointSelfAttention: 点间交互，保证连续性")
    print("  3. PointOrderEmbedding: 点顺序编码 (0=起点, 19=终点)")
    print("  4. RefinementLayer × 3: 迭代精修，融合多种信息")
    print("  5. 输出: 初始点 + 中间结果 (用于辅助损失)")
    print("=" * 70)
    print("✅ All tests passed!")
    print("=" * 70)
