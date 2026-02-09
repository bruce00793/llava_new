"""
Map-Scene Interaction Layer

在 LLM 输出后添加 Cross-Attention，让 Map Features 直接从 Scene Tokens 提取视觉信息。

设计思路：
- LLM 的 Self-Attention 完成了语义融合
- 这个模块让 Map Features "主动询问" Scene Tokens，提取精确的空间几何信息

Flow:
    Map Features (B, 1050, dim) ───┐
                                   ├──→ Cross-Attention ──→ Enhanced Map Features
    Scene Tokens (B, 768, dim) ────┘

每一层包含：
1. Self-Attention: Map Features 之间交流（保证几何一致性）
2. Cross-Attention: Map Features 从 Scene Tokens 提取信息（核心！）
3. FFN: 非线性变换

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class MapSceneInteractionBlock(nn.Module):
    """
    单层 Map-Scene 交互模块
    
    结构：
        Map Features ──→ Self-Attention ──→ Cross-Attention ──→ FFN ──→ Output
                              ↑                    ↑
                         (Map 之间)         (Map ← Scene)
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # ========== Self-Attention (Map Features 之间) ==========
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(embed_dim)
        self.self_attn_dropout = nn.Dropout(dropout)
        
        # ========== Cross-Attention (Map ← Scene) ==========
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(embed_dim)
        self.cross_attn_dropout = nn.Dropout(dropout)
        
        # ========== FFN ==========
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(embed_dim)
        
    def forward(
        self,
        map_features: torch.Tensor,
        scene_tokens: torch.Tensor,
        map_mask: Optional[torch.Tensor] = None,
        scene_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            map_features: (B, num_map_tokens, embed_dim) - Map Features from LLM
            scene_tokens: (B, num_scene_tokens, embed_dim) - Scene Tokens
            map_mask: Optional attention mask for map features
            scene_mask: Optional attention mask for scene tokens
            
        Returns:
            enhanced_map_features: (B, num_map_tokens, embed_dim)
        """
        # 【重要】保存原始 dtype，Attention 计算强制使用 FP32 以避免数值溢出
        original_dtype = map_features.dtype
        
        # 【关键修复】使用 autocast(enabled=False) 彻底禁用 autocast
        # 原因：仅使用 .float() 无效！autocast 会覆盖内部的 F.linear/matmul 操作，
        # 导致 Attention 仍然以 FP16 执行，FP16 Attention 在随机初始化时
        # 极易产生 score 溢出 → inf → softmax(inf) = NaN
        with torch.cuda.amp.autocast(enabled=False):
            # 转换为 FP32 进行 Attention 计算
            map_features = map_features.float()
            scene_tokens = scene_tokens.float()
            
            # ========== 1. Self-Attention ==========
            # Map Features 之间的交流，保证几何一致性
            residual = map_features
            map_features = self.self_attn_norm(map_features)
            
            self_attn_out, _ = self.self_attn(
                query=map_features,
                key=map_features,
                value=map_features,
                attn_mask=map_mask,
            )
            map_features = residual + self.self_attn_dropout(self_attn_out)
            
            # ========== 2. Cross-Attention ==========
            # Map Features 从 Scene Tokens 提取视觉信息（核心！）
            residual = map_features
            map_features = self.cross_attn_norm(map_features)
            
            cross_attn_out, _ = self.cross_attn(
                query=map_features,      # Q: Map Features (我想知道什么)
                key=scene_tokens,        # K: Scene Tokens (图像有什么)
                value=scene_tokens,      # V: Scene Tokens (图像信息)
                key_padding_mask=scene_mask,
            )
            map_features = residual + self.cross_attn_dropout(cross_attn_out)
            
            # ========== 3. FFN ==========
            residual = map_features
            map_features = self.ffn_norm(map_features)
            ffn_out = self.ffn(map_features)
            map_features = residual + ffn_out
        
        # 【重要】转回原始 dtype
        map_features = map_features.to(original_dtype)
        
        return map_features


class MapSceneInteractionLayer(nn.Module):
    """
    多层 Map-Scene 交互模块
    
    在 LLM 输出后、Map Decoder 之前使用，让 Map Features 直接和 Scene Tokens 交互。
    
    Args:
        input_dim: 输入维度 (LLM hidden size, 默认 4096)
        embed_dim: 交互层维度 (默认 256)
        num_heads: 注意力头数 (默认 8)
        num_layers: 交互层数 (默认 3)
        ffn_dim: FFN 中间维度 (默认 1024)
        dropout: Dropout 概率 (默认 0.1)
    """
    
    def __init__(
        self,
        input_dim: int = 4096,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 3,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        
        # ========== Input Projection ==========
        # 将 LLM 输出 (4096) 降维到交互维度 (256)
        self.map_input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.scene_input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        # ========== Interaction Layers ==========
        self.layers = nn.ModuleList([
            MapSceneInteractionBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # ========== Output Projection ==========
        # 将交互后的特征投影回 LLM 维度 (用于残差连接)
        self.output_proj = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, input_dim),
        )
        
        # 残差连接的缩放因子
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)
        
        self._init_weights()
        
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        map_features: torch.Tensor,
        scene_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            map_features: (B, 1050, 4096) - Map Features from LLM (instance + point queries)
            scene_tokens: (B, 768, 4096) - Scene Tokens from Q-Former (经过 LLM 处理后)
            
        Returns:
            enhanced_map_features: (B, 1050, 4096) - Enhanced Map Features
        """
        # 保存原始特征用于残差连接
        map_features_residual = map_features
        
        # 【关键修复】整个 Map-Scene Interaction 在 FP32 下执行
        # 避免 autocast 把投影层和 Attention 转为 FP16
        with torch.cuda.amp.autocast(enabled=False):
            map_features = map_features.float()
            scene_tokens = scene_tokens.float()
            
            # ========== Input Projection ==========
            # (B, 1050, 4096) → (B, 1050, 256)
            map_proj = self.map_input_proj(map_features)
            
            # (B, 768, 4096) → (B, 768, 256)
            scene_proj = self.scene_input_proj(scene_tokens)
            
            # ========== Multi-Layer Interaction ==========
            for layer in self.layers:
                map_proj = layer(map_proj, scene_proj)
            
            # ========== Output Projection ==========
            # (B, 1050, 256) → (B, 1050, 4096)
            map_enhanced = self.output_proj(map_proj)
        
        # ========== Residual Connection ==========
        # 残差连接：保留 LLM 的语义信息 + 添加视觉空间信息
        map_features_residual = map_features_residual.float()
        output = map_features_residual + self.residual_scale * map_enhanced
        
        return output


def build_map_scene_interaction(
    input_dim: int = 4096,
    embed_dim: int = 256,
    num_heads: int = 8,
    num_layers: int = 3,
    ffn_dim: int = 1024,
    dropout: float = 0.1,
) -> MapSceneInteractionLayer:
    """
    构建 Map-Scene 交互层
    
    Args:
        input_dim: LLM hidden size (默认 4096)
        embed_dim: 交互层维度 (默认 256)
        num_heads: 注意力头数 (默认 8)
        num_layers: 层数 (默认 3)
        ffn_dim: FFN 维度 (默认 1024)
        dropout: Dropout (默认 0.1)
        
    Returns:
        MapSceneInteractionLayer
    """
    return MapSceneInteractionLayer(
        input_dim=input_dim,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )


if __name__ == "__main__":
    print("=" * 70)
    print("Testing MapSceneInteractionLayer")
    print("=" * 70)
    
    # 创建模块
    interaction_layer = MapSceneInteractionLayer(
        input_dim=4096,
        embed_dim=256,
        num_heads=8,
        num_layers=3,
    )
    
    # 测试输入
    B = 2
    map_features = torch.randn(B, 1050, 4096)   # Map Features from LLM
    scene_tokens = torch.randn(B, 768, 4096)    # Scene Tokens
    
    print(f"\n[Input]")
    print(f"  map_features: {map_features.shape}")
    print(f"  scene_tokens: {scene_tokens.shape}")
    
    # 前向传播
    enhanced_map_features = interaction_layer(map_features, scene_tokens)
    
    print(f"\n[Output]")
    print(f"  enhanced_map_features: {enhanced_map_features.shape}")
    
    # 验证残差连接
    diff = (enhanced_map_features - map_features).abs().mean()
    print(f"\n[Residual Check]")
    print(f"  Mean difference from input: {diff:.6f}")
    print(f"  (Should be small due to residual connection)")
    
    # 参数统计
    total_params = sum(p.numel() for p in interaction_layer.parameters())
    print(f"\n[Parameters]")
    print(f"  Total: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 梯度测试
    loss = enhanced_map_features.sum()
    loss.backward()
    print(f"\n[Gradient Check]")
    print(f"  map_input_proj grad: {interaction_layer.map_input_proj[1].weight.grad is not None}")
    print(f"  cross_attn grad: {interaction_layer.layers[0].cross_attn.in_proj_weight.grad is not None}")
    
    print("\n" + "=" * 70)
    print("✅ Test passed!")
    print("=" * 70)
