"""
Learnable Instance and Point Queries for Map Detection

Structure:
- 50 instances
- Each instance: 1 instance query + 20 point queries = 21 queries
- Total: 1050 queries

优化设计：
- Instance Query = 内容向量 + 实例位置编码
- Point Query = 内容向量 + 点位置编码

好处：
1. 点位置编码让模型知道"这是第几个点"（起点/中点/终点）
2. 实例位置编码让模型知道"这是第几个实例"
3. 同位置的点共享位置编码，提高泛化能力
4. 加速训练收敛

Author: Auto-generated for Map Detection
Date: 2025-01
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


class MapInstancePointQueries(nn.Module):
    """
    Learnable queries for map element detection with position encoding.
    
    Structure per instance:
        [instance_query, point_1, point_2, ..., point_20]
    
    Total sequence:
        [Inst1(21), Inst2(21), ..., Inst50(21)] = 1050 queries
    
    Query 组成：
        Instance Query = instance_content[i] + instance_position[i]
        Point Query = point_content[i,j] + point_position[j]
    
    关键设计：
        - 同一位置的点（如所有第0个点）共享 point_position[0]
        - 这让模型知道"所有 Point_x_0 都是起点"
        - 加速训练，提高泛化
    """
    
    def __init__(
        self,
        num_instances: int = 50,
        num_points: int = 20,
        embed_dim: int = 4096,
        use_sinusoidal_point_pos: bool = False,  # 是否使用正弦位置编码
    ):
        """
        Args:
            num_instances: Number of map element instances (default: 50)
            num_points: Number of points per instance (default: 20)
            embed_dim: Embedding dimension, must match LLM hidden size (default: 4096 for Vicuna-7B)
            use_sinusoidal_point_pos: If True, use fixed sinusoidal encoding for point positions.
                                      If False (default), use learnable embeddings.
        """
        super().__init__()
        
        self.num_instances = num_instances
        self.num_points = num_points
        self.embed_dim = embed_dim
        self.use_sinusoidal_point_pos = use_sinusoidal_point_pos
        
        # =====================================================================
        # Instance Query = instance_content + instance_position
        # =====================================================================
        
        # 1. Instance 内容向量：可学习，代表"这个实例要提取什么信息"
        #    初始化：正态分布 N(0, 0.02²)
        self.instance_content = nn.Parameter(
            torch.randn(num_instances, embed_dim) * 0.02
        )
        
        # 2. Instance 位置编码：可学习，代表"这是第几个实例"
        #    让不同实例有不同的初始偏向
        #    初始化：正态分布 N(0, 0.02²)
        self.instance_position = nn.Embedding(num_instances, embed_dim)
        nn.init.normal_(self.instance_position.weight, mean=0.0, std=0.02)
        
        # =====================================================================
        # Point Query = point_content + point_position
        # =====================================================================
        
        # 3. Point 内容向量：可学习，代表"这个点要提取什么信息"
        #    [50, 20, 4096]
        #    初始化：正态分布 N(0, 0.02²)
        self.point_content = nn.Parameter(
            torch.randn(num_instances, num_points, embed_dim) * 0.02
        )
        
        # 4. Point 位置编码：代表"这是第几个点"
        #    关键：所有实例的第 j 个点共享同一个 position[j]
        #    这让模型知道：position[0] = 起点，position[19] = 终点
        if use_sinusoidal_point_pos:
            # 使用固定的正弦位置编码（类似 Transformer 原版）
            point_pos = self._create_sinusoidal_encoding(num_points, embed_dim)
            self.register_buffer('point_position', point_pos)
        else:
            # 使用可学习的位置编码（推荐，更灵活）
            self.point_position = nn.Embedding(num_points, embed_dim)
            nn.init.normal_(self.point_position.weight, mean=0.0, std=0.02)
    
    def _create_sinusoidal_encoding(self, num_positions: int, dim: int) -> torch.Tensor:
        """
        创建正弦位置编码（固定，不可学习）
        
        公式：
            PE(pos, 2i) = sin(pos / 10000^(2i/d))
            PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
        
        Args:
            num_positions: 位置数量（20 个点）
            dim: 编码维度（4096）
        
        Returns:
            encoding: [num_positions, dim]
        """
        encoding = torch.zeros(num_positions, dim)
        position = torch.arange(0, num_positions, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim)
        )
        
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        
        return encoding
    
    def forward(self, batch_size: int) -> torch.Tensor:
        """
        Generate query sequence for a batch.
        
        Sequence structure:
            [Inst1_query, Inst1_P1, ..., Inst1_P20,
             Inst2_query, Inst2_P1, ..., Inst2_P20,
             ...
             Inst50_query, Inst50_P1, ..., Inst50_P20]
        
        维度变化：
            instance_content:  [50, 4096]
            instance_position: [50, 4096]
            → instance_query:  [50, 4096] (相加)
            
            point_content:  [50, 20, 4096]
            point_position: [20, 4096] (共享)
            → point_query:  [50, 20, 4096] (广播相加)
            
            最终输出: [B, 1050, 4096]
        
        Args:
            batch_size: Batch size
        
        Returns:
            queries: (batch_size, 1050, embed_dim)
        """
        # ========== 确定目标设备 ==========
        # 使用 instance_content 的设备作为基准（这是 nn.Parameter，不会被 accelerate 移动）
        target_device = self.instance_content.device
        
        # ========== 构建 Instance Queries ==========
        # 直接使用 embedding 的 weight（避免索引操作在多GPU下的问题）
        inst_position = self.instance_position.weight.to(target_device)  # [50, 4096]
        instance_queries = self.instance_content + inst_position  # [50, 4096]
        
        # ========== 构建 Point Queries ==========
        if self.use_sinusoidal_point_pos:
            point_pos = self.point_position.to(target_device)  # [20, 4096] (buffer)
        else:
            point_pos = self.point_position.weight.to(target_device)  # [20, 4096]
        
        # 广播：[50, 20, 4096] + [20, 4096] → [50, 20, 4096]
        point_queries = self.point_content + point_pos.unsqueeze(0)  # [50, 20, 4096]
        
        # ========== 组合成序列 ==========
        # 交替排列：[Inst1, P1_1..P1_20, Inst2, P2_1..P2_20, ...]
        queries_list = []
        
        for i in range(self.num_instances):
            # Add instance query
            queries_list.append(instance_queries[i:i+1])  # [1, 4096]
            
            # Add 20 point queries
            queries_list.append(point_queries[i])  # [20, 4096]
        
        # Concatenate: [1050, 4096]
        queries = torch.cat(queries_list, dim=0)
        
        # Expand to batch: [B, 1050, 4096]
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)
        
        return queries
    
    def get_total_queries(self) -> int:
        """Get total number of queries."""
        return self.num_instances * (1 + self.num_points)
    
    def get_position_encoding_info(self) -> dict:
        """获取位置编码的统计信息（用于调试）"""
        info = {
            'num_instances': self.num_instances,
            'num_points': self.num_points,
            'embed_dim': self.embed_dim,
            'use_sinusoidal_point_pos': self.use_sinusoidal_point_pos,
        }
        
        # Instance position stats
        if hasattr(self.instance_position, 'weight'):
            inst_pos = self.instance_position.weight
            info['instance_position'] = {
                'mean': inst_pos.mean().item(),
                'std': inst_pos.std().item(),
                'shape': list(inst_pos.shape),
            }
        
        # Point position stats
        if self.use_sinusoidal_point_pos:
            point_pos = self.point_position
            info['point_position'] = {
                'type': 'sinusoidal (fixed)',
                'mean': point_pos.mean().item(),
                'std': point_pos.std().item(),
                'shape': list(point_pos.shape),
            }
        else:
            point_pos = self.point_position.weight
            info['point_position'] = {
                'type': 'learnable',
                'mean': point_pos.mean().item(),
                'std': point_pos.std().item(),
                'shape': list(point_pos.shape),
            }
        
        return info


class MapAttentionMask:
    """
    Custom attention mask for map detection queries.
    
    Mask rules:
    1. All tokens can see text + scene queries (causal for text+scene itself)
    2. Instance queries can see all previous tokens + all instance queries
    3. Point queries can see: all previous tokens + their instance query + their sibling point queries
    
    This creates a structured attention pattern that allows:
    - Instance queries to have global instance-level communication
    - Point queries to focus on their own instance's information
    """
    
    @staticmethod
    def create_mask(
        batch_size: int,
        text_len: int,
        scene_len: int = 768,
        num_instances: int = 50,
        num_points: int = 20,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create custom attention mask compatible with transformers.
        
        Args:
            batch_size: Batch size
            text_len: Length of text tokens (including scene tokens if embedded)
            scene_len: Length of scene queries (default: 768, set to 0 if scene is in text)
            num_instances: Number of instances (default: 50)
            num_points: Points per instance (default: 20)
            device: Device for mask
            dtype: Data type for mask
        
        Returns:
            mask: (batch_size, 1, total_len, total_len)
                  1.0 = can attend, 0.0 = cannot attend
                  Format compatible with transformers 4D attention mask
        """
        queries_per_inst = 1 + num_points  # 21
        total_queries = num_instances * queries_per_inst  # 1050
        prefix_len = text_len + scene_len
        total_len = prefix_len + total_queries
        
        if device is None:
            device = torch.device('cpu')
        
        # Initialize mask: (total_len, total_len)
        # Start with all zeros (no attention)
        mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)
        
        # ===== Rule 1: Prefix (text + scene) uses causal attention =====
        # Lower triangular for causal: position i can see positions 0..i
        causal_mask = torch.tril(torch.ones(prefix_len, prefix_len, dtype=dtype, device=device))
        mask[:prefix_len, :prefix_len] = causal_mask
        
        # ===== Rule 2: All queries can see entire prefix =====
        mask[prefix_len:, :prefix_len] = 1.0
        
        # ===== Rule 3: Instance queries can see all instance queries =====
        # This allows global communication between instances
        instance_positions = []
        for i in range(num_instances):
            pos = prefix_len + i * queries_per_inst
            instance_positions.append(pos)
        
        for pos_i in instance_positions:
            for pos_j in instance_positions:
                mask[pos_i, pos_j] = 1.0
        
        # ===== Rule 4: Point queries can see their instance + siblings =====
        # This keeps points focused on their own instance
        for inst_idx in range(num_instances):
            inst_start = prefix_len + inst_idx * queries_per_inst
            inst_query_pos = inst_start
            point_start = inst_start + 1
            point_end = inst_start + queries_per_inst
            
            # Point queries see their instance query
            mask[point_start:point_end, inst_query_pos] = 1.0
            
            # Point queries see each other (within same instance)
            mask[point_start:point_end, point_start:point_end] = 1.0
        
        # ===== Rule 5: Instance queries can also see their own points =====
        # This allows instance query to aggregate point information
        for inst_idx in range(num_instances):
            inst_start = prefix_len + inst_idx * queries_per_inst
            inst_query_pos = inst_start
            point_start = inst_start + 1
            point_end = inst_start + queries_per_inst
            
            # Instance query can see its own points
            mask[inst_query_pos, point_start:point_end] = 1.0
        
        # Expand to 4D format: (batch_size, 1, total_len, total_len)
        # The "1" dimension is for num_heads broadcasting
        mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, -1, -1)
        
        return mask
    
    @staticmethod
    def visualize_mask(mask: torch.Tensor, save_path: str = None):
        """
        Visualize the attention mask pattern (for debugging).
        
        Args:
            mask: (batch, 1, seq, seq) or (seq, seq)
            save_path: Optional path to save the visualization
        """
        import matplotlib.pyplot as plt
        
        if mask.dim() == 4:
            mask_2d = mask[0, 0].cpu().numpy()
        elif mask.dim() == 2:
            mask_2d = mask.cpu().numpy()
        else:
            raise ValueError(f"Unexpected mask dimension: {mask.dim()}")
        
        plt.figure(figsize=(12, 12))
        plt.imshow(mask_2d, cmap='Blues', aspect='auto')
        plt.colorbar(label='Attention (1=attend, 0=mask)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.title('Custom Attention Mask Pattern')
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved mask visualization to {save_path}")
        else:
            plt.show()


class MapQueryExtractor:
    """
    Extract instance and point features from LLM output.
    """
    
    @staticmethod
    def extract_features(
        llm_output: torch.Tensor,
        text_len: int,
        scene_len: int = 768,
        num_instances: int = 50,
        num_points: int = 20,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract instance and point features from LLM output.
        
        Args:
            llm_output: (batch_size, total_len, hidden_size)
            text_len: Length of text tokens
            scene_len: Length of scene queries (default: 768)
            num_instances: Number of instances (default: 50)
            num_points: Points per instance (default: 20)
        
        Returns:
            instance_features: (batch_size, 50, hidden_size)
            point_features: (batch_size, 50, 20, hidden_size)
        """
        batch_size = llm_output.shape[0]
        hidden_size = llm_output.shape[2]
        
        prefix_len = text_len + scene_len
        queries_per_inst = 1 + num_points
        
        # Extract query outputs
        query_outputs = llm_output[:, prefix_len:, :]  # (B, 1050, hidden_size)
        
        # Extract instance queries
        instance_features_list = []
        point_features_list = []
        
        for i in range(num_instances):
            start_idx = i * queries_per_inst
            
            # Instance query
            inst_feat = query_outputs[:, start_idx:start_idx+1, :]  # (B, 1, H)
            instance_features_list.append(inst_feat)
            
            # Point queries
            point_feat = query_outputs[:, start_idx+1:start_idx+queries_per_inst, :]  # (B, 20, H)
            point_features_list.append(point_feat)
        
        # Concatenate
        instance_features = torch.cat(instance_features_list, dim=1)  # (B, 50, H)
        point_features = torch.stack(point_features_list, dim=1)  # (B, 50, 20, H)
        
        return instance_features, point_features


# Simple test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing MapInstancePointQueries with Position Encoding")
    print("=" * 70)
    
    # ========== 测试 1: 可学习位置编码 ==========
    print("\n" + "=" * 70)
    print("[测试 1] 可学习位置编码 (推荐)")
    print("=" * 70)
    
    query_module = MapInstancePointQueries(
        num_instances=50,
        num_points=20,
        embed_dim=4096,
        use_sinusoidal_point_pos=False  # 可学习
    )
    
    # 参数量分析
    print("\n[参数量分析]")
    param_count = sum(p.numel() for p in query_module.parameters())
    print(f"  总参数量: {param_count:,} ({param_count/1e6:.2f}M)")
    print(f"  组成:")
    print(f"    instance_content:  {query_module.instance_content.numel():,}")
    print(f"    instance_position: {query_module.instance_position.weight.numel():,}")
    print(f"    point_content:     {query_module.point_content.numel():,}")
    print(f"    point_position:    {query_module.point_position.weight.numel():,}")
    
    # 位置编码信息
    print("\n[位置编码信息]")
    info = query_module.get_position_encoding_info()
    print(f"  Instance 位置编码:")
    print(f"    shape: {info['instance_position']['shape']}")
    print(f"    mean: {info['instance_position']['mean']:.4f}")
    print(f"    std: {info['instance_position']['std']:.4f}")
    print(f"  Point 位置编码:")
    print(f"    type: {info['point_position']['type']}")
    print(f"    shape: {info['point_position']['shape']}")
    print(f"    mean: {info['point_position']['mean']:.4f}")
    print(f"    std: {info['point_position']['std']:.4f}")
    
    # 生成 Queries
    print("\n[生成 Queries]")
    batch_size = 2
    queries = query_module(batch_size)
    print(f"  输出 shape: {queries.shape}")  # Should be (2, 1050, 4096)
    assert queries.shape == (2, 1050, 4096)
    print("  ✓ Shape 正确!")
    
    # 验证位置编码共享
    print("\n[验证位置编码共享]")
    # 获取第 0 个实例的第 0 个点 和 第 1 个实例的第 0 个点
    # 它们应该共享相同的 point_position[0]
    point_pos = query_module.point_position.weight[0]  # position[0]
    
    # 位置 0: Inst0, 位置 1-20: Inst0 的 20 个点
    # 位置 21: Inst1, 位置 22-41: Inst1 的 20 个点
    inst0_point0 = queries[0, 1, :]   # Inst0 的 Point0
    inst1_point0 = queries[0, 22, :]  # Inst1 的 Point0
    
    # 它们的差异应该只来自 point_content，不来自 point_position
    # (因为 point_position[0] 是共享的)
    content_diff = query_module.point_content[0, 0] - query_module.point_content[1, 0]
    query_diff = inst0_point0 - inst1_point0
    
    # 检查差异是否相等（说明 position 被共享了）
    diff_match = torch.allclose(content_diff, query_diff.cpu(), atol=1e-5)
    print(f"  Inst0_Point0 和 Inst1_Point0 的差异只来自 content: {diff_match}")
    if diff_match:
        print("  ✓ 位置编码正确共享!")
    else:
        print("  ✗ 警告：位置编码可能未正确共享")
    
    # ========== 测试 2: 正弦位置编码 ==========
    print("\n" + "=" * 70)
    print("[测试 2] 正弦位置编码 (固定)")
    print("=" * 70)
    
    query_module_sin = MapInstancePointQueries(
        num_instances=50,
        num_points=20,
        embed_dim=4096,
        use_sinusoidal_point_pos=True  # 正弦
    )
    
    info_sin = query_module_sin.get_position_encoding_info()
    print(f"  Point 位置编码:")
    print(f"    type: {info_sin['point_position']['type']}")
    print(f"    shape: {info_sin['point_position']['shape']}")
    print(f"    mean: {info_sin['point_position']['mean']:.4f}")
    print(f"    std: {info_sin['point_position']['std']:.4f}")
    
    queries_sin = query_module_sin(batch_size)
    print(f"  输出 shape: {queries_sin.shape}")
    assert queries_sin.shape == (2, 1050, 4096)
    print("  ✓ Shape 正确!")
    
    # ========== 测试 3: Attention Mask ==========
    print("\n" + "=" * 70)
    print("[测试 3] MapAttentionMask")
    print("=" * 70)
    
    mask = MapAttentionMask.create_mask(
        batch_size=2,
        text_len=100,
        scene_len=768,
        num_instances=50,
        num_points=20
    )
    total_len = 100 + 768 + 1050
    print(f"  Mask shape: {mask.shape}")
    assert mask.shape == (2, 1, total_len, total_len)
    print("  ✓ Mask 创建成功!")
    
    # ========== 测试 4: 特征提取 ==========
    print("\n" + "=" * 70)
    print("[测试 4] MapQueryExtractor")
    print("=" * 70)
    
    fake_llm_output = torch.randn(2, total_len, 4096)
    inst_feat, point_feat = MapQueryExtractor.extract_features(
        llm_output=fake_llm_output,
        text_len=100,
        scene_len=768,
        num_instances=50,
        num_points=20
    )
    print(f"  Instance features: {inst_feat.shape}")  # (2, 50, 4096)
    print(f"  Point features: {point_feat.shape}")     # (2, 50, 20, 4096)
    assert inst_feat.shape == (2, 50, 4096)
    assert point_feat.shape == (2, 50, 20, 4096)
    print("  ✓ 特征提取成功!")
    
    # ========== 总结 ==========
    print("\n" + "=" * 70)
    print("✅ 所有测试通过!")
    print("=" * 70)
    print("\n设计总结:")
    print("  Instance Query = instance_content[i] + instance_position[i]")
    print("  Point Query    = point_content[i,j] + point_position[j]")
    print("\n关键优势:")
    print("  1. 同位置的点共享 point_position，让模型知道点的顺序")
    print("  2. point_position[0] = 起点先验")
    print("  3. point_position[19] = 终点先验")
    print("  4. 加速训练收敛，提高泛化能力")
    print("=" * 70)

