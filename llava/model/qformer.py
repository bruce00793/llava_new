"""
Q-Former for Multi-View Map Detection
Simplified and adapted from ORION's vision encoder design

Architecture:
    6-view images → Backbone → Position Encoding → Query-based Fusion → Scene Tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PositionEmbeddingSine(nn.Module):
    """
    Sinusoidal position encoding for 2D positions
    
    Note: 强制使用 FP32 计算，因为 temperature 的大指数在 FP16 下会溢出
    """
    def __init__(self, num_pos_feats=128, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, h, w):
        """
        Args:
            h, w: Height and width of feature map
        Returns:
            pos: (h, w, num_pos_feats*2) - always in FP32
        """
        # 全部使用 FP32 计算，避免大指数溢出
        y_embed = torch.arange(h, dtype=torch.float32).unsqueeze(1).repeat(1, w)
        x_embed = torch.arange(w, dtype=torch.float32).unsqueeze(0).repeat(h, 1)
        
        y_embed = y_embed / h  # Normalize to [0, 1]
        x_embed = x_embed / w
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        
        pos = torch.cat([pos_y, pos_x], dim=-1)  # (h, w, num_pos_feats*2)
        return pos


class PositionEmbedding3D(nn.Module):
    """
    优化后的 3D 位置编码 (参考 PETR/BEVFormer/ORION)
    
    目的：
    ======
    告诉模型每个特征图元素在真实世界（自车坐标系）中的位置。
    
    由于深度未知，使用多个深度假设来表示"可能的位置"。
    
    改进点：
    ========
    1. 使用正弦位置编码（Sinusoidal PE）代替直接展平
       - 天然保留位置信息的连续性和周期性
       - 业界标准做法（PETR/BEVFormer/DETR 都使用）
    
    2. 分层处理：先对每个深度独立编码，再聚合
       - 保留深度结构信息
       - 避免不同深度的信息混淆
    
    3. 更合理的 MLP 维度设计
       - 避免维度剧变导致的信息瓶颈
    
    流程：
    ======
    像素(u,v) → K^-1 → 射线方向 → ×32深度 → 32个3D点
                                              ↓
                            正弦编码 (每个点 → 192维)
                                              ↓
                            每深度 MLP (192 → 64)
                                              ↓
                            聚合 MLP (64×32=2048 → 256)
                                              ↓
                            3D 位置编码 (256维)
    """
    def __init__(
        self, 
        embed_dims=256, 
        depth_num=32,           # 深度假设数量（从16增加到32，更密集的深度采样）
        depth_start=1.0,        # 最小深度 (米)
        depth_max=60.0,         # 最大深度 (米)
        use_lid=True,           # 使用 LID 分布（近密远疏）
        pc_range=None,          # BEV 范围
        num_pos_feats=64,       # 每个坐标轴的正弦编码维度
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.depth_num = depth_num
        self.depth_start = depth_start
        self.depth_max = depth_max
        self.use_lid = use_lid
        self.num_pos_feats = num_pos_feats
        
        # BEV 范围：与 MapConfig 保持一致！
        # 格式：[x_min, y_min, z_min, x_max, y_max, z_max]
        # MapTR 使用：[-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        # - x: 左右范围 [-15, 15] = 30m
        # - y: 前后范围 [-30, 30] = 60m  
        # - z: 高度范围 [-2, 2] = 4m
        if pc_range is None:
            pc_range = [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0]
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float32))
        
        # ========== LID 深度分布（近处密集，远处稀疏） ==========
        if use_lid:
            # d_i = d_start + (d_max - d_start) * (i/n)^2
            indices = torch.arange(depth_num, dtype=torch.float32) / max(depth_num - 1, 1)
            depth_bins = depth_start + (depth_max - depth_start) * (indices ** 2)
        else:
            depth_bins = torch.linspace(depth_start, depth_max, depth_num)
        self.register_buffer('depth_bins', depth_bins)
        
        # ========== 简化的分层 MLP 设计 ==========
        # 
        # 设计原则：
        # 1. 深度假设增加到 32 个，采样更密集
        # 2. per_depth_mlp 简化为 1 层，直接压缩
        # 3. aggregation_mlp 渐进式压缩，保持信息流通
        #
        # 正弦编码后每个 3D 点的维度: 3 * num_pos_feats = 192
        sin_embed_dim = 3 * num_pos_feats  # 192
        
        # 第一阶段：对每个深度点独立编码 (192 → 64)
        # 简化为单层，压缩比 3x
        self.per_depth_mlp = nn.Sequential(
            nn.Linear(sin_embed_dim, 64),    # 192 → 64
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
        )
        
        # 第二阶段：聚合所有深度信息 (64 × 32 = 2048 → 256)
        # 渐进式压缩: 2048 → 1024 → 512 → 256
        # 每步压缩 2x
        aggregated_dim = 64 * depth_num  # 2048
        self.aggregation_mlp = nn.Sequential(
            nn.Linear(aggregated_dim, 1024),  # 2048 → 1024 (2x)
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),             # 1024 → 512 (2x)
            nn.LayerNorm(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dims),       # 512 → 256 (2x)
            nn.LayerNorm(embed_dims),
        )
        
    def _sinusoidal_encoding(self, coords, temperature=10000):
        """
        正弦位置编码（参考 DETR/PETR）
        
        将 3D 坐标 (x, y, z) 编码为高维向量
        
        优势：
        - 平滑性：相近的坐标得到相近的编码
        - 唯一性：不同坐标得到不同的编码
        - 无需学习：纯数学变换，训练更稳定
        
        Args:
            coords: (..., 3) 归一化后的 3D 坐标 [-1, 1]
            
        Returns:
            (..., num_pos_feats * 3) 正弦编码
            
        Note:
            调用者 (forward) 使用 @custom_fwd 确保输入是 FP32
        """
        scale = 2 * 3.14159265359  # 2π
        coords = coords * scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=coords.device)
        dim_t = temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        # 对每个坐标轴分别编码
        pos_x = coords[..., 0, None] / dim_t  # (..., num_pos_feats)
        pos_y = coords[..., 1, None] / dim_t
        pos_z = coords[..., 2, None] / dim_t
        
        # sin/cos 交替
        pos_x = torch.stack([pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_y = torch.stack([pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()], dim=-1).flatten(-2)
        pos_z = torch.stack([pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()], dim=-1).flatten(-2)
        
        # 拼接: (..., num_pos_feats * 3)
        return torch.cat([pos_x, pos_y, pos_z], dim=-1)
        
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(self, h, w, cam_intrinsic, cam_extrinsic, img_shape=(448, 800)):
        """
        生成 3D 位置编码
        
        Args:
            h, w: 特征图尺寸
            cam_intrinsic: (3, 3) 相机内参矩阵
            cam_extrinsic: (4, 4) 相机外参矩阵 (cam2ego)
            img_shape: 原始图像尺寸 (H, W)
            
        Returns:
            pos_3d: (h, w, embed_dims) 3D 位置编码
            
        Note:
            使用 @custom_fwd(cast_inputs=torch.float32) 强制 FP32 计算！
            原因：
            1. torch.inverse() 在 FP16 下数值极不稳定 → NaN
            2. 正弦编码中的大指数 (10000^x) 在 FP16 下会溢出
            3. 这是导致训练 NaN 的根本原因！
        """
        device = cam_intrinsic.device
        
        # 输入已被 @custom_fwd 自动转为 FP32
        
        # ========== Step 1: 生成像素网格 ==========
        # 特征图坐标 → 原图像素坐标
        scale_h = img_shape[0] / h
        scale_w = img_shape[1] / w
        
        y_coords = (torch.arange(h, device=device, dtype=torch.float32) + 0.5) * scale_h
        x_coords = (torch.arange(w, device=device, dtype=torch.float32) + 0.5) * scale_w
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing='ij')
        
        # 齐次像素坐标 (h, w, 3)
        pixels = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1)
        
        # ========== Step 2: 反投影到射线方向 ==========
        # K^-1 @ pixel → 归一化相机坐标（射线方向）
        regularization = 1e-6 * torch.eye(3, device=device, dtype=torch.float32)
        K_inv = torch.inverse(cam_intrinsic + regularization)
        ray_dirs = torch.einsum('ij,hwj->hwi', K_inv, pixels)  # (h, w, 3)
        
        # ========== Step 3: 多深度假设 ==========
        # 射线方向 × 深度 → 相机坐标系 3D 点
        depths = self.depth_bins.to(device).float()
        points_cam = ray_dirs.unsqueeze(-1) * depths  # (h, w, 3, depth_num)
        
        # ========== Step 4: 变换到自车坐标系 ==========
        R = cam_extrinsic[:3, :3]
        t = cam_extrinsic[:3, 3]
        points_ego = torch.einsum('ij,hwjd->hwid', R, points_cam) + t.view(1, 1, 3, 1)
        # (h, w, 3, depth_num)
        
        # 转置为 (h, w, depth_num, 3) 方便后续处理
        points_ego = points_ego.permute(0, 1, 3, 2)  # (h, w, 32, 3)
        
        # ========== Step 5: 归一化到 [-1, 1] ==========
        # pc_range 格式：[x_min, y_min, z_min, x_max, y_max, z_max]
        # 与 MapConfig 保持一致！
        pc_range = self.pc_range.to(device).float()
        # 分别对 x, y, z 归一化
        # 索引：x_min=[0], y_min=[1], z_min=[2], x_max=[3], y_max=[4], z_max=[5]
        points_norm = points_ego.clone()
        points_norm[..., 0] = (points_ego[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0]) * 2 - 1  # x: [0] ~ [3]
        points_norm[..., 1] = (points_ego[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1]) * 2 - 1  # y: [1] ~ [4]
        points_norm[..., 2] = (points_ego[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2]) * 2 - 1  # z: [2] ~ [5]
        points_norm = points_norm.clamp(-2.0, 2.0)  # 避免极端值
        
        # ========== Step 6: 正弦位置编码 ==========
        # 对每个 3D 点做正弦编码（FP32）
        sin_embed = self._sinusoidal_encoding(points_norm)  # (h, w, 32, 192)
        
        # ========== Step 7: 分层 MLP 编码 ==========
        # 第一层：每个深度独立编码
        per_depth_feat = self.per_depth_mlp(sin_embed)  # (h, w, 32, 64)
        
        # 第二层：聚合所有深度
        aggregated = per_depth_feat.flatten(-2)  # (h, w, 2048)  (64 × 32 = 2048)
        pos_3d = self.aggregation_mlp(aggregated)  # (h, w, 256)
        
        # 注意：输出保持 FP32，调用者可按需转换
        return pos_3d


class QFormer(nn.Module):
    """
    Q-Former: Multi-view feature aggregation using learnable queries
    
    Simplified from ORION's detection + map head design.
    
    Enhanced Features:
        1. 3D Position Encoding:
            - 32 depth hypotheses (更密集的深度采样)
            - LID distribution (denser near, sparser far)
            - Hierarchical MLP encoder
        
        2. 混合 Query 设计 (参考 ORION):
            - 768 Query = 576 约束 (75%) + 192 自由 (25%)
            - 约束 Query: content + camera + spatial (从特定区域提取特征)
            - 自由 Query: 只有 content (全局场景理解，类似 ORION Extra Queries)
        
        3. 优化方差占比 - 方案 B (约束 Query):
            - Content: 65% (绝对主导，学习语义提取策略)
            - Camera:  20% (辅助，确保 6 视角覆盖)
            - Spatial: 15% (轻微引导，避免完全重叠)
    """
    def __init__(
        self,
        img_backbone,
        img_neck=None,
        embed_dims=256,
        num_queries=768,
        num_decoder_layers=6,
        num_heads=8,
        ffn_dims=2048,
        dropout=0.1,
        llm_hidden_size=4096,
        num_cams=6,             # 相机数量
        # ========== 3D位置编码配置 ==========
        depth_num=32,           # 深度假设数量（32个，更密集的深度采样）
        depth_start=1.0,        # 最小深度 (米)
        depth_max=60.0,         # 最大深度 (米)
        use_lid=True,           # 使用LID分布 (方案B)
        pc_range=None,          # BEV范围，格式 [x_min, y_min, z_min, x_max, y_max, z_max]
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_queries = num_queries
        self.num_cams = num_cams
        
        # ========== Image Encoder ==========
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        
        # ========== Position Encoding ==========
        # 2D position encoding (fallback when no camera params)
        self.position_encoding_2d = PositionEmbeddingSine(num_pos_feats=embed_dims//2)
        
        # 3D position encoding (uses camera intrinsics/extrinsics)
        # Enhanced with: more depth bins (A), LID distribution (B), stronger MLP (C)
        self.position_encoding_3d = PositionEmbedding3D(
            embed_dims=embed_dims,
            depth_num=depth_num,        # 32个深度假设
            depth_start=depth_start,
            depth_max=depth_max,
            use_lid=use_lid,            # 方案B: LID分布
            pc_range=pc_range,          # BEV范围归一化
        )
        
        # Flag to control which position encoding to use
        self.use_3d_pos_encoding = True
        
        # ========== 相机编码 (图像特征和 Query 共享！) ==========
        # 关键设计：图像特征和 Query 使用同一个相机编码
        # 这样可以保证：同相机的 Query-特征 点积更大
        #
        # 原理：
        #   - 图像特征: feat + camera_embed[cam_id]
        #   - Query:    content + camera_embed[cam_id] + spatial
        #   - 点积时: camera_embed[i] · camera_embed[j]
        #     - 当 i=j 时: ||camera_embed[i]||² > 0  ← 自己和自己点积最大
        #     - 当 i≠j 时: ≈ 0（随机向量近似正交）
        #
        self.camera_embed = nn.Embedding(num_cams, embed_dims)
        
        # ========== 混合 Query 设计 (约束 Query + 自由 Query) ==========
        #
        # 设计思想（参考 ORION）:
        #   - 约束 Query (576个): 有相机/空间编码，用于从特定区域提取精确特征
        #   - 自由 Query (192个): 只有内容编码，用于全局场景理解
        #
        # 为什么增加自由 Query 比例 (25%)?
        #   - Scene Tokens 送入 LLM，LLM 需要的是语义信息
        #   - 自由 Query 没有位置先验，更擅长捕获全局/抽象信息
        #   - 类似 ORION 的 Extra Queries，用于整体场景理解
        #
        # Query 组成:
        #   约束 Query = 内容 + 相机编码 + 空间编码
        #   自由 Query = 内容 (无位置先验，类似 ORION Extra Queries)
        #
        # 方差占比设计 (针对约束 Query) - 方案 B:
        #   - 内容 (content):  65% ← 绝对主导，学习语义提取策略
        #   - 相机 (camera):   20% ← 辅助，确保 6 视角覆盖
        #   - 空间 (spatial):  15% ← 轻微引导，避免完全重叠
        #
        # Query 分配:
        #   768 = 576 (约束, 75%) + 192 (自由, 25%)
        #   576 约束 Query ÷ 6 相机 = 96 Query/相机
        #   - FRONT:       Q0-95    (96个)
        #   - FRONT_RIGHT: Q96-191  (96个)
        #   - FRONT_LEFT:  Q192-287 (96个)
        #   - BACK:        Q288-383 (96个)
        #   - BACK_LEFT:   Q384-479 (96个)
        #   - BACK_RIGHT:  Q480-575 (96个)
        #   - FREE:        Q576-767 (192个，无相机/空间编码)
        
        # 约束 Query 数量
        self.num_constrained_queries = 576
        self.num_free_queries = num_queries - self.num_constrained_queries  # 192
        
        # 计算每个相机分配的约束 Query 数量
        queries_per_cam = self.num_constrained_queries // num_cams  # 576 // 6 = 96
        remaining = self.num_constrained_queries % num_cams          # 576 % 6 = 0
        
        # 分配：576 = 96 × 6，正好整除，每相机 96 个
        self.queries_per_cam_list = [queries_per_cam] * num_cams
        for i in range(remaining):
            self.queries_per_cam_list[i] += 1
        # [96, 96, 96, 96, 96, 96]
        
        max_queries_per_cam = max(self.queries_per_cam_list)  # 96
        
        # 1. Query 内容：所有 768 个 Query 都有内容编码
        self.query_content = nn.Embedding(num_queries, embed_dims)
        
        # 2. Query 相机编码：直接使用 self.camera_embed（与图像特征共享！）
        #    只有前 576 个约束 Query 使用
        
        # 3. Query 空间编码：只有约束 Query 使用
        self.spatial_query_embed = nn.Embedding(max_queries_per_cam, embed_dims)
        
        # 4. 构建相机分配索引（只针对约束 Query）
        camera_assignment = []
        spatial_idx = []
        for cam_id, n_queries in enumerate(self.queries_per_cam_list):
            camera_assignment.extend([cam_id] * n_queries)
            spatial_idx.extend(list(range(n_queries)))
        
        self.register_buffer('camera_assignment', torch.tensor(camera_assignment, dtype=torch.long))
        self.register_buffer('spatial_idx', torch.tensor(spatial_idx, dtype=torch.long))
        
        # ========== 初始化（方案 B：65% content, 20% camera, 15% spatial）==========
        #
        # 设计目标：
        #   - 内容绝对主导，学习"提取什么语义"
        #   - 相机编码提供 ~5% 的 attention 差异（确保覆盖即可）
        #   - 空间编码仅提供轻微引导
        #
        # Query 内容：正态分布，绝对主导 (65%)
        # std=0.10 使得向量方差 ≈ 256 × 0.10² ≈ 2.56
        nn.init.normal_(self.query_content.weight, mean=0.0, std=0.10)
        
        # 相机编码：正交初始化，确保 6 个相机完全可区分 (20%)
        # 正交向量保证：同相机点积大，不同相机点积=0
        # 缩放后向量范数 ≈ 0.5 × sqrt(256/6) ≈ 3.3，向量范数² ≈ 10.8
        # 对应方差 ≈ 0.8，占比 20%
        nn.init.orthogonal_(self.camera_embed.weight)
        self.camera_embed.weight.data *= 0.5
        
        # 空间编码：正态分布，轻微引导 (15%)
        # std=0.048 使得向量方差 ≈ 256 × 0.048² ≈ 0.59
        nn.init.normal_(self.spatial_query_embed.weight, mean=0.0, std=0.048)
        
        # ========== Transformer Decoder ==========
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dims,
            nhead=num_heads,
            dim_feedforward=ffn_dims,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        
        # ========== Projector to LLM dimension ==========
        self.projector = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),
            nn.GELU(),
            nn.Linear(embed_dims * 2, llm_hidden_size),
        )
    
    def get_queries(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        构建混合 Query (约束 Query + 自由 Query)
        
        设计思想（参考 ORION）:
        - 约束 Query (576个, 75%): content + camera + spatial，用于从特定区域提取特征
        - 自由 Query (192个, 25%): 只有 content，用于全局场景理解（类似 ORION Extra Queries）
        
        关键设计：约束 Query 和图像特征共享同一个 camera_embed！
        
        这保证了：
        - 同相机的 Query-特征 点积更大
        - 不同相机的 Query-特征 点积较小（正交 = 0）
        
        方差占比（约束 Query）- 方案 B:
        - content: 65% (绝对主导)
        - camera:  20% (辅助)
        - spatial: 15% (轻微引导)
        
        维度变化：
            约束 Query:
                content[:576]:       [576, 256]
                camera_embed:        [576, 256]  (通过 camera_assignment 查表)
                spatial_query_embed: [576, 256]  (通过 spatial_idx 查表)
                → constrained:       [576, 256]
            
            自由 Query:
                content[576:]:       [192, 256]  (只有内容，无位置先验)
                → free:              [192, 256]
            
            合并:                    [768, 256]
            扩展 batch:              [B, 768, 256]
        
        Args:
            batch_size: B
            device: 设备
            dtype: 数据类型
            
        Returns:
            queries: [B, num_queries, embed_dims]
        """
        # ========== 约束 Query (576个, 75%) ==========
        # 这些 Query 有相机/空间编码，会优先关注对应相机区域
        content_constrained = self.query_content.weight[:self.num_constrained_queries]  # [576, 256]
        
        # 相机编码（使用共享的 camera_embed！）
        cam_emb = self.camera_embed(self.camera_assignment)  # [576, 256]
        
        # 空间编码
        spa_emb = self.spatial_query_embed(self.spatial_idx)  # [576, 256]
        
        # 组合（相加）
        constrained_queries = content_constrained + cam_emb + spa_emb  # [576, 256]
        
        # ========== 自由 Query (192个, 25%) ==========
        # 这些 Query 只有内容编码，用于全局场景理解（类似 ORION Extra Queries）
        # 没有相机/空间约束，可以自由探索所有视觉特征
        free_queries = self.query_content.weight[self.num_constrained_queries:]  # [192, 256]
        
        # ========== 合并 ==========
        queries = torch.cat([constrained_queries, free_queries], dim=0)  # [768, 256]
        
        # 扩展 batch 维度
        queries = queries.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 768, 256]
        queries = queries.to(device).to(dtype)
        
        return queries
        
    def extract_img_feat(self, imgs):
        """
        Extract features from multi-view images
        
        Args:
            imgs: (B, N, 3, H, W) where N=6 cameras
            
        Returns:
            img_feats: (B*N, C, h, w) single tensor
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B * N, C, H, W)
        
        # Backbone
        if self.img_neck is not None:
            x = self.img_backbone(imgs)
            if isinstance(x, (list, tuple)):
                x = x[0]  # Take first level
            img_feats = self.img_neck(x)
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[0]
        else:
            img_feats = self.img_backbone(imgs)
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[-1]  # Take last level
        
        return img_feats  # (B*N, C, h, w)
    
    def add_position_encoding(self, img_feats, batch_size, num_cams, 
                               cam_intrinsics=None, cam_extrinsics=None, img_shape=(448, 800)):
        """
        Add position encoding to image features.
        Uses 3D position encoding if camera params are provided, otherwise falls back to 2D.
        
        Args:
            img_feats: (B*N, C, h, w)
            batch_size: B
            num_cams: N (should be 6)
            cam_intrinsics: (B, N, 3, 3) camera intrinsic matrices (optional)
            cam_extrinsics: (B, N, 4, 4) camera extrinsic matrices (optional)
            img_shape: (H, W) original image shape
            
        Returns:
            img_feats_pos: (B, N*h*w, C)
        """
        BN, C, h, w = img_feats.shape
        device = img_feats.device
        dtype = img_feats.dtype
        
        # Decide which position encoding to use
        use_3d = (self.use_3d_pos_encoding and 
                  cam_intrinsics is not None and 
                  cam_extrinsics is not None)
        
        if use_3d:
            # ========== 3D Position Encoding ==========
            # Generate position encoding for each camera using camera params
            pos_embeds = []
            for b in range(batch_size):
                for n in range(num_cams):
                    # Get camera parameters for this view
                    K = cam_intrinsics[b, n]  # (3, 3)
                    E = cam_extrinsics[b, n]  # (4, 4)
                    
                    # Generate 3D position encoding
                    pos_3d = self.position_encoding_3d(h, w, K, E, img_shape)  # (h, w, C)
                    pos_3d = pos_3d.permute(2, 0, 1)  # (C, h, w)
                    pos_embeds.append(pos_3d)
            
            pos_embed = torch.stack(pos_embeds, dim=0)  # (B*N, C, h, w)
        else:
            # ========== 2D Position Encoding (fallback) ==========
            pos_embed = self.position_encoding_2d(h, w)  # (h, w, C)
            pos_embed = pos_embed.to(device).to(dtype)
            pos_embed = pos_embed.reshape(1, h, w, C).permute(0, 3, 1, 2)  # (1, C, h, w)
            pos_embed = pos_embed.expand(BN, -1, -1, -1)
        
        # Camera ID embedding (always added)
        cam_ids = torch.arange(num_cams, device=device)
        cam_ids = cam_ids.repeat(batch_size)  # (B*N,)
        cam_embed = self.camera_embed(cam_ids)  # (B*N, C)
        cam_embed = cam_embed[:, :, None, None].expand(-1, -1, h, w)  # (B*N, C, h, w)
        
        # Add position encoding
        img_feats = img_feats + pos_embed + cam_embed
        
        # Flatten spatial dimensions
        img_feats = img_feats.flatten(2).permute(0, 2, 1)  # (B*N, h*w, C)
        img_feats = img_feats.reshape(batch_size, num_cams * h * w, C)  # (B, N*h*w, C)
        
        return img_feats
    
    def forward(self, imgs, cam_intrinsics=None, cam_extrinsics=None, **kwargs):
        """
        Forward pass of Q-Former
        
        Args:
            imgs: (B, N, 3, H, W) - N=6 camera views
            cam_intrinsics: (B, N, 3, 3) - camera intrinsic matrices
                If provided, enables 3D position encoding for better spatial understanding
            cam_extrinsics: (B, N, 4, 4) - camera extrinsic matrices (cam2ego)
                If provided, enables 3D position encoding for better spatial understanding
            
        Returns:
            scene_tokens: (B, num_queries, llm_hidden_size) - ready for LLM
            
        维度变化：
            imgs:           [B, 6, 3, 448, 800]
                ↓ extract_img_feat
            img_feats:      [B*6, 256, 14, 25]
                ↓ add_position_encoding
            memory:         [B, 2100, 256]      (2100 = 6 × 14 × 25)
                
            queries:        [B, 768, 256]       (混合 Query: 576约束 + 192自由)
                            - 约束 Q0-575:  content + camera + spatial
                            - 自由 Q576-767: 只有 content (全局场景)
                ↓ decoder (Cross-Attention)
            scene_features: [B, 768, 256]
                ↓ projector
            scene_tokens:   [B, 768, 4096]      → 送入 LLM
        """
        batch_size, num_cams, _, H, W = imgs.shape
        device = imgs.device
        dtype = imgs.dtype
        
        # 1. Extract image features
        img_feats = self.extract_img_feat(imgs)  # (B*N, C, h, w)
        
        # 2. Add position encoding (uses 3D if camera params provided)
        memory = self.add_position_encoding(
            img_feats, 
            batch_size, 
            num_cams,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
            img_shape=(H, W)
        )  # (B, N*h*w, C) = (B, 2100, 256)
        
        # 3. Prepare Mixed Queries (约束 Query + 自由 Query)
        # - 约束 Query (576个): content + camera + spatial
        # - 自由 Query (192个): 只有 content (全局场景理解)
        queries = self.get_queries(batch_size, device, memory.dtype)  # (B, 768, 256)
        
        # 4. Decoder: queries attend to multi-view features
        # - 约束 Query 会更关注对应相机的特征（相机编码引导）
        # - 自由 Query 可以自由探索所有特征（全局信息提取）
        #
        # 【关键修复】使用 autocast(enabled=False) 彻底禁用 autocast
        # 原因：仅使用 .float() 无效！autocast 会覆盖 TransformerDecoder
        # 内部 F.linear/matmul 操作，导致 Attention 仍以 FP16 执行
        # FP16 Attention 在随机初始化时极易 score 溢出 → NaN
        original_dtype = queries.dtype
        
        with torch.cuda.amp.autocast(enabled=False):
            queries_fp32 = queries.float()
            memory_fp32 = memory.float()
            
            scene_features = self.decoder(
                tgt=queries_fp32,
                memory=memory_fp32,
            )  # (B, num_queries, C) = (B, 768, 256)
            
            # 5. Project to LLM dimension (也在 FP32 下执行)
            scene_tokens = self.projector(scene_features)  # (B, num_queries, llm_hidden_size) = (B, 768, 4096)
        
        # Clamp 到合理范围，防止后续 FP16 溢出
        scene_tokens = scene_tokens.clamp(-1e4, 1e4)
        
        # 转回原始 dtype
        scene_tokens = scene_tokens.to(original_dtype)
        
        return scene_tokens


def build_qformer(config):
    """
    Build Q-Former from config
    
    Args:
        config: dict with keys:
            - img_backbone: backbone config
            - img_neck: neck config (optional)
            - embed_dims: feature dimension (default 256)
            - num_queries: number of scene queries (default 768)
            - num_decoder_layers: transformer layers (default 6)
            - llm_hidden_size: LLM hidden size (default 4096)
            
            # 3D Position Encoding Config (Enhanced)
            - depth_num: number of depth hypotheses (default 32)
            - depth_start: minimum depth in meters (default 1.0)
            - depth_max: maximum depth in meters (default 60.0)
            - use_lid: use LID depth distribution (default True)
            - pc_range: BEV range [x_min, y_min, z_min, x_max, y_max, z_max]
                        (default [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0])
    
    Returns:
        QFormer instance
    """
    from torchvision.models import resnet50
    
    # Build backbone (default ResNet-50)
    img_backbone_cfg = config.get('img_backbone')
    embed_dims = config.get('embed_dims', 256)
    
    if img_backbone_cfg is None or img_backbone_cfg == 'resnet50':
        # Build ResNet-50
        backbone = resnet50(pretrained=True)
        # Remove avgpool and fc
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        
        # Add neck to reduce ResNet50's 2048 dims to embed_dims
        neck = nn.Sequential(
            nn.Conv2d(2048, embed_dims, kernel_size=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )
    elif isinstance(img_backbone_cfg, str):
        # Handle other backbone names
        raise NotImplementedError(f"Backbone '{img_backbone_cfg}' not implemented")
    else:
        # Assume it's already a nn.Module
        backbone = img_backbone_cfg
        neck = config.get('img_neck', None)
    
    qformer = QFormer(
        img_backbone=backbone,
        img_neck=neck,
        embed_dims=embed_dims,
        num_queries=config.get('num_queries', 768),
        num_decoder_layers=config.get('num_decoder_layers', 6),
        num_heads=config.get('num_heads', 8),
        ffn_dims=config.get('ffn_dims', 2048),
        dropout=config.get('dropout', 0.1),
        llm_hidden_size=config.get('llm_hidden_size', 4096),
        num_cams=config.get('num_cams', 6),             # 相机数量
        # 3D Position Encoding Config (Enhanced - ABC方案)
        depth_num=config.get('depth_num', 32),          # 32个深度假设
        depth_start=config.get('depth_start', 1.0),
        depth_max=config.get('depth_max', 60.0),
        use_lid=config.get('use_lid', True),            # 方案B: LID分布
        pc_range=config.get('pc_range', None),          # BEV范围
    )
    
    return qformer


if __name__ == "__main__":
    # Test Q-Former
    print("Testing Q-Former with Enhanced Features...")
    print("=" * 60)
    print("  1. 3D Position Encoding (32深度假设 + LID分布)")
    print("  2. 混合 Query 设计 (576约束 + 192自由)")
    print("  3. 优化方差占比 - 方案B (Content 65% + Camera 20% + Spatial 15%)")
    print("=" * 60)
    
    config = {
        'embed_dims': 256,
        'num_queries': 768,
        'num_decoder_layers': 6,
        'llm_hidden_size': 4096,
        'num_cams': 6,          # 相机数量
        # Enhanced 3D Position Encoding (ABC方案)
        'depth_num': 32,        # 32个深度假设
        'depth_start': 1.0,
        'depth_max': 60.0,
        'use_lid': True,        # 方案B: LID深度分布
        'pc_range': [-15.0, -30.0, -2.0, 15.0, 30.0, 2.0],  # BEV范围 [x_min, y_min, z_min, x_max, y_max, z_max]，与 MapTR/map_config 一致
    }
    
    print("\nConfig:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    qformer = build_qformer(config)
    
    # Print 3D position encoding info
    print("\n" + "=" * 60)
    print("3D Position Encoding Info:")
    print(f"  depth_num: {qformer.position_encoding_3d.depth_num}")
    print(f"  depth_bins: {qformer.position_encoding_3d.depth_bins.tolist()[:5]}... (first 5)")
    print(f"  use_lid: {qformer.position_encoding_3d.use_lid}")
    
    # Print Mixed Query info (约束 Query + 自由 Query)
    print("\n" + "=" * 60)
    print("混合 Query 设计 (约束 Query + 自由 Query):")
    print(f"  总 Query 数量: {qformer.num_queries}")
    print(f"  约束 Query: {qformer.num_constrained_queries} 个 (有相机/空间编码)")
    print(f"  自由 Query: {qformer.num_free_queries} 个 (只有内容，无位置先验)")
    print(f"  相机数量: {qformer.num_cams}")
    print(f"  每相机约束 Query: {qformer.queries_per_cam_list}")
    print(f"  camera_assignment shape: {qformer.camera_assignment.shape}")
    print(f"  spatial_idx shape: {qformer.spatial_idx.shape}")
    
    print("\n  方差占比 (约束 Query) - 方案 B:")
    print(f"    Content: 65% (绝对主导)")
    print(f"    Camera:  20% (辅助)")
    print(f"    Spatial: 15% (轻微引导)")
    
    # Show camera assignment distribution
    print("\n  Query 分配:")
    cam_names = ['FRONT', 'FRONT_RIGHT', 'FRONT_LEFT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    start_idx = 0
    for cam_id, (cam_name, n_queries) in enumerate(zip(cam_names, qformer.queries_per_cam_list)):
        end_idx = start_idx + n_queries
        print(f"    {cam_name}: Q{start_idx}-{end_idx-1} ({n_queries}个约束)")
        start_idx = end_idx
    print(f"    FREE: Q{start_idx}-{qformer.num_queries-1} ({qformer.num_free_queries}个自由)")
    
    # Test input (H=448 divisible by 32, W=800 divisible by 32)
    imgs = torch.randn(2, 6, 3, 448, 800)  # (B=2, N=6, C=3, H=448, W=800)
    
    # Test with camera parameters (3D position encoding)
    print("\n" + "=" * 60)
    print("Test 1: Forward without camera params (uses 2D fallback)")
    scene_tokens = qformer(imgs)
    print(f"  Input shape: {imgs.shape}")
    print(f"  Output shape: {scene_tokens.shape}")
    assert scene_tokens.shape == (2, 768, 4096), "Output shape mismatch!"
    print("  ✓ Passed!")
    
    # Test with camera parameters
    print("\n" + "=" * 60)
    print("Test 2: Forward with camera params (uses 3D position encoding)")
    cam_intrinsics = torch.randn(2, 6, 3, 3)  # Mock intrinsics
    cam_extrinsics = torch.eye(4).unsqueeze(0).unsqueeze(0).expand(2, 6, -1, -1).clone()  # Identity extrinsics
    
    scene_tokens_3d = qformer(imgs, cam_intrinsics=cam_intrinsics, cam_extrinsics=cam_extrinsics)
    print(f"  Input shape: {imgs.shape}")
    print(f"  cam_intrinsics: {cam_intrinsics.shape}")
    print(f"  cam_extrinsics: {cam_extrinsics.shape}")
    print(f"  Output shape: {scene_tokens_3d.shape}")
    assert scene_tokens_3d.shape == (2, 768, 4096), "Output shape mismatch!"
    print("  ✓ Passed!")
    
    # Test get_queries method
    print("\n" + "=" * 60)
    print("Test 3: get_queries method (混合 Query)")
    queries = qformer.get_queries(batch_size=2, device=torch.device('cpu'), dtype=torch.float32)
    print(f"  Output shape: {queries.shape}")
    print(f"  约束 Query [0:576]: 有相机/空间编码")
    print(f"  自由 Query [576:768]: 只有内容编码")
    assert queries.shape == (2, 768, 256), "Query shape mismatch!"
    print("  ✓ Passed!")
    
    # Parameter count
    print("\n" + "=" * 60)
    print("Parameter Statistics:")
    total_params = sum(p.numel() for p in qformer.parameters())
    pos_3d_params = sum(p.numel() for p in qformer.position_encoding_3d.parameters())
    query_params = (
        qformer.query_content.weight.numel() +
        qformer.spatial_query_embed.weight.numel()
    )
    print(f"  Total Q-Former params: {total_params:,}")
    print(f"  3D Position Encoding params: {pos_3d_params:,}")
    print(f"  混合 Query params: {query_params:,}")
    print(f"    - query_content (768个): {qformer.query_content.weight.numel():,}")
    print(f"    - camera_embed (6个，共享): {qformer.camera_embed.weight.numel():,}")
    print(f"    - spatial_query_embed ({max(qformer.queries_per_cam_list)}个): {qformer.spatial_query_embed.weight.numel():,}")
    
    print("\n" + "=" * 60)
    print("✅ All Q-Former tests passed!")
    print("=" * 60)

