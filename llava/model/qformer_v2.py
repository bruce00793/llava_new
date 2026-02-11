"""
Q-Former V2: 三阶段双流架构（位置-视觉分离 + 压缩瓶颈）

设计思想：
    V1 将 3D 位置编码直接"加"到图像特征上，位置和视觉信息混合在一起。
    V2 让它们分别走独立的编码路径，通过交叉注意力精细融合，
    最后用可学习的汇总查询压缩到 768 个标记（与 V1 输出一致）。

三个阶段：
    第一阶段（各自提炼）：图像和位置分别做自注意力，各自建立内部结构
    第二阶段（位置问图像）：位置编码向图像特征提问"这里有什么"，再全局自注意力统一
    第三阶段（压缩瓶颈）：768 个学习到的查询从 2100 个融合特征中提取最相关的信息

流程图：
    6 摄像头图像 (B, 6, 3, H, W)
              │
      ┌───────┴───────────────┐
      ▼                       ▼
    ResNet-50 + Neck       3D 位置编码器
      │                       │
      ▼                       ▼
    图像特征               3D 位置编码
    (B, 2100, 256)         (B, 2100, 256)
      │                       │
    ═══ 第一阶段：各自提炼 ═══
      │                       │
    5 层自注意力            5 层自注意力
      编码器                  编码器
      │                       │
      ▼                       ▼
    精炼图像特征            精炼位置编码
    (B, 2100, 256)         (B, 2100, 256)
      │                       │
    ═══ 第二阶段：位置问图像 ═══
      │                       │
      └──── 3 层交叉注意力 ────┘
           Q=位置  K,V=图像
                  │
                  ▼
        融合特征 (B, 2100, 256)
                  │
             3 层自注意力
                  │
                  ▼
        精炼融合特征 (B, 2100, 256)
                  │
    ═══ 第三阶段：压缩瓶颈 ═══
                  │
      768 可学习汇总查询 (B, 768, 256)
                  │
          2 层交叉注意力
         Q=768查询  K,V=2100特征
                  │
                  ▼
        压缩场景标记 (B, 768, 256)
                  │
             投影器 256→4096
                  │
                  ▼
         场景标记 (B, 768, 4096)  → 送入 LLM

接口：
    输入: (B, 6, 3, 448, 800) + 可选的摄像头内外参数
    输出: (B, 768, 4096) — 与 V1 完全一致

Author: Auto-generated for Map Detection
Date: 2025-02
"""

import math
import torch
import torch.nn as nn
from typing import Optional

# 复用 V1 的基础组件（骨干网络、位置编码等）
from .qformer import PositionEmbeddingSine, PositionEmbedding3D


# ============================================================
# 基础模块
# ============================================================

class EncoderLayer(nn.Module):
    """
    标准 Transformer 编码器层（自注意力 + 前馈网络）
    使用 Pre-Norm 结构（先归一化再注意力）
    """
    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        
        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        """x: (B, N, C)"""
        # 自注意力 (Pre-Norm)
        residual = x
        x = self.norm1(x)
        x = residual + self.dropout1(self.self_attn(x, x, x)[0])
        
        # 前馈网络 (Pre-Norm)
        residual = x
        x = self.norm2(x)
        x = residual + self.ffn(x)
        
        return x


class CrossAttentionLayer(nn.Module):
    """
    交叉注意力层（Q 从一个来源，K/V 从另一个来源）
    使用 Pre-Norm 结构
    """
    def __init__(self, embed_dim=256, num_heads=8, ffn_dim=1024, dropout=0.1):
        super().__init__()
        
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        
    def forward(self, query, kv):
        """
        query: (B, Nq, C) — 查询方
        kv:    (B, Nkv, C) — 键值方
        """
        # 交叉注意力 (Pre-Norm)
        residual = query
        q = self.norm1(query)
        kv_normed = self.norm_kv(kv)
        query = residual + self.dropout1(
            self.cross_attn(q, kv_normed, kv_normed)[0]
        )
        
        # 前馈网络 (Pre-Norm)
        residual = query
        query = self.norm2(query)
        query = residual + self.ffn(query)
        
        return query


# ============================================================
# Q-Former V2 主体
# ============================================================

class QFormerV2(nn.Module):
    """
    三阶段双流 Q-Former：
    
    第一阶段：图像特征 / 3D 位置编码各自走 5 层自注意力编码器
    第二阶段：位置编码交叉注意力图像特征（3 层），再自注意力（3 层）
    第三阶段：768 个汇总查询从 2100 个融合特征中压缩提取（2 层）
    
    输入输出与 V1 完全一致：
        输入: (B, 6, 3, 448, 800) + 可选摄像头参数
        输出: (B, 768, 4096)
    """
    
    def __init__(
        self,
        img_backbone,
        img_neck=None,
        embed_dims=256,
        num_output_tokens=768,       # 输出标记数（与 V1 一致）
        # 第一阶段：各自提炼
        num_image_encoder_layers=5,
        num_position_encoder_layers=5,
        # 第二阶段：位置问图像
        num_cross_attn_layers=3,
        num_fusion_self_attn_layers=3,
        # 第三阶段：压缩瓶颈
        num_compression_layers=2,
        # 通用参数
        num_heads=8,
        ffn_dims=1024,
        dropout=0.1,
        llm_hidden_size=4096,
        num_cams=6,
        # 3D 位置编码配置
        depth_num=32,
        depth_start=1.0,
        depth_max=60.0,
        use_lid=True,
        pc_range=None,
    ):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.num_output_tokens = num_output_tokens
        self.num_cams = num_cams
        
        # ========== 图像编码器（与 V1 共享） ==========
        self.img_backbone = img_backbone
        self.img_neck = img_neck
        
        # ========== 位置编码（与 V1 共享） ==========
        self.position_encoding_2d = PositionEmbeddingSine(num_pos_feats=embed_dims // 2)
        self.position_encoding_3d = PositionEmbedding3D(
            embed_dims=embed_dims,
            depth_num=depth_num,
            depth_start=depth_start,
            depth_max=depth_max,
            use_lid=use_lid,
            pc_range=pc_range,
        )
        self.use_3d_pos_encoding = True
        
        # ========== 第一阶段：各自提炼 ==========
        # 图像特征自注意力编码器（5 层）
        self.image_encoder = nn.ModuleList([
            EncoderLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_image_encoder_layers)
        ])
        
        # 位置编码自注意力编码器（5 层）
        self.position_encoder = nn.ModuleList([
            EncoderLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_position_encoder_layers)
        ])
        
        # ========== 第二阶段：位置问图像 ==========
        # 交叉注意力：位置(Q) 问 图像(K,V)（3 层）
        self.cross_attn_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_cross_attn_layers)
        ])
        
        # 融合后自注意力（3 层）
        self.fusion_encoder = nn.ModuleList([
            EncoderLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_fusion_self_attn_layers)
        ])
        
        # ========== 第三阶段：压缩瓶颈 ==========
        # 768 个可学习汇总查询
        self.summary_queries = nn.Parameter(
            torch.randn(num_output_tokens, embed_dims) * 0.02
        )
        
        # 压缩交叉注意力：汇总查询(Q) 提取 融合特征(K,V)（2 层）
        self.compression_layers = nn.ModuleList([
            CrossAttentionLayer(embed_dims, num_heads, ffn_dims, dropout)
            for _ in range(num_compression_layers)
        ])
        
        # 输出归一化
        self.output_norm = nn.LayerNorm(embed_dims)
        
        # ========== 投影器：256 → 4096 ==========
        self.projector = nn.Sequential(
            nn.Linear(embed_dims, embed_dims * 2),   # 256 → 512
            nn.GELU(),
            nn.Linear(embed_dims * 2, llm_hidden_size),  # 512 → 4096
        )
        
        # 初始化
        self._init_weights()
        
    def _init_weights(self):
        """权重初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        
        # 汇总查询：小随机初始化
        nn.init.normal_(self.summary_queries, mean=0.0, std=0.02)
    
    def extract_img_feat(self, imgs):
        """
        提取图像特征（与 V1 完全一致）
        
        输入: (B, N, 3, H, W)
        输出: (B*N, C, h, w)
        """
        B, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B * N, C, H, W)
        
        if self.img_neck is not None:
            x = self.img_backbone(imgs)
            if isinstance(x, (list, tuple)):
                x = x[0]
            img_feats = self.img_neck(x)
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[0]
        else:
            img_feats = self.img_backbone(imgs)
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[-1]
        
        return img_feats
    
    def compute_position_encoding(self, img_feats, batch_size, num_cams,
                                   cam_intrinsics=None, cam_extrinsics=None,
                                   img_shape=(448, 800)):
        """
        计算 3D 位置编码（不加到图像特征上，单独返回）
        
        输入: img_feats (B*N, C, h, w)
        输出: position_encoding (B, N*h*w, C)
        """
        BN, C, h, w = img_feats.shape
        device = img_feats.device
        dtype = img_feats.dtype
        
        use_3d = (self.use_3d_pos_encoding and
                  cam_intrinsics is not None and
                  cam_extrinsics is not None)
        
        if use_3d:
            pos_embeds = []
            for b in range(batch_size):
                for n in range(num_cams):
                    K = cam_intrinsics[b, n]
                    E = cam_extrinsics[b, n]
                    pos_3d = self.position_encoding_3d(h, w, K, E, img_shape)
                    pos_3d = pos_3d.permute(2, 0, 1)  # (C, h, w)
                    pos_embeds.append(pos_3d)
            pos_embed = torch.stack(pos_embeds, dim=0)  # (B*N, C, h, w)
        else:
            pos_embed = self.position_encoding_2d(h, w)
            pos_embed = pos_embed.to(device).to(dtype)
            pos_embed = pos_embed.reshape(1, h, w, C).permute(0, 3, 1, 2)
            pos_embed = pos_embed.expand(BN, -1, -1, -1)
        
        # 展平并合并所有摄像头
        pos_embed = pos_embed.flatten(2).permute(0, 2, 1)  # (B*N, h*w, C)
        pos_embed = pos_embed.reshape(batch_size, num_cams * h * w, C)  # (B, 2100, C)
        
        return pos_embed
    
    def forward(self, imgs, cam_intrinsics=None, cam_extrinsics=None, **kwargs):
        """
        前向传播
        
        输入:
            imgs: (B, 6, 3, 448, 800)
            cam_intrinsics: (B, 6, 3, 3) 可选
            cam_extrinsics: (B, 6, 4, 4) 可选
            
        输出:
            scene_tokens: (B, 768, 4096) — 与 V1 输出完全一致
        """
        batch_size, num_cams, _, H, W = imgs.shape
        
        # ===== 提取图像特征和位置编码（分开！不相加）=====
        img_feats_raw = self.extract_img_feat(imgs)  # (B*N, C, h, w)
        
        # 图像特征：展平
        BN, C, h, w = img_feats_raw.shape
        img_features = img_feats_raw.flatten(2).permute(0, 2, 1)  # (B*N, h*w, C)
        img_features = img_features.reshape(batch_size, num_cams * h * w, C)  # (B, 2100, 256)
        
        # 位置编码：单独计算（不加到图像上）
        pos_encoding = self.compute_position_encoding(
            img_feats_raw, batch_size, num_cams,
            cam_intrinsics, cam_extrinsics, (H, W)
        )  # (B, 2100, 256)
        
        # ===== 全部在 FP32 下执行（防止注意力溢出）=====
        original_dtype = img_features.dtype
        
        with torch.cuda.amp.autocast(enabled=False):
            img_features = img_features.float()
            pos_encoding = pos_encoding.float()
            
            # ===== 第一阶段：各自提炼（5+5 层自注意力）=====
            for layer in self.image_encoder:
                img_features = layer(img_features)
            
            for layer in self.position_encoder:
                pos_encoding = layer(pos_encoding)
            
            # ===== 第二阶段：位置问图像（3 层交叉注意力 + 3 层自注意力）=====
            # 位置编码(Q) 向 图像特征(K,V) 提问："这个3D坐标有什么？"
            fused = pos_encoding
            for layer in self.cross_attn_layers:
                fused = layer(query=fused, kv=img_features)
            
            # 全局自注意力：让相邻位置互相交流
            for layer in self.fusion_encoder:
                fused = layer(fused)
            # fused: (B, 2100, 256)
            
            # ===== 第三阶段：压缩瓶颈（768 查询提取 2100 特征）=====
            # 扩展汇总查询到 batch 维度
            queries = self.summary_queries.unsqueeze(0).expand(batch_size, -1, -1)  # (B, 768, 256)
            queries = queries.float()
            
            # 汇总查询(Q) 从 融合特征(K,V) 中提取最相关的信息
            for layer in self.compression_layers:
                queries = layer(query=queries, kv=fused)
            
            # 输出归一化
            scene_features = self.output_norm(queries)  # (B, 768, 256)
            
            # 投影到 LLM 维度
            scene_tokens = self.projector(scene_features)  # (B, 768, 4096)
        
        # 防溢出 + 转回原 dtype
        scene_tokens = scene_tokens.clamp(-1e4, 1e4)
        scene_tokens = scene_tokens.to(original_dtype)
        
        return scene_tokens


# ============================================================
# 构建函数
# ============================================================

def build_qformer_v2(config):
    """
    构建 Q-Former V2
    
    参数与 build_qformer (V1) 完全兼容，额外参数有默认值。
    
    Args:
        config: dict，支持以下键：
            - img_backbone: 骨干网络配置（默认 'resnet50'）
            - embed_dims: 特征维度（默认 256）
            - num_output_tokens: 输出标记数（默认 768）
            - num_image_encoder_layers: 图像编码器层数（默认 5）
            - num_position_encoder_layers: 位置编码器层数（默认 5）
            - num_cross_attn_layers: 交叉注意力层数（默认 3）
            - num_fusion_self_attn_layers: 融合自注意力层数（默认 3）
            - num_compression_layers: 压缩层数（默认 2）
            - llm_hidden_size: LLM 隐藏维度（默认 4096）
            - 3D 位置编码参数（与 V1 一致）
    """
    from torchvision.models import resnet50
    
    embed_dims = config.get('embed_dims', 256)
    
    # 构建骨干网络（与 V1 完全一致）
    img_backbone_cfg = config.get('img_backbone')
    if img_backbone_cfg is None or img_backbone_cfg == 'resnet50':
        backbone = resnet50(pretrained=True)
        backbone = nn.Sequential(*list(backbone.children())[:-2])
        neck = nn.Sequential(
            nn.Conv2d(2048, embed_dims, kernel_size=1),
            nn.BatchNorm2d(embed_dims),
            nn.ReLU(inplace=True)
        )
    elif isinstance(img_backbone_cfg, str):
        raise NotImplementedError(f"Backbone '{img_backbone_cfg}' not implemented")
    else:
        backbone = img_backbone_cfg
        neck = config.get('img_neck', None)
    
    qformer = QFormerV2(
        img_backbone=backbone,
        img_neck=neck,
        embed_dims=embed_dims,
        num_output_tokens=config.get('num_output_tokens', 768),
        # 第一阶段
        num_image_encoder_layers=config.get('num_image_encoder_layers', 5),
        num_position_encoder_layers=config.get('num_position_encoder_layers', 5),
        # 第二阶段
        num_cross_attn_layers=config.get('num_cross_attn_layers', 3),
        num_fusion_self_attn_layers=config.get('num_fusion_self_attn_layers', 3),
        # 第三阶段
        num_compression_layers=config.get('num_compression_layers', 2),
        # 通用
        num_heads=config.get('num_heads', 8),
        ffn_dims=config.get('ffn_dims', 1024),
        dropout=config.get('dropout', 0.1),
        llm_hidden_size=config.get('llm_hidden_size', 4096),
        num_cams=config.get('num_cams', 6),
        # 3D 位置编码
        depth_num=config.get('depth_num', 32),
        depth_start=config.get('depth_start', 1.0),
        depth_max=config.get('depth_max', 60.0),
        use_lid=config.get('use_lid', True),
        pc_range=config.get('pc_range', None),
    )
    
    return qformer
