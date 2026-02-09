"""
Complete Pipeline Test: Raw Data → Detection Head Output
验证从原始数据到检测头输出的完整流程
"""
import sys
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple

print("="*80)
print("Complete Pipeline Validation")
print("="*80)

# ============================================================================
# Step 1: 原始数据 → 预处理
# ============================================================================
print("\n" + "="*80)
print("Step 1: 原始数据预处理 (nuScenes → 模型输入)")
print("="*80)

# 模拟nuScenes原始数据
print("\n[原始数据]")
print("  图像数据:")
raw_images = np.random.randint(0, 255, (6, 900, 1600, 3), dtype=np.uint8)
print(f"    6个相机图像: {raw_images.shape} (6, H, W, 3)")
print(f"    类型: uint8, 范围: [0, 255]")

print("\n  标注数据:")
raw_annotations = {
    'divider': [
        {'points': np.random.rand(10, 2) * 100, 'type': 'road_divider'},
        {'points': np.random.rand(8, 2) * 100, 'type': 'lane_divider'},
    ],
    'crossing': [
        {'points': np.random.rand(4, 2) * 100, 'type': 'ped_crossing'},
    ]
}
print(f"    road_divider: {len([x for x in raw_annotations['divider'] if x['type']=='road_divider'])} 条")
print(f"    lane_divider: {len([x for x in raw_annotations['divider'] if x['type']=='lane_divider'])} 条")
print(f"    ped_crossing: {len(raw_annotations['crossing'])} 个")

# 预处理：归一化、resize
print("\n[预处理]")
batch_size = 2
images = torch.from_numpy(raw_images).float() / 255.0  # [6, H, W, 3]
images = images.permute(0, 3, 1, 2)  # [6, 3, H, W]
images = torch.nn.functional.interpolate(images, size=(224, 224))
images = images.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1)  # [B, 6, 3, 224, 224]

print(f"  图像张量: {images.shape}")
print(f"    格式: [B, num_cameras, C, H, W]")
print(f"    类型: {images.dtype}")
print(f"    范围: [{images.min():.3f}, {images.max():.3f}]")

# 标注处理：转换为模型格式
gt_instances = []
for sample in range(batch_size):
    sample_gts = {
        'labels': torch.tensor([0, 1, 2, 1, 0]),  # 5个实例
        'lines': torch.randn(5, 20, 2)  # 5个实例，每个20个点
    }
    gt_instances.append(sample_gts)

print(f"\n  GT标注: {len(gt_instances)} samples")
print(f"    每个sample包含:")
print(f"      labels: [num_instances] - 类别标签")
print(f"      lines: [num_instances, 20, 2] - 归一化坐标")


# ============================================================================
# Step 2: Q-Former处理
# ============================================================================
print("\n" + "="*80)
print("Step 2: Q-Former处理 (图像 → scene tokens)")
print("="*80)

class MockQFormer(nn.Module):
    """模拟Q-Former"""
    def __init__(self):
        super().__init__()
        self.vision_encoder = nn.Conv2d(3, 768, 1)  # 模拟vision encoder
        self.projection = nn.Linear(768, 4096)  # 投影到4096维
        
    def forward(self, images):
        B, N_cam, C, H, W = images.shape
        # Merge cameras
        x = images.reshape(B * N_cam, C, H, W)
        # Vision encoder
        x = self.vision_encoder(x)  # [B*6, 768, H, W]
        x = x.mean(dim=[2, 3])  # Global pooling → [B*6, 768]
        # Project to 4096 dim
        x = self.projection(x)  # [B*6, 4096]
        # Reshape and expand to 512 tokens
        x = x.reshape(B, N_cam, 4096)  # [B, 6, 4096]
        # Simulate Q-Former: 6 camera features → 512 scene tokens
        # 简单重复扩展（实际Q-Former会做更复杂的cross-attention）
        x = x.unsqueeze(2).repeat(1, 1, 85, 1)  # [B, 6, 85, 4096]
        x = x.reshape(B, -1, 4096)[:, :512, :]  # [B, 512, 4096]
        return x

qformer = MockQFormer()

print("\n[Q-Former输入]")
print(f"  images: {images.shape}")
print(f"    [B, 6_cameras, 3, 224, 224]")

with torch.no_grad():
    scene_tokens = qformer(images)

print("\n[Q-Former输出]")
print(f"  scene_tokens: {scene_tokens.shape}")
print(f"    [B, 512_tokens, 4096_dim]")
print(f"    类型: {scene_tokens.dtype}")
print(f"    统计: mean={scene_tokens.mean():.3f}, std={scene_tokens.std():.3f}")


# ============================================================================
# Step 3: LLM处理
# ============================================================================
print("\n" + "="*80)
print("Step 3: LLM处理 (scene_tokens + text + queries → features)")
print("="*80)

class MockLLM(nn.Module):
    """模拟LLM with map queries"""
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(32000, 4096)
        
        # Map queries
        self.instance_queries = nn.Parameter(torch.randn(50, 4096))
        self.point_queries = nn.Parameter(torch.randn(50, 20, 4096))
        
        # Transformer layers (simplified)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=4096, nhead=32, batch_first=True),
            num_layers=2  # 简化版，实际是32层
        )
    
    def forward(self, text_ids, scene_tokens):
        B = scene_tokens.shape[0]
        
        # Text embedding
        text_embeds = self.embed_tokens(text_ids)  # [B, seq_len, 4096]
        
        # Expand queries
        instance_queries = self.instance_queries.unsqueeze(0).expand(B, -1, -1)  # [B, 50, 4096]
        point_queries = self.point_queries.unsqueeze(0).expand(B, -1, -1, -1)  # [B, 50, 20, 4096]
        point_queries = point_queries.reshape(B, 1000, 4096)  # [B, 1000, 4096]
        
        # Concatenate all inputs
        all_tokens = torch.cat([
            text_embeds,      # [B, text_len, 4096]
            scene_tokens,     # [B, 512, 4096]
            instance_queries, # [B, 50, 4096]
            point_queries     # [B, 1000, 4096]
        ], dim=1)
        
        # Transformer processing
        output = self.transformer(all_tokens)
        
        # Extract features
        text_len = text_embeds.shape[1]
        scene_len = scene_tokens.shape[1]  # 实际长度
        instance_start = text_len + scene_len
        point_start = instance_start + 50
        
        instance_features = output[:, instance_start:instance_start+50, :]  # [B, 50, 4096]
        point_features = output[:, point_start:, :]  # [B, remaining, 4096]
        point_features = point_features[:, :1000, :]  # Take first 1000
        point_features = point_features.reshape(B, 50, 20, 4096)  # [B, 50, 20, 4096]
        
        return instance_features, point_features

llm = MockLLM()

print("\n[LLM输入]")
# Text input
text = "Detect map elements including road dividers, lane dividers and pedestrian crossings."
text_ids = torch.randint(0, 32000, (batch_size, 20))
print(f"  text_ids: {text_ids.shape}")
print(f"    [B, text_length]")
print(f"  scene_tokens: {scene_tokens.shape}")
print(f"    [B, 512, 4096]")
print(f"  instance_queries: [50, 4096] (learnable)")
print(f"  point_queries: [50, 20, 4096] (learnable)")

print("\n[LLM拼接序列]")
print(f"  text_embeds:      [B, 20, 4096]")
print(f"  scene_tokens:     [B, 512, 4096]")
print(f"  instance_queries: [B, 50, 4096]")
print(f"  point_queries:    [B, 1000, 4096] (50*20 flatten)")
print(f"  → total_sequence: [B, 1582, 4096]")

with torch.no_grad():
    instance_features, point_features = llm(text_ids, scene_tokens)

print("\n[LLM输出 - 提取特征]")
print(f"  instance_features: {instance_features.shape}")
print(f"    [B, 50_instances, 4096_dim]")
print(f"    统计: mean={instance_features.mean():.3f}, std={instance_features.std():.3f}")
print(f"\n  point_features: {point_features.shape}")
print(f"    [B, 50_instances, 20_points, 4096_dim]")
print(f"    统计: mean={point_features.mean():.3f}, std={point_features.std():.3f}")


# ============================================================================
# Step 4: 检测头处理
# ============================================================================
print("\n" + "="*80)
print("Step 4: 检测头 (features → predictions)")
print("="*80)

# Import detection head
sys.path.insert(0, '/home/cly/auto/llava_test/LLaVA')
from llava.model.map_detection_head import MapDetectionHead

detection_head = MapDetectionHead(
    hidden_size=4096,
    num_classes=3,
    intermediate_dim=1024,
    bottleneck_dim=256,
    dropout=0.1
)

print("\n[检测头配置]")
print(f"  输入维度: 4096")
print(f"  类别数: 3 (road_divider, lane_divider, ped_crossing)")
print(f"  中间维度: 1024 → 256")
print(f"  参数量: {detection_head.get_num_params():,}")

print("\n[检测头输入]")
print(f"  instance_features: {instance_features.shape}")
print(f"  point_features: {point_features.shape}")

with torch.no_grad():
    pred_classes, pred_points = detection_head(instance_features, point_features)

print("\n[检测头输出]")
print(f"  pred_classes: {pred_classes.shape}")
print(f"    [B, 50, 3] - 类别logits")
print(f"    统计: mean={pred_classes.mean():.3f}, std={pred_classes.std():.3f}")
print(f"\n  pred_points: {pred_points.shape}")
print(f"    [B, 50, 20, 2] - 坐标")
print(f"    范围: [{pred_points.min():.3f}, {pred_points.max():.3f}] (Tanh归一化)")


# ============================================================================
# Step 5: 后处理
# ============================================================================
print("\n" + "="*80)
print("Step 5: 后处理 (predictions → 最终结果)")
print("="*80)

print("\n[后处理操作]")

# 1. 分类后处理
pred_probs = torch.softmax(pred_classes, dim=-1)  # [B, 50, 3]
pred_labels = pred_probs.argmax(dim=-1)  # [B, 50]
pred_scores = pred_probs.max(dim=-1)[0]  # [B, 50]

print(f"  1. 分类:")
print(f"     pred_probs: {pred_probs.shape} - 概率分布")
print(f"     pred_labels: {pred_labels.shape} - 类别ID")
print(f"     pred_scores: {pred_scores.shape} - 置信度")

# 2. 坐标反归一化
img_width, img_height = 1600, 900
coords_x = (pred_points[..., 0] + 1) / 2 * img_width  # [-1,1] → [0, img_width]
coords_y = (pred_points[..., 1] + 1) / 2 * img_height  # [-1,1] → [0, img_height]
pred_coords_real = torch.stack([coords_x, coords_y], dim=-1)

print(f"\n  2. 坐标反归一化:")
print(f"     pred_points (归一化): {pred_points.shape}, 范围 [-1, 1]")
print(f"     pred_coords_real: {pred_coords_real.shape}, 范围 [0, {img_width}] x [0, {img_height}]")

# 3. 置信度过滤
threshold = 0.5
valid_mask = pred_scores > threshold  # [B, 50]
num_valid = valid_mask.sum(dim=1)

print(f"\n  3. 置信度过滤 (threshold={threshold}):")
print(f"     valid_mask: {valid_mask.shape}")
print(f"     每个样本有效预测数: {num_valid.tolist()}")

# 4. 提取有效结果
print(f"\n  4. 提取有效结果 (示例：第一个样本):")
sample_idx = 0
valid_indices = torch.where(valid_mask[sample_idx])[0]
if len(valid_indices) > 0:
    valid_labels_sample = pred_labels[sample_idx, valid_indices]
    valid_scores_sample = pred_scores[sample_idx, valid_indices]
    valid_coords_sample = pred_coords_real[sample_idx, valid_indices]
    
    print(f"     有效实例数: {len(valid_indices)}")
    print(f"     类别: {valid_labels_sample[:5].tolist()}")
    print(f"     置信度: {valid_scores_sample[:5].tolist()}")
    print(f"     坐标形状: {valid_coords_sample.shape} [num_valid, 20, 2]")
else:
    print(f"     无有效预测（所有置信度 < {threshold}）")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "="*80)
print("完整流程总结")
print("="*80)

print("\n数据流转换:")
print("  1. 原始数据:")
print("     └─ images: (6, 900, 1600, 3) uint8")
print("     └─ annotations: dict with points")
print()
print("  2. 预处理:")
print("     └─ images: [B, 6, 3, 224, 224] float32")
print()
print("  3. Q-Former:")
print("     └─ scene_tokens: [B, 512, 4096] float32")
print()
print("  4. LLM:")
print("     ├─ 输入序列: [B, 1582, 4096]")
print("     │   ├─ text_embeds: [B, 20, 4096]")
print("     │   ├─ scene_tokens: [B, 512, 4096]")
print("     │   ├─ instance_queries: [B, 50, 4096]")
print("     │   └─ point_queries: [B, 1000, 4096]")
print("     └─ 输出特征:")
print("         ├─ instance_features: [B, 50, 4096]")
print("         └─ point_features: [B, 50, 20, 4096]")
print()
print("  5. 检测头:")
print("     ├─ pred_classes: [B, 50, 3]")
print("     └─ pred_points: [B, 50, 20, 2]")
print()
print("  6. 后处理:")
print("     ├─ pred_labels: [B, 50]")
print("     ├─ pred_scores: [B, 50]")
print("     ├─ pred_coords_real: [B, 50, 20, 2]")
print("     └─ valid results after filtering")

print("\n" + "="*80)
print("✅ 完整流程验证成功！")
print("="*80)

print("\n关键维度:")
print(f"  Batch size: {batch_size}")
print(f"  Num cameras: 6")
print(f"  Num scene tokens: 512")
print(f"  Num instances: 50")
print(f"  Num points per instance: 20")
print(f"  Hidden size: 4096")
print(f"  Num classes: 3")
print(f"  总参数量: {detection_head.get_num_params():,}")

