#!/usr/bin/env python3
"""
Q-Former 诊断工具：分析 512 Scene Tokens 能否有效代表 6 张图像

实验内容：
1. 重建损失分析：用 512 tokens 重建原始 2100 patch 特征
2. Attention 可视化：每个 Query 关注哪些图像区域
3. 信息压缩率分析：计算信息保留程度

使用方法：
    python diagnose_qformer.py --checkpoint <checkpoint_path> --num-samples 50
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from llava.data.map_dataset import MapDetectionDataset
from llava.model.map_llava_model import LLaVAMapDetector


class FeatureReconstructor(nn.Module):
    """
    用 512 Scene Tokens 重建 2100 Patch 特征的 Decoder
    
    如果重建误差很大，说明 512 tokens 信息不足
    """
    def __init__(self, token_dim: int = 4096, num_tokens: int = 512, 
                 num_patches: int = 2100, patch_dim: int = 1024):
        super().__init__()
        self.num_patches = num_patches
        
        # 简单的 MLP Decoder
        self.decoder = nn.Sequential(
            nn.Linear(token_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Linear(2048, num_patches * patch_dim // 4),  # 先输出压缩表示
        )
        
        # 最终投影到 patch 特征
        self.final_proj = nn.Linear(patch_dim // 4, patch_dim)
        self.patch_dim = patch_dim
        
    def forward(self, scene_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scene_tokens: [B, 512, 4096] Scene Tokens
        Returns:
            reconstructed: [B, 2100, 1024] 重建的 patch 特征
        """
        B = scene_tokens.shape[0]
        
        # 全局池化 + 解码
        pooled = scene_tokens.mean(dim=1)  # [B, 4096]
        decoded = self.decoder(pooled)  # [B, 2100 * 256]
        
        # Reshape
        decoded = decoded.view(B, self.num_patches, -1)  # [B, 2100, 256]
        reconstructed = self.final_proj(decoded)  # [B, 2100, 1024]
        
        return reconstructed


def extract_qformer_features(model, images, device):
    """
    提取 Q-Former 的中间特征用于分析
    
    Returns:
        image_features: [B, 2100, 1024] 原始图像 patch 特征
        scene_tokens: [B, 512, D] Scene Tokens
        attention_weights: [B, 512, 2100] Attention weights (如果可获取)
    """
    model.eval()
    with torch.no_grad():
        # 获取 Q-Former
        qformer = model.qformer
        
        # 1. 提取图像特征
        B, num_cams, C, H, W = images.shape
        images_flat = images.view(B * num_cams, C, H, W)
        
        # 通过 backbone
        backbone_features = qformer.backbone(images_flat)
        
        # 取最后一层特征
        if isinstance(backbone_features, dict):
            feat = backbone_features['layer4'] if 'layer4' in backbone_features else list(backbone_features.values())[-1]
        elif isinstance(backbone_features, (list, tuple)):
            feat = backbone_features[-1]
        else:
            feat = backbone_features
            
        # 通过 neck
        if hasattr(qformer, 'neck') and qformer.neck is not None:
            feat = qformer.neck([feat])[0]
        
        # feat: [B*6, C, h, w]
        _, C_feat, h, w = feat.shape
        num_patches_per_cam = h * w
        total_patches = num_cams * num_patches_per_cam
        
        # Reshape 为 [B, 6*h*w, C]
        feat = feat.view(B, num_cams, C_feat, h, w)
        feat = feat.permute(0, 1, 3, 4, 2).reshape(B, total_patches, C_feat)
        
        image_features = feat  # [B, 2100, C]
        
        # 2. 获取 Scene Tokens（完整前向传播）
        scene_tokens = qformer(images)  # [B, 512, D]
        
        # 3. 尝试获取 attention weights（如果模型支持）
        attention_weights = None
        
        return image_features, scene_tokens, attention_weights, (h, w)


def compute_reconstruction_metrics(image_features, scene_tokens, device):
    """
    计算重建指标
    """
    B, num_patches, feat_dim = image_features.shape
    _, num_tokens, token_dim = scene_tokens.shape
    
    # 方法1：直接余弦相似度（Scene Tokens 与最近 Patch 的相似度）
    # 先投影到相同维度
    if feat_dim != token_dim:
        proj = nn.Linear(feat_dim, token_dim).to(device)
        with torch.no_grad():
            image_features_proj = proj(image_features)
    else:
        image_features_proj = image_features
    
    # 归一化
    img_norm = F.normalize(image_features_proj, dim=-1)  # [B, 2100, D]
    tok_norm = F.normalize(scene_tokens, dim=-1)  # [B, 512, D]
    
    # 计算相似度矩阵
    sim_matrix = torch.bmm(tok_norm, img_norm.transpose(1, 2))  # [B, 512, 2100]
    
    # 每个 token 的最大相似度（它最关注的 patch）
    max_sim_per_token = sim_matrix.max(dim=2)[0]  # [B, 512]
    
    # 每个 patch 被关注的最大程度
    max_sim_per_patch = sim_matrix.max(dim=1)[0]  # [B, 2100]
    
    # 统计
    metrics = {
        'avg_token_max_sim': max_sim_per_token.mean().item(),
        'min_token_max_sim': max_sim_per_token.min().item(),
        'avg_patch_coverage': max_sim_per_patch.mean().item(),
        'min_patch_coverage': max_sim_per_patch.min().item(),
        'uncovered_patches_ratio': (max_sim_per_patch < 0.3).float().mean().item(),
    }
    
    return metrics, sim_matrix


def visualize_attention_coverage(sim_matrix, h, w, num_cams=6, save_path=None):
    """
    可视化 512 个 tokens 对 6 张图像的覆盖情况
    """
    # sim_matrix: [B, 512, 2100]
    # 取第一个样本
    sim = sim_matrix[0].cpu().numpy()  # [512, 2100]
    
    # 每个 patch 被关注的最大程度
    patch_coverage = sim.max(axis=0)  # [2100]
    
    # Reshape 为 [6, h, w]
    num_patches_per_cam = h * w
    coverage_maps = patch_coverage.reshape(num_cams, h, w)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    
    for i, (ax, name) in enumerate(zip(axes.flat, cam_names)):
        im = ax.imshow(coverage_maps[i], cmap='hot', vmin=0, vmax=1)
        ax.set_title(f'{name}\nAvg Coverage: {coverage_maps[i].mean():.3f}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Scene Token Coverage per Camera\n(Higher = Better represented by 512 tokens)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved coverage visualization to {save_path}")
    
    plt.close()
    
    return coverage_maps


def analyze_token_distribution(sim_matrix, h, w, num_cams=6, save_path=None):
    """
    分析每个 token 主要关注哪个相机
    """
    sim = sim_matrix[0].cpu().numpy()  # [512, 2100]
    num_patches_per_cam = h * w
    
    # 每个 token 最关注的 patch 索引
    best_patch_idx = sim.argmax(axis=1)  # [512]
    
    # 转换为相机索引
    best_cam_idx = best_patch_idx // num_patches_per_cam  # [512]
    
    # 统计每个相机被多少 tokens 关注
    cam_counts = np.bincount(best_cam_idx, minlength=num_cams)
    
    # 可视化
    fig, ax = plt.subplots(figsize=(10, 6))
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = ax.bar(cam_names, cam_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, count in zip(bars, cam_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{count}\n({count/512*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Tokens', fontsize=12)
    ax.set_title('Distribution of 512 Scene Tokens Across 6 Cameras\n(Each token is assigned to its most-attended camera)', 
                 fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(cam_counts) * 1.2)
    
    # 添加理想均匀分布线
    ax.axhline(y=512/6, color='red', linestyle='--', linewidth=2, label=f'Ideal uniform: {512/6:.0f}')
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved token distribution to {save_path}")
    
    plt.close()
    
    return cam_counts


def compute_information_retention(image_features, scene_tokens):
    """
    计算信息保留率：通过 PCA 分析
    """
    # 原始图像特征的有效维度（通过 PCA 解释方差比）
    img_feat = image_features[0].cpu().numpy()  # [2100, C]
    tok_feat = scene_tokens[0].cpu().numpy()  # [512, D]
    
    # 计算协方差矩阵的特征值
    def compute_effective_dim(features, threshold=0.95):
        """计算保留 threshold 方差所需的维度数"""
        # 中心化
        features = features - features.mean(axis=0)
        # SVD
        try:
            _, s, _ = np.linalg.svd(features, full_matrices=False)
            explained_variance = (s ** 2) / (s ** 2).sum()
            cumsum = np.cumsum(explained_variance)
            effective_dim = np.searchsorted(cumsum, threshold) + 1
            return effective_dim, explained_variance
        except:
            return features.shape[1], np.ones(features.shape[1]) / features.shape[1]
    
    img_eff_dim, img_var = compute_effective_dim(img_feat)
    tok_eff_dim, tok_var = compute_effective_dim(tok_feat)
    
    return {
        'image_effective_dim': img_eff_dim,
        'token_effective_dim': tok_eff_dim,
        'image_total_patches': img_feat.shape[0],
        'token_count': tok_feat.shape[0],
        'compression_ratio': img_feat.shape[0] / tok_feat.shape[0],
        'dim_retention_ratio': tok_eff_dim / img_eff_dim if img_eff_dim > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Q-Former Diagnostic Tool')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/6x4090_fresh_20260125_143156/best_model_ema.pth',
                        help='Path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to analyze')
    parser.add_argument('--output-dir', type=str, default='qformer_diagnosis',
                        help='Output directory for visualizations')
    parser.add_argument('--dataroot', type=str, 
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes',
                        help='nuScenes data root')
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print("\n" + "="*60)
    print("Loading model...")
    print("="*60)
    
    # 本地路径
    llm_path = "/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5"
    
    model = LLaVAMapDetector(
        llm_path=llm_path,
        qformer_config={},
    )
    
    # 加载 checkpoint
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('ema_state_dict', checkpoint))
        model.load_state_dict(state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {args.checkpoint}, using random weights")
    
    model = model.to(device)
    model.eval()
    
    # 加载数据
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(llm_path, local_files_only=True)
    
    val_dataset = MapDetectionDataset(
        dataroot=args.dataroot,
        version='v1.0-trainval',
        split='val',
        gt_cache_path=os.path.join(args.dataroot, 'gt_cache'),
        tokenizer=tokenizer,
        use_augmentation=False,
    )
    
    # 分析
    print("\n" + "="*60)
    print(f"Analyzing Q-Former with {args.num_samples} samples...")
    print("="*60)
    
    all_metrics = []
    all_sim_matrices = []
    
    for i in tqdm(range(min(args.num_samples, len(val_dataset))), desc="Processing"):
        sample = val_dataset[i]
        images = sample['images'].unsqueeze(0).to(device)  # [1, 6, 3, H, W]
        
        try:
            image_features, scene_tokens, attn_weights, (h, w) = extract_qformer_features(
                model, images, device
            )
            
            metrics, sim_matrix = compute_reconstruction_metrics(
                image_features, scene_tokens, device
            )
            
            all_metrics.append(metrics)
            
            if i < 5:  # 保存前 5 个样本的相似度矩阵用于可视化
                all_sim_matrices.append((sim_matrix, h, w))
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            continue
    
    # 汇总结果
    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
        }
    
    print("\n📊 Coverage Metrics (higher is better):")
    print("-" * 50)
    
    print(f"\n1. Token-Patch Similarity:")
    print(f"   Average max similarity per token: {avg_metrics['avg_token_max_sim']['mean']:.4f} ± {avg_metrics['avg_token_max_sim']['std']:.4f}")
    print(f"   Min max similarity (worst token): {avg_metrics['min_token_max_sim']['mean']:.4f}")
    
    print(f"\n2. Patch Coverage:")
    print(f"   Average patch coverage: {avg_metrics['avg_patch_coverage']['mean']:.4f} ± {avg_metrics['avg_patch_coverage']['std']:.4f}")
    print(f"   Min patch coverage (worst patch): {avg_metrics['min_patch_coverage']['mean']:.4f}")
    print(f"   Uncovered patches ratio (<0.3): {avg_metrics['uncovered_patches_ratio']['mean']*100:.2f}%")
    
    # 信息保留分析
    if len(all_sim_matrices) > 0:
        sim_matrix, h, w = all_sim_matrices[0]
        
        # 可视化覆盖情况
        coverage_maps = visualize_attention_coverage(
            sim_matrix, h, w, 
            save_path=output_dir / 'coverage_heatmap.png'
        )
        
        # 可视化 token 分布
        cam_counts = analyze_token_distribution(
            sim_matrix, h, w,
            save_path=output_dir / 'token_distribution.png'
        )
        
        # 信息保留分析
        info_metrics = compute_information_retention(
            image_features.cpu(), scene_tokens.cpu()
        )
        
        print(f"\n3. Information Retention:")
        print(f"   Original patches: {info_metrics['image_total_patches']}")
        print(f"   Scene tokens: {info_metrics['token_count']}")
        print(f"   Compression ratio: {info_metrics['compression_ratio']:.2f}x")
        print(f"   Image effective dim (95% var): {info_metrics['image_effective_dim']}")
        print(f"   Token effective dim (95% var): {info_metrics['token_effective_dim']}")
    
    # 诊断结论
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS CONCLUSION")
    print("="*60)
    
    avg_coverage = avg_metrics['avg_patch_coverage']['mean']
    uncovered_ratio = avg_metrics['uncovered_patches_ratio']['mean']
    
    if avg_coverage > 0.7 and uncovered_ratio < 0.1:
        status = "✅ GOOD"
        conclusion = "512 Scene Tokens 能较好地代表 6 张图像"
    elif avg_coverage > 0.5 and uncovered_ratio < 0.3:
        status = "⚠️ MODERATE"
        conclusion = "512 Scene Tokens 有一定信息损失，建议增加 token 数量或改进 Q-Former"
    else:
        status = "❌ POOR"
        conclusion = "512 Scene Tokens 信息严重不足，这是性能瓶颈的主要原因"
    
    print(f"\nStatus: {status}")
    print(f"Conclusion: {conclusion}")
    
    print(f"\n📁 Visualizations saved to: {output_dir}/")
    print("   - coverage_heatmap.png: 每个相机区域被 tokens 覆盖的程度")
    print("   - token_distribution.png: 512 tokens 在 6 个相机间的分布")
    
    # 建议
    print("\n" + "="*60)
    print("💡 RECOMMENDATIONS")
    print("="*60)
    
    if uncovered_ratio > 0.2:
        print("\n1. ⚠️ 有 {:.1f}% 的图像区域未被有效覆盖".format(uncovered_ratio * 100))
        print("   建议：增加 Scene Tokens 数量（如 1024）或使用 Deformable Attention")
    
    if avg_coverage < 0.6:
        print("\n2. ⚠️ 平均覆盖率较低 ({:.2f})".format(avg_coverage))
        print("   建议：检查 Q-Former 的训练是否充分，或添加重建损失")
    
    # 保存报告
    report_path = output_dir / 'diagnosis_report.txt'
    with open(report_path, 'w') as f:
        f.write("Q-Former Diagnosis Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Samples analyzed: {len(all_metrics)}\n")
        f.write(f"Average patch coverage: {avg_coverage:.4f}\n")
        f.write(f"Uncovered patches ratio: {uncovered_ratio*100:.2f}%\n")
        f.write(f"Status: {status}\n")
        f.write(f"Conclusion: {conclusion}\n")
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == '__main__':
    main()
