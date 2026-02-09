#!/usr/bin/env python3
"""
Q-Former 简化诊断工具：分析 512 Scene Tokens 能否有效代表 6 张图像

只加载 Q-Former 部分，不需要完整的 LLM

使用方法：
    python diagnose_qformer_simple.py --checkpoint <checkpoint_path> --num-samples 30
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
from llava.model.qformer import MultiViewQFormer, build_qformer


def extract_features_and_tokens(qformer, images, device):
    """
    提取 Q-Former 的中间特征用于分析
    
    Returns:
        image_features: [B, num_patches, C] 原始图像 patch 特征
        scene_tokens: [B, 512, D] Scene Tokens
        h, w: 特征图的空间尺寸
    """
    qformer.eval()
    with torch.no_grad():
        B, num_cams, C, H, W = images.shape
        images_flat = images.view(B * num_cams, C, H, W)
        
        # 1. 通过 backbone 提取特征
        backbone_features = qformer.backbone(images_flat)
        
        # 取最后一层特征
        if isinstance(backbone_features, dict):
            feat = backbone_features.get('layer4', list(backbone_features.values())[-1])
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
        feat_reshaped = feat.view(B, num_cams, C_feat, h, w)
        image_features = feat_reshaped.permute(0, 1, 3, 4, 2).reshape(B, total_patches, C_feat)
        
        # 2. 获取 Scene Tokens（完整 Q-Former 前向传播）
        scene_tokens = qformer(images)  # [B, 512, D]
        
        return image_features, scene_tokens, h, w


def compute_coverage_metrics(image_features, scene_tokens, device):
    """
    计算 Scene Tokens 对图像特征的覆盖程度
    
    核心指标：
    1. 每个 Scene Token 与图像特征的最大相似度
    2. 每个图像 patch 被 Scene Tokens 覆盖的程度
    """
    B, num_patches, feat_dim = image_features.shape
    _, num_tokens, token_dim = scene_tokens.shape
    
    # 投影到相同维度进行比较
    if feat_dim != token_dim:
        # 使用随机初始化的投影（仅用于相对比较）
        proj = nn.Linear(feat_dim, token_dim).to(device)
        nn.init.orthogonal_(proj.weight)
        with torch.no_grad():
            image_features_proj = proj(image_features)
    else:
        image_features_proj = image_features
    
    # 归一化
    img_norm = F.normalize(image_features_proj, dim=-1)  # [B, num_patches, D]
    tok_norm = F.normalize(scene_tokens, dim=-1)  # [B, 512, D]
    
    # 计算相似度矩阵 [B, 512, num_patches]
    sim_matrix = torch.bmm(tok_norm, img_norm.transpose(1, 2))
    
    # 指标计算
    # 1. 每个 token 的最大相似度（它最匹配的 patch）
    max_sim_per_token = sim_matrix.max(dim=2)[0]  # [B, 512]
    
    # 2. 每个 patch 被关注的最大程度（任意 token 与它的最大相似度）
    max_sim_per_patch = sim_matrix.max(dim=1)[0]  # [B, num_patches]
    
    # 3. 统计
    metrics = {
        'avg_token_relevance': max_sim_per_token.mean().item(),
        'min_token_relevance': max_sim_per_token.min().item(),
        'std_token_relevance': max_sim_per_token.std().item(),
        'avg_patch_coverage': max_sim_per_patch.mean().item(),
        'min_patch_coverage': max_sim_per_patch.min().item(),
        'std_patch_coverage': max_sim_per_patch.std().item(),
        'poorly_covered_ratio': (max_sim_per_patch < 0.3).float().mean().item(),
        'well_covered_ratio': (max_sim_per_patch > 0.5).float().mean().item(),
    }
    
    return metrics, sim_matrix


def analyze_token_camera_distribution(sim_matrix, h, w, num_cams=6):
    """
    分析每个 Scene Token 主要关注哪个相机
    """
    sim = sim_matrix[0].cpu().numpy()  # [512, num_patches]
    num_patches_per_cam = h * w
    
    # 每个 token 最关注的 patch
    best_patch_idx = sim.argmax(axis=1)  # [512]
    
    # 转换为相机索引
    best_cam_idx = best_patch_idx // num_patches_per_cam  # [512]
    
    # 统计
    cam_counts = np.bincount(best_cam_idx, minlength=num_cams)
    
    return cam_counts


def visualize_coverage(sim_matrix, h, w, num_cams=6, save_path=None):
    """
    可视化每个相机区域被 Scene Tokens 覆盖的程度
    """
    sim = sim_matrix[0].cpu().numpy()  # [512, num_patches]
    
    # 每个 patch 被覆盖的最大程度
    patch_coverage = sim.max(axis=0)  # [num_patches]
    
    # Reshape 为 [6, h, w]
    num_patches_per_cam = h * w
    coverage_maps = patch_coverage.reshape(num_cams, h, w)
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    
    for i, (ax, name) in enumerate(zip(axes.flat, cam_names)):
        im = ax.imshow(coverage_maps[i], cmap='RdYlGn', vmin=0, vmax=1)
        avg_cov = coverage_maps[i].mean()
        min_cov = coverage_maps[i].min()
        ax.set_title(f'{name}\nAvg: {avg_cov:.3f}, Min: {min_cov:.3f}', fontsize=11)
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.suptitle('Scene Token Coverage per Camera Region\n'
                 '(Green=Well covered, Red=Poorly covered by 512 tokens)', 
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.close()
    
    return coverage_maps


def visualize_token_distribution(cam_counts, save_path=None):
    """
    可视化 512 个 tokens 在 6 个相机间的分布
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
    
    bars = ax.bar(cam_names, cam_counts, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加数值标签
    for bar, count in zip(bars, cam_counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, 
                f'{count}\n({count/512*100:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Number of Tokens', fontsize=12)
    ax.set_title('Distribution of 512 Scene Tokens Across 6 Cameras', fontsize=13, fontweight='bold')
    ax.set_ylim(0, max(cam_counts) * 1.25)
    
    # 理想均匀分布线
    ideal = 512 / 6
    ax.axhline(y=ideal, color='red', linestyle='--', linewidth=2, label=f'Ideal uniform: {ideal:.0f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.close()


def visualize_similarity_histogram(sim_matrix, save_path=None):
    """
    可视化相似度分布
    """
    sim = sim_matrix[0].cpu().numpy()  # [512, num_patches]
    
    # 每个 patch 的最大被覆盖程度
    patch_coverage = sim.max(axis=0)
    
    # 每个 token 的最大相关性
    token_relevance = sim.max(axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Patch coverage distribution
    axes[0].hist(patch_coverage, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    axes[0].axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Poor threshold (0.3)')
    axes[0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Good threshold (0.5)')
    axes[0].set_xlabel('Max Similarity with any Token', fontsize=11)
    axes[0].set_ylabel('Number of Patches', fontsize=11)
    axes[0].set_title('Patch Coverage Distribution\n(How well each patch is represented)', fontsize=12, fontweight='bold')
    axes[0].legend()
    
    # Token relevance distribution
    axes[1].hist(token_relevance, bins=50, color='coral', edgecolor='white', alpha=0.8)
    axes[1].axvline(x=0.3, color='red', linestyle='--', linewidth=2, label='Poor threshold (0.3)')
    axes[1].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Good threshold (0.5)')
    axes[1].set_xlabel('Max Similarity with any Patch', fontsize=11)
    axes[1].set_ylabel('Number of Tokens', fontsize=11)
    axes[1].set_title('Token Relevance Distribution\n(How much each token captures image info)', fontsize=12, fontweight='bold')
    axes[1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {save_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Q-Former Simple Diagnostic Tool')
    parser.add_argument('--checkpoint', type=str, 
                        default='outputs/6x4090_fresh_20260125_143156/best_model_ema.pth',
                        help='Path to checkpoint')
    parser.add_argument('--num-samples', type=int, default=30,
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
    
    # 加载 Q-Former
    print("\n" + "="*60)
    print("Loading Q-Former...")
    print("="*60)
    
    qformer = build_qformer({})  # 使用默认配置
    
    # 加载 checkpoint 中的 Q-Former 权重
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint.get('ema_state_dict', checkpoint))
        
        # 提取 Q-Former 相关的权重
        qformer_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('qformer.'):
                new_key = key[len('qformer.'):]
                qformer_state_dict[new_key] = value
        
        if qformer_state_dict:
            missing, unexpected = qformer.load_state_dict(qformer_state_dict, strict=False)
            print(f"  Loaded {len(qformer_state_dict)} Q-Former weights")
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
        else:
            print("  Warning: No Q-Former weights found in checkpoint")
    else:
        print(f"Warning: Checkpoint not found, using random weights")
    
    qformer = qformer.to(device)
    qformer.eval()
    
    # 加载数据
    print("\n" + "="*60)
    print("Loading dataset...")
    print("="*60)
    
    # 简单的 tokenizer mock（只需要 pad_token_id）
    class MockTokenizer:
        pad_token_id = 0
    
    val_dataset = MapDetectionDataset(
        dataroot=args.dataroot,
        version='v1.0-trainval',
        split='val',
        gt_cache_path=os.path.join(args.dataroot, 'gt_cache'),
        tokenizer=MockTokenizer(),
        use_augmentation=False,
    )
    
    print(f"  Dataset size: {len(val_dataset)}")
    
    # 分析
    print("\n" + "="*60)
    print(f"Analyzing {args.num_samples} samples...")
    print("="*60)
    
    all_metrics = []
    sample_sim_matrix = None
    sample_h, sample_w = None, None
    all_cam_counts = []
    
    for i in tqdm(range(min(args.num_samples, len(val_dataset))), desc="Processing"):
        sample = val_dataset[i]
        images = sample['images'].unsqueeze(0).to(device)  # [1, 6, 3, H, W]
        
        try:
            image_features, scene_tokens, h, w = extract_features_and_tokens(
                qformer, images, device
            )
            
            metrics, sim_matrix = compute_coverage_metrics(
                image_features, scene_tokens, device
            )
            
            cam_counts = analyze_token_camera_distribution(sim_matrix, h, w)
            
            all_metrics.append(metrics)
            all_cam_counts.append(cam_counts)
            
            # 保存第一个样本用于可视化
            if sample_sim_matrix is None:
                sample_sim_matrix = sim_matrix
                sample_h, sample_w = h, w
                
        except Exception as e:
            print(f"Error processing sample {i}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_metrics:
        print("Error: No samples processed successfully!")
        return
    
    # 汇总结果
    print("\n" + "="*60)
    print("DIAGNOSIS RESULTS")
    print("="*60)
    
    # 计算平均指标
    avg_metrics = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics]
        avg_metrics[key] = {
            'mean': np.mean(values),
            'std': np.std(values),
        }
    
    print("\n📊 Coverage Metrics:")
    print("-" * 50)
    
    print(f"\n1. Token Relevance (每个 Token 与图像的相关性):")
    print(f"   平均值: {avg_metrics['avg_token_relevance']['mean']:.4f} ± {avg_metrics['avg_token_relevance']['std']:.4f}")
    print(f"   最小值: {avg_metrics['min_token_relevance']['mean']:.4f}")
    
    print(f"\n2. Patch Coverage (每个图像区域被覆盖的程度):")
    print(f"   平均值: {avg_metrics['avg_patch_coverage']['mean']:.4f} ± {avg_metrics['avg_patch_coverage']['std']:.4f}")
    print(f"   最小值: {avg_metrics['min_patch_coverage']['mean']:.4f}")
    
    print(f"\n3. Coverage Quality:")
    print(f"   覆盖良好 (>0.5): {avg_metrics['well_covered_ratio']['mean']*100:.1f}%")
    print(f"   覆盖较差 (<0.3): {avg_metrics['poorly_covered_ratio']['mean']*100:.1f}%")
    
    # 相机分布
    avg_cam_counts = np.mean(all_cam_counts, axis=0)
    print(f"\n4. Token Distribution per Camera:")
    cam_names = ['FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'BACK', 'BACK_LEFT', 'BACK_RIGHT']
    for name, count in zip(cam_names, avg_cam_counts):
        print(f"   {name}: {count:.1f} tokens ({count/512*100:.1f}%)")
    
    # 生成可视化
    print("\n" + "="*60)
    print("Generating visualizations...")
    print("="*60)
    
    if sample_sim_matrix is not None:
        visualize_coverage(
            sample_sim_matrix, sample_h, sample_w,
            save_path=output_dir / 'coverage_heatmap.png'
        )
        
        visualize_token_distribution(
            avg_cam_counts.astype(int),
            save_path=output_dir / 'token_distribution.png'
        )
        
        visualize_similarity_histogram(
            sample_sim_matrix,
            save_path=output_dir / 'similarity_histogram.png'
        )
    
    # 诊断结论
    print("\n" + "="*60)
    print("🔍 DIAGNOSIS CONCLUSION")
    print("="*60)
    
    avg_coverage = avg_metrics['avg_patch_coverage']['mean']
    poorly_covered = avg_metrics['poorly_covered_ratio']['mean']
    
    if avg_coverage > 0.6 and poorly_covered < 0.15:
        status = "✅ GOOD"
        conclusion = "512 Scene Tokens 能较好地代表 6 张图像信息"
        suggestion = "Q-Former 不是主要瓶颈，建议优化 Map Decoder 或添加迭代精修"
    elif avg_coverage > 0.4 and poorly_covered < 0.35:
        status = "⚠️ MODERATE"
        conclusion = "512 Scene Tokens 有一定信息损失"
        suggestion = "建议：1) 增加 token 数量到 1024 或 2) 在 Q-Former 中使用 Deformable Attention"
    else:
        status = "❌ POOR"
        conclusion = "512 Scene Tokens 信息严重不足，这可能是性能瓶颈"
        suggestion = "强烈建议：1) 增加 token 数量 2) 使用 Deformable Attention 3) 添加重建损失"
    
    print(f"\nStatus: {status}")
    print(f"Conclusion: {conclusion}")
    print(f"Suggestion: {suggestion}")
    
    # 额外分析
    print("\n" + "-"*50)
    print("📈 与 MapTR 对比分析:")
    print("-"*50)
    print(f"  MapTR BEV 特征数量: 200×200 = 40,000")
    print(f"  你的 Scene Tokens:  512")
    print(f"  压缩比: {40000/512:.1f}x")
    print(f"  ")
    print(f"  如果覆盖率 < 60%，说明压缩损失了大量信息")
    print(f"  当前覆盖率: {avg_coverage*100:.1f}%")
    
    print(f"\n📁 Visualizations saved to: {output_dir}/")
    
    # 保存报告
    report_path = output_dir / 'diagnosis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write("Q-Former Diagnosis Report\n")
        f.write("="*60 + "\n\n")
        f.write(f"Samples analyzed: {len(all_metrics)}\n")
        f.write(f"Average patch coverage: {avg_coverage:.4f}\n")
        f.write(f"Poorly covered ratio: {poorly_covered*100:.2f}%\n")
        f.write(f"Well covered ratio: {avg_metrics['well_covered_ratio']['mean']*100:.2f}%\n")
        f.write(f"\nStatus: {status}\n")
        f.write(f"Conclusion: {conclusion}\n")
        f.write(f"Suggestion: {suggestion}\n")
    
    print(f"\n📄 Report saved to: {report_path}")


if __name__ == '__main__':
    main()
