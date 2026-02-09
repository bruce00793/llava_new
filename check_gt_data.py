#!/usr/bin/env python3
"""
GT 数据检查脚本

用途: 快速验证 GT 数据是否正确加载，不启动完整训练

检查内容:
1. GT 缓存是否存在
2. 加载几个样本并打印 GT 内容
3. 检查类别分布
4. 检查坐标范围

使用方法:
    python check_gt_data.py
"""

import os
import sys
import pickle
import numpy as np
import torch
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    print("=" * 70)
    print("GT 数据检查脚本")
    print("=" * 70)
    
    # 配置
    dataroot = "/home/cly/auto/llava_test/LLaVA/data/nuscenes"
    version = "v1.0-mini"
    gt_cache_dir = f"{dataroot}/gt_cache"
    
    print(f"\n[配置]")
    print(f"  数据集: {dataroot}")
    print(f"  版本: {version}")
    print(f"  GT 缓存: {gt_cache_dir}")
    
    # 检查 GT 缓存
    print(f"\n[检查 GT 缓存]")
    train_cache = os.path.join(gt_cache_dir, f"gt_cache_{version.replace('.', '')}_{version.split('-')[1]}_train.pkl")
    val_cache = os.path.join(gt_cache_dir, f"gt_cache_{version.replace('.', '')}_{version.split('-')[1]}_val.pkl")
    
    # 尝试不同的缓存文件名
    possible_caches = [
        os.path.join(gt_cache_dir, "gt_cache_v10-mini_mini_train.pkl"),
        os.path.join(gt_cache_dir, "gt_cache_v10-mini_mini_val.pkl"),
        os.path.join(gt_cache_dir, "gt_cache_v10-trainval_trainval_train.pkl"),
        os.path.join(gt_cache_dir, "gt_cache_v10-trainval_trainval_val.pkl"),
        train_cache,
        val_cache,
    ]
    
    found_caches = []
    for cache_path in possible_caches:
        if os.path.exists(cache_path):
            found_caches.append(cache_path)
            print(f"  ✅ 找到: {os.path.basename(cache_path)}")
    
    if not found_caches:
        # 列出目录内容
        if os.path.exists(gt_cache_dir):
            print(f"\n  目录内容 ({gt_cache_dir}):")
            for f in os.listdir(gt_cache_dir):
                print(f"    - {f}")
        else:
            print(f"  ❌ GT 缓存目录不存在: {gt_cache_dir}")
        return
    
    # 加载并检查 GT 数据
    print(f"\n[加载 GT 数据]")
    
    for cache_path in found_caches[:2]:  # 只检查前两个
        print(f"\n{'='*50}")
        print(f"缓存文件: {os.path.basename(cache_path)}")
        print(f"{'='*50}")
        
        with open(cache_path, 'rb') as f:
            gt_cache = pickle.load(f)
        
        print(f"  样本数量: {len(gt_cache)}")
        
        # 统计
        all_labels = []
        all_num_points = []
        all_x_coords = []
        all_y_coords = []
        
        # 检查前 5 个样本
        print(f"\n  [前 5 个样本详情]")
        sample_tokens = list(gt_cache.keys())[:5]
        
        for i, token in enumerate(sample_tokens):
            gt = gt_cache[token]
            print(f"\n  样本 {i+1}: {token[:16]}...")
            
            if 'labels' in gt and 'points' in gt:
                labels = gt['labels']
                points = gt['points']
                
                print(f"    实例数: {len(labels)}")
                print(f"    类别: {labels.tolist() if hasattr(labels, 'tolist') else labels}")
                
                if len(points) > 0:
                    print(f"    点数组形状: {np.array(points).shape}")
                    
                    # 打印第一个实例的点
                    if len(points) > 0:
                        first_pts = np.array(points[0])
                        print(f"    实例 0 的点 (前 5 个):")
                        for j, pt in enumerate(first_pts[:5]):
                            print(f"      点 {j}: ({pt[0]:.4f}, {pt[1]:.4f})")
                        if len(first_pts) > 5:
                            print(f"      ... 共 {len(first_pts)} 个点")
                
                # 收集统计数据
                all_labels.extend(labels if hasattr(labels, '__iter__') else [labels])
                for pts in points:
                    pts = np.array(pts)
                    all_num_points.append(len(pts))
                    all_x_coords.extend(pts[:, 0].tolist())
                    all_y_coords.extend(pts[:, 1].tolist())
            else:
                print(f"    GT 格式: {list(gt.keys())}")
        
        # 统计全部数据
        print(f"\n  [全部数据统计]")
        
        for token, gt in gt_cache.items():
            if 'labels' in gt and 'points' in gt:
                labels = gt['labels']
                points = gt['points']
                all_labels.extend(labels if hasattr(labels, '__iter__') else [labels])
                for pts in points:
                    pts = np.array(pts)
                    all_num_points.append(len(pts))
                    if len(pts) > 0:
                        all_x_coords.extend(pts[:, 0].tolist())
                        all_y_coords.extend(pts[:, 1].tolist())
        
        # 类别分布
        label_counts = Counter(all_labels)
        class_names = ['divider', 'ped_crossing', 'boundary']
        print(f"\n  类别分布:")
        for label, count in sorted(label_counts.items()):
            name = class_names[label] if label < len(class_names) else f"unknown_{label}"
            print(f"    {name} (类别 {label}): {count}")
        
        # 坐标范围
        if all_x_coords:
            print(f"\n  坐标范围 (归一化后):")
            print(f"    X: [{min(all_x_coords):.4f}, {max(all_x_coords):.4f}]")
            print(f"    Y: [{min(all_y_coords):.4f}, {max(all_y_coords):.4f}]")
            
            # 检查是否在 [-1, 1] 范围内
            x_in_range = all([x >= -1.0 and x <= 1.0 for x in all_x_coords])
            y_in_range = all([y >= -1.0 and y <= 1.0 for y in all_y_coords])
            
            if x_in_range and y_in_range:
                print(f"    ✅ 坐标在 [-1, 1] 范围内")
            else:
                print(f"    ⚠️  部分坐标超出 [-1, 1] 范围")
        
        # 点数统计
        if all_num_points:
            print(f"\n  每个实例的点数:")
            print(f"    最小: {min(all_num_points)}")
            print(f"    最大: {max(all_num_points)}")
            print(f"    平均: {np.mean(all_num_points):.1f}")
            
            if all([n == 20 for n in all_num_points]):
                print(f"    ✅ 所有实例都有 20 个点")
            else:
                print(f"    ⚠️  点数不一致: {Counter(all_num_points)}")
    
    print("\n" + "=" * 70)
    print("✅ GT 数据检查完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
