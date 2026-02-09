"""
将 nuScenes GT 数据转换为文字格式

用于 LLM 文本生成验证实验

输出格式:
<map>
[divider] (-0.35,-0.68)(-0.34,-0.65)(-0.33,-0.62)...(共20个点)
[boundary] (0.70,0.50)(0.71,0.53)...(共20个点)
</map>

坐标归一化（与主训练完全一致！）:
    x: [-15, 15] → [-1, 1]
    y: [-30, 30] → [-1, 1]
    
精度: 2 位小数

Author: Auto-generated for LLM Text Verification
"""

import os
import sys
import json
import pickle
import random
import argparse
from tqdm import tqdm
from typing import Dict, List, Tuple
import numpy as np

# 类别名称
CLASS_NAMES = ['divider', 'ped_crossing', 'boundary']


def normalize_coords(points: np.ndarray) -> np.ndarray:
    """
    将 BEV 坐标归一化到 [-1, 1]（与主训练完全一致！）
    
    原始范围: x: [-15, 15], y: [-30, 30]
    
    公式（与 map_dataset.py 的 _normalize_coords 一致）:
        x_norm = (x - x_min) / (x_max - x_min) * 2 - 1
        y_norm = (y - y_min) / (y_max - y_min) * 2 - 1
    """
    # PC_RANGE = [-15, -30, -2, 15, 30, 2]
    x_min, y_min = -15, -30
    x_max, y_max = 15, 30
    
    normalized = np.zeros_like(points)
    normalized[:, 0] = (points[:, 0] - x_min) / (x_max - x_min) * 2 - 1  # x: [-15, 15] → [-1, 1]
    normalized[:, 1] = (points[:, 1] - y_min) / (y_max - y_min) * 2 - 1  # y: [-30, 30] → [-1, 1]
    
    # Clip 到 [-1, 1]（与主训练一致）
    normalized = np.clip(normalized, -1, 1)
    
    return normalized


def format_instance_as_text(cls_id: int, points: np.ndarray) -> str:
    """
    将单个实例转换为文字格式
    
    Args:
        cls_id: 类别 ID (0, 1, 2)
        points: [20, 2] 坐标数组
    
    Returns:
        格式化的字符串，如: [divider] (-0.35,-0.68)(-0.34,-0.65)...
    """
    # 【验证】类别 ID 必须在 [0, 1, 2] 范围内
    if cls_id < 0 or cls_id > 2:
        raise ValueError(f"Invalid class ID: {cls_id}. Expected 0, 1, or 2.")
    
    cls_name = CLASS_NAMES[cls_id]
    
    # 【验证】点数必须是 20
    if points.shape[0] != 20:
        raise ValueError(f"Expected 20 points, got {points.shape[0]}")
    if points.shape[1] != 2:
        raise ValueError(f"Expected 2D coordinates, got shape {points.shape}")
    
    # 归一化坐标
    norm_points = normalize_coords(points)
    
    # 格式化 20 个点
    points_str = ""
    for i in range(20):
        x, y = norm_points[i]
        points_str += f"({x:.2f},{y:.2f})"
    
    return f"[{cls_name}] {points_str}"


def convert_gt_to_text(gt_data: Dict) -> Tuple[str, Dict]:
    """
    将 GT 数据转换为完整的文字格式
    
    Args:
        gt_data: 包含 'gt_classes' 和 'gt_points' 的字典
    
    Returns:
        (text, stats): 格式化的文字和统计信息
        
        文字格式:
        <map>
        [divider] (-0.35,-0.68)(-0.34,-0.65)...
        [boundary] (0.10,0.15)(0.12,0.18)...
        </map>
    """
    gt_classes = np.array(gt_data['gt_classes'])  # [N] 类别 ID 列表
    gt_points = np.array(gt_data['gt_points'])    # [N, 20, 2] 坐标数组
    
    stats = {
        'num_instances': len(gt_classes),
        'class_counts': {0: 0, 1: 0, 2: 0},
    }
    
    if len(gt_classes) == 0:
        return "<map>\n</map>", stats
    
    # 【验证】gt_classes 和 gt_points 数量必须一致
    if len(gt_classes) != len(gt_points):
        raise ValueError(f"Mismatch: {len(gt_classes)} classes vs {len(gt_points)} points")
    
    # 【验证】gt_points 形状必须是 [N, 20, 2]
    if gt_points.ndim != 3 or gt_points.shape[1] != 20 or gt_points.shape[2] != 2:
        raise ValueError(f"Invalid gt_points shape: {gt_points.shape}. Expected (N, 20, 2)")
    
    # 按类别分组
    lines_by_class = {0: [], 1: [], 2: []}
    
    for cls_id, points in zip(gt_classes, gt_points):
        cls_id = int(cls_id)  # 确保是 int
        line = format_instance_as_text(cls_id, points)
        lines_by_class[cls_id].append(line)
        stats['class_counts'][cls_id] += 1
    
    # 组装输出（按类别顺序：divider → ped_crossing → boundary）
    output_lines = ["<map>"]
    for cls_id in range(3):
        for line in lines_by_class[cls_id]:
            output_lines.append(line)
    output_lines.append("</map>")
    
    return '\n'.join(output_lines), stats


def get_split_tokens(nusc, split: str) -> List[str]:
    """获取指定 split 的 sample tokens"""
    from nuscenes.utils.splits import create_splits_scenes
    
    splits = create_splits_scenes()
    if split == 'train':
        scene_names = splits['train']
    elif split == 'val':
        scene_names = splits['val']
    else:
        raise ValueError(f"Unknown split: {split}")
    
    sample_tokens = []
    for scene in nusc.scene:
        if scene['name'] in scene_names:
            sample_token = scene['first_sample_token']
            while sample_token:
                sample_tokens.append(sample_token)
                sample = nusc.get('sample', sample_token)
                sample_token = sample['next']
    
    return sample_tokens


def main():
    parser = argparse.ArgumentParser(description='Convert GT to text format')
    parser.add_argument('--dataroot', type=str, required=True,
                        help='Path to nuScenes dataset')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--gt-cache-dir', type=str, required=True,
                        help='Path to GT cache directory (from Step 1)')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for text files')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val'],
                        help='Dataset split')
    parser.add_argument('--sample-ratio', type=float, default=0.15,
                        help='Ratio of samples to convert (default: 0.15 = 15%)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for sampling')
    args = parser.parse_args()
    
    # 设置随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载 nuScenes
    from nuscenes import NuScenes
    print(f"Loading nuScenes {args.version} from {args.dataroot}...")
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)
    
    # 获取 sample tokens
    sample_tokens = get_split_tokens(nusc, args.split)
    print(f"Total {args.split} samples: {len(sample_tokens)}")
    
    # 采样 15%
    num_samples = int(len(sample_tokens) * args.sample_ratio)
    random.shuffle(sample_tokens)
    selected_tokens = sample_tokens[:num_samples]
    print(f"Selected {num_samples} samples ({args.sample_ratio*100:.0f}%)")
    
    # GT cache 目录
    gt_ann_dir = os.path.join(args.gt_cache_dir, 'annotations')
    if not os.path.exists(gt_ann_dir):
        print(f"Error: GT annotations directory not found: {gt_ann_dir}")
        sys.exit(1)
    
    # 转换
    converted_data = {}
    total_stats = {
        'total_instances': 0,
        'class_counts': {0: 0, 1: 0, 2: 0},
        'avg_instances_per_sample': 0,
        'samples_with_no_gt': 0,
        'errors': [],
    }
    
    print(f"\nConverting GT to text format...")
    for token in tqdm(selected_tokens):
        # 加载 GT
        gt_file = os.path.join(gt_ann_dir, f'{token}.pkl')
        if not os.path.exists(gt_file):
            print(f"Warning: GT file not found for {token}, skipping...")
            total_stats['errors'].append(f"Missing GT: {token}")
            continue
        
        try:
            with open(gt_file, 'rb') as f:
                gt_data = pickle.load(f)
            
            # 转换为文字
            text, sample_stats = convert_gt_to_text(gt_data)
            converted_data[token] = text
            
            # 统计
            total_stats['total_instances'] += sample_stats['num_instances']
            for cls_id in range(3):
                total_stats['class_counts'][cls_id] += sample_stats['class_counts'][cls_id]
            
            if sample_stats['num_instances'] == 0:
                total_stats['samples_with_no_gt'] += 1
                
        except Exception as e:
            print(f"Error processing {token}: {e}")
            total_stats['errors'].append(f"Error in {token}: {str(e)}")
            continue
    
    if len(converted_data) > 0:
        total_stats['avg_instances_per_sample'] = total_stats['total_instances'] / len(converted_data)
    
    # 打印警告
    if total_stats['samples_with_no_gt'] > 0:
        print(f"\n⚠️ Warning: {total_stats['samples_with_no_gt']} samples have no GT instances!")
    if total_stats['errors']:
        print(f"⚠️ Warning: {len(total_stats['errors'])} errors occurred during conversion!")
    
    # 保存文字 GT
    output_file = os.path.join(args.output_dir, f'{args.split}_text_{args.sample_ratio:.2f}.json')
    with open(output_file, 'w') as f:
        json.dump(converted_data, f, indent=2)
    
    # 保存 token 列表（训练时必须使用这个列表！）
    token_list = list(converted_data.keys())
    token_file = os.path.join(args.output_dir, f'{args.split}_tokens_{args.sample_ratio:.2f}.json')
    with open(token_file, 'w') as f:
        json.dump(token_list, f, indent=2)
    
    # 【重要】创建一个索引文件，方便训练时快速查找
    index_file = os.path.join(args.output_dir, f'{args.split}_index_{args.sample_ratio:.2f}.json')
    index_data = {
        'split': args.split,
        'sample_ratio': args.sample_ratio,
        'seed': args.seed,
        'num_samples': len(token_list),
        'text_file': output_file,
        'token_file': token_file,
        'tokens': token_list,  # 直接包含 token 列表
    }
    with open(index_file, 'w') as f:
        json.dump(index_data, f, indent=2)
    
    print(f"\n【重要】训练时请使用以下文件确保数据一致性:")
    print(f"  Index file: {index_file}")
    print(f"  Token file: {token_file} (共 {len(token_list)} 个样本)")
    
    # 打印统计
    print(f"\n{'='*60}")
    print(f"Conversion completed!")
    print(f"{'='*60}")
    print(f"Output file: {output_file}")
    print(f"Token file: {token_file}")
    print(f"Total samples: {len(converted_data)}")
    print(f"Total instances: {total_stats['total_instances']}")
    print(f"Avg instances per sample: {total_stats['avg_instances_per_sample']:.2f}")
    print(f"Samples with no GT: {total_stats['samples_with_no_gt']}")
    print(f"\nClass distribution:")
    for cls_id, cls_name in enumerate(CLASS_NAMES):
        count = total_stats['class_counts'][cls_id]
        pct = count / total_stats['total_instances'] * 100 if total_stats['total_instances'] > 0 else 0
        print(f"  {cls_name}: {count} ({pct:.1f}%)")
    
    # 打印示例
    print(f"\n{'='*60}")
    print("Sample output:")
    print(f"{'='*60}")
    sample_token = list(converted_data.keys())[0]
    print(f"Token: {sample_token}")
    print(converted_data[sample_token][:500] + "..." if len(converted_data[sample_token]) > 500 else converted_data[sample_token])
    
    # 计算文本长度统计
    text_lengths = [len(text) for text in converted_data.values()]
    print(f"\nText length statistics:")
    print(f"  Min: {min(text_lengths)} chars")
    print(f"  Max: {max(text_lengths)} chars")
    print(f"  Avg: {np.mean(text_lengths):.0f} chars")
    print(f"  Median: {np.median(text_lengths):.0f} chars")
    
    # 【验证】检查生成的文本是否可以被正确解析
    print(f"\n{'='*60}")
    print("Validating generated text format...")
    print(f"{'='*60}")
    
    validation_errors = 0
    import re
    
    # 正则表达式匹配格式
    instance_pattern = r'\[(divider|ped_crossing|boundary)\]\s*(\([^)]+\)){20}'
    coord_pattern = r'\((-?[\d.]+),(-?[\d.]+)\)'
    
    for token, text in list(converted_data.items())[:10]:  # 只检查前 10 个
        # 检查 <map> 标签
        if not text.startswith('<map>') or not text.endswith('</map>'):
            print(f"  ❌ {token}: Missing <map> tags")
            validation_errors += 1
            continue
        
        # 提取内容
        content = text[6:-7].strip()  # 去掉 <map>\n 和 \n</map>
        
        if content:  # 如果有内容
            lines = content.split('\n')
            for line in lines:
                # 检查格式
                coords = re.findall(coord_pattern, line)
                if len(coords) != 20:
                    print(f"  ❌ {token}: Expected 20 coords, got {len(coords)}")
                    validation_errors += 1
                    break
                
                # 检查坐标范围
                for x, y in coords:
                    x, y = float(x), float(y)
                    if x < -1 or x > 1 or y < -1 or y > 1:
                        print(f"  ⚠️ {token}: Coord out of range: ({x}, {y})")
    
    if validation_errors == 0:
        print("  ✅ All sampled texts passed validation!")
    else:
        print(f"  ⚠️ {validation_errors} validation errors found")


if __name__ == '__main__':
    main()
