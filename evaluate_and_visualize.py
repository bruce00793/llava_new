#!/usr/bin/env python3
"""
评估和可视化脚本
用于评估训练好的模型并生成可视化结果
"""

import os
# 强制只使用单卡，避免多 GPU 设备冲突
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import json
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import argparse
from collections import defaultdict

from llava.model.map_llava_model import build_map_detector
from llava.data.map_dataset import MapDetectionDataset
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate and visualize map detection model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--dataroot', type=str, 
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes',
                        help='Path to nuScenes data')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='nuScenes version')
    parser.add_argument('--gt-cache', type=str,
                        default='/home/cly/auto/llava_test/LLaVA/data/nuscenes/gt_cache',
                        help='Path to GT cache')
    parser.add_argument('--output-dir', type=str, default='./eval_results',
                        help='Output directory for results')
    parser.add_argument('--num-samples', type=int, default=100,
                        help='Number of samples to evaluate (0 = all)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--num-vis', type=int, default=10,
                        help='Number of samples to visualize')
    return parser.parse_args()


def compute_chamfer_distance(pred_points, gt_points):
    """计算 Chamfer Distance"""
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    pred_points = np.array(pred_points)
    gt_points = np.array(gt_points)
    
    # Pred -> GT
    dist_pred_to_gt = np.min(np.linalg.norm(
        pred_points[:, None, :] - gt_points[None, :, :], axis=-1
    ), axis=1)
    
    # GT -> Pred
    dist_gt_to_pred = np.min(np.linalg.norm(
        gt_points[:, None, :] - pred_points[None, :, :], axis=-1
    ), axis=1)
    
    return (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2


def compute_instance_chamfer(pred_pts, gt_pts):
    """计算单个实例的 Chamfer Distance (双向平均)"""
    if len(pred_pts) == 0 or len(gt_pts) == 0:
        return float('inf')
    
    pred_pts = np.array(pred_pts).reshape(-1, 2)
    gt_pts = np.array(gt_pts).reshape(-1, 2)
    
    # Pred -> GT: 每个预测点到最近GT点的距离
    dist_pred_to_gt = np.min(np.linalg.norm(
        pred_pts[:, None, :] - gt_pts[None, :, :], axis=-1
    ), axis=1)
    
    # GT -> Pred: 每个GT点到最近预测点的距离
    dist_gt_to_pred = np.min(np.linalg.norm(
        gt_pts[:, None, :] - pred_pts[None, :, :], axis=-1
    ), axis=1)
    
    return (np.mean(dist_pred_to_gt) + np.mean(dist_gt_to_pred)) / 2


def compute_ap_per_class(all_pred_instances, all_gt_instances, cd_threshold):
    """
    计算单个类别在给定 CD 阈值下的 AP (与 MapTR 一致)
    
    Args:
        all_pred_instances: list of dicts, each with 'score', 'points' (已转换为真实坐标)
        all_gt_instances: list of dicts, each with 'points', 'matched' flag
        cd_threshold: Chamfer Distance 阈值 (meters)
    
    Returns:
        AP value
    """
    if len(all_gt_instances) == 0:
        return 0.0, 0, len(all_pred_instances)
    
    num_gt = len(all_gt_instances)
    num_pred = len(all_pred_instances)
    
    if num_pred == 0:
        return 0.0, num_gt, 0
    
    # 标记每个 GT 是否已被匹配
    gt_matched = [False] * num_gt
    
    # 按置信度降序排序预测
    sorted_preds = sorted(all_pred_instances, key=lambda x: x['score'], reverse=True)
    
    tp = np.zeros(num_pred)
    fp = np.zeros(num_pred)
    
    for i, pred in enumerate(sorted_preds):
        pred_pts = pred['points']
        
        # 找到与该预测 CD 最小的 GT
        min_cd = float('inf')
        best_gt_idx = -1
        
        for j, gt in enumerate(all_gt_instances):
            if gt_matched[j]:
                continue
            cd = compute_instance_chamfer(pred_pts, gt['points'])
            if cd < min_cd:
                min_cd = cd
                best_gt_idx = j
        
        # 判断是否匹配
        if min_cd < cd_threshold and best_gt_idx >= 0:
            gt_matched[best_gt_idx] = True
            tp[i] = 1
        else:
            fp[i] = 1
    
    # 计算累积 TP/FP
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    # Precision & Recall
    precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
    recall = tp_cumsum / (num_gt + 1e-6)
    
    # 计算 AP (area under PR curve)
    # 添加起点 (0, 1) 和终点
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))
    
    # 使 precision 单调递减
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # 计算 AP
    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap, num_gt, num_pred


def evaluate_sample(model, sample, device, threshold=0.3, debug=False):
    """评估单个样本"""
    model.eval()
    
    with torch.no_grad():
        images = sample['images'].unsqueeze(0).to(device)
        text_ids = sample['text_ids'].unsqueeze(0).to(device)
        
        # 获取相机参数（如果有）
        cam_intrinsics = sample.get('cam_intrinsics')
        cam_extrinsics = sample.get('cam_extrinsics')
        if cam_intrinsics is not None:
            cam_intrinsics = cam_intrinsics.unsqueeze(0).to(device)
        if cam_extrinsics is not None:
            cam_extrinsics = cam_extrinsics.unsqueeze(0).to(device)
        
        # 前向传播
        outputs = model(
            images=images,
            text_ids=text_ids,
            cam_intrinsics=cam_intrinsics,
            cam_extrinsics=cam_extrinsics,
        )
        
        # 解析预测结果
        pred_logits = outputs['pred_logits'][0]  # (50, 3)
        pred_points = outputs['pred_points'][0]  # (50, 20, 2)
        
        # 获取预测类别和分数
        pred_probs = torch.softmax(pred_logits, dim=-1)
        pred_scores, pred_labels = pred_probs.max(dim=-1)
        
        if debug:
            print(f"  pred_scores range: [{pred_scores.min():.3f}, {pred_scores.max():.3f}]")
            print(f"  pred_labels unique: {pred_labels.unique().tolist()}")
        
        # 过滤低置信度预测 (降低阈值到 0.1)
        valid_mask = (pred_scores > threshold) & (pred_labels < 3)  # 3 类
        
        if debug:
            print(f"  valid predictions: {valid_mask.sum().item()}")
        
        predictions = {
            'labels': pred_labels[valid_mask].cpu().numpy(),
            'scores': pred_scores[valid_mask].cpu().numpy(),
            'points': pred_points[valid_mask].cpu().numpy(),
            'all_labels': pred_labels.cpu().numpy(),  # 保留所有预测用于分析
            'all_scores': pred_scores.cpu().numpy(),
        }
        
    return predictions


def visualize_sample(sample, predictions, gt_data, output_path, class_names):
    """可视化单个样本"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 显示 6 个相机视图
    images = sample['images']  # (6, 3, H, W)
    camera_names = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    
    for i, ax in enumerate(axes.flat):
        img = images[i].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5).clip(0, 1)  # denormalize
        ax.imshow(img)
        ax.set_title(camera_names[i])
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path.replace('.png', '_cameras.png'), dpi=100)
    plt.close()
    
    # BEV 可视化
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    colors = {'divider': 'red', 'boundary': 'blue', 'ped_crossing': 'green'}
    
    # BEV 范围 (与 MapTR 配置一致)
    # pc_range = [-15, -30, -2, 15, 30, 2]
    # x: [-15, 15], y: [-30, 30]
    # 模型和GT的坐标都是归一化到 [-1, 1]
    x_half_range = 15.0  # x: [-1,1] → [-15, 15]
    y_half_range = 30.0  # y: [-1,1] → [-30, 30]
    
    # 绘制 GT (GT 也是归一化坐标，需要转换)
    if gt_data is not None:
        gt_labels = gt_data.get('labels', [])
        gt_points = gt_data.get('points', [])
        gt_masks = gt_data.get('masks', [])
        
        for j, (label, points, mask) in enumerate(zip(gt_labels, gt_points, gt_masks)):
            if mask:
                class_name = class_names.get(int(label), 'unknown')
                color = colors.get(class_name, 'gray')
                pts = np.array(points).copy()
                # GT 坐标转换
                pts[:, 0] = pts[:, 0] * x_half_range  # x
                pts[:, 1] = pts[:, 1] * y_half_range  # y
                ax.plot(pts[:, 0], pts[:, 1], '--', color=color, alpha=0.5, linewidth=2)
    
    # 绘制预测 (转换归一化坐标到真实世界坐标)
    for j in range(len(predictions['labels'])):
        label = predictions['labels'][j]
        score = predictions['scores'][j]
        points = predictions['points'][j].copy()  # (20, 2)
        
        # 坐标转换: [-1, 1] → 真实世界坐标
        points[:, 0] = points[:, 0] * x_half_range  # x: [-1,1] → [-15, 15]
        points[:, 1] = points[:, 1] * y_half_range  # y: [-1,1] → [-30, 30]
        
        class_name = class_names.get(int(label), 'unknown')
        color = colors.get(class_name, 'gray')
        
        ax.plot(points[:, 0], points[:, 1], '-', color=color, linewidth=2,
                label=f'{class_name}: {score:.2f}')
    
    ax.set_xlim(-15, 15)
    ax.set_ylim(-30, 30)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('BEV Map Detection (Solid: Pred, Dashed: GT)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    vis_dir = os.path.join(args.output_dir, 'visualizations')
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 加载模型
    print(f"\n{'='*60}")
    print("Loading model...")
    print(f"{'='*60}")
    
    model = build_map_detector(
        llm_path='/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5',
        freeze_llm=True,
    )
    
    # 加载 checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model = model.to(device)
    model.eval()
    print("✅ Model loaded successfully!")
    
    # 加载本地 tokenizer
    print(f"\n{'='*60}")
    print("Loading tokenizer from local path...")
    print(f"{'='*60}")
    
    llm_path = '/home/cly/auto/llava_test/LLaVA/vicuna-7b-v1.5'
    tokenizer = AutoTokenizer.from_pretrained(llm_path, use_fast=False, local_files_only=True)
    print("✅ Tokenizer loaded from local files!")
    
    # 加载验证数据集
    print(f"\n{'='*60}")
    print("Loading validation dataset...")
    print(f"{'='*60}")
    
    val_dataset = MapDetectionDataset(
        dataroot=args.dataroot,
        version=args.version,
        split='val',
        gt_cache_path=args.gt_cache,
        tokenizer=tokenizer,
    )
    
    num_samples = len(val_dataset) if args.num_samples == 0 else min(args.num_samples, len(val_dataset))
    print(f"Evaluating on {num_samples} samples...")
    
    # 类别名称 (与 MapTR 一致)
    # 0: divider (road_divider + lane_divider 合并)
    # 1: ped_crossing (人行横道)
    # 2: boundary (道路边界)
    class_names = {0: 'divider', 1: 'ped_crossing', 2: 'boundary'}
    
    # 评估
    print(f"\n{'='*60}")
    print("Running evaluation...")
    print(f"{'='*60}")
    
    all_predictions = []
    all_gt = []
    chamfer_distances = defaultdict(list)
    
    # 统计信息
    total_pred_count = {cls: 0 for cls in class_names.values()}
    total_gt_count = {cls: 0 for cls in class_names.values()}
    
    # 用于 mAP 计算的实例级数据 (与 MapTR 一致)
    pred_instances_by_class = defaultdict(list)  # {class_name: [{'score', 'points'}, ...]}
    gt_instances_by_class = defaultdict(list)    # {class_name: [{'points'}, ...]}
    
    for i in tqdm(range(num_samples)):
        sample = val_dataset[i]
        
        # 获取 GT (from MapGroundTruth object)
        gt_obj = sample.get('gt', None)
        if gt_obj is not None and hasattr(gt_obj, 'class_labels'):
            gt_labels = gt_obj.class_labels.numpy() if isinstance(gt_obj.class_labels, torch.Tensor) else np.array(gt_obj.class_labels)
            gt_points = gt_obj.points.numpy() if isinstance(gt_obj.points, torch.Tensor) else np.array(gt_obj.points)
            # MapGroundTruth doesn't have masks, so create all-ones mask
            gt_masks = np.ones(len(gt_labels), dtype=np.float32)
        else:
            gt_labels = np.array([])
            gt_points = np.array([])
            gt_masks = np.array([])
        
        gt_data = {
            'labels': gt_labels,
            'points': gt_points,
            'masks': gt_masks,
        }
        
        # 预测 (第一个样本开启调试)
        try:
            predictions = evaluate_sample(model, sample, device, debug=(i==0))
        except Exception as e:
            print(f"Error evaluating sample {i}: {e}")
            continue
        
        # 调试: 第一个样本打印详细信息
        if i == 0:
            print(f"\n[DEBUG] Sample 0:")
            print(f"  GT labels shape: {gt_data['labels'].shape}, unique: {np.unique(gt_data['labels'])}")
            print(f"  GT masks shape: {gt_data['masks'].shape}, unique: {np.unique(gt_data['masks'])}")
            print(f"  Pred labels: {predictions['labels'][:10]}... (total: {len(predictions['labels'])})")
        
        all_predictions.append(predictions)
        all_gt.append(gt_data)
        
        # 收集实例级别数据 & 计算 Chamfer Distance
        for cls_id, cls_name in class_names.items():
            pred_mask = predictions['labels'] == cls_id
            # GT mask: 如果 masks 全是 1 或者没有 masks，则只用 labels
            if gt_data['masks'].size > 0 and np.any(gt_data['masks'] != 1):
                gt_mask = (gt_data['labels'] == cls_id) & (gt_data['masks'] == 1)
            else:
                gt_mask = (gt_data['labels'] == cls_id)
            
            # 统计
            total_pred_count[cls_name] += np.sum(pred_mask)
            total_gt_count[cls_name] += np.sum(gt_mask)
            
            # 收集预测实例 (用于 mAP)
            # 注意：GT 点已经归一化到 [-1, 1]，预测点也是 [-1, 1]
            # 为了公平比较，需要统一转换到真实世界坐标
            # pc_range = [-15, -30, -2, 15, 30, 2], 所以:
            # x: [-1,1] → [-15, 15] (range=30, half=15)
            # y: [-1,1] → [-30, 30] (range=60, half=30)
            pred_indices = np.where(pred_mask)[0]
            for idx in pred_indices:
                pts = predictions['points'][idx].copy()  # (20, 2)
                # 转换坐标: [-1, 1] → 真实世界坐标
                pts[:, 0] = pts[:, 0] * 15.0  # x: [-1,1] → [-15, 15]
                pts[:, 1] = pts[:, 1] * 30.0  # y: [-1,1] → [-30, 30]
                pred_instances_by_class[cls_name].append({
                    'score': predictions['scores'][idx],
                    'points': pts,
                })
            
            # 收集 GT 实例 (用于 mAP)
            # GT 点也是归一化的 [-1, 1]，需要转换到真实世界坐标
            gt_indices = np.where(gt_mask)[0]
            for idx in gt_indices:
                gt_pts = gt_data['points'][idx].copy()  # (20, 2)
                # 转换坐标: [-1, 1] → 真实世界坐标
                gt_pts[:, 0] = gt_pts[:, 0] * 15.0  # x: [-1,1] → [-15, 15]
                gt_pts[:, 1] = gt_pts[:, 1] * 30.0  # y: [-1,1] → [-30, 30]
                gt_instances_by_class[cls_name].append({
                    'points': gt_pts,
                })
            
            # 计算整体 Chamfer Distance (用于快速评估)
            if np.sum(pred_mask) > 0 and np.sum(gt_mask) > 0:
                pred_pts = predictions['points'][pred_mask].reshape(-1, 2).copy()
                # 正确的坐标转换
                pred_pts[:, 0] = pred_pts[:, 0] * 15.0  # x: [-1,1] → [-15, 15]
                pred_pts[:, 1] = pred_pts[:, 1] * 30.0  # y: [-1,1] → [-30, 30]
                gt_pts = gt_data['points'][gt_mask].reshape(-1, 2).copy()
                gt_pts[:, 0] = gt_pts[:, 0] * 15.0  # x
                gt_pts[:, 1] = gt_pts[:, 1] * 30.0  # y
                cd = compute_chamfer_distance(pred_pts, gt_pts)
                if cd < float('inf'):
                    chamfer_distances[cls_name].append(cd)
        
        # 可视化
        if args.visualize and i < args.num_vis:
            vis_path = os.path.join(vis_dir, f'sample_{i:04d}.png')
            try:
                visualize_sample(sample, predictions, gt_data, vis_path, class_names)
            except Exception as e:
                print(f"Error visualizing sample {i}: {e}")
    
    # 计算 metrics
    print(f"\n{'='*60}")
    print("Computing metrics...")
    print(f"{'='*60}")
    
    results = {}
    
    # 统计信息
    print("\n📈 Prediction Statistics:")
    for cls_name in class_names.values():
        pred_cnt = total_pred_count[cls_name]
        gt_cnt = total_gt_count[cls_name]
        results[f'pred_count_{cls_name}'] = int(pred_cnt)
        results[f'gt_count_{cls_name}'] = int(gt_cnt)
        print(f"  {cls_name}: pred={pred_cnt}, gt={gt_cnt}")
    
    # ============================================
    # mAP 计算 (与 MapTR 一致)
    # ============================================
    print("\n📊 mAP (Mean Average Precision) - MapTR Style:")
    
    # MapTR 使用的 CD 阈值
    cd_thresholds = [0.5, 1.0, 1.5]  # meters
    
    ap_results = {}
    for thresh in cd_thresholds:
        print(f"\n  CD Threshold = {thresh}m:")
        aps = []
        for cls_name in class_names.values():
            pred_instances = pred_instances_by_class[cls_name]
            gt_instances = gt_instances_by_class[cls_name]
            
            ap, num_gt, num_pred = compute_ap_per_class(pred_instances, gt_instances, thresh)
            aps.append(ap)
            results[f'AP_{cls_name}_@{thresh}m'] = float(ap)
            print(f"    {cls_name}: AP={ap:.4f} (GT={num_gt}, Pred={num_pred})")
        
        # mAP = 所有类别 AP 的平均
        valid_aps = [ap for ap in aps if ap > 0]
        mAP = np.mean(valid_aps) if valid_aps else 0.0
        results[f'mAP_@{thresh}m'] = float(mAP)
        print(f"    mAP@{thresh}m: {mAP:.4f}")
        ap_results[thresh] = mAP
    
    # 总 mAP (所有阈值的平均)
    overall_mAP = np.mean(list(ap_results.values()))
    results['mAP'] = float(overall_mAP)
    print(f"\n  📊 Overall mAP (avg of 0.5/1.0/1.5m): {overall_mAP:.4f}")
    
    # ============================================
    # Chamfer Distance (作为参考)
    # ============================================
    print("\n📏 Chamfer Distance (lower is better):")
    for cls_name in class_names.values():
        if chamfer_distances[cls_name]:
            mean_cd = np.mean(chamfer_distances[cls_name])
            results[f'CD_{cls_name}'] = float(mean_cd)
            print(f"  {cls_name}: {mean_cd:.4f}")
        else:
            print(f"  {cls_name}: N/A (no valid matches)")
    
    if chamfer_distances:
        all_cd = [cd for cds in chamfer_distances.values() for cd in cds]
        if all_cd:
            mean_cd_all = np.mean(all_cd)
            results['CD_mean'] = float(mean_cd_all)
            print(f"  Mean: {mean_cd_all:.4f}")
    
    # 保存结果
    results_path = os.path.join(args.output_dir, 'metrics.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_path}")
    if args.visualize:
        print(f"✅ Visualizations saved to: {vis_dir}")
    
    print(f"\n{'='*60}")
    print("Evaluation complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
