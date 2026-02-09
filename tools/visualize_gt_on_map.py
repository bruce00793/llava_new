"""
在整体地图上可视化GT提取结果
展示：整体地图 + 车辆位置 + ROI范围 + 提取的GT
"""

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'llava/data/map_gt'))
sys.path.insert(0, os.path.join(parent_dir, 'llava/model'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MPLPolygon
from matplotlib.transforms import Affine2D
import cv2

from nuscenes.nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap
from pyquaternion import Quaternion

from cache import GTCache
from map_config import MapDetectionConfig


def visualize_gt_on_map(
    sample_token: str,
    dataroot: str = 'data/nuscenes_mini',
    version: str = 'v1.0-mini',
    save_path: str = None
):
    """
    在整体地图上可视化GT提取
    """
    print(f"处理sample: {sample_token}")
    
    # 初始化
    config = MapDetectionConfig()
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=False)
    
    # 获取sample信息
    sample = nusc.get('sample', sample_token)
    scene = nusc.get('scene', sample['scene_token'])
    log = nusc.get('log', scene['log_token'])
    location = log['location']
    
    print(f"场景位置: {location}")
    
    # 获取ego pose
    lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
    ego_translation = np.array(ego_pose['translation'])
    ego_rotation = Quaternion(ego_pose['rotation'])
    
    print(f"车辆世界坐标: ({ego_translation[0]:.1f}, {ego_translation[1]:.1f})")
    
    # 加载地图
    nusc_map = NuScenesMap(dataroot=dataroot, map_name=location)
    
    # 生成GT
    cache = GTCache(
        dataroot=dataroot,
        version=version,
        pc_range=config.PC_RANGE,
        min_arc_length=config.MIN_ARC_LENGTH,
        min_area=config.MIN_AREA
    )
    gt_data = cache.process_one_sample(sample_token)
    
    print(f"GT实例数: {len(gt_data['gt_classes'])}")
    
    # 创建可视化
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # ====== 左图：整体地图视图（世界坐标系）======
    ax1 = axes[0]
    
    # 计算显示范围（以车辆为中心，100米范围）
    view_range = 80
    x_center, y_center = ego_translation[0], ego_translation[1]
    
    # 绘制地图元素（世界坐标）
    # 1. lane_divider
    for record in nusc_map.lane_divider:
        line_token = record['line_token']
        line = nusc_map.get('line', line_token)
        nodes = [nusc_map.get('node', n) for n in line['node_tokens']]
        pts = np.array([[n['x'], n['y']] for n in nodes])
        
        # 只绘制视野范围内的
        if np.any((pts[:, 0] > x_center - view_range) & (pts[:, 0] < x_center + view_range) &
                  (pts[:, 1] > y_center - view_range) & (pts[:, 1] < y_center + view_range)):
            ax1.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1, alpha=0.5)
    
    # 2. road_divider
    for record in nusc_map.road_divider:
        line_token = record['line_token']
        line = nusc_map.get('line', line_token)
        nodes = [nusc_map.get('node', n) for n in line['node_tokens']]
        pts = np.array([[n['x'], n['y']] for n in nodes])
        
        if np.any((pts[:, 0] > x_center - view_range) & (pts[:, 0] < x_center + view_range) &
                  (pts[:, 1] > y_center - view_range) & (pts[:, 1] < y_center + view_range)):
            ax1.plot(pts[:, 0], pts[:, 1], 'b-', linewidth=1, alpha=0.5)
    
    # 3. ped_crossing
    for record in nusc_map.ped_crossing:
        polygon_token = record['polygon_token']
        polygon = nusc_map.get('polygon', polygon_token)
        nodes = [nusc_map.get('node', n) for n in polygon['exterior_node_tokens']]
        pts = np.array([[n['x'], n['y']] for n in nodes])
        
        if np.any((pts[:, 0] > x_center - view_range) & (pts[:, 0] < x_center + view_range) &
                  (pts[:, 1] > y_center - view_range) & (pts[:, 1] < y_center + view_range)):
            poly = MPLPolygon(pts, fill=True, facecolor='orange', edgecolor='orange', alpha=0.3)
            ax1.add_patch(poly)
    
    # 绘制车辆位置
    ax1.plot(x_center, y_center, 'ro', markersize=15, label='Ego车辆', zorder=10)
    
    # 绘制车辆朝向
    yaw = ego_rotation.yaw_pitch_roll[0]
    arrow_len = 15
    ax1.arrow(x_center, y_center, 
              arrow_len * np.cos(yaw), arrow_len * np.sin(yaw),
              head_width=3, head_length=3, fc='red', ec='red', zorder=10)
    
    # 绘制ROI（感兴趣区域）在世界坐标中的位置
    # ROI在ego坐标系中是 [-15,-30] 到 [15,30]
    # 需要转换到世界坐标系
    roi_corners_ego = np.array([
        [config.PC_RANGE[0], config.PC_RANGE[1]],  # 左后
        [config.PC_RANGE[3], config.PC_RANGE[1]],  # 右后
        [config.PC_RANGE[3], config.PC_RANGE[4]],  # 右前
        [config.PC_RANGE[0], config.PC_RANGE[4]],  # 左前
        [config.PC_RANGE[0], config.PC_RANGE[1]],  # 闭合
    ])
    
    # 转换到世界坐标
    rot_matrix = ego_rotation.rotation_matrix[:2, :2]
    roi_corners_world = roi_corners_ego @ rot_matrix.T + ego_translation[:2]
    
    ax1.plot(roi_corners_world[:, 0], roi_corners_world[:, 1], 
             'r--', linewidth=3, label='ROI (感兴趣区域)')
    
    ax1.set_xlim(x_center - view_range, x_center + view_range)
    ax1.set_ylim(y_center - view_range, y_center + view_range)
    ax1.set_aspect('equal')
    ax1.set_xlabel('X (米, 世界坐标)', fontsize=12)
    ax1.set_ylabel('Y (米, 世界坐标)', fontsize=12)
    ax1.set_title(f'整体地图视图 - {location}\n绿色=lane_divider, 蓝色=road_divider, 橙色=ped_crossing', fontsize=12)
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ====== 右图：提取的GT（ego坐标系）======
    ax2 = axes[1]
    
    # 绘制ROI边界
    x_min, y_min = config.PC_RANGE[0], config.PC_RANGE[1]
    x_max, y_max = config.PC_RANGE[3], config.PC_RANGE[4]
    
    ax2.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             'k--', linewidth=2, label='ROI边界')
    
    # 绘制车辆
    ax2.plot(0, 0, 'ro', markersize=15, label='Ego车辆', zorder=10)
    ax2.arrow(0, 0, 0, 8, head_width=2, head_length=2, fc='red', ec='red', zorder=10)
    
    # 绘制提取的GT
    colors = {'divider': 'blue', 'lane_divider': 'green', 'ped_crossing': 'orange'}
    
    if len(gt_data['gt_classes']) > 0:
        for i in range(len(gt_data['gt_classes'])):
            cls_id = gt_data['gt_classes'][i]
            points = gt_data['gt_points'][i]
            is_closed = gt_data['gt_is_closed'][i]
            cls_name = config.CLASS_NAMES[cls_id]
            color = colors[cls_name]
            
            ax2.scatter(points[:, 0], points[:, 1], c=color, s=30, alpha=0.8, zorder=5)
            
            if is_closed:
                ax2.plot(np.append(points[:, 0], points[0, 0]),
                        np.append(points[:, 1], points[0, 1]),
                        c=color, linewidth=2.5, alpha=0.9)
            else:
                ax2.plot(points[:, 0], points[:, 1], c=color, linewidth=2.5, alpha=0.9)
            
            # 标注
            center = points.mean(axis=0)
            ax2.text(center[0], center[1], str(i), fontsize=10, ha='center', va='center',
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='circle', facecolor=color, alpha=0.8))
    else:
        ax2.text(0, 0, 'No GT instances', ha='center', va='center', fontsize=14, color='red')
    
    ax2.set_xlim(x_min - 5, x_max + 5)
    ax2.set_ylim(y_min - 5, y_max + 5)
    ax2.set_aspect('equal')
    ax2.set_xlabel('X (米, ego坐标, 左-右)', fontsize=12)
    ax2.set_ylabel('Y (米, ego坐标, 后-前)', fontsize=12)
    ax2.set_title(f'提取的GT ({len(gt_data["gt_classes"])}个实例)\n从ROI内裁剪并采样为20点', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'GT提取验证: Sample {sample_token[:16]}...\n'
                 f'从整体地图 → ROI裁剪 → ego坐标转换 → 20点采样',
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n可视化已保存到: {save_path}")
    else:
        plt.show()
    
    return gt_data


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='data/nuscenes_mini')
    parser.add_argument('--version', default='v1.0-mini')
    parser.add_argument('--sample', default=None)
    parser.add_argument('--output', default='gt_on_map.png')
    
    args = parser.parse_args()
    
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    if args.sample is None:
        # 使用第三个场景（GT最多的那个）
        sample_token = nusc.scene[2]['first_sample_token']
    else:
        sample_token = args.sample
    
    visualize_gt_on_map(
        sample_token=sample_token,
        dataroot=args.dataroot,
        version=args.version,
        save_path=args.output
    )

