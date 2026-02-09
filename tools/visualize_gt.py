"""
可视化GT数据与原始图像
"""

import sys
import os

# 添加路径
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, os.path.join(parent_dir, 'llava/data/map_gt'))
sys.path.insert(0, os.path.join(parent_dir, 'llava/model'))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPLPolygon
import cv2
from pathlib import Path

# 直接导入模块
from cache import GTCache
from nuscenes_map_api import NuScenesMapAPI
from map_config import MapDetectionConfig


def visualize_gt_with_images(
    sample_token: str,
    dataroot: str = 'data/nuscenes_mini',
    version: str = 'v1.0-mini',
    save_path: str = None
):
    """
    可视化一个sample的GT数据和6个相机图像
    """
    print(f"处理sample: {sample_token}")
    
    # 初始化
    config = MapDetectionConfig()
    api = NuScenesMapAPI(dataroot, version, verbose=False)
    cache = GTCache(
        dataroot=dataroot,
        version=version,
        pc_range=config.PC_RANGE,
        min_arc_length=config.MIN_ARC_LENGTH,
        min_area=config.MIN_AREA
    )
    
    # 生成GT
    print("生成GT数据...")
    gt_data = cache.process_one_sample(sample_token)
    
    print(f"\nGT数据统计:")
    print(f"  实例数: {len(gt_data['gt_classes'])}")
    if len(gt_data['gt_classes']) > 0:
        for cls_id in range(3):
            count = np.sum(gt_data['gt_classes'] == cls_id)
            if count > 0:
                print(f"  {config.CLASS_NAMES[cls_id]}: {count}个")
    
    # 获取相机图像
    sample = api.nusc.get('sample', sample_token)
    cam_names = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']
    
    # 创建可视化
    fig = plt.figure(figsize=(20, 12))
    
    # 上半部分：6个相机图像 (2行3列)
    for idx, cam_name in enumerate(cam_names):
        ax = plt.subplot(3, 3, idx + 1)
        
        # 读取图像
        cam_token = sample['data'][cam_name]
        cam_data = api.nusc.get('sample_data', cam_token)
        img_path = os.path.join(dataroot, cam_data['filename'])
        
        if os.path.exists(img_path):
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
        else:
            ax.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
        ax.set_title(cam_name, fontsize=10)
        ax.axis('off')
    
    # 下半部分：BEV视图显示GT (占据底部3个格子)
    ax_bev = plt.subplot(3, 3, (7, 9))
    
    # 绘制ROI边界
    x_min, y_min = config.PC_RANGE[0], config.PC_RANGE[1]
    x_max, y_max = config.PC_RANGE[3], config.PC_RANGE[4]
    
    ax_bev.plot([x_min, x_max, x_max, x_min, x_min],
                [y_min, y_min, y_max, y_max, y_min],
                'k--', linewidth=2, label='ROI边界')
    
    # 绘制坐标轴
    ax_bev.axhline(y=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    ax_bev.axvline(x=0, color='gray', linestyle=':', linewidth=1, alpha=0.5)
    
    # 绘制车辆位置
    ax_bev.plot(0, 0, 'ro', markersize=15, label='Ego车辆', zorder=10)
    ax_bev.arrow(0, 0, 0, 5, head_width=1.5, head_length=1.5, 
                 fc='red', ec='red', linewidth=2, zorder=10)
    
    # 绘制GT实例
    colors = {'divider': 'blue', 'lane_divider': 'green', 'ped_crossing': 'orange'}
    
    if len(gt_data['gt_classes']) > 0:
        for i in range(len(gt_data['gt_classes'])):
            cls_id = gt_data['gt_classes'][i]
            points = gt_data['gt_points'][i]  # [20, 2]
            is_closed = gt_data['gt_is_closed'][i]
            cls_name = config.CLASS_NAMES[cls_id]
            color = colors[cls_name]
            
            # 绘制点
            ax_bev.scatter(points[:, 0], points[:, 1], c=color, s=20, alpha=0.6, zorder=5)
            
            # 绘制连线
            if is_closed:
                # 多边形：首尾相连
                ax_bev.plot(np.append(points[:, 0], points[0, 0]),
                           np.append(points[:, 1], points[0, 1]),
                           c=color, linewidth=2, alpha=0.8, label=cls_name if i == 0 else '')
            else:
                # 折线：不闭合
                ax_bev.plot(points[:, 0], points[:, 1],
                           c=color, linewidth=2, alpha=0.8, label=cls_name if i == 0 else '')
            
            # 标注实例编号
            center = points.mean(axis=0)
            ax_bev.text(center[0], center[1], str(i), fontsize=8, 
                       ha='center', va='center', color='white',
                       bbox=dict(boxstyle='circle', facecolor=color, alpha=0.7))
    else:
        ax_bev.text(0, 0, 'No GT instances\n(需要Map expansion数据)', 
                   ha='center', va='center', fontsize=12, color='red',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))
    
    ax_bev.set_xlabel('X (米, 左-右)', fontsize=12)
    ax_bev.set_ylabel('Y (米, 后-前)', fontsize=12)
    ax_bev.set_title('BEV视图 - GT标注 (ego坐标系)', fontsize=14, fontweight='bold')
    ax_bev.set_xlim(x_min - 2, x_max + 2)
    ax_bev.set_ylim(y_min - 2, y_max + 2)
    ax_bev.set_aspect('equal')
    ax_bev.grid(True, alpha=0.3)
    ax_bev.legend(loc='upper right', fontsize=10)
    
    plt.suptitle(f'Sample: {sample_token[:16]}... | GT实例数: {len(gt_data["gt_classes"])}',
                 fontsize=16, fontweight='bold')
    
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
    parser.add_argument('--dataroot', default='data/nuscenes_mini', help='nuScenes数据路径')
    parser.add_argument('--version', default='v1.0-mini', help='数据集版本')
    parser.add_argument('--sample', default=None, help='sample token (不指定则使用第一个)')
    parser.add_argument('--output', default='gt_visualization.png', help='输出图像路径')
    
    args = parser.parse_args()
    
    # 获取第一个sample
    from nuscenes.nuscenes import NuScenes
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    
    if args.sample is None:
        sample_token = nusc.scene[0]['first_sample_token']
        print(f"使用第一个sample: {sample_token}")
    else:
        sample_token = args.sample
    
    # 生成可视化
    visualize_gt_with_images(
        sample_token=sample_token,
        dataroot=args.dataroot,
        version=args.version,
        save_path=args.output
    )

