#!/usr/bin/env python3
"""训练 Loss 可视化脚本"""

import matplotlib.pyplot as plt
import numpy as np

# 数据
epochs = list(range(1, 25))

train_total = [29.1268,23.9766,22.7165,21.8246,20.7640,19.7259,18.6947,17.8799,
               17.0640,16.4913,15.6313,15.0872,14.2890,13.8302,13.1903,12.7988,
               12.3421,12.0818,11.6014,11.4546,11.3259,11.2711,11.2072,11.1493]
val_total = [62.0065,67.3784,40.0483,36.8083,36.3847,35.8189,35.4174,34.9434,
             34.9251,35.0662,34.5061,35.0202,34.7688,34.4617,33.5489,33.3654,
             33.2531,32.6667,32.6051,32.1976,32.0255,31.9925,31.5181,31.5705]

train_pts = [5.6749,4.6619,4.4149,4.2417,4.0348,3.8324,3.6303,3.4728,
             3.3165,3.2059,3.0402,2.9342,2.7797,2.6903,2.5660,2.4902,
             2.4007,2.3508,2.2564,2.2276,2.2019,2.1919,2.1790,2.1678]
val_pts = [11.9421,13.1738,7.8124,7.1934,7.1194,7.0135,6.9363,6.8444,
           6.8431,6.8767,6.7689,6.8737,6.8248,6.7648,6.5827,6.5472,
           6.5241,6.4077,6.3956,6.3137,6.2802,6.2746,6.1777,6.1870]

train_cls = [0.3744,0.3320,0.3194,0.3066,0.2937,0.2807,0.2703,0.2567,
             0.2395,0.2296,0.2139,0.2069,0.1941,0.1881,0.1789,0.1728,
             0.1682,0.1628,0.1586,0.1573,0.1572,0.1547,0.1549,0.1542]
val_cls = [1.1455,0.7522,0.4909,0.4190,0.3924,0.3742,0.3665,0.3592,
           0.3534,0.3400,0.3296,0.3247,0.3211,0.3175,0.3166,0.3134,
           0.3152,0.3130,0.3124,0.3133,0.3112,0.3086,0.3136,0.3165]

train_dir = [0.7335,0.6165,0.5790,0.5639,0.5475,0.5288,0.5074,0.5044,
             0.4877,0.4850,0.4777,0.4698,0.4557,0.4554,0.4466,0.4417,
             0.4410,0.4376,0.4316,0.4291,0.4302,0.4327,0.4296,0.4337]
val_dir = [0.9611,0.9634,0.8676,0.6698,0.6157,0.5703,0.5820,0.5758,
           0.5522,0.5328,0.5111,0.5034,0.4993,0.4947,0.4899,0.4898,
           0.4784,0.4737,0.4682,0.4665,0.4697,0.4748,0.4704,0.4643]

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图表
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Total Loss
ax1 = axes[0, 0]
ax1.plot(epochs, train_total, 'b-o', label='Train', linewidth=2, markersize=5)
ax1.plot(epochs, val_total, 'r-o', label='Val', linewidth=2, markersize=5)
ax1.fill_between(epochs, train_total, val_total, alpha=0.2, color='red')
ax1.set_title('Total Loss (Train vs Val)', fontsize=14, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, 25, 2))

# 2. Points Loss (最严重的过拟合)
ax2 = axes[0, 1]
ax2.plot(epochs, train_pts, 'b-o', label='Train', linewidth=2, markersize=5)
ax2.plot(epochs, val_pts, 'r-o', label='Val', linewidth=2, markersize=5)
ax2.fill_between(epochs, train_pts, val_pts, alpha=0.2, color='red')
ax2.set_title('Points Loss (Main Overfitting Source)', fontsize=14, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Loss', fontsize=12)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, 25, 2))
# 添加注释
ax2.annotate('Val loss stagnates\nafter epoch 8', 
             xy=(12, 6.8), xytext=(16, 9),
             fontsize=10, color='red',
             arrowprops=dict(arrowstyle='->', color='red', lw=1.5))

# 3. Classification Loss
ax3 = axes[1, 0]
ax3.plot(epochs, train_cls, 'b-o', label='Train', linewidth=2, markersize=5)
ax3.plot(epochs, val_cls, 'r-o', label='Val', linewidth=2, markersize=5)
ax3.fill_between(epochs, train_cls, val_cls, alpha=0.2, color='orange')
ax3.set_title('Classification Loss', fontsize=14, fontweight='bold')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, 25, 2))

# 4. Generalization Gap
ax4 = axes[1, 1]
gap_total = [v - t for t, v in zip(train_total, val_total)]
gap_pts = [v - t for t, v in zip(train_pts, val_pts)]
gap_cls = [v - t for t, v in zip(train_cls, val_cls)]

ax4.plot(epochs, gap_total, 'g-o', label='Total Gap', linewidth=2, markersize=5)
ax4.plot(epochs, gap_pts, 'm-s', label='Points Gap', linewidth=2, markersize=5)
ax4.plot(epochs, gap_cls, 'c-^', label='Cls Gap', linewidth=2, markersize=5)
ax4.set_title('Generalization Gap (Val - Train)', fontsize=14, fontweight='bold')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Gap', fontsize=12)
ax4.legend(fontsize=11)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, 25, 2))
ax4.axhline(y=0, color='k', linestyle='--', alpha=0.5)

# 调整布局
plt.tight_layout()

# 保存图片
output_path = '/home/cly/auto/llava_test/LLaVA/outputs/6x4090_fresh_20260125_143156/training_loss_analysis.png'
plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
print(f"图片已保存到: {output_path}")

# 打印过拟合分析
print("\n" + "="*60)
print("过拟合分析总结")
print("="*60)
print(f"\n{'Epoch':<8} {'Train Pts':<12} {'Val Pts':<12} {'Gap':<12} {'Gap Ratio':<12}")
print("-"*56)
for i, epoch in enumerate([1, 8, 15, 24]):
    idx = epoch - 1
    gap = val_pts[idx] - train_pts[idx]
    ratio = val_pts[idx] / train_pts[idx]
    print(f"{epoch:<8} {train_pts[idx]:<12.4f} {val_pts[idx]:<12.4f} {gap:<12.4f} {ratio:<12.2f}x")

print("\n关键发现:")
print("  1. Points Loss 过拟合最严重 (Gap Ratio: 2.85x)")
print("  2. Epoch 8 后验证 loss 基本停滞")
print("  3. 训练 loss 持续下降，但验证 loss 不再跟随")
print("  4. 建议: 添加正则化、数据增强、迭代精修")
print("="*60)

plt.show()
