# coding=utf-8
"""Unprocesses sRGB images into realistic raw data using NumPy.

Unprocessing Images for Learned Raw Denoising
http://timothybrooks.com/tech/unprocessing
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import cv2
from scipy.interpolate import interp2d
from utils.image import pad_to_size
import matplotlib.pyplot as plt
def inverse_smoothstep(image):
  """Approximately inverts a global tone mapping curve."""
  image = np.clip(image, 0.0, 1.0)
  return 0.5 - np.sin(np.arcsin(1.0 - 2.0 * image) / 3.0)


def gamma_expansion(image):
  """Converts from gamma to linear space."""
  return np.maximum(image, 1e-8) ** 2.2


def apply_ccm(image, ccm):
  """Applies a color correction matrix."""
  shape = image.shape
  image = image.reshape(-1, 3)
  image = np.dot(image, ccm.T)
  return image.reshape(shape)


def safe_invert_gains(image, rgb_gain, red_gain, blue_gain):
  """Inverts gains while safely handling saturated pixels."""
  gains = np.array([1.0 / red_gain, 1.0, 1.0 / blue_gain]) / rgb_gain
  gains = gains[np.newaxis, np.newaxis, :]

  # Prevents dimming of saturated pixels by smoothly masking gains near white.
  gray = np.mean(image, axis=-1, keepdims=True)
  inflection = 0.9
  mask = (np.maximum(gray - inflection, 0.0) / (1.0 - inflection)) ** 2.0
  safe_gains = np.maximum(mask + (1.0 - mask) * gains, gains)
  return image * safe_gains


def mosaic(image):
  """Extracts RGGB Bayer planes from an RGB image."""
  assert image.shape[2] == 3, "Image must have 3 channels"
  h, w = image.shape[0], image.shape[1]
  
  red = image[0::2, 0::2, 0]
  green_red = image[0::2, 1::2, 1]
  green_blue = image[1::2, 0::2, 1]
  blue = image[1::2, 1::2, 2]
  
  bayer_4ch = np.stack([red, green_red, green_blue, blue], axis=-1)
  return bayer_4ch  # [H/2, W/2, 4]


def random_ccm():
  """Generates random RGB -> Camera color correction matrices."""
  xyz2cams = np.array([
      [[1.0234, -0.2969, -0.2266],
       [-0.5625, 1.6328, -0.0469],
       [-0.0703, 0.2188, 0.6406]],
      [[0.4913, -0.0541, -0.0202],
       [-0.613, 1.3513, 0.2906],
       [-0.1564, 0.2151, 0.7183]],
      [[0.838, -0.263, -0.0639],
       [-0.2887, 1.0725, 0.2496],
       [-0.0627, 0.1427, 0.5438]],
      [[0.6596, -0.2079, -0.0562],
       [-0.4782, 1.3016, 0.1933],
       [-0.097, 0.1581, 0.5181]]
  ])
  
  num_ccms = len(xyz2cams)
  weights = np.random.uniform(1e-8, 1e8, (num_ccms, 1, 1))
  weights_sum = np.sum(weights, axis=0)
  xyz2cam = np.sum(xyz2cams * weights, axis=0) / weights_sum

  # Multiplies with RGB -> XYZ to get RGB -> Camera CCM.
  rgb2xyz = np.array([[0.4124564, 0.3575761, 0.1804375],
                      [0.2126729, 0.7151522, 0.0721750],
                      [0.0193339, 0.1191920, 0.9503041]])
  rgb2cam = np.dot(xyz2cam, rgb2xyz)

  # Normalizes each row.
  rgb2cam = rgb2cam / np.sum(rgb2cam, axis=-1, keepdims=True)
  return rgb2cam


def random_gains():
  """Generates random gains for brightening and white balance."""
  rgb_gain = 1.0 / np.random.normal(0.8, 0.1)
  red_gain = np.random.uniform(1.9, 2.4)
  blue_gain = np.random.uniform(1.5, 1.9)
  return rgb_gain, red_gain, blue_gain


def bayer_4ch_to_raw_hw_bilinear(bayer_4ch):
  """使用双线性插值的demosaic，将 [H/2, W/2, 4] 转换为 [H, W, 1]。"""
  h, w = bayer_4ch.shape[0], bayer_4ch.shape[1]
  h_out, w_out = h * 2, w * 2
  
  red = bayer_4ch[:, :, 0]
  green_red = bayer_4ch[:, :, 1]
  green_blue = bayer_4ch[:, :, 2]
  blue = bayer_4ch[:, :, 3]
  green = (green_red + green_blue) / 2.0
  
  # 创建坐标
  x_in = np.arange(w) * 2 + 0.5
  y_in = np.arange(h) * 2 + 0.5
  x_out = np.arange(w_out)
  y_out = np.arange(h_out)
  
  # 双线性插值
  try:
    f_r = interp2d(x_in, y_in, red, kind='linear', fill_value=0)
    f_g = interp2d(x_in, y_in, green, kind='linear', fill_value=0)
    f_b = interp2d(x_in, y_in, blue, kind='linear', fill_value=0)
    
    red_interp = f_r(x_out, y_out)
    green_interp = f_g(x_out, y_out)
    blue_interp = f_b(x_out, y_out)
  except:
    # 如果插值失败，回退到简单重复
    red_interp = np.repeat(np.repeat(red, 2, axis=0), 2, axis=1)
    green_interp = np.repeat(np.repeat(green, 2, axis=0), 2, axis=1)
    blue_interp = np.repeat(np.repeat(blue, 2, axis=0), 2, axis=1)
  
  rgb = np.stack([red_interp, green_interp, blue_interp], axis=-1)
  raw = np.mean(rgb, axis=2, keepdims=True).astype(np.float32)
  
  return raw


def rgb_to_bayer_4ch(image, use_random_params=False, rgb2cam=None, rgb_gain=1.0, 
                     red_gain=2.0, blue_gain=1.9):
  """将RGB图像转换为 [H/2, W/2, 4] 的Bayer pattern（4个通道分开）。
  
  Args:
    image: RGB图像 [H, W, 3]，值域0-1
    use_random_params: 是否使用随机参数，如果True则忽略其他参数
    rgb2cam: 颜色校正矩阵，如果None则使用默认值
    rgb_gain: RGB增益
    red_gain: 红色增益
    blue_gain: 蓝色增益
    
  Returns:
    bayer_4ch: Bayer pattern格式 [H/2, W/2, 4] (R, G_red, G_blue, B)
  """
  image = image.astype(np.float32)
  
  if use_random_params:
    rgb2cam = random_ccm()
    rgb_gain, red_gain, blue_gain = random_gains()
  else:
    if rgb2cam is None:
      rgb2cam = np.array([[0.4913, -0.0541, -0.0202],
                          [-0.613, 1.3513, 0.2906],
                          [-0.1564, 0.2151, 0.7183]], dtype=np.float32)
  
  # 反向处理流程
  processed = inverse_smoothstep(image)
  processed = gamma_expansion(processed)
  processed = apply_ccm(processed, rgb2cam)
  processed = safe_invert_gains(processed, rgb_gain, red_gain, blue_gain)
  
  processed = np.clip(processed, 0.0, 1.0)
  
  # 应用Bayer马赛克
  bayer_4ch = mosaic(processed)  # [H/2, W/2, 4]
  
  return bayer_4ch.astype(np.float32)


def rgb_to_bayer(image, use_random_params=False, rgb2cam=None, rgb_gain=1.0, 
                 red_gain=2.0, blue_gain=1.9, output_format='hw1'):
  """将RGB图像转换为Bayer pattern格式。
  
  Args:
    image: RGB图像 [H, W, 3]，值域0-1
    use_random_params: 是否使用随机参数，如果True则忽略其他参数
    rgb2cam: 颜色校正矩阵，如果None则使用默认值
    rgb_gain: RGB增益
    red_gain: 红色增益
    blue_gain: 蓝色增益
    output_format: 输出格式
      - 'hw1': 返回 [H, W, 1] 的重建raw图像（使用demosaic）
      - '4ch': 返回 [H/2, W/2, 4] 的Bayer pattern（4个通道分开）
    
  Returns:
    bayer_image: 根据output_format返回相应格式
      - 'hw1': [H, W, 1] 的raw图像
      - '4ch': [H/2, W/2, 4] 的Bayer pattern
  """
  # 获取Bayer 4通道格式
  bayer_4ch = rgb_to_bayer_4ch(image, use_random_params, rgb2cam, 
                                rgb_gain, red_gain, blue_gain)
  
  # 根据输出格式进行处理
  if output_format == '4ch':
    return bayer_4ch
  elif output_format == 'hw1':
    # 使用demosaic转换为[H, W, 1]
    bayer_hw1 = bayer_4ch_to_raw_hw_bilinear(bayer_4ch)
    bayer_hw1 = np.clip(bayer_hw1 * (1.0 / 0.56), 0.0, 1.0)
    return bayer_hw1
  else:
    raise ValueError("output_format must be '4ch' or 'hw1'")


def add_noise(image, shot_noise=0.01, read_noise=0.0005):
  """Adds random shot (proportional to image) and read (independent) noise."""
  variance = image * shot_noise + read_noise
  noise = np.random.normal(0, 1, image.shape) * np.sqrt(variance)
  return image + noise


def visualize_images(image=None, bayer_pattern=None, output_path='visualize.png'):
  """Visualizes RGB image and Bayer pattern image.
  
  Args:
    image: RGB image [H, W, 3], value range 0-1 or 0-255
    bayer_pattern: Bayer pattern image [H, W, 1]
    output_path: Output file path
  """
  import matplotlib.pyplot as plt
  from matplotlib.gridspec import GridSpec
  
  # Process inputs
  if image is not None:
    image_np = image.copy()
    if isinstance(image_np, np.ndarray):
      max_val = image_np.max()
    else:
      raise ValueError("only support numpy array input for image")
    if max_val > 1.1:
      image_np = image_np / 255.0
  else:
    image_np = None
  
  if bayer_pattern is not None:
    bayer_np = np.squeeze(bayer_pattern)  # [H, W]
    # Convert to 0-1 range
    if bayer_np.max() > 1.1:
      bayer_np = bayer_np / 255.0
  else:
    bayer_np = None
  
  # Create figure
  if image_np is not None and bayer_np is not None:
    plt.switch_backend('Agg')
    fig = plt.figure(figsize=(14, 6))
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)
    
    # Display RGB image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(np.clip(image_np, 0, 1))
    ax1.set_title('RGB Image')
    ax1.axis('off')
    
    # Display Bayer pattern
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(np.clip(bayer_np, 0, 1), cmap='gray')
    ax2.set_title('Bayer Pattern (Raw)')
    ax2.axis('off')
    
  elif image_np is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.clip(image_np, 0, 1))
    ax.set_title('RGB Image')
    ax.axis('off')
    
  elif bayer_np is not None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(np.clip(bayer_np, 0, 1), cmap='gray')
    ax.set_title('Bayer Pattern (Raw)')
    ax.axis('off')
  else:
    print("Error: At least one image must be provided")
    return
  
  plt.tight_layout()
  plt.savefig(output_path, dpi=100, bbox_inches='tight')
  print("Image saved to: {}".format(output_path))
  plt.close()

class SignificantTokensStats:
  """全局类，统计所有帧中显著token检测的信息"""
  
  # 输出文件名常量
  STATS_OUTPUT_PATH = "significant_tokens_stats.png"
  
  def __init__(self):
    """初始化统计类"""
    self.frames_data = []  # 存储每一帧的数据
    self.frame_count = 0
  
  def record(self, diff, patch_counts, significant_indices, t, k, num_patches_h, num_patches_w):
    """记录一帧的显著token检测信息
    
    Args:
      diff: 像素差异数组 [H, W]
      patch_counts: 每个patch的变化像素数 [num_patches_h, num_patches_w]
      significant_indices: 显著patch的索引 numpy.ndarray
      t: 像素差值阈值
      k: 显著性阈值
      num_patches_h: patch网格高度
      num_patches_w: patch网格宽度
    """
    self.frame_count += 1
    total_patches = num_patches_h * num_patches_w
    
    # 计算当前帧的统计信息
    diff_min = float(diff.min())
    diff_max = float(diff.max())
    diff_mean = float(diff.mean())
    diff_std = float(diff.std())
    
    changed_ratio = float((diff > t).mean())
    
    patch_counts_mean = float(patch_counts.mean())
    patch_counts_std = float(patch_counts.std())
    patch_counts_min = int(patch_counts.min())
    patch_counts_max = int(patch_counts.max())
    
    significant_count = len(significant_indices)
    significant_ratio = significant_count / total_patches
    
    # 存储数据
    frame_data = {
      'frame': self.frame_count,
      'diff_min': diff_min,
      'diff_max': diff_max,
      'diff_mean': diff_mean,
      'diff_std': diff_std,
      'changed_ratio': changed_ratio,
      'patch_counts_mean': patch_counts_mean,
      'patch_counts_std': patch_counts_std,
      'patch_counts_min': patch_counts_min,
      'patch_counts_max': patch_counts_max,
      'significant_count': significant_count,
      'significant_ratio': significant_ratio,
      'total_patches': total_patches,
      't': t,
      'k': k,
    }
    self.frames_data.append(frame_data)
    
    # 实时打印当前帧信息
    # self._print_current_frame(frame_data)
  
  def _print_current_frame(self, frame_data):
    """打印当前帧的信息"""
    import sys
    stats_info = (
      f"\r【显著token检测】Frame {frame_data['frame']:3d} | "
      f"diff: [{frame_data['diff_min']:.4f}, {frame_data['diff_max']:.4f}] "
      f"μ={frame_data['diff_mean']:.4f} σ={frame_data['diff_std']:.4f} | "
      f"changed%: {100*frame_data['changed_ratio']:.1f}% (t={frame_data['t']}) | "
      f"patch_counts: μ={frame_data['patch_counts_mean']:.1f} σ={frame_data['patch_counts_std']:.1f} "
      f"[{frame_data['patch_counts_min']}, {frame_data['patch_counts_max']}] | "
      f"significant: {frame_data['significant_count']}/{frame_data['total_patches']} "
      f"({100*frame_data['significant_ratio']:.1f}%) (k={frame_data['k']})"
    )
    sys.stderr.write(stats_info)
    sys.stderr.flush()
  
  def get_statistics(self):
    """获取累计统计信息"""
    if len(self.frames_data) == 0:
      return None
    
    # 将列表转换为字典的字典，然后提取各个字段
    diff_means = np.array([d['diff_mean'] for d in self.frames_data])
    diff_stds = np.array([d['diff_std'] for d in self.frames_data])
    changed_ratios = np.array([d['changed_ratio'] for d in self.frames_data])
    patch_counts_means = np.array([d['patch_counts_mean'] for d in self.frames_data])
    patch_counts_stds = np.array([d['patch_counts_std'] for d in self.frames_data])
    significant_ratios = np.array([d['significant_ratio'] for d in self.frames_data])
    
    stats = {
      'total_frames': len(self.frames_data),
      'diff_mean_avg': float(diff_means.mean()),
      'diff_mean_std': float(diff_means.std()),
      'diff_std_avg': float(diff_stds.mean()),
      'changed_ratio_avg': float(changed_ratios.mean()),
      'changed_ratio_std': float(changed_ratios.std()),
      'patch_counts_mean_avg': float(patch_counts_means.mean()),
      'patch_counts_mean_std': float(patch_counts_means.std()),
      'patch_counts_std_avg': float(patch_counts_stds.mean()),
      'significant_ratio_avg': float(significant_ratios.mean()),
      'significant_ratio_std': float(significant_ratios.std()),
    }
    return stats
  
  def plot_statistics(self, output_path=None):
    """绘制累计统计信息"""
    if output_path is None:
      output_path = self.STATS_OUTPUT_PATH
    
    if len(self.frames_data) == 0:
      print("No data to plot")
      return
    
    import matplotlib.pyplot as plt
    
    # 提取数据
    frames = [d['frame'] for d in self.frames_data]
    diff_means = [d['diff_mean'] for d in self.frames_data]
    diff_stds = [d['diff_std'] for d in self.frames_data]
    changed_ratios = [d['changed_ratio'] * 100 for d in self.frames_data]
    patch_counts_means = [d['patch_counts_mean'] for d in self.frames_data]
    patch_counts_stds = [d['patch_counts_std'] for d in self.frames_data]
    significant_ratios = [d['significant_ratio'] * 100 for d in self.frames_data]
    
    # 获取平均值
    stats = self.get_statistics()
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'Significant Tokens Statistics (Total Frames: {len(self.frames_data)})', fontsize=16)
    
    # 1. diff mean over frames
    axes[0, 0].plot(frames, diff_means, 'b-', linewidth=2)
    axes[0, 0].axhline(y=stats['diff_mean_avg'], color='r', linestyle='--', label=f"Avg: {stats['diff_mean_avg']:.4f}")
    axes[0, 0].set_xlabel('Frame')
    axes[0, 0].set_ylabel('Mean Diff')
    axes[0, 0].set_title('Pixel Difference Mean')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. diff std over frames
    axes[0, 1].plot(frames, diff_stds, 'g-', linewidth=2)
    axes[0, 1].axhline(y=stats['diff_std_avg'], color='r', linestyle='--', label=f"Avg: {stats['diff_std_avg']:.4f}")
    axes[0, 1].set_xlabel('Frame')
    axes[0, 1].set_ylabel('Std Diff')
    axes[0, 1].set_title('Pixel Difference Std')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. changed ratio over frames
    axes[0, 2].plot(frames, changed_ratios, 'c-', linewidth=2)
    axes[0, 2].axhline(y=stats['changed_ratio_avg']*100, color='r', linestyle='--', 
                       label=f"Avg: {stats['changed_ratio_avg']*100:.1f}%")
    axes[0, 2].set_xlabel('Frame')
    axes[0, 2].set_ylabel('Changed Ratio (%)')
    axes[0, 2].set_title('Pixel Change Ratio')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. patch counts mean over frames
    axes[1, 0].plot(frames, patch_counts_means, 'm-', linewidth=2)
    axes[1, 0].axhline(y=stats['patch_counts_mean_avg'], color='r', linestyle='--', 
                       label=f"Avg: {stats['patch_counts_mean_avg']:.1f}")
    axes[1, 0].set_xlabel('Frame')
    axes[1, 0].set_ylabel('Mean Patch Count')
    axes[1, 0].set_title('Patch Counts Mean')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. patch counts std over frames
    axes[1, 1].plot(frames, patch_counts_stds, 'orange', linewidth=2)
    axes[1, 1].axhline(y=stats['patch_counts_std_avg'], color='r', linestyle='--', 
                       label=f"Avg: {stats['patch_counts_std_avg']:.1f}")
    axes[1, 1].set_xlabel('Frame')
    axes[1, 1].set_ylabel('Std Patch Count')
    axes[1, 1].set_title('Patch Counts Std')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. significant ratio over frames
    axes[1, 2].plot(frames, significant_ratios, 'r-', linewidth=2)
    axes[1, 2].axhline(y=stats['significant_ratio_avg']*100, color='b', linestyle='--', 
                       label=f"Avg: {stats['significant_ratio_avg']*100:.1f}%")
    axes[1, 2].set_xlabel('Frame')
    axes[1, 2].set_ylabel('Significant Ratio (%)')
    axes[1, 2].set_title('Significant Token Ratio')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    print(f"\nStatistics saved to: {output_path}")
    plt.close()
    
    # 打印统计摘要
    self._print_summary(stats)
    return stats['significant_ratio_avg']*100
  
  def _print_summary(self, stats):
    """打印统计摘要"""
    print("\n" + "="*80)
    print("SIGNIFICANT TOKENS STATISTICS SUMMARY")
    print("="*80)
    print(f"Total Frames: {stats['total_frames']}")
    print(f"\nPixel Difference (diff):")
    print(f"  Mean: {stats['diff_mean_avg']:.4f} ± {stats['diff_mean_std']:.4f}")
    print(f"  Std:  {stats['diff_std_avg']:.4f}")
    print(f"\nPixel Change Ratio:")
    print(f"  Average: {stats['changed_ratio_avg']*100:.1f}% ± {stats['changed_ratio_std']*100:.1f}%")
    print(f"\nPatch Counts:")
    print(f"  Mean: {stats['patch_counts_mean_avg']:.1f} ± {stats['patch_counts_mean_std']:.1f}")
    print(f"  Std:  {stats['patch_counts_std_avg']:.1f}")
    print(f"\nSignificant Token Ratio:")
    print(f"  Average: {stats['significant_ratio_avg']*100:.1f}% ± {stats['significant_ratio_std']*100:.1f}%")
    print("="*80 + "\n")
  
  def reset(self):
    """重置统计数据"""
    self.frames_data = []
    self.frame_count = 0


# 创建全局实例
_significant_tokens_stats = SignificantTokensStats()

#672 k=60 tau=0.004还行
#1024 k=80, t=0.003还行 
def get_significant_tokens(bayer_image, last_bayer_image, input_size_chw, patch_size,  k=80 ,t=0.005, device='cpu',output_kt=False):
    """根据Bayer图像的变化，确定显著的token索引。
    
    Args:
      bayer_image: 当前帧Bayer图像 [H, W, 1]
      last_bayer_image: 上一帧Bayer图像 [H, W, 1]，如果为None则返回所有token索引
      input_size_chw: 输入图像大小 [C, H, W]，例如 [3, 672, 672]
      patch_size: patch大小 [patch_h, patch_w]，例如 [16, 16]
      k: 显著性阈值，变化像素数超过k则认为token显著
      t: 像素差值阈值，差值超过t则认为像素变化较大
      device: 设备类型，与传入图像一致
      
    Returns:
      keep_index: torch.Tensor of shape (1, n)，包含所有显著token的索引，device与输入一致
    """
    if output_kt:
      return k,t
    # 获取图像尺寸
    _, h, w = input_size_chw
    patch_h, patch_w = patch_size
    
    # 计算patch数量
    num_patches_h = h // patch_h
    num_patches_w = w // patch_w
    
    # 如果last_bayer_image为None，返回所有token的索引
    if last_bayer_image is None:
      indices = np.arange(num_patches_h * num_patches_w, dtype=np.int32)
      return torch.from_numpy(indices).unsqueeze(0).to(device)  # (1, n)
    
    # 检查k的有效性
    max_pixels_per_patch = patch_size[0] * patch_size[1]
    if k < 0 or k > max_pixels_per_patch:
      raise ValueError(f"k must be in range [0, {max_pixels_per_patch}], got {k}")
    
    # 确保bayer_image形状正确
    bayer_image = np.squeeze(bayer_image)  # [H, W]
    last_bayer_image = np.squeeze(last_bayer_image)  # [H, W]
    
    # 计算差异
    diff = np.abs(bayer_image - last_bayer_image)
    
    # 标记变化像素
    changed_pixels = (diff > t).astype(np.int32)
    
    # 将图像reshape为patch网格形式 
    # [H, W] -> [num_patches_h, patch_h, num_patches_w, patch_w]
    patches = changed_pixels[:num_patches_h*patch_h, :num_patches_w*patch_w].reshape(
        num_patches_h, patch_h, num_patches_w, patch_w)
    
    # 对每个patch求和，得到 [num_patches_h, num_patches_w]
    patch_counts = patches.sum(axis=(1, 3))
    
    # 找出超过阈值k的patch
    significant_mask = patch_counts > k
    
    # 获取显著patch的索引
    significant_h, significant_w = np.where(significant_mask)
    significant_indices = significant_h * num_patches_w + significant_w
    

    # 记录统计信息
    _significant_tokens_stats.record(diff, patch_counts, significant_indices, t, k, num_patches_h, num_patches_w)
    
    # If no significant indices found, use the patch with maximum count
    if len(significant_indices) == 0:
      max_idx = np.argmax(patch_counts)
      significant_indices = np.array([max_idx], dtype=np.int32)
      print("\nWarning: No significant patches found. Using patch with maximum change count instead.")
    else:
      print()  # 换行
    
    # 转换为torch tensor，形状为(1, n)，并移到指定device
    return torch.from_numpy(significant_indices.astype(np.int64)).unsqueeze(0).to(device)
def split_image_into_patches(image_hwc, patch_size):
  """将HWC格式的图像分割成patch块。
  
  Args:
    image_hwc: 输入图像 numpy.ndarray [H, W, C]
    patch_size: patch大小 [patch_h, patch_w]，例如 [16, 16]
    
  Returns:
    patches: numpy.ndarray of shape [num_patches_h, num_patches_w, patch_h, patch_w, C]
             即 [num_patches_h * num_patches_w, patch_h, patch_w, C] 如果拉平的话
    patch_grid_shape: tuple (num_patches_h, num_patches_w) patch网格的形状
  """
  # 确保输入是numpy数组，并转换为HWC格式
  if isinstance(image_hwc, torch.Tensor):
    image_hwc = image_hwc.detach().cpu().numpy()
  
  if not isinstance(image_hwc, np.ndarray):
    raise TypeError("image_hwc must be numpy.ndarray or torch.Tensor")
  
  # 如果是4D张量 [B, C, H, W]，取第一个batch并转换为 [H, W, C]
  if image_hwc.ndim == 4:
    image_hwc = image_hwc[0].transpose(1, 2, 0)  # [B, C, H, W] -> [H, W, C]
  # 如果是3D张量 [C, H, W]，转换为 [H, W, C]
  elif image_hwc.ndim == 3 and image_hwc.shape[0] in [1, 3]:
    image_hwc = image_hwc.transpose(1, 2, 0)  # [C, H, W] -> [H, W, C]
  
  h, w, c = image_hwc.shape
  patch_h, patch_w = patch_size
  
  # 检查图像是否能被patch_size整除
  if h % patch_h != 0 or w % patch_w != 0:
    raise ValueError(f"Image size [{h}, {w}] must be divisible by patch_size [{patch_h}, {patch_w}]")
  
  # 计算patch数量
  num_patches_h = h // patch_h
  num_patches_w = w // patch_w
  
  # 方法1: 返回4D数组 [num_patches_h, num_patches_w, patch_h, patch_w, C]
  patches = image_hwc.reshape(
      num_patches_h, patch_h,
      num_patches_w, patch_w,
      c
  ).transpose(0, 2, 1, 3, 4)  # [num_patches_h, num_patches_w, patch_h, patch_w, C]
  
  # 方法2（可选）: 返回2D数组 [num_patches_h * num_patches_w, patch_h, patch_w, C]
  patches_flat = patches.reshape(-1, patch_h, patch_w, c)
  
  return patches_flat, (num_patches_h, num_patches_w)

def combine_patches_to_image(patches, patch_grid_shape, patch_size):
  """将patch块重新组合成完整图像（分割函数的反函数）。
  
  Args:
    patches: numpy.ndarray [num_patches_h * num_patches_w, patch_h, patch_w, C]
    patch_grid_shape: tuple (num_patches_h, num_patches_w)
    patch_size: [patch_h, patch_w]
    
  Returns:
    image_hwc: numpy.ndarray [H, W, C]
  """
  num_patches_h, num_patches_w = patch_grid_shape
  patch_h, patch_w = patch_size
  c = patches.shape[-1]
  
  # 先reshape为 [num_patches_h, num_patches_w, patch_h, patch_w, C]
  patches = patches.reshape(num_patches_h, num_patches_w, patch_h, patch_w, c)
  
  # 转换为 [num_patches_h, patch_h, num_patches_w, patch_w, C]
  image = patches.transpose(0, 2, 1, 3, 4)
  
  # reshape为 [H, W, C]
  image = image.reshape(num_patches_h * patch_h, num_patches_w * patch_w, c)
  
  return image

def reuse_non_significant_tokens(image, last_image, keep_index, patch_size):
  """将image中不在keep_index中的patch替换成last_image。
  
  Args:
    image: 当前帧图像 torch.Tensor [B, C, H, W]，例如 [1, 3, 672, 672]
    last_image: 上一帧图像 torch.Tensor [B, C, H, W]，如果为None则直接返回image
    keep_index: 需要保留的patch索引 torch.Tensor [1, n]，例如 [[0, 5, 10, ...]]
    patch_size: patch大小 [patch_h, patch_w]，例如 [16, 16]
    
  Returns:
    output_image: 复用后的图像 torch.Tensor [B, C, H, W]
  """
  
  # 如果last_image为None，直接返回image
  if last_image is None:
    return image
  
  # 获取图像尺寸
  b, c, h, w = image.shape
  patch_h, patch_w = patch_size
  
  # 计算patch数量
  num_patches_h = h // patch_h
  num_patches_w = w // patch_w
  total_patches = num_patches_h * num_patches_w
  
  # 创建mask：True表示需要替换，False表示保留
  replace_mask = torch.ones(total_patches, dtype=torch.bool, device=image.device)
  # 将keep_index展平为1D
  keep_index_flat = keep_index.view(-1)
  replace_mask[keep_index_flat] = False
  
  # 转换mask为 [num_patches_h, num_patches_w]
  replace_mask = replace_mask.reshape(num_patches_h, num_patches_w)
  
  # 扩展mask到 [B, C, H, W]
  replace_mask = replace_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, num_patches_h, num_patches_w]
  replace_mask = replace_mask.repeat_interleave(patch_h, dim=2).repeat_interleave(patch_w, dim=3)  # [1, 1, H, W]
  replace_mask = replace_mask.repeat(b, c, 1, 1)  # [B, C, H, W]
  
  # 使用mask进行替换
  output_image = torch.where(replace_mask, last_image, image)
  
  return output_image


def fast_mask_visulize(x:torch.Tensor=None, mask_indeces:torch.Tensor=None, input_shape=(3, 672, 672), embed_patch_size=(16, 16), save_path="debug_visualize.png", alpha=0.5):
    """
    Visualize the mask indices on the image by overlaying white rectangles.
    
    Args:
        x (torch.Tensor): Input image tensor of shape (batch, channel, height, width) or (channel, height, width)
                          If None, a black image will be created.
        mask_indeces (torch.Tensor): 1D tensor containing indices of patches to highlight
        input_shape (tuple): Expected shape of the input image (channel, height, width)
        embed_patch_size (tuple): Size of each patch (height, width)
        save_path (str): Path to save the visualization
        alpha (float): Opacity of the white overlay (0.0 to 1.0)
    
    Returns:
        numpy.ndarray: Visualization image with overlaid patches
    """
    h, w = input_shape[-2:]
    
    if x is None:
        # Create a black image if x is None
        image = np.zeros((h, w, 3), dtype=np.uint8)
    else:
        # Process existing image
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).float()
        x = pad_to_size(x, input_shape[-2:])
        
        # Convert to numpy for visualization
        if isinstance(x, torch.Tensor):
            if x.dim() == 4:  # (batch, channel, height, width)
                x = x[0]  # Take the first image in batch
            x = x.detach().cpu().numpy()
        
        # Make sure the image is in (C, H, W) format
        # assert x.shape[0] == input_shape[0], f"Expected {input_shape[0]} channels, got {x.shape[0]}"
        
        # Transpose to (H, W, C) for visualization
        image = np.transpose(x, (1, 2, 0))
        
        # Normalize to [0, 255] if needed
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # Convert to RGB if needed
        if image.shape[2] == 1:  # Grayscale or single channel
            image = np.repeat(image, 3, axis=2)
    
    # Calculate patch dimensions
    patch_h, patch_w = embed_patch_size
    num_patches_h = h // patch_h
    num_patches_w = w // patch_w
    total_patches = num_patches_h * num_patches_w
    
    # Create a mask overlay
    mask = np.zeros((h, w), dtype=np.uint8)
    
    if mask_indeces is not None:
        # Convert tensor to numpy if needed
        if isinstance(mask_indeces, torch.Tensor):
            mask_indeces=mask_indeces.view(-1)
            mask_indeces = mask_indeces.detach().cpu().numpy()
            
        
        # Fill the mask for each patch index
        for idx in mask_indeces:
            if idx < 0 or idx >= total_patches:
                continue  # Skip invalid indices
                
            # Convert flat index to 2D coordinates
            patch_y = idx // num_patches_w
            patch_x = idx % num_patches_w
            
            # Fill the corresponding rectangle in the mask
            y_start = patch_y * patch_h
            y_end = min((patch_y + 1) * patch_h, h)
            x_start = patch_x * patch_w
            x_end = min((patch_x + 1) * patch_w, w)
            
            mask[y_start:y_end, x_start:x_end] = 1
    
    # If x is None, directly create a binary visualization (black background, white patches)
    if x is None:
        result = np.zeros((h, w, 3), dtype=np.uint8)
        mask_3d = np.stack([mask] * 3, axis=2)
        result[mask_3d == 1] = 255
    else:
        # Create white overlay with alpha blending for regular image
        white_overlay = np.ones_like(image) * 255
        mask_3d = np.stack([mask] * 3, axis=2)
        blended = cv2.addWeighted(image, 1.0 - alpha, white_overlay, alpha, 0)
        result = np.where(mask_3d == 1, blended, image).astype(np.uint8)
    
    # Save the visualization
    cv2.imwrite(save_path, cv2.cvtColor(result, cv2.COLOR_RGB2BGR))

def visualize_detection(frame, result, mask=None, img_size=None, patch_size=None, alpha=0.5, save_path="debug_visualize.png"):
    """
    Visualize detection results on the frame with an optional alpha mask overlay.
    
    Args:
        frame: torch.Tensor image (C, H, W)
        result: dict with keys 'boxes', 'scores', 'labels'
        mask: Optional tensor mask of 0s and 1s for patches
        img_size: Tuple of (H, W) for original image size
        patch_size: Tuple of (H, W) for patch size
        alpha: Opacity for mask overlay (0.0 = no overlay, 1.0 = fully opaque white)
        save_path: Path to save the visualization (default: "debug_visualize.png")
    
    Returns:
        numpy array of visualized image
    """
    # Convert frame to numpy and transpose from (C,H,W) to (H,W,C)
    if isinstance(frame, torch.Tensor):
        frame = frame.cpu().numpy()
        if frame.shape[0] == 3:  # If in CHW format
            frame = np.transpose(frame, (1, 2, 0))
    
    # Normalize if needed
    if frame.max() <= 1.0:
        frame = (frame * 255).astype(np.uint8)
    else:
        frame = frame.astype(np.uint8)
    
    # Convert to BGR for OpenCV
    image = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Move results to CPU and convert to numpy
    boxes = result['boxes'].cpu().numpy()
    scores = result['scores'].cpu().numpy()
    labels = result['labels'].cpu().numpy()
    
    # NFS dataset category names (1-based indexing, convert from 0-based)
    category_names = {
        1: "person",
        2: "aircraft",
        3: "airboard",
        4: "ball",
        5: "face",
        6: "bicycle",
        7: "bird",
        8: "dollar",
        9: "cup",
        10: "animal",
        11: "vehicle",
        12: "drone",
        13: "fish",
        14: "motorcycle",
        15: "bag",
        16: "shuffleboard",
        17: "yoyo"
    }
    
    # Colors for different classes (17 classes)
    colors = plt.cm.hsv(np.linspace(0, 1, 18))[:, :3] * 255
    
    # Draw boxes, scores and labels
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = box.astype(int)
        # Convert from 0-based to 1-based category ID for lookup
        category_id = int(label) + 1
        color = colors[category_id % len(colors)].tolist()
        
        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Create label text
        label_name = category_names.get(category_id, f"class_{category_id}")
        label_text = f"{label_name}: {score:.2f}"
        
        # Put text above the box
        cv2.putText(image, label_text, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Overlay mask if provided (optional)
    if mask is not None and alpha > 0 and img_size is not None and patch_size is not None:
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        if mask.ndim == 3 and mask.shape[0] == 1:  # If mask has shape [1, H, W]
            mask = mask[0]
        
        patches_h = img_size[0] // patch_size[0]
        patches_w = img_size[1] // patch_size[1]
        
        # If mask is in patch space, resize it to match the image dimensions
        if mask.shape[0] == patches_h and mask.shape[1] == patches_w:
            mask_fullsize = np.zeros((img_size[0], img_size[1]), dtype=np.float32)
            for i in range(patches_h):
                for j in range(patches_w):
                    y_start = i * patch_size[0]
                    y_end = (i + 1) * patch_size[0]
                    x_start = j * patch_size[1]
                    x_end = (j + 1) * patch_size[1]
                    mask_fullsize[y_start:y_end, x_start:x_end] = mask[i, j]
            mask = mask_fullsize
        else:
            # Resize mask to match image dimensions
            mask = cv2.resize(mask, (img_size[1], img_size[0]), interpolation=cv2.INTER_NEAREST)
        
        # Create white overlay
        white_overlay = np.ones_like(image) * 255
        
        # Blend image with white overlay using alpha as scalar
        blended = cv2.addWeighted(image, 1.0 - alpha, white_overlay, alpha, 0)
        
        # Apply blended result only where mask is 1
        mask_expanded = np.expand_dims(mask, axis=2).astype(np.uint8)
        image = np.where(mask_expanded == 1, blended, image).astype(np.uint8)
    
    # Convert back to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save the image to the specified path
    cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    
    return rgb_image
def update_patches(image_hwc, last_image_hwc, keep_index, patch_size):
  """将image_hwc中在keep_index里面的patch更新到last_image_hwc里面。
  
  Args:
    image_hwc: 当前帧图像 numpy.ndarray [H, W, C]
    last_image_hwc: 上一帧图像 numpy.ndarray [H, W, C]，如果为None则直接返回image_hwc
    keep_index: 需要更新的patch索引 numpy.ndarray 或 torch.Tensor，例如 [0, 5, 10, ...]
    patch_size: patch大小 [patch_h, patch_w]，例如 [16, 16]
    
  Returns:
    output_image: 更新后的图像 numpy.ndarray [H, W, C]
  """
  
  # 如果last_image_hwc为None，直接返回image
  if last_image_hwc is None:
    return image_hwc.copy() if isinstance(image_hwc, np.ndarray) else image_hwc
  
  # 确保image_hwc是numpy数组
  if not isinstance(image_hwc, np.ndarray):
    raise TypeError("image_hwc must be numpy.ndarray")
  
  # 获取图像尺寸
  h, w, c = image_hwc.shape
  patch_h, patch_w = patch_size
  
  # 计算patch数量
  num_patches_h = h // patch_h
  num_patches_w = w // patch_w
  total_patches = num_patches_h * num_patches_w
  
  # 转换keep_index为numpy数组
  if isinstance(keep_index, torch.Tensor):
    keep_index = keep_index.view(-1).detach().cpu().numpy()
  keep_index = np.asarray(keep_index, dtype=np.int32)
  
  # 验证keep_index有效性
  if len(keep_index) > 0 and (keep_index.min() < 0 or keep_index.max() >= total_patches):
    raise ValueError(f"keep_index out of range [0, {total_patches})")
  
  # 创建mask：True表示需要更新
  update_mask = np.zeros(total_patches, dtype=bool)
  update_mask[keep_index] = True
  
  # 转换mask为 [num_patches_h, num_patches_w]
  update_mask = update_mask.reshape(num_patches_h, num_patches_w)
  
  # 扩展mask到 [H, W, C]
  update_mask = np.repeat(np.repeat(update_mask, patch_h, axis=0), patch_w, axis=1)  # [H, W]
  update_mask = np.stack([update_mask] * c, axis=2)  # [H, W, C]
  
  # 使用mask进行更新
  output_image = np.where(update_mask, image_hwc, last_image_hwc)
  
  return output_image

def print_significant_tokens_stats(diff, patch_counts, significant_indices, t, k, num_patches_h, num_patches_w):
  """打印显著token检测的统计信息。
  
  Args:
    diff: 像素差异数组 [H, W]
    patch_counts: 每个patch的变化像素数 [num_patches_h, num_patches_w]
    significant_indices: 显著patch的索引 numpy.ndarray
    t: 像素差值阈值
    k: 显著性阈值
    num_patches_h: patch网格高度
    num_patches_w: patch网格宽度
  """
  # 计算patch_counts的统计信息
  patch_counts_mean = patch_counts.mean()
  patch_counts_std = patch_counts.std()
  patch_counts_max = patch_counts.max()
  patch_counts_min = patch_counts.min()
  
  # 使用stderr和\r实现覆盖输出
  import sys
  stats_info = (
    f"\r【显著token检测】 "
    f"diff: [{diff.min():.4f}, {diff.max():.4f}] μ={diff.mean():.4f} σ={diff.std():.4f} | "
    f"changed%: {100*(diff > t).mean():.1f}% (t={t}) | "
    f"patch_counts: μ={patch_counts_mean:.1f} σ={patch_counts_std:.1f} [{patch_counts_min}, {patch_counts_max}] | "
    f"significant: {len(significant_indices)}/{num_patches_h * num_patches_w} ({100*len(significant_indices)/(num_patches_h*num_patches_w):.1f}%) (k={k})"
  )
  sys.stderr.write(stats_info)
  sys.stderr.flush()