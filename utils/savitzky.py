import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ----------------------------
# 1. 模拟带毛刺的数据（你可以替换成你自己的数据）
# ----------------------------
np.random.seed(42)  # 为了可重复性
depth = np.linspace(2000, 3500, 500)  # 深度范围：2000–3500 米
true_trend = 0.002 * (depth - 2000) + 15  # 真实趋势：缓慢上升
noise = np.random.normal(0, 0.8, size=depth.shape)  # 高斯噪声
pred_pp = true_trend + noise  # 带毛刺的预测孔隙压力

# ----------------------------
# 2. 应用 Savitzky-Golay 滤波器平滑
# ----------------------------
# 参数说明：
# - window_length: 必须是奇数，代表滑动窗口大小（点数）
# - polyorder: 多项式阶数，通常 2 或 3
# - mode: 边界处理方式（'interp', 'mirror', 'constant' 等）

window_length = 51   # 窗口大小（根据数据密度调整，越大越平滑）
polyorder = 3        # 多项式阶数（必须 < window_length）

smoothed_pp = savgol_filter(pred_pp, window_length=window_length, polyorder=polyorder)

# ----------------------------
# 3. 绘图对比
# ----------------------------
plt.figure(figsize=(8, 6))
plt.plot(depth, pred_pp, color='lightblue', linewidth=1, label='原始 Pred Pp（带毛刺）')
plt.plot(depth, smoothed_pp, color='red', linewidth=2, label='Savitzky-Golay 平滑后')
plt.xlabel('深度 (m)')
plt.ylabel('孔隙压力 (MPa)')
plt.title('Savitzky-Golay 滤波平滑效果')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()