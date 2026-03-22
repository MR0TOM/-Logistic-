"""
植物生长Logistic模型拟合与生长速率分析
Plant Growth Logistic Model Fitting and Growth Rate Analysis

技术栈：Python, NumPy, SciPy, Matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# 1. Logistic模型定义
# ============================================================

def logistic_model(t, W_max, r, t_mid):
    """
    Logistic生长模型
    W(t) = W_max / (1 + exp(-r * (t - t_mid)))
    
    参数:
        t: 时间
        W_max: 最大生物量
        r: 生长速率参数
        t_mid: 生长拐点时间
    """
    return W_max / (1 + np.exp(-r * (t - t_mid)))

def logistic_derivative(t, W_max, r, t_mid):
    """
    Logistic模型的导数（生长速率）
    dW/dt = W_max * r * exp(-r*(t-t_mid)) / (1 + exp(-r*(t-t_mid)))^2
    """
    exp_term = np.exp(-r * (t - t_mid))
    return W_max * r * exp_term / (1 + exp_term)**2

def logistic_residuals(params, t, W):
    """残差函数，用于least_squares"""
    W_max, r, t_mid = params
    return logistic_model(t, W_max, r, t_mid) - W

# ============================================================
# 2. 生成模拟数据
# ============================================================

# 真实参数
W_max_true = 100.0  # 最大生物量
r_true = 0.3        # 生长速率
t_mid_true = 15.0   # 拐点时间

# 生成时间序列
t = np.linspace(0, 30, 50)
np.random.seed(42)

# 生成真实生物量
W_true = logistic_model(t, W_max_true, r_true, t_mid_true)

# 添加测量噪声（相对误差约5%）
noise_level = 0.05
W_noisy = W_true * (1 + noise_level * np.random.randn(len(t)))

print("=" * 70)
print("植物生长Logistic模型拟合与生长速率分析")
print("=" * 70)
print(f"\n真实参数:")
print(f"  W_max (最大生物量) = {W_max_true}")
print(f"  r (生长速率) = {r_true}")
print(f"  t_mid (拐点时间) = {t_mid_true}")
print(f"\n数据点数: {len(t)}")
print(f"噪声水平: {noise_level*100:.1f}%")

# ============================================================
# 3. 非线性拟合 - 使用两种优化算法
# ============================================================

print("\n" + "=" * 70)
print("3. 非线性拟合结果对比")
print("=" * 70)

# 初始猜测值
p0 = [80.0, 0.2, 10.0]

# 3.1 Levenberg-Marquardt (LM) 算法
print("\n【算法1: Levenberg-Marquardt (LM)】")
try:
    popt_lm, pcov_lm = curve_fit(
        logistic_model, t, W_noisy, 
        p0=p0, 
        method='lm',
        maxfev=10000
    )
    W_max_lm, r_lm, t_mid_lm = popt_lm
    
    # 计算标准误差
    perr_lm = np.sqrt(np.diag(pcov_lm))
    
    # 计算R²
    W_pred_lm = logistic_model(t, *popt_lm)
    ss_res_lm = np.sum((W_noisy - W_pred_lm)**2)
    ss_tot_lm = np.sum((W_noisy - np.mean(W_noisy))**2)
    r_squared_lm = 1 - (ss_res_lm / ss_tot_lm)
    
    print(f"  拟合参数:")
    print(f"    W_max = {W_max_lm:.4f} ± {perr_lm[0]:.4f}")
    print(f"    r = {r_lm:.4f} ± {perr_lm[1]:.4f}")
    print(f"    t_mid = {t_mid_lm:.4f} ± {perr_lm[2]:.4f}")
    print(f"  R² = {r_squared_lm:.6f}")
    print(f"  残差平方和 (RSS) = {ss_res_lm:.4f}")
    
except Exception as e:
    print(f"  LM算法失败: {e}")
    popt_lm = None

# 3.2 Trust Region Reflective (TRF) 算法
print("\n【算法2: Trust Region Reflective (TRF)】")
try:
    # 设置参数边界
    bounds = ([0, 0, 0], [200, 2, 50])
    
    popt_trf, pcov_trf = curve_fit(
        logistic_model, t, W_noisy, 
        p0=p0, 
        method='trf',
        bounds=bounds,
        maxfev=10000
    )
    W_max_trf, r_trf, t_mid_trf = popt_trf
    
    # 计算标准误差
    perr_trf = np.sqrt(np.diag(pcov_trf))
    
    # 计算R²
    W_pred_trf = logistic_model(t, *popt_trf)
    ss_res_trf = np.sum((W_noisy - W_pred_trf)**2)
    ss_tot_trf = np.sum((W_noisy - np.mean(W_noisy))**2)
    r_squared_trf = 1 - (ss_res_trf / ss_tot_trf)
    
    print(f"  拟合参数:")
    print(f"    W_max = {W_max_trf:.4f} ± {perr_trf[0]:.4f}")
    print(f"    r = {r_trf:.4f} ± {perr_trf[1]:.4f}")
    print(f"    t_mid = {t_mid_trf:.4f} ± {perr_trf[2]:.4f}")
    print(f"  R² = {r_squared_trf:.6f}")
    print(f"  残差平方和 (RSS) = {ss_res_trf:.4f}")
    
except Exception as e:
    print(f"  TRF算法失败: {e}")
    popt_trf = None

# 选择最优拟合结果
if popt_lm is not None and popt_trf is not None:
    if r_squared_lm >= r_squared_trf:
        best_params = popt_lm
        best_cov = pcov_lm
        best_method = "LM"
        best_r2 = r_squared_lm
    else:
        best_params = popt_trf
        best_cov = pcov_trf
        best_method = "TRF"
        best_r2 = r_squared_trf
elif popt_lm is not None:
    best_params = popt_lm
    best_cov = pcov_lm
    best_method = "LM"
    best_r2 = r_squared_lm
else:
    best_params = popt_trf
    best_cov = pcov_trf
    best_method = "TRF"
    best_r2 = r_squared_trf

W_max_fit, r_fit, t_mid_fit = best_params
print(f"\n【最优拟合结果 ({best_method})】")
print(f"  W_max = {W_max_fit:.4f}")
print(f"  r = {r_fit:.4f}")
print(f"  t_mid = {t_mid_fit:.4f}")

# ============================================================
# 4. 协方差矩阵与参数相关性分析
# ============================================================

print("\n" + "=" * 70)
print("4. 协方差矩阵与参数相关性")
print("=" * 70)

print("\n协方差矩阵:")
print(best_cov)

# 计算相关系数矩阵
correlation_matrix = np.zeros_like(best_cov)
for i in range(3):
    for j in range(3):
        correlation_matrix[i, j] = best_cov[i, j] / np.sqrt(best_cov[i, i] * best_cov[j, j])

print("\n参数相关系数矩阵:")
param_names = ['W_max', 'r', 't_mid']
print(f"{'':>10}", end='')
for name in param_names:
    print(f"{name:>12}", end='')
print()
for i, name in enumerate(param_names):
    print(f"{name:>10}", end='')
    for j in range(3):
        print(f"{correlation_matrix[i, j]:>12.4f}", end='')
    print()

# ============================================================
# 5. 生长速率分析
# ============================================================

print("\n" + "=" * 70)
print("5. 生长速率分析")
print("=" * 70)

# 使用拟合参数计算生长速率
t_fine = np.linspace(0, 30, 500)
growth_rate = logistic_derivative(t_fine, W_max_fit, r_fit, t_mid_fit)

# 找到最大生长速率点
max_rate_idx = np.argmax(growth_rate)
t_max_rate = t_fine[max_rate_idx]
max_rate_value = growth_rate[max_rate_idx]

print(f"\n最大生长速率分析:")
print(f"  最大生长速率 = {max_rate_value:.4f}")
print(f"  最大生长速率时刻 = {t_max_rate:.4f}")
print(f"  拐点时间 t_mid = {t_mid_fit:.4f}")
print(f"  差异 (t_max_rate - t_mid) = {t_max_rate - t_mid_fit:.6f}")

# 理论验证：Logistic模型中，最大生长速率确实发生在 t = t_mid
theoretical_max_rate = W_max_fit * r_fit / 4
print(f"\n理论最大生长速率 (W_max * r / 4) = {theoretical_max_rate:.4f}")
print(f"  计算值与理论值差异 = {abs(max_rate_value - theoretical_max_rate):.6f}")

# ============================================================
# 6. 分段拟合分析
# ============================================================

print("\n" + "=" * 70)
print("6. 分段拟合分析（前期指数 vs 后期饱和阶段）")
print("=" * 70)

# 分割数据：前期（t < t_mid）和后期（t > t_mid）
early_mask = t < t_mid_fit
late_mask = t >= t_mid_fit

t_early = t[early_mask]
W_early = W_noisy[early_mask]
t_late = t[late_mask]
W_late = W_noisy[late_mask]

print(f"\n前期数据点 (t < t_mid): {len(t_early)} 个")
print(f"后期数据点 (t >= t_mid): {len(t_late)} 个")

# 前期拟合
if len(t_early) >= 3:
    try:
        popt_early, _ = curve_fit(logistic_model, t_early, W_early, p0=p0, method='lm')
        W_pred_early = logistic_model(t_early, *popt_early)
        r2_early = 1 - np.sum((W_early - W_pred_early)**2) / np.sum((W_early - np.mean(W_early))**2)
        
        print(f"\n【前期拟合 (指数生长阶段)】")
        print(f"  W_max = {popt_early[0]:.4f}")
        print(f"  r = {popt_early[1]:.4f}")
        print(f"  t_mid = {popt_early[2]:.4f}")
        print(f"  R² = {r2_early:.4f}")
    except Exception as e:
        print(f"前期拟合失败: {e}")
        popt_early = None
else:
    print("前期数据点不足，无法拟合")
    popt_early = None

# 后期拟合
if len(t_late) >= 3:
    try:
        popt_late, _ = curve_fit(logistic_model, t_late, W_late, p0=p0, method='lm')
        W_pred_late = logistic_model(t_late, *popt_late)
        r2_late = 1 - np.sum((W_late - W_pred_late)**2) / np.sum((W_late - np.mean(W_late))**2)
        
        print(f"\n【后期拟合 (饱和阶段)】")
        print(f"  W_max = {popt_late[0]:.4f}")
        print(f"  r = {popt_late[1]:.4f}")
        print(f"  t_mid = {popt_late[2]:.4f}")
        print(f"  R² = {r2_late:.4f}")
    except Exception as e:
        print(f"后期拟合失败: {e}")
        popt_late = None
else:
    print("后期数据点不足，无法拟合")
    popt_late = None

# 参数稳定性比较
if popt_early is not None and popt_late is not None:
    print(f"\n【参数稳定性比较】")
    print(f"{'参数':>10} {'全数据':>12} {'前期':>12} {'后期':>12} {'变异系数%':>12}")
    for i, name in enumerate(['W_max', 'r', 't_mid']):
        values = [best_params[i], popt_early[i], popt_late[i]]
        cv = np.std(values) / np.mean(values) * 100
        print(f"{name:>10} {values[0]:>12.4f} {values[1]:>12.4f} {values[2]:>12.4f} {cv:>12.2f}%")

# ============================================================
# 7. 初始值敏感性测试
# ============================================================

print("\n" + "=" * 70)
print("7. 初始值敏感性测试")
print("=" * 70)

test_initial_values = [
    [50.0, 0.1, 5.0],
    [80.0, 0.2, 10.0],
    [100.0, 0.3, 15.0],
    [120.0, 0.4, 20.0],
    [150.0, 0.5, 25.0],
]

print(f"\n{'初始值 (W_max, r, t_mid)':>30} {'拟合结果 (W_max, r, t_mid)':>35} {'迭代次数':>10}")
print("-" * 90)

for init in test_initial_values:
    try:
        popt_test, _, info, msg, ier = curve_fit(
            logistic_model, t, W_noisy, 
            p0=init, 
            method='lm',
            full_output=True
        )
        print(f"{str(init):>30} {str([f'{p:.3f}' for p in popt_test]):>35} {info['nfev']:>10}")
    except Exception as e:
        print(f"{str(init):>30} {'拟合失败':>35} {'-':>10}")

print("\n结论: Logistic模型拟合对初始值相对稳健")

# ============================================================
# 8. 95%预测区间计算
# ============================================================

print("\n" + "=" * 70)
print("8. 95%预测区间")
print("=" * 70)

def calculate_prediction_interval(t_eval, t_data, W_data, W_max, r, t_mid, pcov, confidence=0.95):
    """
    计算预测区间
    使用Delta方法近似计算
    """
    n = len(t_data)
    p = 3  # 参数个数
    n_eval = len(t_eval)
    
    # 计算雅可比矩阵（数值近似）
    eps = 1e-8
    J = np.zeros((n_eval, p))
    
    for i in range(p):
        params_plus = np.array([W_max, r, t_mid])
        params_minus = np.array([W_max, r, t_mid])
        params_plus[i] += eps
        params_minus[i] -= eps
        
        J[:, i] = (logistic_model(t_eval, *params_plus) - logistic_model(t_eval, *params_minus)) / (2 * eps)
    
    # 预测方差
    W_pred = logistic_model(t_eval, W_max, r, t_mid)
    var_pred = np.sum(J @ pcov * J, axis=1)
    
    # 残差标准差（使用原始数据计算）
    residuals = W_data - logistic_model(t_data, W_max, r, t_mid)
    mse = np.sum(residuals**2) / (n - p)
    
    # t分布临界值
    t_val = t_dist.ppf((1 + confidence) / 2, n - p)
    
    # 预测区间
    margin = t_val * np.sqrt(var_pred + mse)
    
    return W_pred - margin, W_pred + margin

# 计算预测区间
t_pred = np.linspace(0, 30, 100)
lower_bound, upper_bound = calculate_prediction_interval(t_pred, t, W_noisy, W_max_fit, r_fit, t_mid_fit, best_cov)
W_pred_curve = logistic_model(t_pred, W_max_fit, r_fit, t_mid_fit)

print(f"\n在 t = {t_mid_fit:.1f} (拐点) 处的预测区间:")
idx_mid = np.argmin(np.abs(t_pred - t_mid_fit))
print(f"  预测值: {W_pred_curve[idx_mid]:.4f}")
print(f"  95%预测区间: [{lower_bound[idx_mid]:.4f}, {upper_bound[idx_mid]:.4f}]")
print(f"  区间宽度: {upper_bound[idx_mid] - lower_bound[idx_mid]:.4f}")

# ============================================================
# 9. 绘图
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：生物量实测点与Logistic拟合曲线
ax1 = axes[0]
ax1.scatter(t, W_noisy, c='blue', s=50, alpha=0.6, label='实测数据 (含噪声)', zorder=5)
ax1.plot(t_fine, logistic_model(t_fine, W_max_true, r_true, t_mid_true), 
         'g--', linewidth=2, label='真实曲线', zorder=3)
ax1.plot(t_pred, W_pred_curve, 'r-', linewidth=2, label='Logistic拟合', zorder=4)
ax1.fill_between(t_pred, lower_bound, upper_bound, alpha=0.2, color='red', label='95%预测区间')

# 标记拐点
ax1.axvline(x=t_mid_fit, color='orange', linestyle=':', alpha=0.7, label=f'拐点 t={t_mid_fit:.2f}')
ax1.scatter([t_mid_fit], [W_max_fit/2], c='orange', s=100, marker='*', zorder=6)

ax1.set_xlabel('时间 t', fontsize=12)
ax1.set_ylabel('生物量 W', fontsize=12)
ax1.set_title('生物量生长曲线拟合', fontsize=14)
ax1.legend(loc='lower right')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 30)
ax1.set_ylim(0, 120)

# 添加参数文本
param_text = f'拟合参数:\n'
param_text += f'W_max = {W_max_fit:.2f}\n'
param_text += f'r = {r_fit:.3f}\n'
param_text += f't_mid = {t_mid_fit:.2f}\n'
param_text += f'R² = {best_r2:.4f}'
ax1.text(0.02, 0.98, param_text, transform=ax1.transAxes, fontsize=10,
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# 右图：生长速率曲线
ax2 = axes[1]
growth_rate_curve = logistic_derivative(t_fine, W_max_fit, r_fit, t_mid_fit)
ax2.plot(t_fine, growth_rate_curve, 'b-', linewidth=2, label='生长速率 dW/dt')

# 标记最大生长速率点
ax2.scatter([t_max_rate], [max_rate_value], c='red', s=150, marker='*', 
            zorder=5, label=f'最大速率点 ({t_max_rate:.2f}, {max_rate_value:.2f})')
ax2.axvline(x=t_max_rate, color='red', linestyle='--', alpha=0.5)
ax2.axhline(y=max_rate_value, color='red', linestyle='--', alpha=0.5)

# 添加拐点线
ax2.axvline(x=t_mid_fit, color='orange', linestyle=':', alpha=0.7, label=f'拐点 t_mid={t_mid_fit:.2f}')

ax2.set_xlabel('时间 t', fontsize=12)
ax2.set_ylabel('生长速率 dW/dt', fontsize=12)
ax2.set_title('生长速率曲线', fontsize=14)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 30)

# 添加速率分析文本
rate_text = f'生长速率分析:\n'
rate_text += f'最大速率 = {max_rate_value:.3f}\n'
rate_text += f'最大速率时刻 = {t_max_rate:.2f}\n'
rate_text += f'拐点时间 = {t_mid_fit:.2f}\n'
rate_text += f'理论最大速率 = {theoretical_max_rate:.3f}'
ax2.text(0.98, 0.98, rate_text, transform=ax2.transAxes, fontsize=10,
         verticalalignment='top', horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

plt.tight_layout()
plt.savefig('logistic_growth_analysis.png', dpi=150, bbox_inches='tight')
print("\n图表已保存为: logistic_growth_analysis.png")
plt.show()

# ============================================================
# 10. 模型解释与总结
# ============================================================

print("\n" + "=" * 70)
print("10. 模型解释与总结")
print("=" * 70)

print("""
【Logistic生长模型解释】

1. 模型形式: W(t) = W_max / (1 + exp(-r * (t - t_mid)))
   
   - W_max: 环境容纳量，表示植物能达到的最大生物量
   - r: 内禀增长率，反映植物生长潜力
   - t_mid: 生长拐点，此时生长速率最大

2. 生长阶段划分:
   
   前期 (t << t_mid): 近似指数增长
     W(t) ≈ W_max * exp(-r*t_mid) * exp(r*t) / 2
     生长速率随时间增加
   
   拐点 (t = t_mid): 
     W = W_max / 2
     dW/dt = W_max * r / 4 (最大生长速率)
   
   后期 (t >> t_mid): 生长饱和
     W(t) ≈ W_max
     生长速率趋近于0

3. 生物学意义:
   
   - 前期：资源充足，植物快速积累生物量
   - 中期：资源竞争加剧，生长速率达到峰值
   - 后期：资源限制，生长趋于停滞

4. 拟合质量评估:
   
   - R²接近1表示拟合良好
   - 参数标准误差小表示估计可靠
   - 分段拟合参数变异小表示模型稳定

5. 应用价值:
   
   - 预测作物产量
   - 优化收获时间
   - 评估环境对生长的影响
   - 比较不同品种/处理下的生长特性
""")

print("=" * 70)
print("分析完成!")
print("=" * 70)
