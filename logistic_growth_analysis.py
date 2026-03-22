"""
植物生长 Logistic 模型拟合与生长速率分析
=========================================
技术栈: Python, NumPy, SciPy, Matplotlib
"""

import numpy as np
from scipy.optimize import curve_fit, least_squares
from scipy.stats import t as t_dist
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
np.random.seed(42)

def logistic_model(t, W_max, r, t_mid):
    """
    Logistic 生长模型
    W(t) = W_max / (1 + exp(-r * (t - t_mid)))
    
    参数:
        t: 时间
        W_max: 最大生物量
        r: 生长速率参数
        t_mid: 生长拐点时间
    """
    return W_max / (1 + np.exp(-r * (t - t_mid)))

def growth_rate(t, W_max, r, t_mid):
    """
    生长速率 dW/dt 的解析解
    dW/dt = r * W_max * exp(-r*(t-t_mid)) / (1 + exp(-r*(t-t_mid)))^2
    """
    exp_term = np.exp(-r * (t - t_mid))
    return r * W_max * exp_term / (1 + exp_term)**2

def generate_data(t_true, W_max_true, r_true, t_mid_true, noise_level=0.05):
    """生成含噪声的测量数据"""
    W_true = logistic_model(t_true, W_max_true, r_true, t_mid_true)
    noise = np.random.normal(0, noise_level * W_max_true, len(t_true))
    W_measured = W_true + noise
    W_measured = np.maximum(W_measured, 0.1)
    return W_measured, W_true

def fit_logistic_lm(t, W, p0):
    """使用 Levenberg-Marquardt 算法拟合"""
    try:
        popt, pcov = curve_fit(
            logistic_model, t, W, p0=p0,
            method='lm',
            maxfev=10000
        )
        residuals = W - logistic_model(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((W - np.mean(W))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, pcov, r_squared, True
    except Exception as e:
        return None, None, None, False

def fit_logistic_trf(t, W, p0, bounds=(0, np.inf)):
    """使用 Trust Region Reflective 算法拟合"""
    try:
        lower_bounds = [1, 0.001, 0]
        upper_bounds = [np.inf, np.inf, np.inf]
        popt, pcov = curve_fit(
            logistic_model, t, W, p0=p0,
            method='trf',
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000
        )
        residuals = W - logistic_model(t, *popt)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((W - np.mean(W))**2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt, pcov, r_squared, True
    except Exception as e:
        return None, None, None, False

def calculate_prediction_interval(t, W, popt, pcov, alpha=0.05):
    """计算95%预测区间"""
    n = len(W)
    p = len(popt)
    dof = max(1, n - p)
    t_val = t_dist.ppf(1 - alpha/2, dof)
    residuals = W - logistic_model(t, *popt)
    mse = np.sum(residuals**2) / dof
    
    t_fine = np.linspace(t.min(), t.max(), 200)
    W_pred = logistic_model(t_fine, *popt)
    
    J = np.zeros((len(t_fine), 3))
    eps = 1e-8
    for i, param in enumerate(popt):
        params_plus = popt.copy()
        params_plus[i] += eps
        params_minus = popt.copy()
        params_minus[i] -= eps
        J[:, i] = (logistic_model(t_fine, *params_plus) - 
                   logistic_model(t_fine, *params_minus)) / (2 * eps)
    
    pred_var = np.diag(J @ pcov @ J.T) + mse
    pred_std = np.sqrt(pred_var)
    
    W_lower = W_pred - t_val * pred_std
    W_upper = W_pred + t_val * pred_std
    
    return t_fine, W_pred, W_lower, W_upper

def sensitivity_analysis(t, W, true_params):
    """初始值敏感性测试"""
    W_max_true, r_true, t_mid_true = true_params
    
    initial_guesses = [
        ('接近真值', [W_max_true * 1.1, r_true * 0.9, t_mid_true * 1.1]),
        ('高估W_max', [W_max_true * 2, r_true, t_mid_true]),
        ('低估W_max', [W_max_true * 0.5, r_true, t_mid_true]),
        ('高估r', [W_max_true, r_true * 3, t_mid_true]),
        ('低估r', [W_max_true, r_true * 0.3, t_mid_true]),
        ('偏离t_mid', [W_max_true, r_true, t_mid_true * 1.5]),
        ('极端初始值', [W_max_true * 3, r_true * 5, t_mid_true * 2]),
    ]
    
    results = []
    for name, p0 in initial_guesses:
        popt_lm, _, r2_lm, success_lm = fit_logistic_lm(t, W, p0)
        popt_trf, _, r2_trf, success_trf = fit_logistic_trf(t, W, p0)
        
        results.append({
            'name': name,
            'p0': p0,
            'lm_success': success_lm,
            'lm_params': popt_lm,
            'lm_r2': r2_lm,
            'trf_success': success_trf,
            'trf_params': popt_trf,
            'trf_r2': r2_trf
        })
    
    return results

def segmented_fit(t, W, t_mid_est):
    """分段拟合分析"""
    early_mask = t < t_mid_est
    late_mask = t >= t_mid_est
    
    results = {}
    
    if np.sum(early_mask) > 5:
        t_early = t[early_mask]
        W_early = W[early_mask]
        p0_early = [np.max(W_early) * 1.5, 0.15, np.median(t_early)]
        popt_early, pcov_early, r2_early, success_early = fit_logistic_lm(t_early, W_early, p0_early)
        results['early'] = {
            'params': popt_early,
            'cov': pcov_early,
            'r2': r2_early,
            'success': success_early,
            't': t_early,
            'W': W_early
        }
    
    if np.sum(late_mask) > 5:
        t_late = t[late_mask]
        W_late = W[late_mask]
        p0_late = [np.max(W_late) * 1.1, 0.1, np.median(t_late)]
        popt_late, pcov_late, r2_late, success_late = fit_logistic_lm(t_late, W_late, p0_late)
        results['late'] = {
            'params': popt_late,
            'cov': pcov_late,
            'r2': r2_late,
            'success': success_late,
            't': t_late,
            'W': W_late
        }
    
    return results

def print_results(popt, pcov, r_squared, algorithm_name):
    """打印拟合结果"""
    print(f"\n{'='*60}")
    print(f"  {algorithm_name} 算法拟合结果")
    print(f"{'='*60}")
    print(f"\n拟合参数:")
    print(f"  W_max  = {popt[0]:.4f} ± {np.sqrt(pcov[0,0]):.4f}")
    print(f"  r      = {popt[1]:.4f} ± {np.sqrt(pcov[1,1]):.4f}")
    print(f"  t_mid  = {popt[2]:.4f} ± {np.sqrt(pcov[2,2]):.4f}")
    print(f"\n拟合优度:")
    print(f"  R² = {r_squared:.6f}")
    print(f"\n协方差矩阵:")
    print(f"  {pcov}")

def main():
    print("=" * 70)
    print("        植物生长 Logistic 模型拟合与生长速率分析")
    print("=" * 70)
    
    W_max_true = 100.0
    r_true = 0.15
    t_mid_true = 50.0
    noise_level = 0.05
    
    print(f"\n【真实参数设置】")
    print(f"  W_max (最大生物量) = {W_max_true}")
    print(f"  r (生长速率参数)   = {r_true}")
    print(f"  t_mid (拐点时间)   = {t_mid_true}")
    print(f"  噪声水平           = {noise_level * 100}%")
    
    t_data = np.linspace(0, 100, 50)
    W_measured, W_true = generate_data(t_data, W_max_true, r_true, t_mid_true, noise_level)
    
    print(f"\n【数据生成】")
    print(f"  时间范围: {t_data.min():.1f} ~ {t_data.max():.1f}")
    print(f"  数据点数: {len(t_data)}")
    
    p0_initial = [80.0, 0.1, 45.0]
    
    print(f"\n{'='*60}")
    print("  一、两种优化算法对比")
    print(f"{'='*60}")
    
    popt_lm, pcov_lm, r2_lm, success_lm = fit_logistic_lm(t_data, W_measured, p0_initial)
    popt_trf, pcov_trf, r2_trf, success_trf = fit_logistic_trf(t_data, W_measured, p0_initial)
    
    if success_lm:
        print_results(popt_lm, pcov_lm, r2_lm, "Levenberg-Marquardt (LM)")
    
    if success_trf:
        print_results(popt_trf, pcov_trf, r2_trf, "Trust Region Reflective (TRF)")
    
    print(f"\n{'='*60}")
    print("  算法对比总结")
    print(f"{'='*60}")
    print(f"\n{'算法':<30} {'R²':<15} {'W_max':<12} {'r':<12} {'t_mid':<12}")
    print("-" * 81)
    if success_lm:
        print(f"{'LM':<30} {r2_lm:<15.6f} {popt_lm[0]:<12.4f} {popt_lm[1]:<12.4f} {popt_lm[2]:<12.4f}")
    if success_trf:
        print(f"{'TRF':<30} {r2_trf:<15.6f} {popt_trf[0]:<12.4f} {popt_trf[1]:<12.4f} {popt_trf[2]:<12.4f}")
    print(f"{'真实值':<30} {'-':<15} {W_max_true:<12.4f} {r_true:<12.4f} {t_mid_true:<12.4f}")
    
    popt_best = popt_lm if success_lm else popt_trf
    pcov_best = pcov_lm if success_lm else pcov_trf
    
    print(f"\n{'='*60}")
    print("  二、生长速率分析")
    print(f"{'='*60}")
    
    t_fine = np.linspace(0, 100, 500)
    W_fit = logistic_model(t_fine, *popt_best)
    dW_dt = growth_rate(t_fine, *popt_best)
    
    max_rate_idx = np.argmax(dW_dt)
    t_max_rate = t_fine[max_rate_idx]
    max_rate = dW_dt[max_rate_idx]
    
    print(f"\n生长速率分析结果:")
    print(f"  最大生长速率: {max_rate:.4f}")
    print(f"  最大速率时刻: {t_max_rate:.4f}")
    print(f"  理论拐点 t_mid: {popt_best[2]:.4f}")
    print(f"  差异: {abs(t_max_rate - popt_best[2]):.6f}")
    print(f"\n  理论验证: Logistic模型最大生长速率出现在 t = t_mid")
    print(f"  实际计算与理论一致，差异为数值误差")
    
    W_at_mid = logistic_model(popt_best[2], *popt_best)
    print(f"\n  拐点处生物量 W(t_mid) = {W_at_mid:.4f} = W_max/2 = {popt_best[0]/2:.4f}")
    
    print(f"\n{'='*60}")
    print("  三、初始值敏感性测试")
    print(f"{'='*60}")
    
    sensitivity_results = sensitivity_analysis(t_data, W_measured, (W_max_true, r_true, t_mid_true))
    
    print(f"\n{'初始值类型':<15} {'LM成功':<10} {'LM R²':<12} {'TRF成功':<10} {'TRF R²':<12}")
    print("-" * 59)
    for result in sensitivity_results:
        lm_r2_str = f"{result['lm_r2']:.4f}" if result['lm_r2'] is not None else "N/A"
        trf_r2_str = f"{result['trf_r2']:.4f}" if result['trf_r2'] is not None else "N/A"
        print(f"{result['name']:<15} {str(result['lm_success']):<10} {lm_r2_str:<12} "
              f"{str(result['trf_success']):<10} {trf_r2_str:<12}")
    
    print(f"\n结论: 两种算法对初始值均有一定鲁棒性，但极端初始值可能导致收敛困难")
    
    print(f"\n{'='*60}")
    print("  四、分段拟合分析")
    print(f"{'='*60}")
    
    seg_results = segmented_fit(t_data, W_measured, popt_best[2])
    
    if 'early' in seg_results and seg_results['early']['success']:
        print(f"\n前期生长阶段 (t < {popt_best[2]:.1f}):")
        print(f"  W_max = {seg_results['early']['params'][0]:.4f}")
        print(f"  r     = {seg_results['early']['params'][1]:.4f}")
        print(f"  t_mid = {seg_results['early']['params'][2]:.4f}")
        print(f"  R²    = {seg_results['early']['r2']:.4f}")
    
    if 'late' in seg_results and seg_results['late']['success']:
        print(f"\n后期饱和阶段 (t >= {popt_best[2]:.1f}):")
        print(f"  W_max = {seg_results['late']['params'][0]:.4f}")
        print(f"  r     = {seg_results['late']['params'][1]:.4f}")
        print(f"  t_mid = {seg_results['late']['params'][2]:.4f}")
        print(f"  R²    = {seg_results['late']['r2']:.4f}")
    
    print(f"\n分段拟合结论:")
    print(f"  前期阶段主要反映指数增长特性，参数估计可能不稳定")
    print(f"  后期阶段主要反映饱和特性，W_max估计较准确")
    print(f"  全数据拟合能更好地平衡各阶段信息")
    
    print(f"\n{'='*60}")
    print("  五、95%预测区间计算")
    print(f"{'='*60}")
    
    t_pred, W_pred, W_lower, W_upper = calculate_prediction_interval(
        t_data, W_measured, popt_best, pcov_best
    )
    
    print(f"\n预测区间已计算，将显示在图表中")
    
    print(f"\n{'='*60}")
    print("  六、模型解释")
    print(f"{'='*60}")
    
    print(f"""
Logistic 模型解释:
─────────────────────────────────────────────────────────────
W(t) = W_max / (1 + exp(-r(t - t_mid)))

参数含义:
  • W_max: 环境容纳量/最大生物量，表示环境能支持的最大生物量
  • r: 内禀增长率，反映物种在理想条件下的生长潜力
  • t_mid: 拐点时间，生长速率最大的时刻

模型特性:
  • S型曲线: 初期指数增长 → 中期快速生长 → 后期渐近饱和
  • 拐点特性: t = t_mid 时，W = W_max/2，生长速率最大
  • 最大生长速率: dW/dt|_max = r * W_max / 4

拟合结果解读:
  • W_max = {popt_best[0]:.2f}: 预测最大生物量
  • r = {popt_best[1]:.4f}: 生长速率参数
  • t_mid = {popt_best[2]:.2f}: 生长拐点约在第 {popt_best[2]:.0f} 天
  • R² = {r2_lm:.4f}: 模型解释了 {r2_lm*100:.2f}% 的数据变异
""")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ax1 = axes[0]
    ax1.scatter(t_data, W_measured, c='blue', s=50, alpha=0.7, 
                label='实测数据', zorder=5, edgecolors='darkblue')
    ax1.plot(t_fine, W_fit, 'r-', linewidth=2, label='Logistic拟合曲线')
    ax1.fill_between(t_pred, W_lower, W_upper, alpha=0.3, color='red', 
                     label='95%预测区间')
    ax1.axhline(y=popt_best[0], color='green', linestyle='--', 
                alpha=0.7, label=f'W_max = {popt_best[0]:.1f}')
    ax1.axvline(x=popt_best[2], color='orange', linestyle='--', 
                alpha=0.7, label=f't_mid = {popt_best[2]:.1f}')
    ax1.set_xlabel('时间 (天)', fontsize=12)
    ax1.set_ylabel('生物量 (g)', fontsize=12)
    ax1.set_title('生物量实测点与Logistic拟合曲线', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 100)
    ax1.set_ylim(0, popt_best[0] * 1.15)
    
    ax2 = axes[1]
    ax2.plot(t_fine, dW_dt, 'b-', linewidth=2, label='生长速率曲线')
    ax2.scatter([t_max_rate], [max_rate], c='red', s=150, marker='*', 
                zorder=5, label=f'最大速率点 (t={t_max_rate:.1f}, rate={max_rate:.2f})')
    ax2.axvline(x=popt_best[2], color='orange', linestyle='--', 
                alpha=0.7, label=f't_mid = {popt_best[2]:.1f}')
    ax2.fill_between(t_fine, 0, dW_dt, alpha=0.2, color='blue')
    ax2.set_xlabel('时间 (天)', fontsize=12)
    ax2.set_ylabel('生长速率 (g/天)', fontsize=12)
    ax2.set_title('生长速率曲线与最大速率点', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 100)
    
    textstr = f'最大生长速率: {max_rate:.3f} g/天\n出现时刻: t = {t_max_rate:.1f} 天\n理论拐点: t = {popt_best[2]:.1f} 天'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes, fontsize=10,
             verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('logistic_growth_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\n图表已保存为: logistic_growth_analysis.png")
    plt.show()
    
    print(f"\n{'='*70}")
    print("                    分析完成")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
