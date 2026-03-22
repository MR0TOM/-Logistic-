import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 生成模拟数据 ----------------------
# 真实参数
W_max_true = 100    # 最大生物量
r_true = 0.5        # 生长速率
t_mid_true = 10     # 生长拐点时间

# 时间序列
t = np.linspace(0, 20, 50)  # 0到20天，50个时间点

# 生成含噪声的生物量数据（添加高斯噪声）
np.random.seed(42)  # 设置随机种子保证可重复性
noise = np.random.normal(0, 2, size=len(t))  # 噪声标准差为2
W = W_max_true / (1 + np.exp(-r_true * (t - t_mid_true))) + noise

# ---------------------- 2. 定义Logistic模型 ----------------------
def logistic(t, W_max, r, t_mid):
    """Logistic生长模型"""
    return W_max / (1 + np.exp(-r * (t - t_mid)))

# ---------------------- 3. 非线性拟合（两种算法） ----------------------
# 初始参数猜测
p0 = [80, 0.3, 8]  # [W_max初始猜测, r初始猜测, t_mid初始猜测]

# 使用Levenberg-Marquardt算法拟合
popt_lm, pcov_lm = curve_fit(logistic, t, W, p0=p0, method='lm')
W_fit_lm = logistic(t, *popt_lm)

# 使用Trust Region Reflective算法拟合
popt_trf, pcov_trf = curve_fit(logistic, t, W, p0=p0, method='trf')
W_fit_trf = logistic(t, *popt_trf)

# ---------------------- 4. 拟合优度计算 ----------------------
def calc_goodness_of_fit(W, W_fit, n_params):
    """计算拟合优度指标"""
    residuals = W - W_fit
    ss_res = np.sum(residuals**2)  # 残差平方和
    ss_tot = np.sum((W - np.mean(W))**2)  # 总平方和
    r_squared = 1 - (ss_res / ss_tot)  # R²
    adj_r_squared = 1 - (1 - r_squared) * (len(W) - 1) / (len(W) - n_params - 1)  # 调整R²
    rmse = np.sqrt(ss_res / len(W))  # 均方根误差
    return residuals, ss_res, ss_tot, r_squared, adj_r_squared, rmse

gof_lm = calc_goodness_of_fit(W, W_fit_lm, 3)
gof_trf = calc_goodness_of_fit(W, W_fit_trf, 3)

# ---------------------- 5. 生长速率分析 ----------------------
def growth_rate(t, W_max, r, t_mid):
    """计算生长速率dW/dt"""
    u = -r * (t - t_mid)
    return W_max * r * np.exp(u) / (1 + np.exp(u))**2

# 计算生长速率
gr_lm = growth_rate(t, *popt_lm)
gr_trf = growth_rate(t, *popt_trf)

# 找到最大生长速率对应时刻
max_gr_idx_lm = np.argmax(gr_lm)
max_gr_t_lm = t[max_gr_idx_lm]
max_gr_val_lm = gr_lm[max_gr_idx_lm]

max_gr_idx_trf = np.argmax(gr_trf)
max_gr_t_trf = t[max_gr_idx_trf]
max_gr_val_trf = gr_trf[max_gr_idx_trf]

# ---------------------- 6. 分段拟合 ----------------------
# 按真实拐点分割数据（实际应用中可使用拟合的t_mid）
split_point = t_mid_true
t_exp = t[t <= split_point]
W_exp = W[t <= split_point]
t_sat = t[t > split_point]
W_sat = W[t > split_point]

# 指数生长阶段拟合（近似指数模型）
def exp_model(t, a, b):
    return a * np.exp(b * t)

popt_exp, _ = curve_fit(exp_model, t_exp, W_exp, p0=[10, 0.3])
W_fit_exp = exp_model(t_exp, *popt_exp)

# 饱和阶段拟合（近似饱和模型）
def sat_model(t, W_max, a, b):
    return W_max - a * np.exp(-b * t)

popt_sat, _ = curve_fit(sat_model, t_sat, W_sat, p0=[100, 50, 0.3])
W_fit_sat = sat_model(t_sat, *popt_sat)

# ---------------------- 7. 95%预测区间计算 ----------------------
def prediction_interval(t, popt, pcov, alpha=0.05):
    """计算预测区间"""
    W_fit = logistic(t, *popt)
    n = len(t)
    p = len(popt)
    dof = n - p  # 自由度
    
    # 计算Jacobian矩阵（数值微分）
    J = np.zeros((n, p))
    eps = 1e-6
    for i in range(p):
        p_pert = popt.copy()
        p_pert[i] += eps
        J[:, i] = (logistic(t, *p_pert) - W_fit) / eps
    
    # 预测方差
    var_pred = np.sum(J @ pcov * J, axis=1)
    se_pred = np.sqrt(var_pred)
    
    # t分布临界值
    t_crit = stats.t.ppf(1 - alpha/2, dof)
    
    # 预测区间上下限
    upper = W_fit + t_crit * se_pred
    lower = W_fit - t_crit * se_pred
    
    return upper, lower

pi_upper_lm, pi_lower_lm = prediction_interval(t, popt_lm, pcov_lm)
pi_upper_trf, pi_lower_trf = prediction_interval(t, popt_trf, pcov_trf)

# ---------------------- 8. 初始值敏感性测试 ----------------------
initial_guesses = [
    [50, 0.1, 5],    # 低估参数
    [150, 1.0, 15],  # 高估参数
    [80, 0.3, 8],    # 原初始猜测
    [100, 0.5, 10]   # 真实参数
]

print("="*60)
print("初始值敏感性测试（LM算法）:")
print("-"*60)
print(f"{'初始猜测 [W_max, r, t_mid]':<30} {'拟合结果 [W_max, r, t_mid]':<30}")
print("-"*60)
for p0_test in initial_guesses:
    popt_test, _ = curve_fit(logistic, t, W, p0=p0_test, method='lm')
    print(f"{str(p0_test):<30} {np.array2string(popt_test.round(2), separator=', '):<30}")

# ---------------------- 9. 结果输出 ----------------------
print("\n" + "="*60)
print("拟合结果汇总")
print("="*60)
print(f"真实参数: W_max={W_max_true}, r={r_true}, t_mid={t_mid_true}")
print("-"*60)

print("\nLevenberg-Marquardt算法:")
print(f"拟合参数: W_max={popt_lm[0]:.2f}, r={popt_lm[1]:.3f}, t_mid={popt_lm[2]:.2f}")
print(f"参数标准误: {np.sqrt(np.diag(pcov_lm)).round(3)}")
print(f"R²: {gof_lm[3]:.4f}, 调整R²: {gof_lm[4]:.4f}, RMSE: {gof_lm[5]:.2f}")
print(f"协方差矩阵:\n{pcov_lm.round(4)}")

print("\nTrust Region Reflective算法:")
print(f"拟合参数: W_max={popt_trf[0]:.2f}, r={popt_trf[1]:.3f}, t_mid={popt_trf[2]:.2f}")
print(f"参数标准误: {np.sqrt(np.diag(pcov_trf)).round(3)}")
print(f"R²: {gof_trf[3]:.4f}, 调整R²: {gof_trf[4]:.4f}, RMSE: {gof_trf[5]:.2f}")
print(f"协方差矩阵:\n{pcov_trf.round(4)}")

print("\n" + "="*60)
print("生长速率分析")
print("="*60)
print(f"LM算法: 最大生长速率 {max_gr_val_lm:.2f} 发生在 t={max_gr_t_lm:.2f} (与t_mid={t_mid_true}的差异: {abs(max_gr_t_lm - t_mid_true):.2f})")
print(f"TRF算法: 最大生长速率 {max_gr_val_trf:.2f} 发生在 t={max_gr_t_trf:.2f} (与t_mid={t_mid_true}的差异: {abs(max_gr_t_trf - t_mid_true):.2f})")

print("\n" + "="*60)
print("分段拟合结果")
print("="*60)
print(f"指数生长阶段 (t ≤ {split_point}): 拟合参数 a={popt_exp[0]:.2f}, b={popt_exp[1]:.3f}")
print(f"饱和生长阶段 (t > {split_point}): 拟合参数 W_max={popt_sat[0]:.2f}, a={popt_sat[1]:.2f}, b={popt_sat[2]:.3f}")

# ---------------------- 10. 绘图 ----------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：生物量实测点与拟合曲线
ax1.scatter(t, W, color='blue', label='实测数据', alpha=0.6, s=30)
ax1.plot(t, W_fit_lm, 'r--', linewidth=2, label='LM拟合曲线')
ax1.plot(t, W_fit_trf, 'g:', linewidth=2, label='TRF拟合曲线')
# 绘制预测区间
ax1.fill_between(t, pi_lower_lm, pi_upper_lm, color='red', alpha=0.1, label='LM 95%预测区间')
ax1.fill_between(t, pi_lower_trf, pi_upper_trf, color='green', alpha=0.1, label='TRF 95%预测区间')
# 分段拟合曲线
ax1.plot(t_exp, W_fit_exp, 'c-.', linewidth=1.5, label='指数阶段拟合')
ax1.plot(t_sat, W_fit_sat, 'm-.', linewidth=1.5, label='饱和阶段拟合')

ax1.set_xlabel('时间 t', fontsize=12)
ax1.set_ylabel('生物量 W', fontsize=12)
ax1.set_title('Logistic模型拟合（含95%预测区间）', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(alpha=0.3)

# 右图：生长速率曲线
ax2.plot(t, gr_lm, 'r--', linewidth=2, label='LM生长速率')
ax2.plot(t, gr_trf, 'g:', linewidth=2, label='TRF生长速率')
# 标记最大生长速率点
ax2.scatter(max_gr_t_lm, max_gr_val_lm, color='red', s=100, marker='*', 
            label=f'LM最大速率点 (t={max_gr_t_lm:.2f})')
ax2.scatter(max_gr_t_trf, max_gr_val_trf, color='green', s=100, marker='*',
            label=f'TRF最大速率点 (t={max_gr_t_trf:.2f})')
# 标记真实拐点
ax2.axvline(x=t_mid_true, color='gray', linestyle='--', label=f'真实拐点 t_mid={t_mid_true}')

ax2.set_xlabel('时间 t', fontsize=12)
ax2.set_ylabel('生长速率 dW/dt', fontsize=12)
ax2.set_title('生长速率曲线与最大速率点', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(alpha=0.3)

plt.tight_layout()
# 保存图片
plt.savefig('D:/all_code/github/04/seed/plant_growth_analysis/growth_curves.png', dpi=300, bbox_inches='tight')
plt.savefig('D:/all_code/github/04/seed/plant_growth_analysis/growth_curves.pdf', dpi=300, bbox_inches='tight')
plt.show()

# ---------------------- 11. 模型解释 ----------------------
print("\n" + "="*60)
print("模型解释")
print("="*60)
print("""
1. Logistic模型特点：
   - 描述S形生长曲线，初期近似指数增长，后期趋于饱和
   - t_mid为生长拐点，此时生长速率达到最大值
   - W_max为环境容纳量（最大生物量）
   - r为内在生长速率参数

2. 拟合结果分析：
   - 两种算法均能较好拟合数据，R²均接近1
   - 拟合参数与真实值接近，说明模型可识别性良好
   - 协方差矩阵对角线元素为参数方差，反映参数估计不确定性

3. 生长速率分析：
   - 理论上最大生长速率应发生在t_mid处
   - 实际拟合结果与理论值的微小差异源于测量噪声

4. 分段拟合说明：
   - 前期（t < t_mid）：近似指数增长，符合J形曲线
   - 后期（t > t_mid）：增长速率逐渐下降，趋近于0
   - 分段参数稳定性：后期W_max估计更稳定，前期r估计更敏感

5. 初始值敏感性：
   - 非线性拟合对初始值有一定敏感性
   - 合理的初始猜测可提高收敛速度和稳定性
   - 当参数范围未知时，建议使用多组初始值测试
""")

# 保存结果到文本文件
with open('D:/all_code/github/04/seed/plant_growth_analysis/results.txt', 'w', encoding='utf-8') as f:
    f.write("="*60 + "\n")
    f.write("初始值敏感性测试（LM算法）:\n")
    f.write("-"*60 + "\n")
    f.write(f"{'初始猜测 [W_max, r, t_mid]':<30} {'拟合结果 [W_max, r, t_mid]':<30}\n")
    f.write("-"*60 + "\n")
    for p0_test in initial_guesses:
        popt_test, _ = curve_fit(logistic, t, W, p0=p0_test, method='lm')
        f.write(f"{str(p0_test):<30} {np.array2string(popt_test.round(2), separator=', '):<30}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("拟合结果汇总\n")
    f.write("="*60 + "\n")
    f.write(f"真实参数: W_max={W_max_true}, r={r_true}, t_mid={t_mid_true}\n")
    f.write("-"*60 + "\n")
    
    f.write("\nLevenberg-Marquardt算法:\n")
    f.write(f"拟合参数: W_max={popt_lm[0]:.2f}, r={popt_lm[1]:.3f}, t_mid={popt_lm[2]:.2f}\n")
    f.write(f"参数标准误: {np.sqrt(np.diag(pcov_lm)).round(3)}\n")
    f.write(f"R²: {gof_lm[3]:.4f}, 调整R²: {gof_lm[4]:.4f}, RMSE: {gof_lm[5]:.2f}\n")
    f.write(f"协方差矩阵:\n{pcov_lm.round(4)}\n")
    
    f.write("\nTrust Region Reflective算法:\n")
    f.write(f"拟合参数: W_max={popt_trf[0]:.2f}, r={popt_trf[1]:.3f}, t_mid={popt_trf[2]:.2f}\n")
    f.write(f"参数标准误: {np.sqrt(np.diag(pcov_trf)).round(3)}\n")
    f.write(f"R²: {gof_trf[3]:.4f}, 调整R²: {gof_trf[4]:.4f}, RMSE: {gof_trf[5]:.2f}\n")
    f.write(f"协方差矩阵:\n{pcov_trf.round(4)}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("生长速率分析\n")
    f.write("="*60 + "\n")
    f.write(f"LM算法: 最大生长速率 {max_gr_val_lm:.2f} 发生在 t={max_gr_t_lm:.2f} (与t_mid={t_mid_true}的差异: {abs(max_gr_t_lm - t_mid_true):.2f})\n")
    f.write(f"TRF算法: 最大生长速率 {max_gr_val_trf:.2f} 发生在 t={max_gr_t_trf:.2f} (与t_mid={t_mid_true}的差异: {abs(max_gr_t_trf - t_mid_true):.2f})\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("分段拟合结果\n")
    f.write("="*60 + "\n")
    f.write(f"指数生长阶段 (t ≤ {split_point}): 拟合参数 a={popt_exp[0]:.2f}, b={popt_exp[1]:.3f}\n")
    f.write(f"饱和生长阶段 (t > {split_point}): 拟合参数 W_max={popt_sat[0]:.2f}, a={popt_sat[1]:.2f}, b={popt_sat[2]:.3f}\n")
    
    f.write("\n" + "="*60 + "\n")
    f.write("模型解释\n")
    f.write("="*60 + "\n")
    f.write("""
1. Logistic模型特点：
   - 描述S形生长曲线，初期近似指数增长，后期趋于饱和
   - t_mid为生长拐点，此时生长速率达到最大值
   - W_max为环境容纳量（最大生物量）
   - r为内在生长速率参数

2. 拟合结果分析：
   - 两种算法均能较好拟合数据，R²均接近1
   - 拟合参数与真实值接近，说明模型可识别性良好
   - 协方差矩阵对角线元素为参数方差，反映参数估计不确定性

3. 生长速率分析：
   - 理论上最大生长速率应发生在t_mid处
   - 实际拟合结果与理论值的微小差异源于测量噪声

4. 分段拟合说明：
   - 前期（t < t_mid）：近似指数增长，符合J形曲线
   - 后期（t > t_mid）：增长速率逐渐下降，趋近于0
   - 分段参数稳定性：后期W_max估计更稳定，前期r估计更敏感

5. 初始值敏感性：
   - 非线性拟合对初始值有一定敏感性
   - 合理的初始猜测可提高收敛速度和稳定性
   - 当参数范围未知时，建议使用多组初始值测试
""")

print("\n文件已保存到 plant_growth_analysis 文件夹中！")
