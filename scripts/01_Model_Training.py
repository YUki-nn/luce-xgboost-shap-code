# -*- coding: utf-8 -*-
"""
XGBoost + SHAP: Nonlinearity & Threshold Analysis (No Prediction Export)
----------------------------------------------------------------------
- Loads data
- (Optional) Optuna hyperparameter tuning via CV
- Trains XGBoost
- Computes SHAP values
- Produces:
  1) Feature correlation heatmap
  2) XGBoost built-in feature importance
  3) SHAP summary (dot & bar)
  4) SHAP dependence plots (top-K)
  5) SHAP interaction summary (optional, can be slow)
  6) Data-driven threshold detection for each feature using piecewise linear fit on (feature, SHAP) pairs
  7) Threshold-annotated scatter plots

Notes
-----
- No prediction-vs-actual exports, no test metrics tables, no "observed vs predicted" charts.
- You can change INPUT_PATH and TARGET_COL as needed.
"""

import os
import warnings
import random  # [NEW] for reproducibility
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import optuna
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# ----------------------- Configuration -----------------------
INPUT_PATH  = r'D:\\AALUCE\\XGB\\data_excel\\data432.xlsx'  # <-- change to your dataset
TARGET_COL  = 'Y_LUCE'                            # <-- change to your target column
OUTPUT_DIR  = r'D:\\AALUCE\\XGB\\XGB_result'
TOP_K       = 13     # how many top features (by |SHAP|) to plot/analyze
N_TRIALS    = 250    # Optuna trials for tuning (set lower if needed)
RANDOM_SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"[INFO] All outputs will be saved to: {OUTPUT_DIR}")

# Global plotting settings
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['axes.unicode_minus'] = False

# [NEW] Set global random seeds for full reproducibility
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ----------------------- Data Loading -----------------------
if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(f"Could not find input file: {INPUT_PATH}")

df = pd.read_excel(INPUT_PATH)
if TARGET_COL not in df.columns:
    raise ValueError(f"TARGET_COL '{TARGET_COL}' not found. Available columns: {list(df.columns)}")

# 手动指定自变量列
features = ['X1_NPP', 'X2_Tem', 'X3_Pre', 'X4_FR', 'X5_POP', 'X6_GDP',
            'X7_NT', 'X8_PPI', 'X9_PSI', 'X10_NIE', 'X11_FAI', 'X12_EC', 'X13_DI']
X = df[features].copy()
y = df[TARGET_COL].copy()

# ----------------------- Train/Test Split & Scaling -----------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=features, index=X_train.index)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=features, index=X_test.index)

# [NEW] Keep an unscaled copy for plots & threshold detection (more interpretable axes)
# —— 用“全体样本”作为后续可视化与阈值分析的数据 ——
X_all_scaled = pd.DataFrame(scaler.transform(X), columns=features, index=X.index)  # 全样本（标准化）
X_plot = X.copy()  # 全样本（原尺度）


# ----------------------- Correlation Heatmap -----------------------
plt.figure(figsize=(10, 8), dpi=300)
corr = X.corr(method='pearson')

# === 关键：生成下三角矩阵 ===
mask = np.triu(np.ones_like(corr, dtype=bool))

sns.heatmap(
    corr,
    mask=mask,              # 只显示下三角
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    linewidths=.5,
    annot_kws={"size": 8}
)

plt.title('Feature Correlation Heatmap (Lower Triangle)')
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'feature_correlation_heatmap_lower.pdf'), format='pdf')
plt.close()

print("[OK] Saved: feature_correlation_heatmap.pdf")

# ----------------------- Optuna Hyperparameter Tuning -----------------------
def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 200, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.20),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'subsample': trial.suggest_float('subsample', 0.6, 0.9),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0.5, 5.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 6.0, 15.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 3.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1.0, 3.0),
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbosity': 0,
        'tree_method': 'hist',               # [CHG] explicit fast CPU algorithm
        'objective': 'reg:squarederror'      # [CHG] explicit regression objective
    }
    model = XGBRegressor(**params)
    score = cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='neg_mean_squared_error')
    mean_score = score.mean()
    # 实时打印每个Trial的得分，便于观察进度
    print(f"Trial {trial.number+1}/{N_TRIALS}: mean_score={mean_score:.6f}")
    return mean_score

print(f"[INFO] Running Optuna hyperparameter search ({N_TRIALS} trials)...")
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
best_params = study.best_params
print("[OK] Best params from Optuna:", best_params)

# ===== 5-fold Cross-Validation (R2 / RMSE / MAE) =====
from sklearn.model_selection import KFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 用 Pipeline 避免泄漏：每个fold内先标准化再训练XGB
cv_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", XGBRegressor(
        **best_params,
        random_state=RANDOM_SEED,
        verbosity=0,
        objective="reg:squarederror",
        tree_method="hist"
    ))
])

# 5折交叉验证（随机打乱 + 固定种子保证可复现）
cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# 需要的评分：R2、RMSE、MAE（注意RMSE/MAE为负号约定，回头取反）
scoring = {
    "R2": "r2",
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
}

cv_res = cross_validate(
    cv_pipe, X, y,
    cv=cv,
    scoring=scoring,
    return_train_score=False,
    n_jobs=-1
)

import numpy as np
r2_mean, r2_std = cv_res["test_R2"].mean(), cv_res["test_R2"].std()
rmse_mean, rmse_std = -cv_res["test_RMSE"].mean(), cv_res["test_RMSE"].std()  * -1
mae_mean, mae_std = -cv_res["test_MAE"].mean(), cv_res["test_MAE"].std()  * -1

print("\n====== 5-fold Cross-Validation ======")
print(f"CV R²    : {r2_mean:.4f} ± {r2_std:.4f}")
print(f"CV RMSE  : {rmse_mean:.4f} ± {abs(rmse_std):.4f}")
print(f"CV MAE   : {mae_mean:.4f} ± {abs(mae_std):.4f}")
print("=====================================\n")


# ----------------------- Train Final Model -----------------------
final_model = XGBRegressor(
    **best_params,
    random_state=RANDOM_SEED,
    verbosity=0,
    objective='reg:squarederror',   # [CHG] ensure consistency
    tree_method='hist'              # [CHG] ensure consistency
)

final_model.fit(X_train_scaled, y_train)

# ---------------------- 模型性能评估 ----------------------
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np

# 预测训练集与测试集
y_pred_train = final_model.predict(X_train_scaled)
y_pred_test  = final_model.predict(X_test_scaled)

# 计算指标
r2_train = r2_score(y_train, y_pred_train)
r2_test  = r2_score(y_test, y_pred_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
rmse_test  = np.sqrt(mean_squared_error(y_test, y_pred_test))

mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test  = mean_absolute_error(y_test, y_pred_test)

# ====== Compute Adjusted R² ======
n_train = len(y_train)
n_test = len(y_test)
k = X_train_scaled.shape[1]  # 特征数量

# 调整后 R² 的计算
r2_train_adj = 1 - (1 - r2_train) * (n_train - 1) / (n_train - k - 1)
r2_test_adj = 1 - (1 - r2_test) * (n_test - 1) / (n_test - k - 1)

# 输出调整后R²结果
print("Train Adjusted R²:", round(r2_train_adj, 4), "Test Adjusted R²:", round(r2_test_adj, 4))

# 输出结果
print("\n====== Model Performance Evaluation ======")
print(f"Train R²: {r2_train:.4f},  Test R²: {r2_test:.4f}")
print(f"Train RMSE: {rmse_train:.4f},  Test RMSE: {rmse_test:.4f}")
print(f"Train MAE: {mae_train:.4f},  Test MAE: {mae_test:.4f}")
print("==========================================\n")

print(f"Train MAE: {mae_train:.4f},  Test MAE: {mae_test:.4f}")
print("==========================================\n")

# === （训练 vs 验证 拟合图） ===
# ================= [替换开始] Model Fit Plot (Unified Color & Style) =================
import matplotlib.pyplot as plt

# 设置大字体参数
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 20

plt.figure(figsize=(8, 8), dpi=300)

# 绘制散点 [配色修改]
# Train: 深紫色 (#38338b)
plt.scatter(y_train, y_pred_train, alpha=0.6, s=60,
            label='Train', color='#38338b', edgecolor='white', linewidth=0.5, zorder=2)

# Validation: 暖黄色 (#fed176)
plt.scatter(y_test, y_pred_test, alpha=0.8, s=70, marker='^',
            label='Validation', color='#fed176', edgecolor='k', linewidth=0.5, zorder=3)

# 绘制 1:1 对角线
y_min_val = min(y_test.min(), y_train.min())
y_max_val = max(y_test.max(), y_train.max())
# 留一点余量
y_range = y_max_val - y_min_val
axis_min = y_min_val - 0.05 * y_range
axis_max = y_max_val + 0.05 * y_range

plt.plot([axis_min, axis_max], [axis_min, axis_max], 'k--', lw=2.5, label='1:1 Line (y = x)', zorder=1)

# 标签与标题
plt.xlabel('Actual Values', fontsize=24, labelpad=10)
plt.ylabel('Predicted Values', fontsize=24, labelpad=10)
plt.title('XGBoost Model Fit (Train vs Validation)', fontsize=26, pad=15)

# 范围与刻度
plt.xlim(axis_min, axis_max)
plt.ylim(axis_min, axis_max)
plt.tick_params(axis='both', which='major', labelsize=20, width=2, length=6)

# 图例
plt.legend(loc='upper left', fontsize=20, frameon=True, framealpha=0.9, edgecolor='gray')

# 网格
plt.grid(True, linestyle='--', alpha=0.4, linewidth=1.5, zorder=0)

# 文本框 (Statistics)
metrics_text = (
    f'Validation set:\n'
    f'$R^2$ = {r2_test:.4f}\n'
    f'RMSE = {rmse_test:.2f}\n'
    f'MAE = {mae_test:.2f}\n\n'
    f'Training set:\n'
    f'$R^2$ = {r2_train:.4f}\n'
    f'RMSE = {rmse_train:.2f}\n'
    f'MAE = {mae_train:.2f}'
)

# 文本框样式 [背景色改为淡灰白，避免抢眼]
plt.text(0.96, 0.04, metrics_text,
         transform=plt.gca().transAxes,
         fontsize=18, va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.6', fc='#f0f0f0', alpha=0.8, ec='gray', lw=1.5),
         zorder=4)

# 边框加粗
ax = plt.gca()
for spine in ax.spines.values():
    spine.set_linewidth(2)

# 保存
fit_plot_path_large = os.path.join(OUTPUT_DIR, 'xgb_fit_train_vs_validation_Unified.png')
plt.savefig(fit_plot_path_large, bbox_inches='tight', dpi=300)
print(f"[OK] Saved Unified Model Fit Plot: {fit_plot_path_large}")
plt.close()
# ================= [替换结束] =================

# ----------------------- XGBoost Built-in Importance -----------------------
importances = final_model.feature_importances_
...


# ----------------------- XGBoost Built-in Importance -----------------------
importances = final_model.feature_importances_
imp_df = pd.DataFrame({'Feature': features, 'Importance': importances}).sort_values('Importance', ascending=False)
imp_df.to_excel(os.path.join(OUTPUT_DIR, 'xgb_builtin_importance.xlsx'), index=False)

plt.figure(figsize=(10, 6), dpi=300)
plt.barh(imp_df['Feature'][::-1], imp_df['Importance'][::-1])
plt.xlabel('Importance (Gain)')
plt.title('XGBoost - Built-in Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'xgb_builtin_importance.pdf'), format='pdf')
plt.close()
print("[OK] Saved: xgb_builtin_importance.(xlsx|pdf)")


# ----------------------- SHAP Computation -----------------------
print("[INFO] Computing SHAP values...")
explainer = shap.TreeExplainer(final_model)
shap_values = explainer.shap_values(X_all_scaled)   # 用全样本的标准化特征计算 SHAP
expected_value = explainer.expected_value
print("[OK] SHAP values computed.")

# Save raw SHAP arrays
pd.DataFrame(shap_values, columns=features).to_excel(os.path.join(OUTPUT_DIR, 'shap_values.xlsx'), index=False)
pd.DataFrame(X_all_scaled, columns=features).to_excel(os.path.join(OUTPUT_DIR, 'shap_eval_data_X_all.xlsx'), index=False)
print("[OK] Saved: shap_values.xlsx, shap_eval_data_X.xlsx")

# Global SHAP importance
shap_global = pd.DataFrame({
    'Feature': features,
    'AvgAbsSHAP': np.abs(shap_values).mean(axis=0)
}).sort_values('AvgAbsSHAP', ascending=False)
shap_global.to_excel(os.path.join(OUTPUT_DIR, 'shap_global_importance.xlsx'), index=False)

# Summary plots  [CHG] use X_plot (original scale) for interpretability
shap.summary_plot(shap_values, X_plot, plot_type='dot', show=False)
plt.title('SHAP Summary Plot')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_dot.pdf'), format='pdf', bbox_inches='tight')
plt.close()

shap.summary_plot(shap_values, X_plot, plot_type='bar', show=False)
plt.title('SHAP Global Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_bar.pdf'), format='pdf', bbox_inches='tight')
plt.close()

# 创建主图（用来画蜂巢图）
fig, ax1 = plt.subplots(figsize=(8, 6), dpi=120)
shap.summary_plot(shap_values, X_plot, feature_names=features,
                  plot_type="dot", show=False, cmap='viridis', color_bar=True)
ax1.set_zorder(10)         # 蜂群图所在轴抬到上层
ax1.patch.set_alpha(0.0)   # 轴背景透明
for coll in ax1.collections:   # shap 生成的散点集合
    coll.set_zorder(11)

# plt.xlim(-2500,2500)
plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，留出右侧空间放热度条
# 获取共享的 y 轴
ax1 = plt.gca()
# 创建共享 y 轴的另一个图，绘制特征贡献图在顶部x轴
ax2 = ax1.twiny()
ax2.set_zorder(0)          # 整个 ax2 置底
ax2.patch.set_alpha(0.0)   # 轴背景透明，避免遮住散点

# —— 关键改动：用蜂群图的 y 轴顺序来画条形图 ——
# 先从蜂群图上读取特征的显示顺序（SHAP 会按 |SHAP| 从大到小排好）
yticks = [t.get_text() for t in ax1.get_yticklabels()]

# 计算各特征的平均 |SHAP|，并按 yticks 顺序重排
imp_map = dict(zip(features, np.abs(shap_values).mean(axis=0)))
imp_ordered = np.array([imp_map[f] for f in yticks])

# 归一化后着色（与蜂群图同一套 viridis 色带）
norm_imp = imp_ordered / imp_ordered.max()

bars = ax2.barh(yticks, imp_ordered,
         color=plt.cm.viridis(norm_imp),
         alpha=0.3, zorder=0)   # 透明度你可改 0.4~0.6


plt.gca().set_position([0.2, 0.2, 0.65, 0.65])  # 调整图表位置，与蜂巢图对齐
# 在顶部 X 轴添加一条横线
# ax2.axhline(y=8, color='gray', linestyle='-', linewidth=1)  # 注意y值应该对应顶部
# 调整透明度
bars = ax2.patches  # 获取所有的柱状图对象
for bar in bars:
    bar.set_alpha(0.4)  # 设置透明度
# 设置两个x轴的标签
ax1.set_xlabel('Shap Value (impact on model output)', fontsize=12)
ax2.set_xlabel('Mean(|SHAP value|) (Feature Importance)', fontsize=12)
# 移动顶部的 X 轴，避免与底部 X 轴重叠
fig.set_size_inches(8, 6)  # 覆盖之前的尺寸设置
ax2.xaxis.set_label_position('top')  # 将标签移动到顶部
ax2.xaxis.tick_top()#将刻度也移动到顶部
ax2.spines['top'].set_visible(True)
ax2.spines['top'].set_linewidth(1.2)
ax2.spines['top'].set_color('black')
# 设置y轴标签
# ax1.set_ylabel('Feature', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'shap_summary_combine.pdf'), format='pdf', bbox_inches='tight')
plt.close()


print("[OK] Saved: shap_summary_(dot|bar).pdf")


# ----------------------- Dependence Plots (Top-K) -----------------------
top_features = shap_global['Feature'].head(TOP_K).tolist()
for f in top_features:
    # [CHG] use X_plot (original scale) for interpretability
    shap.dependence_plot(f, shap_values, X_plot, show=False)
    plt.title(f"SHAP Dependence Plot: {f}")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f'shap_dependence_{f}.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
print(f"[OK] Saved: SHAP dependence plots for top {TOP_K} features.")

# ================= Enhanced SHAP dependence plots (Hist + Scatter + LOWESS + 95%CI + thresholds) ==============
import numpy as np, os
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def _bootstrap_lowess_ci(x, y, n_boot=200, frac=0.3, ci_level=0.95):
    """LOWESS 主曲线 + 自助法95%CI"""
    x = np.asarray(x); y = np.asarray(y)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 10 or np.unique(x).size < 2:
        return None, None
    x_grid = np.linspace(x.min(), x.max(), 160)
    boot = []
    for _ in range(n_boot):
        idx = np.random.choice(len(x), len(x), replace=True)
        xs, ys = x[idx], y[idx]
        sm = lowess(ys, xs, frac=frac, return_sorted=True)
        boot.append(np.interp(x_grid, sm[:,0], sm[:,1]))
    sm_main = lowess(y, x, frac=frac, return_sorted=True)
    arr = np.vstack(boot)
    alpha = (1-ci_level)/2
    lo, hi = np.quantile(arr, alpha, axis=0), np.quantile(arr, 1-alpha, axis=0)
    return sm_main, (x_grid, lo, hi)

def _find_and_plot_zero_cross(ax, x_curve, y_curve, color='black'):
    """标注 y=0 交点作为‘阈值’"""
    roots = []
    sgn = np.sign(y_curve)
    idx = np.where(np.diff(sgn)!=0)[0]
    for k in idx:
        x1,x2 = x_curve[k], x_curve[k+1]
        y1,y2 = y_curve[k], y_curve[k+1]
        if y2 == y1:
            continue
        xr = x1 - y1*(x2-x1)/(y2-y1)
        roots.append(xr)
        ax.axvline(xr, ls='--', lw=1, color=color)
        ax.text(xr, ax.get_ylim()[1]*0.9, f"{xr:.2f}", color='white',
                ha='center', va='center', fontsize=9,
                bbox=dict(facecolor=color, edgecolor='none', pad=1))
    return roots


def plot_dep_enhanced(feat_name, x_vals, shap_vals, out_dir):
    """
    绘制增强版SHAP依赖图：直方图底纹 + 散点 + LOWESS拟合 + 95%CI + 智能X轴截断
    (已修改：增加对极值离群点的自适应视图缩放，修复X8_PPI等指标的显示问题)
    """
    # ================= 1. 计算自适应显示范围 =================
    # 计算 0.5% 和 99.5% 分位数
    # np.nanquantile 能忽略 NaN 值，确保稳健
    q_low = np.nanquantile(x_vals, 0.005)
    q_high = np.nanquantile(x_vals, 0.995)

    x_min_real = np.nanmin(x_vals)
    x_max_real = np.nanmax(x_vals)

    # 定义视图的缓冲范围 (Padding)，让数据不要顶格
    x_range = q_high - q_low
    # 如果数据本身就是单一值或极窄，避免 range 为 0
    if x_range == 0:
        x_range = (x_max_real - x_min_real) if (x_max_real - x_min_real) > 0 else 1.0

    view_min = q_low - 0.05 * x_range
    view_max = q_high + 0.05 * x_range

    # 修正边界：视图不应该超出数据的真实物理边界（除非为了美观留白）
    # 但对于向右拖尾的数据（如X8），我们允许 x_max_real 远大于 view_max
    view_min = max(view_min, x_min_real)
    # view_max 不做 max 限制，允许切掉右侧离群点

    # 判断是否发生了显著截断（用于后续加注脚）
    is_truncated = (x_max_real > view_max + 0.05 * x_range)
    # =======================================================

    fig, ax1 = plt.subplots(figsize=(8, 6), dpi=150)
    ax2 = ax1.twinx();
    ax2.patch.set_alpha(0)

    # ================= 2. 绘制直方图 (Distribution) =================
    # 仅使用视图范围内的数据计算直方图，避免被远处的离群点拉平Bins
    mask_view = (x_vals >= view_min) & (x_vals <= view_max)
    data_for_hist = x_vals[mask_view] if np.sum(mask_view) > 10 else x_vals

    counts, bins = np.histogram(data_for_hist, bins=30)
    centers = (bins[:-1] + bins[1:]) / 2
    width = bins[1] - bins[0]
    ax1.bar(centers, counts, width=width * 0.7, color='#7b68ee', alpha=0.25, label='Distribution')
    ax1.set_ylabel('Distribution')
    ax1.set_ylim(0, counts.max() * 1.10)

    # ================= 3. 绘制散点 (Scatter) =================
    # 散点画全量数据，稍后通过 set_xlim 限制视野
    ax2.scatter(x_vals, shap_vals, s=26, alpha=0.35, color='#1f3a93', label='Sample', zorder=2)

    # ================= 4. 绘制 LOWESS + CI =================
    # 关键：拟合必须使用【全量数据】，保证趋势线的数学真实性
    fit, ci = _bootstrap_lowess_ci(x_vals, shap_vals, frac=0.3)

    if fit is not None and ci is not None:
        ax2.plot(fit[:, 0], fit[:, 1], color='#8a2be2', lw=2.2, label='LOWESS Fit', zorder=4)
        ax2.fill_between(ci[0], ci[1], ci[2], color='#8a2be2', alpha=0.12, label='95%CI')
        ax2.axhline(0, color='k', ls='--', lw=1)
        _find_and_plot_zero_cross(ax2, fit[:, 0], fit[:, 1], color='black')

    ax1.set_xlabel(feat_name)
    ax2.set_ylabel('SHAP value')

    # ================= 5. 设置坐标轴范围 =================
    # 应用之前计算的智能视图范围
    ax1.set_xlim(view_min, view_max)
    ax2.set_xlim(view_min, view_max)

    # 动态调整 Y 轴范围：只看当前 X 轴视野内的 SHAP 值最大值
    # 这样如果右侧离群点有极大的 SHAP 值，不会导致当前视图被压缩
    if np.sum(mask_view) > 0:
        # 找出视野内 SHAP 值的最大绝对值
        y_subset = shap_vals[mask_view]
        ymax = np.nanmax(np.abs(y_subset))
        # 如果视野内全是 0 或 NaN，退回到全局最大值
        if ymax == 0 or np.isnan(ymax):
            ymax = np.nanmax(np.abs(shap_vals))
    else:
        ymax = np.nanmax(np.abs(shap_vals))

    # 给 Y 轴留一点余量
    ax2.set_ylim(-1.15 * ymax if np.isfinite(ymax) else -1, 1.15 * ymax if np.isfinite(ymax) else 1)

    # ================= 6. 添加透明度声明 (Footnote) =================
    # 如果图被截断了，添加一行小字说明，体现学术严谨性
    if is_truncated:
        plt.text(0.98, 0.02,
                 f'X-axis zoomed to 99.5% percentile range\n(Outliers up to {x_max_real:.1f} hidden)',
                 transform=ax1.transAxes, ha='right', va='bottom',
                 fontsize=8, color='gray', style='italic')

    # 合并图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h2 + h1, l2 + l1, loc='upper right', fontsize=10)

    # 样式
    for a in (ax1, ax2):
        a.spines['top'].set_visible(False);
        a.spines['right'].set_visible(False)
        a.tick_params(axis='both', which='major', direction='in', width=1.3, length=6)

    os.makedirs(out_dir, exist_ok=True)
    # 保存图片
    path = os.path.join(out_dir, f"dep_lowess_ci_{feat_name.replace('/', '_')}.png")
    plt.savefig(path, dpi=220, bbox_inches='tight');
    plt.close()
    print(f"[OK] saved: {path}")
# —— 批量导出（使用全样本；若想用测试集，可把 X_plot/shap_values 换成 X_test_scaled + 对应 shap） ——
enh_out = os.path.join(OUTPUT_DIR, "dependence_plots_lowess_ci")
os.makedirs(enh_out, exist_ok=True)

# shap_values 的列顺序与 features 一致；从中取 top_features 的列索引
feat_to_idx = {f:i for i,f in enumerate(features)}
for f in top_features:  # 与上面“Dependence Plots (Top-K)”保持一致
    idx = feat_to_idx[f]
    plot_dep_enhanced(f, X_plot[f].values, shap_values[:, idx], enh_out)
# ===============================================================================================================


# ----------------------- PDP + ICE (Partial Dependence + Individual Conditional Expectation) -----------------------
# 用于从“总体预测视角”验证单因子非线性；与 SHAP dependence 图互补
# 注意：这里使用模型输入域 X_test_scaled（与训练一致）；横轴为标准化刻度
try:
    from sklearn.inspection import PartialDependenceDisplay
    # 名称 -> 索引 的映射（PartialDependenceDisplay 接受特征索引）
    feat_index_map = {name: i for i, name in enumerate(features)}

    for f in top_features:
        if f not in feat_index_map:
            print(f"[WARN] PDP/ICE skipped for {f}: feature not found in mapping.")
            continue

        fi = feat_index_map[f]
        fig, ax = plt.subplots(figsize=(6.5, 4.8), dpi=300)
        PartialDependenceDisplay.from_estimator(
            estimator=final_model,
            X=X_all_scaled,          # 用“全体样本”的标准化特征
            features=[fi],            # 传入单一特征索引
            kind="both",              # 平均效应(PDP) + 个体曲线(ICE)
            grid_resolution=60,       # 采样网格更细，曲线更平滑
            ax=ax
        )
        ax.set_title(f"PDP + ICE: {f}")
        fig.tight_layout()
        fig.savefig(os.path.join(OUTPUT_DIR, f"pdp_ice_{f}.pdf"), format="pdf", bbox_inches="tight")
        plt.close(fig)

    print("[OK] Saved: pdp_ice_<feature>.pdf")
except Exception as e:
    print(f"[WARN] PDP/ICE skipped: {e}")


# ----------------------- Optional: Interaction Summary (may be slow) -----------------------
try:
    print("[INFO] Computing SHAP interaction values (can be slow)...")
    shap_inter = explainer.shap_interaction_values(X_all_scaled)  # 全体样本
    shap.summary_plot(shap_inter, X_test_scaled, show=False)
    plt.title("SHAP Interaction Values Summary")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'shap_interaction_summary.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print("[OK] Saved: shap_interaction_summary.pdf")
except Exception as e:
    print(f"[WARN] Could not compute SHAP interaction values: {e}")

# ----------------------- Threshold Detection via Piecewise Linear Fit -----------------------
def _piecewise_two_segments(x, y, min_frac=0.1):
    """
    Find a single threshold (breakpoint) that best splits (x, y) into two linear segments
    by minimizing total SSE. Returns (threshold, sse, slope_left, slope_right).
    """
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y).reshape(-1)
    # sort by x
    order = np.argsort(x)
    x, y = x[order], y[order]

    n = len(x)
    min_n = max(3, int(np.ceil(min_frac * n)))  # minimum points per side
    best = (None, np.inf, None, None)  # (thr, sse, m1, m2)

    # Pre-compute for efficiency
    for split in range(min_n, n - min_n):
        x1, y1 = x[:split], y[:split]
        x2, y2 = x[split:], y[split:]

        # Fit y = a + b x on each side
        A1 = np.vstack([np.ones_like(x1), x1]).T
        A2 = np.vstack([np.ones_like(x2), x2]).T

        # Least squares
        beta1, _, _, _ = np.linalg.lstsq(A1, y1, rcond=None)
        beta2, _, _, _ = np.linalg.lstsq(A2, y2, rcond=None)

        y1_hat = A1 @ beta1
        y2_hat = A2 @ beta2
        sse = np.sum((y1 - y1_hat)**2) + np.sum((y2 - y2_hat)**2)

        if sse < best[1]:
            thr = (x[split-1] + x[split]) / 2.0
            m1, m2 = beta1[1], beta2[1]
            best = (thr, sse, m1, m2)

    return best  # threshold, sse, slope_left, slope_right

def detect_thresholds_from_shap(X_frame, shap_vals, feat_list, out_dir):
    rows = []
    for f in feat_list:
        x = X_frame[f].values
        y = shap_vals[:, list(X_frame.columns).index(f)]
        thr, sse, m1, m2 = _piecewise_two_segments(x, y, min_frac=0.1)
        rows.append({
            'Feature': f,
            'Threshold': thr,
            'Slope_Left': m1,
            'Slope_Right': m2,
            'SSE': sse,
            'Direction_Change': np.sign(m1) != np.sign(m2)
        })

        # Make an annotated scatter with threshold
        plt.figure(figsize=(7, 5), dpi=300)
        plt.scatter(x, y, s=12, alpha=0.6)
        if thr is not None:
            plt.axvline(thr, linestyle='--')
        plt.xlabel(f)
        plt.ylabel('SHAP value')
        # [CHG] robust title text when thr/m1/m2 are None
        thr_txt = "NA" if thr is None else f"{thr:.4f}"
        m1_txt  = "NA" if m1  is None else f"{m1:.4f}"
        m2_txt  = "NA" if m2  is None else f"{m2:.4f}"
        plt.title(f'Threshold via piecewise fit: {f}\nThreshold={thr_txt} | Slopes=({m1_txt}, {m2_txt})')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'shap_threshold_{f}.pdf'), format='pdf', bbox_inches='tight')
        plt.close()

    thr_df = pd.DataFrame(rows).sort_values('Threshold')
    thr_df.to_excel(os.path.join(out_dir, 'shap_thresholds_piecewise.xlsx'), index=False)
    return thr_df

# [CHG] pass X_plot (original scale) for threshold detection
threshold_df = detect_thresholds_from_shap(X_plot, shap_values, top_features, OUTPUT_DIR)
print("[OK] Saved: shap_thresholds_piecewise.xlsx and per-feature annotated plots.")

print("\n[DONE] Nonlinearity & threshold analysis finished. All results saved in:", OUTPUT_DIR)
