import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.signal import find_peaks
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

def load_file(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                try:
                    rows.append([int(parts[0]), float(parts[1]), float(parts[2]),
                                 float(parts[3]), float(parts[4]), float(parts[5]),
                                 float(parts[6]), int(parts[7])])
                except ValueError:
                    pass
    return rows

def extract_features(filepath):
    rows = load_file(filepath)
    if len(rows) < 60:
        return None
    
    df = pd.DataFrame(rows, columns=['sensor','x','y','z','roll','pitch','yaw','ts'])
    df = df.drop_duplicates(subset=['sensor','ts'])
    
    s1 = df[df['sensor']==1].reset_index(drop=True)
    s2 = df[df['sensor']==2].reset_index(drop=True)
    
    if len(s1) < 50 or len(s2) < 50:
        return None
    
    # Align by common timestamps
    ts1 = set(s1['ts'].values)
    ts2 = set(s2['ts'].values)
    common = sorted(ts1 & ts2)
    if len(common) < 50:
        return None
    
    s1c = s1[s1['ts'].isin(common)].sort_values('ts').reset_index(drop=True)
    s2c = s2[s2['ts'].isin(common)].sort_values('ts').reset_index(drop=True)
    
    # Distance between two sensors (finger opening distance)
    dx = s1c['x'].values - s2c['x'].values
    dy = s1c['y'].values - s2c['y'].values
    dz = s1c['z'].values - s2c['z'].values
    dist = np.sqrt(dx**2 + dy**2 + dz**2)
    
    ts_arr = s1c['ts'].values
    duration = (ts_arr[-1] - ts_arr[0]) / 1000.0
    fs = len(ts_arr) / duration  # approx sampling rate
    
    # Smooth the distance signal
    from scipy.ndimage import uniform_filter1d
    dist_smooth = uniform_filter1d(dist, size=max(3, int(fs*0.05)))
    
    # Find peaks (open position = maximum opening = tap events)
    min_distance = max(5, int(fs * 0.15))  # at least 150ms between taps
    peaks, peak_props = find_peaks(dist_smooth, 
                                    distance=min_distance,
                                    prominence=np.std(dist_smooth)*0.3)
    
    # Find valleys (closed position)
    valleys, _ = find_peaks(-dist_smooth, distance=min_distance)
    
    n_peaks = len(peaks)
    
    feats = {}
    feats['duration'] = duration
    feats['n_samples'] = len(ts_arr)
    feats['fs'] = fs
    feats['n_taps'] = n_peaks
    
    # Tapping rate (Hz)
    feats['tap_rate'] = n_peaks / duration if duration > 0 else 0
    
    # Amplitude features
    feats['dist_mean'] = np.mean(dist)
    feats['dist_std'] = np.std(dist)
    feats['dist_range'] = np.max(dist) - np.min(dist)
    
    if n_peaks >= 3:
        # Inter-tap intervals
        peak_ts = ts_arr[peaks]
        iti = np.diff(peak_ts)  # in ms
        feats['iti_mean'] = np.mean(iti)
        feats['iti_std'] = np.std(iti)
        feats['iti_cv'] = np.std(iti) / np.mean(iti) if np.mean(iti) > 0 else 0
        feats['iti_min'] = np.min(iti)
        feats['iti_max'] = np.max(iti)
        
        # Peak amplitudes
        peak_amps = dist_smooth[peaks]
        feats['amp_mean'] = np.mean(peak_amps)
        feats['amp_std'] = np.std(peak_amps)
        feats['amp_cv'] = np.std(peak_amps) / np.mean(peak_amps) if np.mean(peak_amps) > 0 else 0
        
        # Amplitude decrement (slope of amplitude over time, key PD feature)
        if n_peaks >= 4:
            x_norm = np.arange(n_peaks) / n_peaks
            slope, intercept, r, p, se = stats.linregress(x_norm, peak_amps)
            feats['amp_slope'] = slope  # negative = decrement
            feats['amp_slope_norm'] = slope / feats['amp_mean'] if feats['amp_mean'] > 0 else 0
        else:
            feats['amp_slope'] = 0
            feats['amp_slope_norm'] = 0
        
        # Hesitations: ITI > 1.5x mean
        feats['hesitation_count'] = np.sum(iti > 1.5 * np.mean(iti))
        feats['hesitation_rate'] = feats['hesitation_count'] / n_peaks
        
        # Velocity (mean absolute first derivative of distance signal)
        dt_arr = np.diff(ts_arr) / 1000.0
        dt_arr = np.where(dt_arr <= 0, 0.001, dt_arr)
        vel = np.abs(np.diff(dist_smooth)) / dt_arr
        feats['vel_mean'] = np.mean(vel)
        feats['vel_max'] = np.max(vel)
        
    else:
        # Not enough taps detected
        feats['iti_mean'] = np.nan
        feats['iti_std'] = np.nan
        feats['iti_cv'] = np.nan
        feats['iti_min'] = np.nan
        feats['iti_max'] = np.nan
        feats['amp_mean'] = np.nan
        feats['amp_std'] = np.nan
        feats['amp_cv'] = np.nan
        feats['amp_slope'] = np.nan
        feats['amp_slope_norm'] = np.nan
        feats['hesitation_count'] = np.nan
        feats['hesitation_rate'] = np.nan
        feats['vel_mean'] = np.nan
        feats['vel_max'] = np.nan
    
    # Gyroscope-based features (rotation dynamics)
    for ax in ['roll', 'pitch', 'yaw']:
        vals = s1c[ax].values
        feats[f's1_{ax}_range'] = np.max(vals) - np.min(vals)
        feats[f's1_{ax}_std'] = np.std(vals)
    
    return feats

# ── Main ──────────────────────────────────────────────────────────────────────
records = []
for label_str in ['0', '1']:
    folder = f'/home/user/LSTM-VAE-ON-PD/{label_str}/'
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        feats = extract_features(fpath)
        if feats is None:
            print(f"[SKIP] {label_str}/{fname}")
            continue
        feats['label'] = int(label_str)
        feats['file'] = fname
        feats['folder'] = label_str
        records.append(feats)

df_all = pd.DataFrame(records)
print(f"有效文件: {len(df_all)}  (label0={sum(df_all['label']==0)}, label1={sum(df_all['label']==1)})")

# Feature columns
feat_cols = [c for c in df_all.columns if c not in ['label','file','folder']]
df_feat = df_all[feat_cols].copy()

# Fill NaN with column median
for col in feat_cols:
    df_feat[col].fillna(df_feat[col].median(), inplace=True)

X = df_feat.values
y = df_all['label'].values

# ── Step 1: Print group statistics ────────────────────────────────────────────
print("\n" + "="*80)
print("关键特征：Label=0 vs Label=1 均值对比")
print("="*80)
key_cols = ['tap_rate','iti_mean','iti_cv','amp_mean','amp_slope_norm',
            'hesitation_rate','vel_mean','dist_range']
for col in key_cols:
    v0 = df_all[df_all['label']==0][col].median()
    v1 = df_all[df_all['label']==1][col].median()
    stat, pval = stats.mannwhitneyu(df_all[df_all['label']==0][col].dropna(),
                                     df_all[df_all['label']==1][col].dropna(),
                                     alternative='two-sided')
    print(f"  {col:<25} label0={v0:8.3f}  label1={v1:8.3f}  p={pval:.4f}")

# ── Step 2: LDA cross-val predicted labels ────────────────────────────────────
scaler = StandardScaler()
X_sc = scaler.fit_transform(X)

lda = LinearDiscriminantAnalysis()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lda_proba = cross_val_predict(lda, X_sc, y, cv=cv, method='predict_proba')
lda_pred = np.argmax(lda_proba, axis=1)

rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
rf_proba = cross_val_predict(rf, X_sc, y, cv=cv, method='predict_proba')
rf_pred = np.argmax(rf_proba, axis=1)

# ── Step 3: Isolation Forest per-class outlier detection ─────────────────────
# For each class separately, find outliers within that class
iso_scores = np.zeros(len(df_all))
for lbl in [0, 1]:
    mask = y == lbl
    iso = IsolationForest(contamination=0.1, random_state=42)
    scores = iso.fit_predict(X_sc[mask])
    iso_scores[mask] = scores  # -1 = outlier, 1 = inlier

# ── Step 4: Z-score outlier per class ────────────────────────────────────────
z_outlier = np.zeros(len(df_all), dtype=bool)
for lbl in [0, 1]:
    mask = np.where(y == lbl)[0]
    X_sub = X_sc[mask]
    z = np.abs(X_sub)
    # Flag if more than 3 features are extreme (|z|>2.5 within class)
    extreme_count = np.sum(z > 2.5, axis=1)
    z_outlier[mask] = extreme_count >= 4

# ── Step 5: Combine flags ─────────────────────────────────────────────────────
df_all['lda_pred'] = lda_pred
df_all['lda_proba_wrong'] = [lda_proba[i, 1-y[i]] for i in range(len(y))]
df_all['rf_pred'] = rf_pred
df_all['rf_proba_wrong'] = [rf_proba[i, 1-y[i]] for i in range(len(y))]
df_all['iso_outlier'] = iso_scores == -1
df_all['z_outlier'] = z_outlier

# A file is "suspicious" if:
#  - Both LDA and RF predict it as the wrong label, OR
#  - Isolation Forest flags it as outlier AND at least one classifier predicts wrong label
df_all['lda_wrong'] = df_all['lda_pred'] != df_all['label']
df_all['rf_wrong'] = df_all['rf_pred'] != df_all['label']
df_all['both_wrong'] = df_all['lda_wrong'] & df_all['rf_wrong']
df_all['flag_suspicious'] = df_all['both_wrong'] | (df_all['iso_outlier'] & (df_all['lda_wrong'] | df_all['rf_wrong']))
df_all['flag_outlier_only'] = df_all['iso_outlier'] & ~df_all['both_wrong']

# ── Step 6: Print results ─────────────────────────────────────────────────────
print("\n" + "="*80)
print("整体分类性能（5-fold cross-val）")
print("="*80)
cm_lda = confusion_matrix(y, lda_pred)
cm_rf = confusion_matrix(y, rf_pred)
acc_lda = np.mean(lda_pred == y)
acc_rf = np.mean(rf_pred == y)
print(f"  LDA 准确率: {acc_lda:.3f}  混淆矩阵: {cm_lda.tolist()}")
print(f"  RF  准确率: {acc_rf:.3f}  混淆矩阵: {cm_rf.tolist()}")

print("\n" + "="*80)
print("【高度可疑】两个分类器都预测为另一类（很可能特征不符合所在文件夹）")
print("="*80)
suspicious = df_all[df_all['flag_suspicious']].sort_values(['label','rf_proba_wrong'], ascending=[True, False])
if len(suspicious) == 0:
    print("  无")
for _, row in suspicious.iterrows():
    print(f"\n  [Label={row['label']} → 疑似应为 {1-row['label']}]  {row['folder']}/{row['file']}")
    print(f"    LDA预测={row['lda_pred']}(错误概率={row['lda_proba_wrong']:.3f})  RF预测={row['rf_pred']}(错误概率={row['rf_proba_wrong']:.3f})  IsoForest异常={row['iso_outlier']}")
    print(f"    tap_rate={row['tap_rate']:.2f}Hz  iti_cv={row['iti_cv']:.3f}  amp_mean={row['amp_mean']:.2f}  amp_slope_norm={row['amp_slope_norm']:.3f}  hesitation={row['hesitation_rate']:.3f}")

print("\n" + "="*80)
print("【组内异常】IsolationForest 标记为本类群中的离群点（特征极端但分类器未误判）")
print("="*80)
outlier_only = df_all[df_all['flag_outlier_only']].sort_values('label')
if len(outlier_only) == 0:
    print("  无")
for _, row in outlier_only.iterrows():
    print(f"\n  [Label={row['label']}]  {row['folder']}/{row['file']}")
    print(f"    LDA预测={row['lda_pred']}  RF预测={row['rf_pred']}  z_outlier={row['z_outlier']}")
    print(f"    tap_rate={row['tap_rate']:.2f}Hz  iti_cv={row['iti_cv']:.3f}  amp_mean={row['amp_mean']:.2f}  amp_slope_norm={row['amp_slope_norm']:.3f}  hesitation={row['hesitation_rate']:.3f}")

print("\n" + "="*80)
print("总结")
print("="*80)
print(f"  高度可疑（建议移除或重新核查）: {df_all['flag_suspicious'].sum()} 个文件")
print(f"  组内离群（特征极端，酌情处理）: {df_all['flag_outlier_only'].sum()} 个文件")
print(f"  完全正常: {(~df_all['flag_suspicious'] & ~df_all['flag_outlier_only']).sum()} 个文件")

# Save full results
df_all.drop(columns=['lda_pred','rf_pred','lda_wrong','rf_wrong']).to_csv(
    '/home/user/LSTM-VAE-ON-PD/label_consistency_check.csv', index=False)
print("\n详细结果已保存到 label_consistency_check.csv")
