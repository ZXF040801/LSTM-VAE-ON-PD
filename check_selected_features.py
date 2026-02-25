import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, welch
from scipy.ndimage import uniform_filter1d
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_predict
import warnings
warnings.filterwarnings('ignore')

SELECTED_FEATURES = [
    'dist_iti_mean', 'dist_tap_rate', 'dist_iti_cv', 'dist_tap_jitter',
    's2_z_tap_rate',
    'dist_pow_0_2hz', 'dist_pow_2_5hz',
    's1_roll_pow_0_2hz', 's1_pitch_pow_0_2hz', 's1_roll_dom_freq',
    's1_pitch_vel_rms', 's1_x_vel_rms', 's2_pitch_vel_std',
    'dist_iqr', 'dist_std',
]

def load_file(filepath):
    rows = []
    with open(filepath) as f:
        for line in f:
            p = line.strip().split()
            if len(p) == 8:
                try:
                    rows.append([int(p[0]), float(p[1]), float(p[2]),
                                 float(p[3]), float(p[4]), float(p[5]),
                                 float(p[6]), int(p[7])])
                except ValueError:
                    pass
    return rows

def band_power(sig, fs, flo, fhi):
    """Power in [flo, fhi] Hz band relative to total power."""
    if len(sig) < 16:
        return np.nan
    nperseg = min(len(sig), 256)
    f, psd = welch(sig, fs=fs, nperseg=nperseg)
    total = np.trapezoid(psd, f)
    if total == 0:
        return np.nan
    band = np.trapezoid(psd[(f >= flo) & (f <= fhi)], f[(f >= flo) & (f <= fhi)])
    return band / total

def dominant_freq(sig, fs):
    if len(sig) < 16:
        return np.nan
    nperseg = min(len(sig), 256)
    f, psd = welch(sig, fs=fs, nperseg=nperseg)
    return f[np.argmax(psd)]

def velocity_rms(sig, ts_ms):
    dt = np.diff(ts_ms) / 1000.0
    dt = np.where(dt <= 0, 0.001, dt)
    vel = np.diff(sig) / dt
    return np.sqrt(np.mean(vel**2))

def velocity_std(sig, ts_ms):
    dt = np.diff(ts_ms) / 1000.0
    dt = np.where(dt <= 0, 0.001, dt)
    vel = np.diff(sig) / dt
    return np.std(vel)

def detect_taps(sig, fs):
    """Detect peaks (open = max distance = tap event)."""
    smooth = uniform_filter1d(sig, size=max(3, int(fs * 0.05)))
    min_dist = max(5, int(fs * 0.15))
    peaks, _ = find_peaks(smooth, distance=min_dist,
                           prominence=np.std(smooth) * 0.3)
    return peaks, smooth

def extract_features(filepath):
    rows = load_file(filepath)
    if len(rows) < 60:
        return None
    df = pd.DataFrame(rows, columns=['sensor','x','y','z','roll','pitch','yaw','ts'])
    df = df.drop_duplicates(subset=['sensor','ts'])

    s1 = df[df['sensor']==1].sort_values('ts').reset_index(drop=True)
    s2 = df[df['sensor']==2].sort_values('ts').reset_index(drop=True)
    if len(s1) < 50 or len(s2) < 50:
        return None

    # Align on common timestamps
    common = sorted(set(s1['ts'].values) & set(s2['ts'].values))
    if len(common) < 50:
        return None
    s1c = s1[s1['ts'].isin(common)].sort_values('ts').reset_index(drop=True)
    s2c = s2[s2['ts'].isin(common)].sort_values('ts').reset_index(drop=True)

    ts = s1c['ts'].values
    duration = (ts[-1] - ts[0]) / 1000.0
    if duration <= 0:
        return None
    fs = len(ts) / duration

    # Distance signal
    dx = s1c['x'].values - s2c['x'].values
    dy = s1c['y'].values - s2c['y'].values
    dz = s1c['z'].values - s2c['z'].values
    dist = np.sqrt(dx**2 + dy**2 + dz**2)

    feats = {}

    # ── Tap detection on dist ─────────────────────────────────────────────────
    peaks_d, dist_smooth = detect_taps(dist, fs)
    n_taps = len(peaks_d)
    feats['dist_tap_rate'] = n_taps / duration if duration > 0 else 0

    if n_taps >= 3:
        peak_ts = ts[peaks_d]
        iti = np.diff(peak_ts)           # ms
        feats['dist_iti_mean'] = np.mean(iti)
        feats['dist_iti_cv']   = np.std(iti) / np.mean(iti) if np.mean(iti) > 0 else 0
        # jitter = mean abs difference of consecutive ITIs
        feats['dist_tap_jitter'] = np.mean(np.abs(np.diff(iti))) if len(iti) > 1 else 0
    else:
        feats['dist_iti_mean'] = np.nan
        feats['dist_iti_cv']   = np.nan
        feats['dist_tap_jitter'] = np.nan

    # ── s2_z tap rate ─────────────────────────────────────────────────────────
    peaks_s2z, _ = detect_taps(s2c['z'].values, fs)
    feats['s2_z_tap_rate'] = len(peaks_s2z) / duration

    # ── Frequency domain on dist ──────────────────────────────────────────────
    feats['dist_pow_0_2hz'] = band_power(dist, fs, 0.1, 2.0)
    feats['dist_pow_2_5hz'] = band_power(dist, fs, 2.0, 5.0)

    # ── Frequency domain on s1 roll and pitch ─────────────────────────────────
    feats['s1_roll_pow_0_2hz']  = band_power(s1c['roll'].values,  fs, 0.1, 2.0)
    feats['s1_pitch_pow_0_2hz'] = band_power(s1c['pitch'].values, fs, 0.1, 2.0)
    feats['s1_roll_dom_freq']   = dominant_freq(s1c['roll'].values, fs)

    # ── Velocity RMS/STD ──────────────────────────────────────────────────────
    feats['s1_pitch_vel_rms'] = velocity_rms(s1c['pitch'].values, ts)
    feats['s1_x_vel_rms']     = velocity_rms(s1c['x'].values,     ts)
    feats['s2_pitch_vel_std'] = velocity_std(s2c['pitch'].values,  ts)

    # ── Distance statistics ───────────────────────────────────────────────────
    feats['dist_iqr'] = float(np.percentile(dist, 75) - np.percentile(dist, 25))
    feats['dist_std'] = float(np.std(dist))

    return feats

# ── Load all files ────────────────────────────────────────────────────────────
records = []
for lbl in ['0', '1']:
    folder = f'/home/user/LSTM-VAE-ON-PD/{lbl}/'
    for fname in sorted(os.listdir(folder)):
        feats = extract_features(os.path.join(folder, fname))
        if feats is None:
            continue
        feats['label'] = int(lbl)
        feats['file']  = fname
        feats['folder'] = lbl
        records.append(feats)

df = pd.DataFrame(records)
print(f"有效文件: {len(df)}  (label0={sum(df['label']==0)}, label1={sum(df['label']==1)})")

# ── 1. Group statistics & Cohen's d ──────────────────────────────────────────
print("\n" + "="*90)
print(f"{'特征':<25} {'Label0中位数':>12} {'Label1中位数':>12} {'Cohen|d|':>10} {'p值':>10}  {'区分度'}")
print("="*90)
effect_sizes = {}
for col in SELECTED_FEATURES:
    v0 = df[df['label']==0][col].dropna()
    v1 = df[df['label']==1][col].dropna()
    med0, med1 = v0.median(), v1.median()
    pooled_std = np.sqrt((v0.std()**2 + v1.std()**2) / 2)
    d = abs(med0 - med1) / pooled_std if pooled_std > 0 else 0
    _, pval = stats.mannwhitneyu(v0, v1, alternative='two-sided')
    mark = '★★★' if d > 0.6 else ('★★' if d > 0.4 else ('★' if d > 0.2 else ''))
    effect_sizes[col] = d
    print(f"  {col:<25} {med0:>12.4f} {med1:>12.4f} {d:>10.3f} {pval:>10.4f}  {mark}")

# ── 2. Classifier with SELECTED_FEATURES ─────────────────────────────────────
df_feat = df[SELECTED_FEATURES].copy()
for col in SELECTED_FEATURES:
    df_feat[col].fillna(df_feat[col].median(), inplace=True)

X = StandardScaler().fit_transform(df_feat.values)
y = df['label'].values

lda = LinearDiscriminantAnalysis()
rf  = RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced')
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

lda_proba = cross_val_predict(lda, X, y, cv=cv, method='predict_proba')
rf_proba  = cross_val_predict(rf,  X, y, cv=cv, method='predict_proba')
lda_pred  = np.argmax(lda_proba, axis=1)
rf_pred   = np.argmax(rf_proba,  axis=1)

acc_lda = np.mean(lda_pred == y)
acc_rf  = np.mean(rf_pred  == y)
print(f"\n{'='*90}")
print(f"分类器性能（5-fold, SELECTED_FEATURES）")
print(f"  LDA 准确率: {acc_lda:.3f}")
print(f"  RF  准确率: {acc_rf:.3f}")

# ── 3. Flag suspicious files ──────────────────────────────────────────────────
df['lda_pred'] = lda_pred
df['rf_pred']  = rf_pred
df['lda_proba_wrong'] = [lda_proba[i, 1-y[i]] for i in range(len(y))]
df['rf_proba_wrong']  = [rf_proba[i,  1-y[i]] for i in range(len(y))]
df['both_wrong'] = (df['lda_pred'] != df['label']) & (df['rf_pred'] != df['label'])

# Isolation Forest per class
iso_flag = np.zeros(len(df), dtype=bool)
for lbl in [0, 1]:
    mask = (y == lbl)
    iso = IsolationForest(contamination=0.10, random_state=42)
    pred = iso.fit_predict(X[mask])
    idx = np.where(mask)[0]
    iso_flag[idx] = pred == -1

df['iso_outlier'] = iso_flag
df['suspicious'] = df['both_wrong'] | (df['iso_outlier'] & (df['lda_pred'] != df['label']))

# Confidence tiers
df['tier'] = 'normal'
df.loc[df['suspicious'] & (df['rf_proba_wrong'] >= 0.65), 'tier'] = 'HIGH'
df.loc[df['suspicious'] & (df['rf_proba_wrong'] >= 0.50) & (df['rf_proba_wrong'] < 0.65), 'tier'] = 'MED'
df.loc[df['iso_outlier'] & ~df['suspicious'], 'tier'] = 'OUTLIER'

# ── 4. Print results ──────────────────────────────────────────────────────────
print(f"\n{'='*90}")
print("★★ 高度可疑 (RF≥65%，两分类器均误判) ★★")
print(f"{'='*90}")
high = df[df['tier']=='HIGH'].sort_values(['label','rf_proba_wrong'], ascending=[True,False])
print(f"共 {len(high)} 个")
for lbl in [0, 1]:
    sub = high[high['label']==lbl]
    print(f"\n  [folder {lbl}] 共 {len(sub)} 个文件")
    for _, r in sub.iterrows():
        print(f"    {r['file']}")
        print(f"      tap_rate={r['dist_tap_rate']:.2f}Hz  iti_mean={r['dist_iti_mean']:.0f}ms  iti_cv={r['dist_iti_cv']:.3f}  jitter={r['dist_tap_jitter']:.1f}  RF_err={r['rf_proba_wrong']:.3f}")
        print(f"      dist_pow_0_2hz={r['dist_pow_0_2hz']:.3f}  dist_pow_2_5hz={r['dist_pow_2_5hz']:.3f}  dom_freq={r['s1_roll_dom_freq']:.2f}Hz  vel_rms_pitch={r['s1_pitch_vel_rms']:.1f}")

print(f"\n{'='*90}")
print("△ 中度可疑 (RF 50-65%) △")
print(f"{'='*90}")
med_sus = df[df['tier']=='MED'].sort_values(['label','rf_proba_wrong'], ascending=[True,False])
print(f"共 {len(med_sus)} 个")
for lbl in [0, 1]:
    sub = med_sus[med_sus['label']==lbl]
    print(f"\n  [folder {lbl}] 共 {len(sub)} 个文件")
    for _, r in sub.iterrows():
        print(f"    {r['file']}")
        print(f"      tap_rate={r['dist_tap_rate']:.2f}Hz  iti_mean={r['dist_iti_mean']:.0f}ms  iti_cv={r['dist_iti_cv']:.3f}  jitter={r['dist_tap_jitter']:.1f}  RF_err={r['rf_proba_wrong']:.3f}")

print(f"\n{'='*90}")
print("◇ 组内离群 (IsoForest, 分类器未误判) ◇")
print(f"{'='*90}")
out_only = df[df['tier']=='OUTLIER'].sort_values('label')
print(f"共 {len(out_only)} 个")
for _, r in out_only.iterrows():
    print(f"  [folder {r['label']}] {r['file']}")
    print(f"    tap_rate={r['dist_tap_rate']:.2f}Hz  iti_cv={r['dist_iti_cv']:.3f}  dist_std={r['dist_std']:.2f}  RF_err={r['rf_proba_wrong']:.3f}")

print(f"\n{'='*90}")
print("汇总")
print(f"{'='*90}")
print(f"  HIGH (强烈建议移除):  {len(high)} 个")
print(f"  MED  (建议核查):      {len(med_sus)} 个")
print(f"  OUTLIER (组内离群):   {len(out_only)} 个")
print(f"  正常:                 {sum(df['tier']=='normal')} 个")

df.to_csv('/home/user/LSTM-VAE-ON-PD/selected_features_check.csv', index=False)
print("\n详细结果保存至 selected_features_check.csv")
