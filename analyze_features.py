"""
UPDRS 0 vs 1 Finger Tapping Feature Analysis
Extracts discriminative features from 6-DOF (position + orientation) data
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. DATA LOADING
# ─────────────────────────────────────────────
def load_file(filepath):
    """
    File format: sensor_id  x  y  z  roll  pitch  yaw  timestamp
    sensor 01 = thumb, sensor 02 = index finger
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                try:
                    sid   = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    roll, pitch, yaw = float(parts[4]), float(parts[5]), float(parts[6])
                    ts    = int(parts[7])
                    rows.append([sid, x, y, z, roll, pitch, yaw, ts])
                except:
                    continue
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=['sensor','x','y','z','roll','pitch','yaw','ts'])
    return df


def get_sensor(df, sid):
    s = df[df['sensor'] == sid].copy().sort_values('ts').reset_index(drop=True)
    return s


# ─────────────────────────────────────────────
# 2. FEATURE EXTRACTION
# ─────────────────────────────────────────────
def compute_features(df, fname):
    """Compute ~40 features from a single recording."""
    feats = {}
    ok = True

    # ---- split by sensor ----
    s1 = get_sensor(df, 1)   # thumb (or finger A)
    s2 = get_sensor(df, 2)   # index finger (or finger B)

    if len(s1) < 30 or len(s2) < 30:
        feats['_invalid'] = 'too_short'
        return feats

    # ---- time axis (seconds) ----
    t1 = (s1['ts'].values - s1['ts'].values[0]) / 1000.0
    t2 = (s2['ts'].values - s2['ts'].values[0]) / 1000.0
    duration1 = t1[-1] - t1[0]
    duration2 = t2[-1] - t2[0]

    # Minimal duration check (< 3 s → likely corrupted)
    if duration1 < 3.0 or duration2 < 3.0:
        feats['_invalid'] = 'too_short_duration'
        return feats

    feats['duration_s1'] = duration1
    feats['duration_s2'] = duration2

    # ─── Sensor-level position features ───
    for sid, s, t, tag in [(1, s1, t1, 's1'), (2, s2, t2, 's2')]:
        for col in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
            vals = s[col].values
            feats[f'{tag}_{col}_mean']  = np.mean(vals)
            feats[f'{tag}_{col}_std']   = np.std(vals)
            feats[f'{tag}_{col}_range'] = np.ptp(vals)  # max - min (amplitude proxy)
            feats[f'{tag}_{col}_iqr']   = np.percentile(vals, 75) - np.percentile(vals, 25)

        # Velocity (1st derivative)
        dt = np.diff(t)
        dt = np.where(dt == 0, 1e-6, dt)  # avoid /0
        for col in ['x', 'y', 'z']:
            v = np.diff(s[col].values) / dt
            feats[f'{tag}_{col}_vel_mean'] = np.mean(np.abs(v))
            feats[f'{tag}_{col}_vel_std']  = np.std(v)
            feats[f'{tag}_{col}_vel_max']  = np.max(np.abs(v))

        # Combined 3-D position speed
        pos = s[['x','y','z']].values
        dpos = np.diff(pos, axis=0)
        speed_3d = np.sqrt((dpos**2).sum(axis=1)) / dt
        feats[f'{tag}_speed3d_mean'] = np.mean(speed_3d)
        feats[f'{tag}_speed3d_max']  = np.max(speed_3d)
        feats[f'{tag}_speed3d_std']  = np.std(speed_3d)

        # --- Amplitude decrement (UPDRS key: bradykinesia) ---
        # Split recording into 3 thirds and compare amplitude of each third
        z = s['z'].values   # z typically captures open-close movement
        n = len(z)
        thirds = np.array_split(z, 3)
        amp_thirds = [np.ptp(t_) for t_ in thirds]
        feats[f'{tag}_amp_decrement'] = (amp_thirds[0] - amp_thirds[2]) / (amp_thirds[0] + 1e-6)
        # Normalised slope of amplitude over thirds
        feats[f'{tag}_amp_slope'] = np.polyfit([0, 1, 2], amp_thirds, 1)[0]

        # --- Dominant frequency (FFT on z) ---
        # Resample to uniform grid first (approximate)
        fs_approx = len(z) / (t[-1] - t[0])  # mean sample rate
        freqs = fftfreq(len(z), 1.0 / fs_approx)
        mag   = np.abs(fft(z - np.mean(z)))
        pos_mask = freqs > 0
        dom_freq  = freqs[pos_mask][np.argmax(mag[pos_mask])]
        feats[f'{tag}_dom_freq_hz'] = dom_freq
        # Power in 1-5 Hz band (typical tapping)
        band_mask = pos_mask & (freqs >= 1.0) & (freqs <= 5.0)
        feats[f'{tag}_power_1_5hz'] = np.sum(mag[band_mask]**2)
        total_power = np.sum(mag[pos_mask]**2) + 1e-6
        feats[f'{tag}_pwr_ratio_1_5hz'] = feats[f'{tag}_power_1_5hz'] / total_power

        # --- Peak detection → tapping events ---
        # Use z-axis (or magnitude); find local maxima
        smooth = np.convolve(z, np.ones(5)/5, mode='same')
        pks, pk_props = signal.find_peaks(smooth,
                                          prominence=np.std(smooth)*0.3,
                                          distance=int(fs_approx * 0.15))
        if len(pks) >= 3:
            ipi = np.diff(t[pks])        # inter-peak intervals
            feats[f'{tag}_tap_rate']     = len(pks) / duration1
            feats[f'{tag}_ipi_mean']     = np.mean(ipi)
            feats[f'{tag}_ipi_std']      = np.std(ipi)
            feats[f'{tag}_ipi_cv']       = np.std(ipi) / (np.mean(ipi) + 1e-6)  # coefficient of variation
            # Rhythm regularity: lower CV → more regular
            # Decrement in peak-to-peak amplitude over time
            pk_amps = smooth[pks]
            if len(pk_amps) >= 3:
                half = len(pk_amps) // 2
                feats[f'{tag}_pk_amp_early'] = np.mean(pk_amps[:half])
                feats[f'{tag}_pk_amp_late']  = np.mean(pk_amps[half:])
                feats[f'{tag}_pk_amp_decrement'] = (
                    (feats[f'{tag}_pk_amp_early'] - feats[f'{tag}_pk_amp_late'])
                    / (feats[f'{tag}_pk_amp_early'] + 1e-6)
                )
            else:
                feats[f'{tag}_pk_amp_early'] = np.nan
                feats[f'{tag}_pk_amp_late']  = np.nan
                feats[f'{tag}_pk_amp_decrement'] = np.nan
        else:
            for k in ['tap_rate','ipi_mean','ipi_std','ipi_cv',
                      'pk_amp_early','pk_amp_late','pk_amp_decrement']:
                feats[f'{tag}_{k}'] = np.nan
            feats['_invalid'] = 'too_few_peaks'

    # ─── Inter-sensor distance features ───
    # Align by common time range
    t_start = max(s1['ts'].min(), s2['ts'].min())
    t_end   = min(s1['ts'].max(), s2['ts'].max())
    if t_end > t_start:
        s1_trim = s1[(s1['ts'] >= t_start) & (s1['ts'] <= t_end)]
        s2_trim = s2[(s2['ts'] >= t_start) & (s2['ts'] <= t_end)]
        # Use shorter one and nearest-neighbor match
        n_common = min(len(s1_trim), len(s2_trim))
        if n_common >= 10:
            p1 = s1_trim[['x','y','z']].values[:n_common]
            p2 = s2_trim[['x','y','z']].values[:n_common]
            dist3d = np.sqrt(((p1 - p2)**2).sum(axis=1))
            feats['intersensor_dist_mean'] = np.mean(dist3d)
            feats['intersensor_dist_std']  = np.std(dist3d)
            feats['intersensor_dist_range'] = np.ptp(dist3d)
            feats['intersensor_dist_min']  = np.min(dist3d)
        else:
            for k in ['intersensor_dist_mean','intersensor_dist_std',
                      'intersensor_dist_range','intersensor_dist_min']:
                feats[k] = np.nan

    return feats


# ─────────────────────────────────────────────
# 3. LOAD ALL FILES
# ─────────────────────────────────────────────
base = '/home/user/LSTM-VAE-ON-PD'
records = []

for label in [0, 1]:
    folder = os.path.join(base, str(label))
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(folder, fname)
        df = load_file(fpath)
        if df is None or len(df) == 0:
            records.append({'fname': fname, 'label': label, '_invalid': 'empty'})
            continue
        feats = compute_features(df, fname)
        feats['fname']  = fname
        feats['label']  = label
        feats['fpath']  = fpath
        records.append(feats)

all_df = pd.DataFrame(records)
print(f"Total files loaded: {len(all_df)}")
print(f"  UPDRS=0: {(all_df['label']==0).sum()}")
print(f"  UPDRS=1: {(all_df['label']==1).sum()}")

# ─────────────────────────────────────────────
# 4. IDENTIFY INVALID / OUTLIER FILES
# ─────────────────────────────────────────────
all_df = all_df.reset_index(drop=True)

# 4a. Files flagged as invalid by the parser
if '_invalid' in all_df.columns:
    invalid_mask = all_df['_invalid'].notna()
else:
    invalid_mask = pd.Series([False] * len(all_df), index=all_df.index)

cols_to_show = [c for c in ['fname','label','_invalid'] if c in all_df.columns]
invalid_files = all_df[invalid_mask][cols_to_show].copy()
print(f"\nFiles flagged as INVALID ({len(invalid_files)}):")
print(invalid_files.to_string(index=False))

# Keep only valid files for further analysis
valid_df = all_df[~invalid_mask].copy().reset_index(drop=True)
print(f"\nValid files: {len(valid_df)}")

# ─────────────────────────────────────────────
# 5. OUTLIER DETECTION (IQR method on key features)
# ─────────────────────────────────────────────
key_outlier_features = [
    's1_z_range', 's2_z_range',
    's1_speed3d_mean', 's2_speed3d_mean',
    's1_tap_rate', 's2_tap_rate',
    's1_ipi_cv', 's2_ipi_cv',
    'intersensor_dist_mean',
]

outlier_files = set()
outlier_reasons = {}

for feat in key_outlier_features:
    if feat not in valid_df.columns:
        continue
    for lbl in [0, 1]:
        sub = valid_df[valid_df['label'] == lbl][feat].dropna()
        if len(sub) < 5:
            continue
        Q1 = sub.quantile(0.25)
        Q3 = sub.quantile(0.75)
        IQR = Q3 - Q1
        lo  = Q1 - 3.0 * IQR
        hi  = Q3 + 3.0 * IQR
        out_idx = valid_df[
            (valid_df['label'] == lbl) &
            (valid_df[feat].notna()) &
            ((valid_df[feat] < lo) | (valid_df[feat] > hi))
        ].index
        for idx in out_idx:
            fn = valid_df.loc[idx, 'fname']
            outlier_files.add(fn)
            reason = f"{feat}={valid_df.loc[idx,feat]:.3f} (class{lbl}: [{lo:.3f},{hi:.3f}])"
            outlier_reasons.setdefault(fn, []).append(reason)

print(f"\nOutlier files detected (IQR ×3): {len(outlier_files)}")
for fn, reasons in sorted(outlier_reasons.items()):
    lbl = valid_df[valid_df['fname']==fn]['label'].values[0]
    print(f"  [{lbl}] {fn}")
    for r in reasons[:3]:
        print(f"       {r}")

# ─────────────────────────────────────────────
# 6. FEATURE IMPORTANCE (Mann-Whitney U + effect size)
# ─────────────────────────────────────────────
feature_cols = [c for c in valid_df.columns
                if c not in ('fname','label','fpath','_invalid')]

results = []
grp0 = valid_df[valid_df['label'] == 0]
grp1 = valid_df[valid_df['label'] == 1]

for feat in feature_cols:
    v0 = grp0[feat].dropna().values.astype(float)
    v1 = grp1[feat].dropna().values.astype(float)
    if len(v0) < 5 or len(v1) < 5:
        continue
    try:
        stat, pval = stats.mannwhitneyu(v0, v1, alternative='two-sided')
    except:
        continue
    # Effect size: rank-biserial correlation
    n0, n1 = len(v0), len(v1)
    rbc = 1 - (2 * stat) / (n0 * n1)
    # Means
    m0, m1 = np.mean(v0), np.mean(v1)
    # Cohen's d
    pooled_std = np.sqrt((np.std(v0)**2 + np.std(v1)**2) / 2)
    cohens_d = (m1 - m0) / (pooled_std + 1e-9)
    results.append({
        'feature': feat,
        'p_value': pval,
        'rank_biserial': abs(rbc),
        'cohens_d': abs(cohens_d),
        'mean_UPDRS0': m0,
        'mean_UPDRS1': m1,
        'direction': 'UPDRS1>' if m1 > m0 else 'UPDRS0>',
    })

results_df = pd.DataFrame(results).sort_values('rank_biserial', ascending=False)

print("\n" + "="*80)
print("TOP DISCRIMINATIVE FEATURES (sorted by rank-biserial effect size)")
print("="*80)
top20 = results_df.head(25)
pd.set_option('display.float_format', '{:.4f}'.format)
pd.set_option('display.max_colwidth', 50)
print(top20[['feature','p_value','rank_biserial','cohens_d','mean_UPDRS0','mean_UPDRS1','direction']].to_string(index=False))

print("\n" + "="*80)
print("STATISTICALLY SIGNIFICANT FEATURES (p < 0.05)")
print("="*80)
sig_df = results_df[results_df['p_value'] < 0.05].copy()
print(sig_df[['feature','p_value','rank_biserial','cohens_d','mean_UPDRS0','mean_UPDRS1','direction']].to_string(index=False))

# ─────────────────────────────────────────────
# 7. SUMMARY OF FILES TO DELETE
# ─────────────────────────────────────────────
to_delete_invalid = invalid_files['fpath'].tolist() if 'fpath' in invalid_files.columns else []
# Build fpath for outliers
outlier_fpaths = valid_df[valid_df['fname'].isin(outlier_files)]['fpath'].tolist()
all_to_delete = list(set(to_delete_invalid + outlier_fpaths))

print("\n" + "="*80)
print(f"FILES TO DELETE (total: {len(all_to_delete)})")
print("="*80)
for fp in sorted(all_to_delete):
    lbl = '0' if '/0/' in fp else '1'
    print(f"  [{lbl}] {os.path.basename(fp)}")

# Save lists
with open('/home/user/LSTM-VAE-ON-PD/outliers_to_delete.txt', 'w') as f:
    for fp in sorted(all_to_delete):
        f.write(fp + '\n')

# Save feature importance table
results_df.to_csv('/home/user/LSTM-VAE-ON-PD/feature_importance.csv', index=False)
sig_df.to_csv('/home/user/LSTM-VAE-ON-PD/significant_features.csv', index=False)

print("\nSaved: outliers_to_delete.txt, feature_importance.csv, significant_features.csv")
