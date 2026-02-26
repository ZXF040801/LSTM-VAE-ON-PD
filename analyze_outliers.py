"""
Outlier detection based on clinical features for UPDRS 0 vs 1 classification.
Features: tap rhythm, power band ratios, angular velocity of s1_pitch.
Data columns: sensor_id | x | y | z | roll | pitch | yaw | timestamp
"""

import os
import numpy as np
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


# ── 1. File loading ────────────────────────────────────────────────────────────

def load_file(filepath):
    """Parse a tapping file into paired (s1, s2, timestamps) arrays."""
    data_by_time = {}
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue
            try:
                sid = parts[0].strip()
                vals = [float(p) for p in parts[1:7]]   # x y z roll pitch yaw
                ts   = int(parts[7])
                if sid not in ('01', '02'):
                    continue
                if ts not in data_by_time:
                    data_by_time[ts] = {}
                data_by_time[ts][sid] = vals
            except Exception:
                continue

    paired = sorted(ts for ts, d in data_by_time.items()
                    if '01' in d and '02' in d)
    if len(paired) < 50:
        return None, None, None

    s1 = np.array([data_by_time[t]['01'] for t in paired])   # (N,6)
    s2 = np.array([data_by_time[t]['02'] for t in paired])   # (N,6)
    ts_arr = np.array(paired, dtype=np.float64)
    return s1, s2, ts_arr


# ── 2. Feature helpers ─────────────────────────────────────────────────────────

def detect_taps(dist):
    kernel = np.ones(5) / 5.0
    smooth = np.convolve(dist, kernel, mode='same')
    mu, sigma = np.mean(smooth), np.std(smooth)
    if sigma < 1e-6:
        return np.array([], dtype=int)
    peaks, _ = find_peaks(smooth,
                          height=mu + 0.3 * sigma,
                          distance=10,
                          prominence=0.3 * sigma)
    return peaks


def power_band_ratio(signal, fs, f_low, f_high):
    sig = signal - np.mean(signal)
    Y   = np.abs(np.fft.rfft(sig)) ** 2          # power spectrum
    P_total = np.sum(Y)
    if P_total < 1e-12:
        return 0.0
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / fs)
    mask  = (freqs >= f_low) & (freqs < f_high)
    return float(np.sum(Y[mask]) / P_total)


# ── 3. Feature extraction ──────────────────────────────────────────────────────

def extract_features(filepath):
    s1, s2, ts = load_file(filepath)
    if s1 is None:
        return None

    # Estimate sampling rate from timestamps (ms → Hz)
    diffs = np.diff(ts)
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return None
    fs = float(np.clip(1000.0 / np.median(diffs), 20, 200))
    dt = 1.0 / fs

    duration = (ts[-1] - ts[0]) / 1000.0   # seconds
    if duration < 2.0:
        return None

    # Inter-finger distance (3-D Euclidean)
    dist = np.sqrt(np.sum((s1[:, :3] - s2[:, :3]) ** 2, axis=1))

    peaks = detect_taps(dist)

    feats = {}

    # --- Rhythm features ---
    if len(peaks) >= 2:
        peak_t = ts[peaks] / 1000.0
        iti    = np.diff(peak_t)
        iti    = iti[iti > 0]
        if len(iti) >= 1:
            feats['dist_iti_mean']    = float(np.mean(iti))
            feats['dist_iti_cv']      = float(np.std(iti) / (np.mean(iti) + 1e-8))
            feats['dist_tap_jitter']  = (float(np.mean(np.abs(np.diff(iti))) / (np.mean(iti) + 1e-8))
                                         if len(iti) >= 2 else 0.0)
        else:
            feats['dist_iti_mean'] = feats['dist_iti_cv'] = feats['dist_tap_jitter'] = np.nan
    else:
        feats['dist_iti_mean'] = feats['dist_iti_cv'] = feats['dist_tap_jitter'] = np.nan

    feats['dist_tap_rate'] = float(len(peaks) / duration) if len(peaks) >= 3 else 0.0

    # --- Power band features ---
    s1_roll  = s1[:, 3]
    s1_pitch = s1[:, 4]
    s1_x     = s1[:, 0]
    s2_y     = s2[:, 1]

    feats['s1_roll_pow_2_5hz']   = power_band_ratio(s1_roll,  fs, 2, 5)
    feats['dist_pow_0_2hz']      = power_band_ratio(dist,     fs, 0, 2)
    feats['dist_pow_2_5hz']      = power_band_ratio(dist,     fs, 2, 5)
    feats['s1_x_pow_0_2hz']      = power_band_ratio(s1_x,     fs, 0, 2)
    feats['s1_roll_pow_0_2hz']   = power_band_ratio(s1_roll,  fs, 0, 2)
    feats['s1_x_pow_2_5hz']      = power_band_ratio(s1_x,     fs, 2, 5)
    feats['s2_y_pow_0_2hz']      = power_band_ratio(s2_y,     fs, 0, 2)
    feats['s1_pitch_pow_0_2hz']  = power_band_ratio(s1_pitch, fs, 0, 2)

    # --- Angular velocity features ---
    vel = np.diff(s1_pitch) / dt
    feats['s1_pitch_vel_rms'] = float(np.sqrt(np.mean(vel ** 2)))
    feats['s1_pitch_vel_std'] = float(np.std(vel))
    feats['s1_pitch_vel_p95'] = float(np.percentile(np.abs(vel), 95))

    return feats


# ── 4. Run extraction ──────────────────────────────────────────────────────────

BASE   = '/home/user/LSTM-VAE-ON-PD'
FEATS  = ['dist_iti_mean', 'dist_tap_rate', 'dist_iti_cv', 'dist_tap_jitter',
          's1_roll_pow_2_5hz', 'dist_pow_0_2hz', 'dist_pow_2_5hz',
          's1_x_pow_0_2hz', 's1_roll_pow_0_2hz', 's1_x_pow_2_5hz',
          's2_y_pow_0_2hz', 's1_pitch_pow_0_2hz',
          's1_pitch_vel_rms', 's1_pitch_vel_std', 's1_pitch_vel_p95']

all_data = {}
skipped  = []

for label in ('0', '1'):
    folder = os.path.join(BASE, label)
    for fname in sorted(os.listdir(folder)):
        if not fname.endswith('.txt'):
            continue
        fpath = os.path.join(folder, fname)
        feats = extract_features(fpath)
        if feats is not None:
            all_data[(label, fname)] = feats
        else:
            skipped.append(f"{label}/{fname}")

class0 = {k: v for k, v in all_data.items() if k[0] == '0'}
class1 = {k: v for k, v in all_data.items() if k[0] == '1'}

print(f"Processed  — Class 0: {len(class0)}  |  Class 1: {len(class1)}")
if skipped:
    print(f"Skipped ({len(skipped)}): {skipped[:5]} …")


# ── 5. Class means and expected direction ──────────────────────────────────────

def get_vals(class_dict, feat):
    return np.array([v[feat] for v in class_dict.values()
                     if feat in v and not np.isnan(v.get(feat, np.nan))])

class_med = {}   # feat → (median_0, median_1)
print(f"\n{'Feature':<28}  {'Med_0':>9}  {'Med_1':>9}  {'Dir':>6}")
for f in FEATS:
    v0 = get_vals(class0, f)
    v1 = get_vals(class1, f)
    m0 = np.median(v0) if len(v0) else np.nan
    m1 = np.median(v1) if len(v1) else np.nan
    class_med[f] = (m0, m1)
    direction = '1>0' if (not np.isnan(m0) and not np.isnan(m1) and m1 > m0) else '0>1'
    print(f"{f:<28}  {m0:>9.4f}  {m1:>9.4f}  {direction:>6}")


# ── 6. Outlier scoring ─────────────────────────────────────────────────────────
#
#  For every feature, "higher in class X" is the expected pattern.
#  A file in class X that shows the OPPOSITE pattern contributes a
#  "wrong-direction" vote. We also weight by z-score magnitude.
#
#  We signed-normalise each feature globally so that positive = "looks like UPDRS-1".
#  Then for each file we compute the mean signed z-score as its "class-1 likeness".
#
# Expected sign (+1 ↔ feature higher in class-1), determined EMPIRICALLY from data:
# Rhythm features: UPDRS-1 is slower → longer ITI, lower tap rate, more variable
# Power features: UPDRS-0 taps sharper/faster → more 2-5Hz energy in dist/roll/x
#                 UPDRS-1 taps slower/rounder → more 0-2Hz energy in all signals
# Velocity: UPDRS-0 moves faster/more vigorously → higher angular velocity
EXPECTED_SIGN = {
    'dist_iti_mean':       +1,   # 1>0 empirically: UPDRS-1 taps slower
    'dist_tap_rate':       -1,   # 0>1 empirically: UPDRS-0 taps faster
    'dist_iti_cv':         +1,   # 1>0 empirically: UPDRS-1 more variable
    'dist_tap_jitter':     +1,   # 1>0 empirically: UPDRS-1 more jittery
    's1_roll_pow_2_5hz':   -1,   # 0>1 empirically: UPDRS-0 sharper taps → more harmonics
    'dist_pow_0_2hz':      +1,   # 1>0 empirically: UPDRS-1 slow movement → low-freq dominant
    'dist_pow_2_5hz':      -1,   # 0>1 empirically: UPDRS-0 fast taps → 2-5Hz harmonic content
    's1_x_pow_0_2hz':      +1,   # 1>0 empirically
    's1_roll_pow_0_2hz':   +1,   # 1>0 empirically
    's1_x_pow_2_5hz':      -1,   # 0>1 empirically
    's2_y_pow_0_2hz':      +1,   # 1>0 empirically
    's1_pitch_pow_0_2hz':  +1,   # 1>0 empirically
    's1_pitch_vel_rms':    -1,   # 0>1 empirically: UPDRS-0 faster movement
    's1_pitch_vel_std':    -1,   # 0>1 empirically: UPDRS-0 more dynamic
    's1_pitch_vel_p95':    -1,   # 0>1 empirically: UPDRS-0 higher peak velocity
}

# Global (combined) mean & std for z-scoring
all_keys = list(all_data.keys())
feat_stats = {}
for f in FEATS:
    vals = np.array([all_data[k][f] for k in all_keys
                     if f in all_data[k] and not np.isnan(all_data[k].get(f, np.nan))])
    feat_stats[f] = (np.mean(vals), np.std(vals) + 1e-9)

# Compute "UPDRS-1 likeness score" for every file
scores = {}
for key, feats in all_data.items():
    signed_z = []
    for f in FEATS:
        v = feats.get(f, np.nan)
        if np.isnan(v):
            continue
        mu, sigma = feat_stats[f]
        z = (v - mu) / sigma                   # raw z-score
        sz = EXPECTED_SIGN[f] * z              # flip so positive = more class-1-like
        signed_z.append(sz)
    scores[key] = np.mean(signed_z) if signed_z else 0.0

# Separate scores by class
scores0 = {k: scores[k] for k in class0}    # class 0 files → expect LOW score
scores1 = {k: scores[k] for k in class1}    # class 1 files → expect HIGH score


# ── 7. Report outliers ────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("OUTLIERS IN CLASS 0  (score > 0 means 'looks like UPDRS-1')")
print("=" * 70)
sorted0 = sorted(scores0.items(), key=lambda x: x[1], reverse=True)
c0_outliers = [(k, s) for k, s in sorted0 if s > 0]
for (label, fname), sc in c0_outliers:
    feats = all_data[(label, fname)]
    print(f"  score={sc:+.3f}  {fname}")
    print(f"    iti_mean={feats.get('dist_iti_mean', float('nan')):.3f}s  "
          f"tap_rate={feats.get('dist_tap_rate', 0):.2f}Hz  "
          f"iti_cv={feats.get('dist_iti_cv', float('nan')):.3f}  "
          f"dist_pow_2_5={feats.get('dist_pow_2_5hz', 0):.3f}  "
          f"vel_p95={feats.get('s1_pitch_vel_p95', 0):.1f}")

print(f"\nTotal class-0 outliers (score>0): {len(c0_outliers)}")

print("\n" + "=" * 70)
print("OUTLIERS IN CLASS 1  (score < 0 means 'looks like UPDRS-0')")
print("=" * 70)
sorted1 = sorted(scores1.items(), key=lambda x: x[1])
c1_outliers = [(k, s) for k, s in sorted1 if s < 0]
for (label, fname), sc in c1_outliers:
    feats = all_data[(label, fname)]
    print(f"  score={sc:+.3f}  {fname}")
    print(f"    iti_mean={feats.get('dist_iti_mean', float('nan')):.3f}s  "
          f"tap_rate={feats.get('dist_tap_rate', 0):.2f}Hz  "
          f"iti_cv={feats.get('dist_iti_cv', float('nan')):.3f}  "
          f"dist_pow_2_5={feats.get('dist_pow_2_5hz', 0):.3f}  "
          f"vel_p95={feats.get('s1_pitch_vel_p95', 0):.1f}")

print(f"\nTotal class-1 outliers (score<0): {len(c1_outliers)}")


# ── 8. Summary ────────────────────────────────────────────────────────────────

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Current counts:  Class-0={len(class0)},  Class-1={len(class1)}")
print(f"Suggested to REMOVE from class 0 (clearly mis-fitting): "
      f"{len(c0_outliers)} files")
print(f"Suggested to REMOVE from class 1 (clearly mis-fitting): "
      f"{len(c1_outliers)} files")
after0 = len(class0) - len(c0_outliers)
after1 = len(class1) - len(c1_outliers)
print(f"After removal:   Class-0={after0},  Class-1={after1}  "
      f"(ratio {after0/max(after1,1):.2f}:1)")

# Ranked top-N outliers most strongly out-of-place
print("\n── Top 10 STRONGEST class-0 outliers ──")
for (label, fname), sc in sorted0[:10]:
    print(f"  {sc:+.3f}  {fname}")

print("\n── Top 10 STRONGEST class-1 outliers ──")
for (label, fname), sc in sorted1[:10]:
    print(f"  {sc:+.3f}  {fname}")
