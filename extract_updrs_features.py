"""
UPDRS 0 vs 1 手指敲击测试特征提取与分析
数据格式: sensor_id(01/02)  x  y  z  roll  pitch  yaw  timestamp
"""

import os
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

# ─── 1. 数据加载 ───────────────────────────────────────────────────────────────

def load_file(filepath):
    """加载单个文件，返回 sensor01 和 sensor02 的 DataFrame"""
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                rows.append({
                    'sensor': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'z': float(parts[3]),
                    'roll': float(parts[4]),
                    'pitch': float(parts[5]),
                    'yaw': float(parts[6]),
                    'ts': int(parts[7])
                })
    df = pd.DataFrame(rows)
    s1 = df[df['sensor'] == 1].reset_index(drop=True)
    s2 = df[df['sensor'] == 2].reset_index(drop=True)
    return s1, s2


def load_folder(folder_path, label):
    """加载整个文件夹的所有文件"""
    records = []
    files = sorted(os.listdir(folder_path))
    print(f"  Loading {len(files)} files from label={label} ...")
    for fname in files:
        fpath = os.path.join(folder_path, fname)
        try:
            s1, s2 = load_file(fpath)
            if len(s1) < 50 or len(s2) < 50:
                continue
            records.append({'label': label, 'fname': fname, 's1': s1, 's2': s2})
        except Exception as e:
            print(f"    [SKIP] {fname}: {e}")
    return records


# ─── 2. 特征提取 ──────────────────────────────────────────────────────────────

def stat_features(arr, prefix):
    """统计特征：均值、标准差、范围、四分位距、偏度、峰度"""
    return {
        f'{prefix}_mean':   np.mean(arr),
        f'{prefix}_std':    np.std(arr),
        f'{prefix}_range':  np.ptp(arr),
        f'{prefix}_iqr':    np.percentile(arr, 75) - np.percentile(arr, 25),
        f'{prefix}_skew':   stats.skew(arr),
        f'{prefix}_kurt':   stats.kurtosis(arr),
        f'{prefix}_rms':    np.sqrt(np.mean(arr**2)),
        f'{prefix}_p95':    np.percentile(np.abs(arr), 95),
    }


def velocity_features(arr, ts, prefix):
    """速度与加速度统计特征"""
    dt = np.diff(ts) / 1000.0  # ms -> s
    dt = np.where(dt == 0, 1e-3, dt)
    vel = np.diff(arr) / dt
    acc = np.diff(vel) / dt[:-1]
    feat = {}
    feat.update(stat_features(vel, f'{prefix}_vel'))
    feat[f'{prefix}_vel_std'] = np.std(vel)
    feat[f'{prefix}_acc_std'] = np.std(acc)
    feat[f'{prefix}_acc_max'] = np.max(np.abs(acc))
    return feat


def freq_features(arr, ts, prefix, fs_nominal=64.0):
    """频域特征：主频率、频带能量"""
    n = len(arr)
    if n < 32:
        return {}
    # 重采样到均匀时间轴
    t_start, t_end = ts[0], ts[-1]
    duration = (t_end - t_start) / 1000.0  # seconds
    if duration <= 0:
        return {}
    fs = n / duration
    yf = np.abs(fft(arr - np.mean(arr)))[:n // 2]
    xf = fftfreq(n, 1.0 / fs)[:n // 2]

    if len(xf) == 0:
        return {}

    dominant_freq = xf[np.argmax(yf)]
    total_power = np.sum(yf**2)

    def band_power(f_low, f_high):
        mask = (xf >= f_low) & (xf < f_high)
        return np.sum(yf[mask]**2) / (total_power + 1e-10)

    return {
        f'{prefix}_dom_freq':     dominant_freq,
        f'{prefix}_pow_0_2hz':    band_power(0.0, 2.0),
        f'{prefix}_pow_2_5hz':    band_power(2.0, 5.0),
        f'{prefix}_pow_5_8hz':    band_power(5.0, 8.0),
        f'{prefix}_pow_8_12hz':   band_power(8.0, 12.0),
        f'{prefix}_spectral_entropy': _spectral_entropy(yf),
    }


def _spectral_entropy(yf):
    p = yf**2
    p = p / (p.sum() + 1e-10)
    return -np.sum(p * np.log(p + 1e-10))


def tap_features(signal_arr, ts, prefix):
    """
    敲击检测特征：
    利用信号的局部极值检测敲击事件，提取频率、幅度、时间间隔抖动
    """
    # 平滑
    if len(signal_arr) < 10:
        return {}
    smoothed = np.convolve(signal_arr, np.ones(5)/5, mode='same')
    # 找波峰（代表敲击顶点）
    height_thresh = np.mean(smoothed) + 0.3 * np.std(smoothed)
    peaks, props = find_peaks(smoothed,
                               height=height_thresh,
                               distance=10,
                               prominence=np.std(smoothed) * 0.3)
    if len(peaks) < 3:
        return {
            f'{prefix}_tap_count':      len(peaks),
            f'{prefix}_tap_rate':       0.0,
            f'{prefix}_iti_mean':       0.0,
            f'{prefix}_iti_std':        0.0,
            f'{prefix}_iti_cv':         0.0,
            f'{prefix}_tap_amp_mean':   0.0,
            f'{prefix}_tap_amp_std':    0.0,
            f'{prefix}_tap_amp_cv':     0.0,
            f'{prefix}_tap_jitter':     0.0,
            f'{prefix}_tap_shimmer':    0.0,
            f'{prefix}_tap_decay':      0.0,
        }

    tap_times = ts[peaks] / 1000.0  # seconds
    duration = (ts[-1] - ts[0]) / 1000.0
    iti = np.diff(tap_times)
    amps = smoothed[peaks]

    iti_mean = np.mean(iti)
    iti_std  = np.std(iti)
    iti_cv   = iti_std / (iti_mean + 1e-10)

    amp_mean = np.mean(amps)
    amp_std  = np.std(amps)
    amp_cv   = amp_std / (amp_mean + 1e-10)

    # Jitter: 相邻 ITI 差值的均值（类语音抖动）
    jitter = np.mean(np.abs(np.diff(iti))) / (iti_mean + 1e-10)
    # Shimmer: 相邻幅度差值均值
    shimmer = np.mean(np.abs(np.diff(amps))) / (amp_mean + 1e-10)

    # 幅度趋势（敲击是否随时间衰减 -> 疲劳）
    if len(amps) >= 4:
        slope, _, _, _, _ = stats.linregress(np.arange(len(amps)), amps)
        decay = slope / (amp_mean + 1e-10)
    else:
        decay = 0.0

    return {
        f'{prefix}_tap_count':    len(peaks),
        f'{prefix}_tap_rate':     len(peaks) / (duration + 1e-10),
        f'{prefix}_iti_mean':     iti_mean,
        f'{prefix}_iti_std':      iti_std,
        f'{prefix}_iti_cv':       iti_cv,
        f'{prefix}_tap_amp_mean': amp_mean,
        f'{prefix}_tap_amp_std':  amp_std,
        f'{prefix}_tap_amp_cv':   amp_cv,
        f'{prefix}_tap_jitter':   jitter,
        f'{prefix}_tap_shimmer':  shimmer,
        f'{prefix}_tap_decay':    decay,
    }


def inter_sensor_features(s1, s2):
    """双传感器相对运动特征（拇指-食指之间的距离变化）"""
    min_len = min(len(s1), len(s2))
    s1 = s1.iloc[:min_len]
    s2 = s2.iloc[:min_len]

    # 欧氏距离
    dist = np.sqrt((s1['x'].values - s2['x'].values)**2 +
                   (s1['y'].values - s2['y'].values)**2 +
                   (s1['z'].values - s2['z'].values)**2)
    ts = s1['ts'].values

    feat = {}
    feat.update(stat_features(dist, 'dist'))
    feat.update(freq_features(dist, ts, 'dist'))
    feat.update(tap_features(dist, ts, 'dist'))
    feat.update(velocity_features(dist, ts, 'dist'))

    # 各轴相对位移
    for ax in ['x', 'y', 'z']:
        rel = s1[ax].values - s2[ax].values
        feat.update(stat_features(rel, f'rel_{ax}'))

    # 各姿态角相对值
    for ang in ['roll', 'pitch', 'yaw']:
        rel = s1[ang].values - s2[ang].values
        feat.update(stat_features(rel, f'rel_{ang}'))

    return feat


def extract_features(record):
    """从单条记录提取全量特征"""
    s1 = record['s1']
    s2 = record['s2']
    ts1 = s1['ts'].values
    ts2 = s2['ts'].values

    feat = {'label': record['label'], 'fname': record['fname']}

    # Sensor 1 各通道
    for col in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
        arr = s1[col].values
        feat.update(stat_features(arr, f's1_{col}'))
        feat.update(freq_features(arr, ts1, f's1_{col}'))
        feat.update(velocity_features(arr, ts1, f's1_{col}'))

    # Sensor 2 各通道
    for col in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']:
        arr = s2[col].values
        feat.update(stat_features(arr, f's2_{col}'))
        feat.update(freq_features(arr, ts2, f's2_{col}'))
        feat.update(velocity_features(arr, ts2, f's2_{col}'))

    # 敲击检测（用 s1 的 z 轴位移，通常最能反映手指敲击）
    feat.update(tap_features(s1['z'].values, ts1, 's1_z'))
    feat.update(tap_features(s2['z'].values, ts2, 's2_z'))

    # 双传感器相对运动
    feat.update(inter_sensor_features(s1, s2))

    return feat


# ─── 3. 统计检验与排序 ────────────────────────────────────────────────────────

def analyze_features(df_feat):
    """对每个特征做 Mann-Whitney U 检验 + Cohen's d，返回排序结果"""
    label_col = 'label'
    fname_col = 'fname'
    feature_cols = [c for c in df_feat.columns if c not in [label_col, fname_col]]

    group0 = df_feat[df_feat[label_col] == 0]
    group1 = df_feat[df_feat[label_col] == 1]

    results = []
    for col in feature_cols:
        g0 = group0[col].dropna().values
        g1 = group1[col].dropna().values
        if len(g0) < 3 or len(g1) < 3:
            continue
        try:
            stat, pval = stats.mannwhitneyu(g0, g1, alternative='two-sided')
            # Cohen's d
            pooled_std = np.sqrt((np.std(g0)**2 + np.std(g1)**2) / 2.0)
            cohen_d = (np.mean(g1) - np.mean(g0)) / (pooled_std + 1e-10)
            results.append({
                'feature':   col,
                'mean_0':    np.mean(g0),
                'std_0':     np.std(g0),
                'mean_1':    np.mean(g1),
                'std_1':     np.std(g1),
                'p_value':   pval,
                'cohen_d':   cohen_d,
                'abs_cohen': abs(cohen_d),
            })
        except Exception:
            continue

    result_df = pd.DataFrame(results).sort_values('abs_cohen', ascending=False)
    return result_df


# ─── 4. 主流程 ───────────────────────────────────────────────────────────────

def main():
    base = '/home/user/LSTM-VAE-ON-PD'
    folder0 = os.path.join(base, '0')
    folder1 = os.path.join(base, '1')

    print("=" * 60)
    print("Loading data ...")
    records0 = load_folder(folder0, label=0)
    records1 = load_folder(folder1, label=1)
    print(f"  UPDRS=0: {len(records0)} files  |  UPDRS=1: {len(records1)} files")

    print("\nExtracting features ...")
    all_feats = []
    for rec in records0 + records1:
        try:
            all_feats.append(extract_features(rec))
        except Exception as e:
            print(f"  [SKIP] {rec['fname']}: {e}")

    df_feat = pd.DataFrame(all_feats)
    df_feat.to_csv(os.path.join(base, 'features_all.csv'), index=False)
    print(f"  Feature matrix shape: {df_feat.shape}")
    print(f"  Saved → features_all.csv")

    print("\nStatistical analysis (Mann-Whitney U + Cohen's d) ...")
    result_df = analyze_features(df_feat)
    result_df.to_csv(os.path.join(base, 'feature_importance.csv'), index=False)
    print(f"  Saved → feature_importance.csv")

    print("\n" + "=" * 60)
    print("TOP 40 最具区分力特征 (按 |Cohen's d| 排序)")
    print("=" * 60)
    top = result_df.head(40)
    print(f"{'Feature':<35} {'mean_0':>8} {'mean_1':>8} {'Cohen_d':>8} {'p-value':>10}")
    print("-" * 75)
    for _, row in top.iterrows():
        sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
        print(f"{row['feature']:<35} {row['mean_0']:>8.3f} {row['mean_1']:>8.3f} "
              f"{row['cohen_d']:>8.3f} {row['p_value']:>10.4f} {sig}")

    # 按特征类别汇总
    print("\n" + "=" * 60)
    print("TOP 10 按特征类别")
    print("=" * 60)
    categories = {
        'Tap detection (ITI/Jitter)': ['iti', 'tap_jitter', 'tap_shimmer', 'tap_rate', 'tap_count', 'tap_amp'],
        'Inter-sensor Distance':      ['dist_'],
        'Freq Domain':                ['dom_freq', 'pow_', 'spectral_entropy'],
        'Velocity/Acceleration':      ['_vel', '_acc'],
        'Position Stats':             ['s1_x_', 's1_y_', 's1_z_', 's2_x_', 's2_y_', 's2_z_'],
        'Orientation Stats':          ['s1_roll', 's1_pitch', 's1_yaw', 's2_roll', 's2_pitch', 's2_yaw'],
        'Relative Motion':            ['rel_'],
    }
    for cat, keys in categories.items():
        mask = result_df['feature'].apply(
            lambda f: any(k in f for k in keys))
        sub = result_df[mask].head(3)
        if len(sub) == 0:
            continue
        print(f"\n[{cat}]")
        for _, row in sub.iterrows():
            sig = '***' if row['p_value'] < 0.001 else ('**' if row['p_value'] < 0.01 else ('*' if row['p_value'] < 0.05 else ''))
            print(f"  {row['feature']:<38} |d|={row['abs_cohen']:.3f}  p={row['p_value']:.4f} {sig}")

    print("\n完成！结果已保存到:")
    print("  features_all.csv        — 所有样本的完整特征矩阵")
    print("  feature_importance.csv  — 特征区分力排名")


if __name__ == '__main__':
    main()
