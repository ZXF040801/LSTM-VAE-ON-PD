import os
import numpy as np
import pandas as pd

def load_file(filepath):
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 8:
                try:
                    rows.append({
                        'sensor': int(parts[0]),
                        'x': float(parts[1]),
                        'y': float(parts[2]),
                        'z': float(parts[3]),
                        'roll': float(parts[4]),
                        'pitch': float(parts[5]),
                        'yaw': float(parts[6]),
                        'ts': int(parts[7]),
                    })
                except ValueError:
                    pass
    return pd.DataFrame(rows)

results = []

for label in ['0', '1']:
    folder = f'/home/user/LSTM-VAE-ON-PD/{label}/'
    for fname in sorted(os.listdir(folder)):
        fpath = os.path.join(folder, fname)
        df = load_file(fpath)
        
        issues = []
        
        if df.empty:
            issues.append('EMPTY_FILE')
            results.append({'label': label, 'file': fname, 'issues': issues,
                           'n_s1': 0, 'n_s2': 0, 'duration_s': 0})
            continue
        
        s1 = df[df['sensor'] == 1].reset_index(drop=True)
        s2 = df[df['sensor'] == 2].reset_index(drop=True)
        n1, n2 = len(s1), len(s2)
        
        # Check sensor counts
        if n1 < 50:
            issues.append(f'S1_TOO_SHORT(n={n1})')
        if n2 < 50:
            issues.append(f'S2_TOO_SHORT(n={n2})')
        
        # Duration
        duration = 0
        if n1 > 1:
            duration = (s1['ts'].max() - s1['ts'].min()) / 1000.0
            if duration < 5.0:
                issues.append(f'DURATION_SHORT({duration:.1f}s)')
            if duration > 60.0:
                issues.append(f'DURATION_LONG({duration:.1f}s)')
        
        # Timestamp monotonicity (check jumps)
        if n1 > 1:
            dts = np.diff(s1['ts'].values)
            if np.any(dts < 0):
                issues.append('TS_NOT_MONOTONIC_S1')
            if np.any(dts == 0):
                issues.append(f'TS_DUPLICATE_S1(n={np.sum(dts==0)})')
            # Large gaps (> 500ms)
            big_gaps = np.sum(dts > 500)
            if big_gaps > 0:
                issues.append(f'TS_GAP_S1(n={big_gaps})')
        
        if n2 > 1:
            dts2 = np.diff(s2['ts'].values)
            if np.any(dts2 < 0):
                issues.append('TS_NOT_MONOTONIC_S2')
            big_gaps2 = np.sum(dts2 > 500)
            if big_gaps2 > 0:
                issues.append(f'TS_GAP_S2(n={big_gaps2})')
        
        # Sensor count imbalance
        if n1 > 0 and n2 > 0:
            ratio = max(n1, n2) / min(n1, n2)
            if ratio > 1.5:
                issues.append(f'SENSOR_IMBALANCE(s1={n1},s2={n2})')
        
        # NaN / Inf
        for col in ['x','y','z','roll','pitch','yaw']:
            if n1 > 0 and (s1[col].isna().any() or np.isinf(s1[col].values).any()):
                issues.append(f'NAN_INF_S1_{col.upper()}')
            if n2 > 0 and (s2[col].isna().any() or np.isinf(s2[col].values).any()):
                issues.append(f'NAN_INF_S2_{col.upper()}')
        
        # Constant channels (std ~ 0 means sensor not moving at all)
        if n1 > 10:
            for col in ['x','y','z','roll','pitch','yaw']:
                if s1[col].std() < 0.01:
                    issues.append(f'CONSTANT_S1_{col.upper()}')
        if n2 > 10:
            for col in ['x','y','z','roll','pitch','yaw']:
                if s2[col].std() < 0.01:
                    issues.append(f'CONSTANT_S2_{col.upper()}')
        
        # Extreme outliers: any channel value way out of range
        # Typical range: position ~[-50,50], angles ~[-200,200]
        if n1 > 0:
            for col in ['x','y','z']:
                vals = s1[col].values
                if np.max(np.abs(vals)) > 500:
                    issues.append(f'EXTREME_S1_{col.upper()}(max={np.max(np.abs(vals)):.0f})')
            for col in ['roll','pitch','yaw']:
                vals = s1[col].values
                if np.max(np.abs(vals)) > 1000:
                    issues.append(f'EXTREME_S1_{col.upper()}(max={np.max(np.abs(vals)):.0f})')
        
        if n2 > 0:
            for col in ['x','y','z']:
                vals = s2[col].values
                if np.max(np.abs(vals)) > 500:
                    issues.append(f'EXTREME_S2_{col.upper()}(max={np.max(np.abs(vals)):.0f})')
            for col in ['roll','pitch','yaw']:
                vals = s2[col].values
                if np.max(np.abs(vals)) > 1000:
                    issues.append(f'EXTREME_S2_{col.upper()}(max={np.max(np.abs(vals)):.0f})')
        
        # Sampling rate (should be ~64 Hz, flag if <30 or >120)
        if n1 > 1 and duration > 0:
            fs = n1 / duration
            if fs < 30 or fs > 150:
                issues.append(f'FS_ABNORMAL_S1({fs:.1f}Hz)')
        
        results.append({
            'label': label,
            'file': fname,
            'issues': issues,
            'n_s1': n1,
            'n_s2': n2,
            'duration_s': round(duration, 2),
        })

# Print summary
anomalies = [r for r in results if r['issues']]
clean = [r for r in results if not r['issues']]

print(f"总文件数: {len(results)}  (label0={sum(1 for r in results if r['label']=='0')}, label1={sum(1 for r in results if r['label']=='1')})")
print(f"无异常:   {len(clean)}")
print(f"有异常:   {len(anomalies)}")
print()
print("=" * 80)
print("有异常的文件详情：")
print("=" * 80)
for r in anomalies:
    print(f"[label={r['label']}] {r['file']}")
    print(f"  n_s1={r['n_s1']}, n_s2={r['n_s2']}, duration={r['duration_s']}s")
    for issue in r['issues']:
        print(f"  * {issue}")
    print()

# Stats per issue type
from collections import Counter
all_issues = [issue for r in anomalies for issue in r['issues']]
# group by prefix
prefixes = Counter()
for i in all_issues:
    prefix = i.split('(')[0]
    prefixes[prefix] += 1
print("=" * 80)
print("异常类型统计：")
for k, v in prefixes.most_common():
    print(f"  {k:<40} {v} 个文件")
