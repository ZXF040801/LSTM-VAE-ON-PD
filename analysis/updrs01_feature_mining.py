import csv
import math
import os
import statistics
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict

XLSX_PATH = 'All-Ruijin-labels.xlsx'
DATA_DIR = 'data'
OUT_CSV = 'analysis/updrs01_feature_summary.csv'
OUT_MD = 'analysis/updrs01_feature_report.md'

NS = {'a': 'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}


def parse_xlsx_rows(path):
    z = zipfile.ZipFile(path)
    shared = []
    if 'xl/sharedStrings.xml' in z.namelist():
        root = ET.fromstring(z.read('xl/sharedStrings.xml'))
        for si in root.findall('a:si', NS):
            txt = ''.join((t.text or '') for t in si.findall('.//a:t', NS))
            shared.append(txt)

    ws = ET.fromstring(z.read('xl/worksheets/sheet1.xml'))
    rows = ws.findall('.//a:sheetData/a:row', NS)

    def col_index(cell_ref):
        letters = ''.join(ch for ch in cell_ref if ch.isalpha())
        idx = 0
        for ch in letters:
            idx = idx * 26 + (ord(ch) - ord('A') + 1)
        return idx - 1

    def cell_val(c):
        t = c.attrib.get('t')
        v = c.find('a:v', NS)
        if v is None:
            return ''
        txt = v.text or ''
        if t == 's':
            if txt.isdigit() and int(txt) < len(shared):
                return shared[int(txt)]
        return txt

    header = {}
    first = rows[0]
    for c in first.findall('a:c', NS):
        idx = col_index(c.attrib.get('r', 'A1'))
        header[idx] = cell_val(c).strip()

    records = []
    for r in rows[1:]:
        row_dict = {name: '' for name in header.values()}
        for c in r.findall('a:c', NS):
            idx = col_index(c.attrib.get('r', 'A1'))
            key = header.get(idx)
            if key is not None:
                row_dict[key] = cell_val(c).strip()
        records.append(row_dict)
    return records


def mean_or_nan(values):
    return statistics.mean(values) if values else float('nan')


def std_or_nan(values):
    return statistics.pstdev(values) if len(values) > 1 else float('nan')


def safe_div(a, b):
    return a / b if b else float('nan')


def series_features(series):
    if len(series) < 3:
        return None
    # series item: (t_ms, x,y,z, roll,pitch,yaw)
    ts = [r[0] for r in series]
    x = [r[1] for r in series]
    y = [r[2] for r in series]
    z = [r[3] for r in series]
    rll = [r[4] for r in series]
    ptc = [r[5] for r in series]
    yw = [r[6] for r in series]

    dt = []
    v_pos = []
    v_ang = []
    for i in range(1, len(series)):
        dtt = (ts[i] - ts[i - 1]) / 1000.0
        if dtt <= 0:
            continue
        dx, dy, dz = x[i] - x[i - 1], y[i] - y[i - 1], z[i] - z[i - 1]
        dr, dp, dyw = rll[i] - rll[i - 1], ptc[i] - ptc[i - 1], yw[i] - yw[i - 1]
        dt.append(dtt)
        v_pos.append(math.sqrt(dx * dx + dy * dy + dz * dz) / dtt)
        v_ang.append(math.sqrt(dr * dr + dp * dp + dyw * dyw) / dtt)

    if not dt:
        return None

    acc = []
    for i in range(1, len(v_pos)):
        dtt = dt[i]
        if dtt > 0:
            acc.append(abs(v_pos[i] - v_pos[i - 1]) / dtt)

    duration = (ts[-1] - ts[0]) / 1000.0 if ts[-1] > ts[0] else float('nan')

    return {
        'duration_s': duration,
        'sample_count': len(series),
        'pos_range_x': max(x) - min(x),
        'pos_range_y': max(y) - min(y),
        'pos_range_z': max(z) - min(z),
        'pos_std_x': std_or_nan(x),
        'pos_std_y': std_or_nan(y),
        'pos_std_z': std_or_nan(z),
        'speed_mean': mean_or_nan(v_pos),
        'speed_std': std_or_nan(v_pos),
        'speed_cv': safe_div(std_or_nan(v_pos), mean_or_nan(v_pos)),
        'ang_speed_mean': mean_or_nan(v_ang),
        'ang_speed_std': std_or_nan(v_ang),
        'ang_speed_cv': safe_div(std_or_nan(v_ang), mean_or_nan(v_ang)),
        'acc_abs_mean': mean_or_nan(acc),
    }


def read_data_file(path):
    by_id = defaultdict(list)
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) < 8:
                continue
            try:
                fid = parts[0]
                vals = [float(p) for p in parts[1:7]]
                t = float(parts[7])
            except ValueError:
                continue
            by_id[fid].append((t, *vals))
    # sort each stream by timestamp
    for fid in list(by_id.keys()):
        by_id[fid].sort(key=lambda x: x[0])
    return by_id


def aggregate_file_features(by_id):
    feats = []
    for fid, series in by_id.items():
        f = series_features(series)
        if f:
            feats.append(f)
    if not feats:
        return None
    keys = feats[0].keys()
    out = {}
    for k in keys:
        vals = [d[k] for d in feats if not math.isnan(d[k])]
        out[k] = mean_or_nan(vals)
    return out


def cohen_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return float('nan')
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = ((len(a)-1)*va + (len(b)-1)*vb) / (len(a)+len(b)-2)
    if pooled <= 0:
        return float('nan')
    return (mb - ma) / math.sqrt(pooled)


def main():
    records = parse_xlsx_rows(XLSX_PATH)
    selected = []
    for r in records:
        score_raw = r.get('FT Clinical UPDRS Score', '').strip()
        fname = r.get('Data Filename', '').strip()
        if not fname:
            continue
        try:
            score = int(float(score_raw))
        except ValueError:
            continue
        if score not in (0, 1):
            continue
        selected.append((score, fname))

    per_file = []
    missing = []
    for score, fname in selected:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            missing.append(fname)
            continue
        by_id = read_data_file(path)
        feats = aggregate_file_features(by_id)
        if not feats:
            continue
        feats['label'] = score
        feats['file'] = fname
        per_file.append(feats)

    feature_names = [k for k in per_file[0].keys() if k not in ('label', 'file')]

    groups = {0: defaultdict(list), 1: defaultdict(list)}
    for row in per_file:
        lbl = row['label']
        for k in feature_names:
            v = row[k]
            if not math.isnan(v):
                groups[lbl][k].append(v)

    summary = []
    for k in feature_names:
        g0 = groups[0][k]
        g1 = groups[1][k]
        summary.append({
            'feature': k,
            'n0': len(g0),
            'mean0': mean_or_nan(g0),
            'std0': std_or_nan(g0),
            'n1': len(g1),
            'mean1': mean_or_nan(g1),
            'std1': std_or_nan(g1),
            'cohen_d_1_vs_0': cohen_d(g0, g1),
            'relative_change_1_vs_0': safe_div(mean_or_nan(g1) - mean_or_nan(g0), mean_or_nan(g0)),
        })

    summary.sort(key=lambda d: abs(d['cohen_d_1_vs_0']) if not math.isnan(d['cohen_d_1_vs_0']) else -1, reverse=True)

    with open(OUT_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)

    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write('# UPDRS 0 vs 1 特征分析（基于 FT Clinical UPDRS Score）\n\n')
        f.write(f'- 标签文件: `{XLSX_PATH}`\n')
        f.write(f'- 原始符合条件记录数（标签=0或1）: **{len(selected)}**\n')
        f.write(f'- 成功匹配并提取特征的数据文件数: **{len(per_file)}**\n')
        f.write(f'- 缺失数据文件数: **{len(missing)}**\n')
        f.write(f'- 0类样本数: **{sum(1 for r in per_file if r["label"]==0)}**\n')
        f.write(f'- 1类样本数: **{sum(1 for r in per_file if r["label"]==1)}**\n\n')

        f.write('## 区分能力最强的候选特征（按 |Cohen d| 排序）\n\n')
        f.write('| feature | mean(0) | mean(1) | change(1 vs 0) | cohen d |\n')
        f.write('|---|---:|---:|---:|---:|\n')
        for r in summary[:10]:
            ch = r['relative_change_1_vs_0']
            d = r['cohen_d_1_vs_0']
            f.write(f"| {r['feature']} | {r['mean0']:.4f} | {r['mean1']:.4f} | {ch*100:.2f}% | {d:.3f} |\n")

        f.write('\n## 特征解释建议\n\n')
        f.write('- `speed_mean`: 空间位置平均速度，反映敲击动作整体运动快慢。\n')
        f.write('- `speed_std` / `speed_cv`: 速度波动与稳定性，轻度帕金森常见节律不稳。\n')
        f.write('- `acc_abs_mean`: 速度变化率绝对值，可近似动作“顿挫感”或不平滑程度。\n')
        f.write('- `pos_range_z` / `pos_std_z`: z轴幅度和离散度，反映抬指高度与运动振幅。\n')
        f.write('- `ang_speed_mean` / `ang_speed_std`: 姿态角速度及其稳定性，反映手部旋转控制能力。\n')

        f.write('\n## 输出文件\n\n')
        f.write(f'- 详细统计: `{OUT_CSV}`\n')
        f.write(f'- 本报告: `{OUT_MD}`\n')


if __name__ == '__main__':
    main()
