"""
数据预处理模块
从 All-Ruijin-labels.xlsx 读取标签，从 data/ 文件夹读取传感器数据。

数据格式（无表头，空格分隔）：
  sensor_id  x  y  z  roll  pitch  yaw  timestamp(ms)

输出：归一化的 6 维时序张量
  通道：[dist, s1_pitch, s1_roll, s1_x, s1_z, s2_y]
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

# ─── 全局配置 ───────────────────────────────────────────────────────────────
SEQ_LEN   = 256   # 序列长度（约 4 秒 @ 64Hz，覆盖 5-8 次完整敲击）
INPUT_DIM = 6     # 输入通道数
CHANNELS  = ['dist', 's1_pitch', 's1_roll', 's1_x', 's1_z', 's2_y']


# ─── 文件读取 ────────────────────────────────────────────────────────────────

def load_sensor_file(filepath):
    """
    读取单个传感器文件，返回 (s1_df, s2_df)。
    传感器 ID 为 1 或 2（文件中为 "01"/"02"）。
    """
    rows = []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 8:
                continue
            try:
                rows.append({
                    'sensor': int(parts[0]),
                    'x':      float(parts[1]),
                    'y':      float(parts[2]),
                    'z':      float(parts[3]),
                    'roll':   float(parts[4]),
                    'pitch':  float(parts[5]),
                    'yaw':    float(parts[6]),
                    'ts':     int(parts[7]),
                })
            except ValueError:
                continue

    if not rows:
        return None, None

    df = pd.DataFrame(rows)
    s1 = df[df['sensor'] == 1].reset_index(drop=True)
    s2 = df[df['sensor'] == 2].reset_index(drop=True)
    return s1, s2


# ─── 序列构建 ────────────────────────────────────────────────────────────────

def build_sequence(s1, s2):
    """
    从两个传感器构建 6 维时序数组。

    通道选择依据特征重要性分析（|Cohen's d|）：
      dim 0 - dist     : 两指欧氏距离    |d|=0.86（最强）
      dim 1 - s1_pitch : 传感器1 pitch  |d|=0.58
      dim 2 - s1_roll  : 传感器1 roll   |d|=0.76
      dim 3 - s1_x     : 传感器1 x轴    |d|=0.74
      dim 4 - s1_z     : 传感器1 z轴    敲击主方向
      dim 5 - s2_y     : 传感器2 y轴    |d|=0.63

    Returns: (T, 6) float32 数组
    """
    min_len = min(len(s1), len(s2))
    s1 = s1.iloc[:min_len]
    s2 = s2.iloc[:min_len]

    dist = np.sqrt(
        (s1['x'].values - s2['x'].values) ** 2 +
        (s1['y'].values - s2['y'].values) ** 2 +
        (s1['z'].values - s2['z'].values) ** 2
    )

    seq = np.stack([
        dist,
        s1['pitch'].values,
        s1['roll'].values,
        s1['x'].values,
        s1['z'].values,
        s2['y'].values,
    ], axis=-1).astype(np.float32)  # (T, 6)

    return seq


def pad_or_crop(seq, length=SEQ_LEN):
    """
    将序列截断或填充（零填充）到固定长度。
    截断时取中间段（更稳定的运动区间）。
    """
    T = len(seq)
    if T >= length:
        start = (T - length) // 2
        return seq[start:start + length]
    else:
        pad = np.zeros((length - T, seq.shape[1]), dtype=np.float32)
        return np.vstack([seq, pad])


# ─── 数据集加载 ──────────────────────────────────────────────────────────────

def load_dataset(base_dir, label_values=(0, 1)):
    """
    从 Excel 标签文件 + data/ 文件夹加载数据。

    Returns:
        X : (N, SEQ_LEN, INPUT_DIM) float32
        y : (N,) int64
        fnames : list of str
    """
    excel_path = os.path.join(base_dir, 'All-Ruijin-labels.xlsx')
    data_dir   = os.path.join(base_dir, 'data')

    df = pd.read_excel(excel_path)
    df = df[df['Label Value'].isin(label_values)].reset_index(drop=True)

    print(f"  Excel 标签分布:")
    print(df['Label Value'].value_counts().sort_index().to_string(header=False))

    sequences, labels, fnames = [], [], []
    skip = 0

    for _, row in df.iterrows():
        fname = str(row['Data Filename']).strip()
        label = int(row['Label Value'])
        fpath = os.path.join(data_dir, fname)

        if not os.path.exists(fpath):
            skip += 1
            continue

        s1, s2 = load_sensor_file(fpath)
        if s1 is None or len(s1) < 50 or s2 is None or len(s2) < 50:
            skip += 1
            continue

        seq = build_sequence(s1, s2)
        seq = pad_or_crop(seq, SEQ_LEN)

        sequences.append(seq)
        labels.append(label)
        fnames.append(fname)

    print(f"  加载成功: {len(sequences)} 条，跳过: {skip} 条")

    X = np.stack(sequences, axis=0)   # (N, T, C)
    y = np.array(labels, dtype=np.int64)
    return X, y, fnames


# ─── Dataset / DataLoader ────────────────────────────────────────────────────

class PDDataset(Dataset):
    """PyTorch Dataset，支持可选的 numpy 数组或张量。"""

    def __init__(self, X, y):
        if isinstance(X, np.ndarray):
            self.X = torch.from_numpy(X).float()
        else:
            self.X = X.float()
        if isinstance(y, np.ndarray):
            self.y = torch.from_numpy(y).long()
        else:
            self.y = y.long()

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── 完整预处理流程 ──────────────────────────────────────────────────────────

def prepare_data(base_dir,
                 test_size=0.2,
                 val_size=0.1,
                 random_state=42,
                 batch_size=32):
    """
    完整数据准备流程:
      加载 → 全局 StandardScaler 归一化 → 划分 train/val/test → DataLoader

    注意：Scaler 仅在训练集上 fit，防止数据泄露。

    Returns: dict 包含 DataLoader、原始数组、scaler、pos_weight
    """
    print("=" * 55)
    print("  数据加载中...")
    X, y, fnames = load_dataset(base_dir)

    # ── 划分数据集（分层抽样保持类别比例）──────────────────────────────────
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state)

    val_ratio = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_ratio, stratify=y_train_full, random_state=random_state)

    # ── 全局归一化（逐通道 StandardScaler）─────────────────────────────────
    N_tr, T, C = X_train.shape
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, C))          # fit only on train

    X_train = scaler.transform(X_train.reshape(-1, C)).reshape(N_tr, T, C).astype(np.float32)
    X_val   = scaler.transform(X_val.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)
    X_test  = scaler.transform(X_test.reshape(-1, C)).reshape(-1, T, C).astype(np.float32)

    print(f"\n  划分结果：")
    print(f"    训练集: {X_train.shape}  标签分布: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"    验证集: {X_val.shape}    标签分布: {dict(zip(*np.unique(y_val,   return_counts=True)))}")
    print(f"    测试集: {X_test.shape}   标签分布: {dict(zip(*np.unique(y_test,  return_counts=True)))}")

    # ── 类别权重（BCEWithLogitsLoss 的 pos_weight）─────────────────────────
    n0 = int(np.sum(y_train == 0))
    n1 = int(np.sum(y_train == 1))
    pos_weight = torch.tensor([n0 / max(n1, 1)], dtype=torch.float32)
    print(f"\n  类别权重 pos_weight = {pos_weight.item():.3f}  (n0={n0}, n1={n1})")

    # ── DataLoader ──────────────────────────────────────────────────────────
    train_loader = DataLoader(PDDataset(X_train, y_train),
                              batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(PDDataset(X_val,   y_val),
                              batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(PDDataset(X_test,  y_test),
                              batch_size=batch_size, shuffle=False)

    print("=" * 55)

    return {
        'train_loader': train_loader,
        'val_loader':   val_loader,
        'test_loader':  test_loader,
        'X_train': X_train, 'y_train': y_train,
        'X_val':   X_val,   'y_val':   y_val,
        'X_test':  X_test,  'y_test':  y_test,
        'scaler':  scaler,
        'pos_weight': pos_weight,
    }
