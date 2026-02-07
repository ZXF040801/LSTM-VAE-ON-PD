import pandas as pd
import numpy as np
import os
from scipy import signal
from scipy.interpolate import interp1d
from sklearn.model_selection import train_test_split
from collections import Counter
import json
import warnings

warnings.filterwarnings('ignore')

class Config:
    """预处理配置参数"""

    # ====== 请修改这里的路径 ======
    EXCEL_PATH = r'D:\Final work\DataForStudentProject-HWU\All-Ruijin-labels.xlsx'
    DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\finger-tapping'
    OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'
    # ==============================

    # 任务配置
    TASK_SHEET = 'Finger-Tapping-Release'
    TASK_NAME = 'finger_tapping'

    # UPDRS评分列名
    UPDRS_COLUMN = 'Clinical UPDRS Score'

    # 只保留这些评分的样本
    VALID_SCORES = [0, 1]
    git
    push - u
    origin
    main  # 分支名替换为你的实际分支（如 master）
    # 信号处理参数
    SAMPLING_RATE = 60  # Hz
    TARGET_LENGTH = 360  # 目标序列长度 (6秒 * 60Hz)

    # 滤波参数
    FILTER_LOWCUT = 0.5  # Hz
    FILTER_HIGHCUT = 20  # Hz
    FILTER_ORDER = 4

    # 数据集划分
    TEST_SIZE = 0.2
    VAL_SIZE = 0.1
    RANDOM_STATE = 42

    # 传感器配置
    N_CHANNELS = 12  # 2个传感器 × 6个通道


# ============================================================================
# 标签处理函数 (基于UPDRS评分)
# ============================================================================

def classify_by_updrs(score, valid_scores=[0, 1]):
    """
    根据Clinical UPDRS Score分类

    参数:
        score: UPDRS评分值
        valid_scores: 有效的评分列表 [0, 1]

    返回:
        标签值 (0或1) 或 None (如果不在有效范围内)

    分类规则:
        - 0分 → label=0
        - 1分 → label=1
        - 其他 → None (排除)
    """
    if pd.isna(score):
        return None

    try:
        score_int = int(score)
    except (ValueError, TypeError):
        return None

    if score_int in valid_scores:
        return score_int  # 0分→0, 1分→1
    else:
        return None  # 排除2,3,4分


def load_labels(excel_path, sheet_name, updrs_column, valid_scores):
    """加载并处理标签数据 (基于UPDRS评分)"""
    print(f"  读取Excel文件: {excel_path}")
    print(f"  Sheet: {sheet_name}")
    print(f"  UPDRS列: {updrs_column}")
    print(f"  有效评分: {valid_scores}")

    df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # 检查UPDRS列是否存在
    if updrs_column not in df.columns:
        print(f"\n  错误: 找不到列 '{updrs_column}'")
        print(f"  可用列: {list(df.columns)}")
        return None

    # 显示原始UPDRS分布
    print(f"\n  原始UPDRS评分分布:")
    updrs_counts = df[updrs_column].value_counts(dropna=False).sort_index()
    for score, count in updrs_counts.items():
        if pd.isna(score):
            status = "✗ 排除 (空值)"
            print(f"    NaN: {count} 个样本 {status}")
        else:
            status = "✓ 保留" if int(score) in valid_scores else "✗ 排除"
            print(f"    {int(score)}分: {count} 个样本 {status}")

    # 创建标签
    df['label'] = df[updrs_column].apply(lambda x: classify_by_updrs(x, valid_scores))

    # 过滤掉无效样本
    original_count = len(df)
    df = df.dropna(subset=['label', 'Data Filename', 'Patient ID'])
    df['label'] = df['label'].astype(int)
    df['Patient ID'] = df['Patient ID'].astype(str)

    filtered_count = len(df)

    print(f"\n  筛选结果:")
    print(f"    原始样本数: {original_count}")
    print(f"    筛选后样本数: {filtered_count}")
    print(f"    排除样本数: {original_count - filtered_count}")
    print(f"\n  保留样本分布:")
    print(f"    0分 (label=0): {(df['label'] == 0).sum()}")
    print(f"    1分 (label=1): {(df['label'] == 1).sum()}")

    return df


# ============================================================================
# 时间序列数据处理函数
# ============================================================================

def load_sensor_data(file_path):
    """
    加载单个加速度计数据文件

    数据格式:
        传感器ID  acc_x  acc_y  acc_z  gyro_x  gyro_y  gyro_z  时间戳
        01        ...    ...    ...    ...     ...     ...     1600846162395
        02        ...    ...    ...    ...     ...     ...     1600846162395
    """
    try:
        sensor1_data = []
        sensor2_data = []

        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue

                sensor_id = parts[0]
                values = [float(x) for x in parts[1:7]]

                if sensor_id == '01':
                    sensor1_data.append(values)
                elif sensor_id == '02':
                    sensor2_data.append(values)

        if len(sensor1_data) == 0 or len(sensor2_data) == 0:
            return None

        return {
            'sensor1': np.array(sensor1_data),
            'sensor2': np.array(sensor2_data)
        }

    except Exception as e:
        return None


def process_sequence(file_path, config):
    """
    处理单个序列: 加载 → 滤波 → 标准化 → 重采样
    """
    # 1. 加载数据
    sensor_data = load_sensor_data(file_path)
    if sensor_data is None:
        return None

    # 2. 合并两个传感器的数据
    min_len = min(len(sensor_data['sensor1']), len(sensor_data['sensor2']))
    if min_len < 50:
        return None

    combined = np.hstack([
        sensor_data['sensor1'][:min_len],
        sensor_data['sensor2'][:min_len]
    ])  # (timesteps, 12)

    # 3. 带通滤波
    nyquist = config.SAMPLING_RATE / 2
    low = config.FILTER_LOWCUT / nyquist
    high = config.FILTER_HIGHCUT / nyquist
    b, a = signal.butter(config.FILTER_ORDER, [low, high], btype='band')

    filtered = np.zeros_like(combined)
    for i in range(combined.shape[1]):
        try:
            filtered[:, i] = signal.filtfilt(b, a, combined[:, i])
        except:
            filtered[:, i] = combined[:, i]

    # 4. Z-score标准化 (按通道)
    mean = np.mean(filtered, axis=0, keepdims=True)
    std = np.std(filtered, axis=0, keepdims=True)
    std[std < 1e-8] = 1
    normalized = (filtered - mean) / std

    # 5. 重采样到目标长度
    current_len = len(normalized)
    if current_len == config.TARGET_LENGTH:
        return normalized

    x_old = np.linspace(0, 1, current_len)
    x_new = np.linspace(0, 1, config.TARGET_LENGTH)

    resampled = np.zeros((config.TARGET_LENGTH, normalized.shape[1]))
    for i in range(normalized.shape[1]):
        f = interp1d(x_old, normalized[:, i], kind='linear', fill_value='extrapolate')
        resampled[:, i] = f(x_new)

    return resampled


# ============================================================================
# 数据集构建函数
# ============================================================================

def find_data_file(filename, data_folder):
    """查找数据文件"""
    full_path = os.path.join(data_folder, filename)
    if os.path.exists(full_path):
        return full_path

    if filename.startswith('PD-Ruijin'):
        alt_filename = filename.replace('PD-Ruijin', 'PD-Monitor - Ruijin')
        alt_path = os.path.join(data_folder, alt_filename)
        if os.path.exists(alt_path):
            return alt_path

    if 'PD-Monitor' in filename:
        alt_filename = filename.replace('PD-Monitor - Ruijin', 'PD-Ruijin')
        alt_path = os.path.join(data_folder, alt_filename)
        if os.path.exists(alt_path):
            return alt_path

    return None


def split_by_patient(df, test_size=0.2, val_size=0.1, random_state=42):
    """按患者ID分层划分数据集"""
    patient_labels = df.groupby('Patient ID')['label'].agg(lambda x: x.mode()[0])
    patient_ids = patient_labels.index.tolist()
    labels = patient_labels.values

    # 检查是否有足够的样本进行分层
    label_counts = Counter(labels)
    min_count = min(label_counts.values())

    if min_count < 2:
        print(f"  警告: 某类别患者数量过少 ({label_counts})，无法分层划分")
        train_val_ids, test_ids = train_test_split(
            patient_ids, test_size=test_size, random_state=random_state
        )
        adjusted_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids, test_size=adjusted_val_size, random_state=random_state
        )
    else:
        train_val_ids, test_ids, train_val_labels, _ = train_test_split(
            patient_ids, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state
        )

        adjusted_val_size = val_size / (1 - test_size)
        train_ids, val_ids = train_test_split(
            train_val_ids,
            test_size=adjusted_val_size,
            stratify=train_val_labels,
            random_state=random_state
        )

    return train_ids, val_ids, test_ids


def build_dataset(df, patient_ids, data_folder, config):
    """为指定患者列表构建数据集"""
    X, y, metadata = [], [], []

    subset_df = df[df['Patient ID'].isin(patient_ids)]

    success = 0
    failed = 0
    not_found = 0

    for idx, row in subset_df.iterrows():
        filename = row['Data Filename']

        file_path = find_data_file(filename, data_folder)

        if file_path is None:
            not_found += 1
            continue

        sequence = process_sequence(file_path, config)

        if sequence is None:
            failed += 1
            continue

        X.append(sequence)
        y.append(row['label'])
        metadata.append({
            'patient_id': row['Patient ID'],
            'hand': row.get('Hand', 'unknown'),
            'updrs_score': int(row['label']),
            'filename': filename
        })

        success += 1

    print(f"    成功: {success}, 失败: {failed}, 未找到: {not_found}")

    if len(X) == 0:
        return np.array([]), np.array([]), []

    return np.array(X), np.array(y), metadata


# ============================================================================
# 主预处理流程
# ============================================================================

def preprocess_data(config=None):
    """完整的预处理流程"""

    if config is None:
        config = Config()

    print("=" * 60)
    print("    帕金森病 Finger Tapping 数据预处理 (UPDRS版)")
    print("=" * 60)
    print("\n分类规则:")
    print("  - UPDRS 0分 → label=0 (正常)")
    print("  - UPDRS 1分 → label=1 (轻度异常)")
    print("  - UPDRS 2/3/4分 → 排除")

    # 检查路径
    print("\n[检查路径]")
    print(f"  Excel文件: {config.EXCEL_PATH}")
    print(f"  数据文件夹: {config.DATA_FOLDER}")
    print(f"  输出文件夹: {config.OUTPUT_FOLDER}")

    if not os.path.exists(config.EXCEL_PATH):
        print(f"\n错误: Excel文件不存在!")
        return None

    if not os.path.exists(config.DATA_FOLDER):
        print(f"\n错误: 数据文件夹不存在!")
        return None

    os.makedirs(config.OUTPUT_FOLDER, exist_ok=True)

    # 步骤1: 加载标签
    print("\n[步骤1] 加载标签数据 (基于UPDRS评分)...")
    df = load_labels(
        config.EXCEL_PATH,
        config.TASK_SHEET,
        config.UPDRS_COLUMN,
        config.VALID_SCORES
    )

    if df is None or len(df) == 0:
        print("错误: 没有有效数据!")
        return None

    # 步骤2: 划分数据集
    print("\n[步骤2] 按患者划分数据集...")
    train_ids, val_ids, test_ids = split_by_patient(
        df, config.TEST_SIZE, config.VAL_SIZE, config.RANDOM_STATE
    )
    print(f"  训练集患者: {len(train_ids)}")
    print(f"  验证集患者: {len(val_ids)}")
    print(f"  测试集患者: {len(test_ids)}")

    # 步骤3-5: 构建数据集
    print("\n[步骤3] 处理训练集...")
    train_X, train_y, train_meta = build_dataset(df, train_ids, config.DATA_FOLDER, config)

    print("\n[步骤4] 处理验证集...")
    val_X, val_y, val_meta = build_dataset(df, val_ids, config.DATA_FOLDER, config)

    print("\n[步骤5] 处理测试集...")
    test_X, test_y, test_meta = build_dataset(df, test_ids, config.DATA_FOLDER, config)

    # 打印统计
    print("\n" + "=" * 60)
    print("                    数据集统计")
    print("=" * 60)

    for name, X, y in [('训练集', train_X, train_y),
                       ('验证集', val_X, val_y),
                       ('测试集', test_X, test_y)]:
        if len(X) > 0:
            print(f"\n{name}:")
            print(f"  形状: {X.shape}")
            print(f"  UPDRS 0分: {(y == 0).sum()}, UPDRS 1分: {(y == 1).sum()}")
            print(f"  1分比例: {np.mean(y):.1%}")
        else:
            print(f"\n{name}: 空")

    # 步骤6: 保存数据
    print("\n[步骤6] 保存数据...")

    if len(train_X) > 0:
        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'train_data.npz'),
            X=train_X, y=train_y
        )
        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'val_data.npz'),
            X=val_X, y=val_y
        )
        np.savez_compressed(
            os.path.join(config.OUTPUT_FOLDER, 'test_data.npz'),
            X=test_X, y=test_y
        )

        # 保存元数据
        with open(os.path.join(config.OUTPUT_FOLDER, 'metadata.json'), 'w') as f:
            json.dump({
                'train_meta': train_meta,
                'val_meta': val_meta,
                'test_meta': test_meta,
                'config': {
                    'sampling_rate': config.SAMPLING_RATE,
                    'target_length': config.TARGET_LENGTH,
                    'n_channels': config.N_CHANNELS,
                    'channel_names': [
                        'thumb_acc_x', 'thumb_acc_y', 'thumb_acc_z',
                        'thumb_gyro_x', 'thumb_gyro_y', 'thumb_gyro_z',
                        'index_acc_x', 'index_acc_y', 'index_acc_z',
                        'index_gyro_x', 'index_gyro_y', 'index_gyro_z'
                    ],
                    'label_mapping': {
                        '0': 'UPDRS Score 0 (Normal)',
                        '1': 'UPDRS Score 1 (Slight)'
                    }
                },
                'statistics': {
                    'train_samples': len(train_X),
                    'val_samples': len(val_X),
                    'test_samples': len(test_X),
                    'train_class_0': int((train_y == 0).sum()),
                    'train_class_1': int((train_y == 1).sum()),
                    'val_class_0': int((val_y == 0).sum()),
                    'val_class_1': int((val_y == 1).sum()),
                    'test_class_0': int((test_y == 0).sum()),
                    'test_class_1': int((test_y == 1).sum())
                }
            }, f, indent=2, ensure_ascii=False)

        print(f"  数据已保存到: {config.OUTPUT_FOLDER}")
        print(f"    - train_data.npz")
        print(f"    - val_data.npz")
        print(f"    - test_data.npz")
        print(f"    - metadata.json")
    else:
        print("  警告: 没有成功处理任何数据!")

    print("\n" + "=" * 60)
    print("                    预处理完成!")
    print("=" * 60)

    return {
        'train': (train_X, train_y, train_meta),
        'val': (val_X, val_y, val_meta),
        'test': (test_X, test_y, test_meta)
    }


# ============================================================================
# 数据加载工具函数
# ============================================================================

def load_processed_data(data_folder):
    """加载预处理后的数据"""
    train_data = np.load(os.path.join(data_folder, 'train_data.npz'))
    val_data = np.load(os.path.join(data_folder, 'val_data.npz'))
    test_data = np.load(os.path.join(data_folder, 'test_data.npz'))

    return (
        train_data['X'], train_data['y'],
        val_data['X'], val_data['y'],
        test_data['X'], test_data['y']
    )


# ============================================================================
# 主程序入口
# ============================================================================

if __name__ == "__main__":
    data = preprocess_data()