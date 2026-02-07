
import pandas as pd
import numpy as np
import os
import json

# ============================================================================
# 配置 - 请修改为你的实际路径
# ============================================================================

EXCEL_PATH = r'D:\Final work\DataForStudentProject-HWU\All-Ruijin-labels.xlsx'
DATA_FOLDER = r'D:\Final work\DataForStudentProject-HWU\finger-tapping'
OUTPUT_FOLDER = r'D:\Final work\DataForStudentProject-HWU\processed_data'


# ============================================================================
# 验证1: 标签分类逻辑
# ============================================================================

def verify_label_classification():
    """验证PD标签分类是否正确"""

    print("=" * 70)
    print("  验证1: PD vs Non-PD 标签分类逻辑")
    print("=" * 70)

    # 分类函数 (与预处理代码相同)
    def classify_pd_status(notes):
        if pd.isna(notes):
            return 0
        notes_str = str(notes).strip().lower()
        if notes_str == '' or notes_str == '  ':
            return 0
        if 'pd' in notes_str or 'pds' in notes_str:
            return 1
        return 0

    # 测试用例
    test_cases = [
        # (Notes值, 预期标签, 说明)
        (None, 0, "空值 → 健康对照"),
        ("", 0, "空字符串 → 健康对照"),
        ("  ", 0, "空格 → 健康对照"),
        ("PD", 1, "PD → 帕金森"),
        ("pd", 1, "pd(小写) → 帕金森"),
        ("PDS", 1, "PDS → 帕金森综合征"),
        ("pds", 1, "pds(小写) → 帕金森综合征"),
        ("PD tremor", 1, "PD tremor → 帕金森"),
        ("ET +PD", 1, "ET +PD → 帕金森(混合)"),
        ("ET", 0, "ET → 非PD(原发性震颤)"),
        ("MSA", 0, "MSA → 非PD(多系统萎缩)"),
        ("PSP", 0, "PSP → 非PD(进行性核上性麻痹)"),
        ("tremor", 0, "tremor → 非PD"),
        ("bilateral", 0, "bilateral → 非PD"),
        ("ET tremor", 0, "ET tremor → 非PD"),
    ]

    print("\n  测试用例:")
    print(f"  {'Notes值':<20} {'预期':<6} {'实际':<6} {'结果':<8} {'说明'}")
    print("  " + "-" * 65)

    all_pass = True
    for notes, expected, description in test_cases:
        actual = classify_pd_status(notes)
        status = "✓ 通过" if actual == expected else "✗ 失败"
        if actual != expected:
            all_pass = False
        notes_display = repr(notes) if notes is not None else "None"
        print(f"  {notes_display:<20} {expected:<6} {actual:<6} {status:<8} {description}")

    print("\n  " + "-" * 65)
    if all_pass:
        print("  ✓ 所有测试用例通过! 标签分类逻辑正确")
    else:
        print("  ✗ 存在失败的测试用例! 请检查分类逻辑")

    return all_pass


# ============================================================================
# 验证2: Excel数据中的实际标签分布
# ============================================================================

def verify_excel_labels():
    """验证Excel中的实际标签分布"""

    print("\n" + "=" * 70)
    print("  验证2: Excel数据中的实际标签分布")
    print("=" * 70)

    if not os.path.exists(EXCEL_PATH):
        print(f"\n  ✗ 错误: Excel文件不存在: {EXCEL_PATH}")
        return False

    # 加载Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name='Finger-Tapping-Release')

    # 分类函数
    def classify_pd_status(notes):
        if pd.isna(notes):
            return 0
        notes_str = str(notes).strip().lower()
        if notes_str == '' or notes_str == '  ':
            return 0
        if 'pd' in notes_str or 'pds' in notes_str:
            return 1
        return 0

    df['label'] = df['Notes'].apply(classify_pd_status)

    print(f"\n  总记录数: {len(df)}")
    print(f"  唯一患者数: {df['Patient ID'].nunique()}")

    # 统计Notes值
    print("\n  Notes列的所有唯一值及其标签:")
    print(f"  {'Notes值':<25} {'数量':<8} {'分配标签':<10} {'类别'}")
    print("  " + "-" * 60)

    notes_counts = df.groupby('Notes', dropna=False).agg({
        'label': 'first',
        'Patient ID': 'count'
    }).reset_index()
    notes_counts.columns = ['Notes', 'label', 'count']
    notes_counts = notes_counts.sort_values('count', ascending=False)

    for _, row in notes_counts.iterrows():
        notes_val = row['Notes'] if pd.notna(row['Notes']) else "(空/NaN)"
        label = row['label']
        count = row['count']
        category = "PD" if label == 1 else "Non-PD"
        print(f"  {str(notes_val):<25} {count:<8} {label:<10} {category}")

    # 总计
    print("\n  " + "-" * 60)
    print(f"  {'总计PD样本:':<25} {(df['label'] == 1).sum()}")
    print(f"  {'总计Non-PD样本:':<25} {(df['label'] == 0).sum()}")
    print(f"  {'PD比例:':<25} {df['label'].mean():.1%}")

    # 检查分布是否合理
    pd_count = (df['label'] == 1).sum()
    non_pd_count = (df['label'] == 0).sum()

    if pd_count > 0 and non_pd_count > 0:
        print("\n  ✓ 标签分布合理，两个类别都有样本")
        return True
    else:
        print("\n  ✗ 警告: 某个类别没有样本!")
        return False


# ============================================================================
# 验证3: 数据文件匹配
# ============================================================================

def verify_file_matching():
    """验证Excel中的文件名与实际文件的匹配"""

    print("\n" + "=" * 70)
    print("  验证3: 数据文件匹配")
    print("=" * 70)

    if not os.path.exists(EXCEL_PATH):
        print(f"\n  ✗ 错误: Excel文件不存在")
        return False

    if not os.path.exists(DATA_FOLDER):
        print(f"\n  ✗ 错误: 数据文件夹不存在: {DATA_FOLDER}")
        return False

    # 加载Excel
    df = pd.read_excel(EXCEL_PATH, sheet_name='Finger-Tapping-Release')
    df = df.dropna(subset=['Data Filename'])

    # 获取实际文件列表
    actual_files = set(os.listdir(DATA_FOLDER))

    print(f"\n  Excel中的文件名数量: {len(df)}")
    print(f"  数据文件夹中的文件数量: {len(actual_files)}")

    # 文件匹配函数
    def find_file(filename):
        if filename in actual_files:
            return filename
        # 尝试替换前缀
        if filename.startswith('PD-Ruijin'):
            alt = filename.replace('PD-Ruijin', 'PD-Monitor - Ruijin')
            if alt in actual_files:
                return alt
        if 'PD-Monitor' in filename:
            alt = filename.replace('PD-Monitor - Ruijin', 'PD-Ruijin')
            if alt in actual_files:
                return alt
        return None

    # 统计匹配情况
    matched = 0
    unmatched = []

    for filename in df['Data Filename']:
        if find_file(filename):
            matched += 1
        else:
            unmatched.append(filename)

    print(f"\n  匹配成功: {matched}")
    print(f"  匹配失败: {len(unmatched)}")
    print(f"  匹配率: {matched / len(df) * 100:.1f}%")

    if len(unmatched) > 0:
        print(f"\n  前10个未匹配的文件名:")
        for fn in unmatched[:10]:
            print(f"    - {fn}")

    if matched / len(df) > 0.9:
        print("\n  ✓ 文件匹配率良好 (>90%)")
        return True
    else:
        print("\n  ✗ 警告: 文件匹配率较低!")
        return False


# ============================================================================
# 验证4: 数据文件格式
# ============================================================================

def verify_data_format():
    """验证原始数据文件的格式"""

    print("\n" + "=" * 70)
    print("  验证4: 原始数据文件格式")
    print("=" * 70)

    if not os.path.exists(DATA_FOLDER):
        print(f"\n  ✗ 错误: 数据文件夹不存在")
        return False

    # 找一个示例文件
    files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('.txt')]
    if len(files) == 0:
        print("\n  ✗ 错误: 没有找到.txt文件")
        return False

    sample_file = os.path.join(DATA_FOLDER, files[0])
    print(f"\n  示例文件: {files[0]}")

    # 读取前几行
    with open(sample_file, 'r') as f:
        lines = [f.readline() for _ in range(10)]

    print("\n  文件前5行:")
    for i, line in enumerate(lines[:5]):
        print(f"    {i + 1}: {line.strip()}")

    # 解析格式
    print("\n  数据格式分析:")

    sensor1_count = 0
    sensor2_count = 0
    timestamps = []

    with open(sample_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 100:  # 只分析前100行
                break
            parts = line.strip().split()
            if len(parts) >= 8:
                if parts[0] == '01':
                    sensor1_count += 1
                elif parts[0] == '02':
                    sensor2_count += 1
                timestamps.append(int(parts[7]))

    print(f"    传感器01 (拇指) 行数: {sensor1_count}")
    print(f"    传感器02 (食指) 行数: {sensor2_count}")

    if len(timestamps) >= 2:
        # 计算采样间隔
        intervals = np.diff(timestamps[:50])
        avg_interval = np.mean(intervals[intervals > 0])
        est_sampling_rate = 1000 / avg_interval if avg_interval > 0 else 0

        print(f"    平均采样间隔: {avg_interval:.1f} ms")
        print(f"    估计采样率: {est_sampling_rate:.1f} Hz")

        # 时间戳转换
        first_ts = timestamps[0]
        from datetime import datetime
        dt = datetime.fromtimestamp(first_ts / 1000)
        print(f"    首个时间戳: {first_ts} → {dt.strftime('%Y-%m-%d %H:%M:%S')}")

    print("\n  数据列含义:")
    print("    列1: 传感器ID (01=拇指, 02=食指)")
    print("    列2-4: 加速度计 (acc_x, acc_y, acc_z)")
    print("    列5-7: 陀螺仪 (gyro_x, gyro_y, gyro_z)")
    print("    列8: Unix时间戳 (毫秒)")

    print("\n  ✓ 数据格式验证完成")
    return True


# ============================================================================
# 验证5: 预处理后的数据
# ============================================================================

def verify_processed_data():
    """验证预处理后的数据"""

    print("\n" + "=" * 70)
    print("  验证5: 预处理后的数据")
    print("=" * 70)

    train_path = os.path.join(OUTPUT_FOLDER, 'train_data.npz')

    if not os.path.exists(train_path):
        print(f"\n  ✗ 预处理数据不存在，请先运行 pd_preprocessing.py")
        return False

    # 加载数据
    train_data = np.load(train_path)
    val_data = np.load(os.path.join(OUTPUT_FOLDER, 'val_data.npz'))
    test_data = np.load(os.path.join(OUTPUT_FOLDER, 'test_data.npz'))

    X_train, y_train = train_data['X'], train_data['y']
    X_val, y_val = val_data['X'], val_data['y']
    X_test, y_test = test_data['X'], test_data['y']

    print(f"\n  训练集: {X_train.shape}, PD={sum(y_train == 1)}, Non-PD={sum(y_train == 0)}")
    print(f"  验证集: {X_val.shape}, PD={sum(y_val == 1)}, Non-PD={sum(y_val == 0)}")
    print(f"  测试集: {X_test.shape}, PD={sum(y_test == 1)}, Non-PD={sum(y_test == 0)}")

    # 验证数据形状
    expected_shape = (360, 12)  # (timesteps, channels)

    print(f"\n  数据形状验证:")
    print(f"    期望每个样本形状: {expected_shape}")
    print(f"    实际样本形状: {X_train[0].shape}")

    if X_train[0].shape == expected_shape:
        print("    ✓ 形状正确")
    else:
        print("    ✗ 形状不匹配!")
        return False

    # 验证数据值
    print(f"\n  数据值统计 (训练集):")
    print(f"    均值: {X_train.mean():.4f} (期望接近0)")
    print(f"    标准差: {X_train.std():.4f} (期望接近1)")
    print(f"    最小值: {X_train.min():.4f}")
    print(f"    最大值: {X_train.max():.4f}")

    # 验证标签
    print(f"\n  标签验证:")
    print(f"    唯一标签值: {np.unique(y_train)}")

    if set(np.unique(y_train)) == {0, 1}:
        print("    ✓ 标签值正确 (0和1)")
    else:
        print("    ✗ 标签值异常!")
        return False

    # 检查是否有NaN
    if np.isnan(X_train).any():
        print("\n    ✗ 警告: 数据中存在NaN值!")
        return False
    else:
        print("\n    ✓ 数据中无NaN值")

    # 加载元数据
    meta_path = os.path.join(OUTPUT_FOLDER, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        print(f"\n  元数据:")
        print(f"    通道名称: {metadata['config']['channel_names'][:6]}...")
        print(f"    采样率: {metadata['config']['sampling_rate']} Hz")
        print(f"    序列长度: {metadata['config']['target_length']}")

    print("\n  ✓ 预处理数据验证通过")
    return True


# ============================================================================
# 验证6: 具体样本检查
# ============================================================================

def verify_specific_samples():
    """检查具体样本的标签是否正确"""

    print("\n" + "=" * 70)
    print("  验证6: 具体样本抽查")
    print("=" * 70)

    if not os.path.exists(EXCEL_PATH):
        return False

    df = pd.read_excel(EXCEL_PATH, sheet_name='Finger-Tapping-Release')

    def classify_pd_status(notes):
        if pd.isna(notes):
            return 0
        notes_str = str(notes).strip().lower()
        if notes_str == '' or notes_str == '  ':
            return 0
        if 'pd' in notes_str or 'pds' in notes_str:
            return 1
        return 0

    df['label'] = df['Notes'].apply(classify_pd_status)

    print("\n  随机抽取10个PD样本:")
    print(f"  {'Patient ID':<15} {'Notes':<20} {'标签':<6} {'Hand'}")
    print("  " + "-" * 55)

    pd_samples = df[df['label'] == 1].sample(min(10, len(df[df['label'] == 1])))
    for _, row in pd_samples.iterrows():
        notes = str(row['Notes'])[:18] if pd.notna(row['Notes']) else "(空)"
        hand = row.get('Hand', 'N/A')
        print(f"  {str(row['Patient ID']):<15} {notes:<20} {row['label']:<6} {hand}")

    print("\n  随机抽取10个Non-PD样本:")
    print(f"  {'Patient ID':<15} {'Notes':<20} {'标签':<6} {'Hand'}")
    print("  " + "-" * 55)

    non_pd_samples = df[df['label'] == 0].sample(min(10, len(df[df['label'] == 0])))
    for _, row in non_pd_samples.iterrows():
        notes = str(row['Notes'])[:18] if pd.notna(row['Notes']) else "(空)"
        hand = row.get('Hand', 'N/A')
        print(f"  {str(row['Patient ID']):<15} {notes:<20} {row['label']:<6} {hand}")

    print("\n  请人工检查上述样本的标签是否正确分配")
    return True


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("            预处理代码验证工具")
    print("=" * 70)
    print(f"\n  Excel路径: {EXCEL_PATH}")
    print(f"  数据文件夹: {DATA_FOLDER}")
    print(f"  输出文件夹: {OUTPUT_FOLDER}")

    results = {}

    # 验证1
    results['标签分类逻辑'] = verify_label_classification()

    # 验证2
    results['Excel标签分布'] = verify_excel_labels()

    # 验证3
    results['文件匹配'] = verify_file_matching()

    # 验证4
    results['数据格式'] = verify_data_format()

    # 验证5
    results['预处理数据'] = verify_processed_data()

    # 验证6
    results['样本抽查'] = verify_specific_samples()

    # 总结
    print("\n" + "=" * 70)
    print("                    验证总结")
    print("=" * 70)

    print(f"\n  {'验证项目':<20} {'结果'}")
    print("  " + "-" * 40)

    all_pass = True
    for name, passed in results.items():
        status = "✓ 通过" if passed else "✗ 失败/跳过"
        print(f"  {name:<20} {status}")
        if not passed:
            all_pass = False

    print("\n" + "=" * 70)
    if all_pass:
        print("  ✓ 所有验证通过! 预处理代码正确实现了项目要求")
    else:
        print("  ! 部分验证未通过，请检查上述详细信息")
    print("=" * 70)


if __name__ == "__main__":
    main()