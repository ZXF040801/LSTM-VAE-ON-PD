"""
移除特征高度可疑（tier=HIGH）的文件，移到 quarantine/ 目录备份而非直接删除。
用法：python3 remove_suspicious.py [--dry-run]
"""

import os
import shutil
import argparse
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
CSV  = os.path.join(BASE, 'selected_features_check.csv')
QUARANTINE = os.path.join(BASE, 'quarantine')

def main(dry_run=False):
    df = pd.read_csv(CSV)
    high = df[df['tier'] == 'HIGH'].copy()

    print(f"共找到 HIGH 级别可疑文件：{len(high)} 个")
    print(f"  folder 0：{sum(high['folder']=='0')} 个")
    print(f"  folder 1：{sum(high['folder']=='1')} 个")
    print()

    moved, missing = [], []

    for _, row in high.sort_values(['folder','file']).iterrows():
        src = os.path.join(BASE, str(row['folder']), row['file'])
        dst_dir = os.path.join(QUARANTINE, str(row['folder']))
        dst = os.path.join(dst_dir, row['file'])

        if not os.path.exists(src):
            missing.append(f"  [缺失] {row['folder']}/{row['file']}")
            continue

        tag = f"  {'[DRY-RUN] ' if dry_run else ''}移动 {row['folder']}/{row['file']}"
        tag += f"  (tap_rate={row['dist_tap_rate']:.2f}Hz, RF_err={row['rf_proba_wrong']:.3f})"
        print(tag)

        if not dry_run:
            os.makedirs(dst_dir, exist_ok=True)
            shutil.move(src, dst)
        moved.append(row['file'])

    print()
    print("=" * 60)
    if dry_run:
        print(f"[DRY-RUN] 实际未移动任何文件，去掉 --dry-run 参数后执行。")
    else:
        print(f"已移动 {len(moved)} 个文件至 quarantine/")
        print(f"备份目录：{QUARANTINE}")
        # 更新剩余文件数
        for lbl in ['0', '1']:
            n = len(os.listdir(os.path.join(BASE, lbl)))
            print(f"  folder {lbl} 剩余：{n} 个文件")

    if missing:
        print(f"\n以下文件在磁盘上不存在（可能已移除）：")
        for m in missing:
            print(m)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true',
                        help='只打印，不实际移动文件')
    args = parser.parse_args()
    main(dry_run=args.dry_run)
