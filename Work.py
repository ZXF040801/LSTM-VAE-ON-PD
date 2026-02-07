import pandas as pd
import os

# ==========================================
# 配置路径
# ==========================================
# 请将此处修改为您本地 Excel 文件的实际路径
# 参考您之前的代码，路径可能是:
excel_path = r'D:\Final work\DataForStudentProject-HWU\All-Ruijin-labels.xlsx'

# 如果文件在当前目录下，可以直接用文件名:
# excel_path = 'All-Ruijin-labels.xlsx'

sheet_name = 'Finger-Tapping-Release'


# ==========================================
# 执行统计
# ==========================================
def analyze_updrs_scores(file_path, sheet):
    print(f"正在读取 Excel 文件: {file_path}")
    print(f"工作表 (Sheet): {sheet}")

    if not os.path.exists(file_path):
        print(f"错误: 找不到文件 {file_path}")
        return

    try:
        # 读取指定的 Sheet
        df = pd.read_excel(file_path, sheet_name=sheet)

        # 检查是否存在 'Clinical UPDRS Score' 列
        target_col = 'Clinical UPDRS Score'

        if target_col in df.columns:
            # 统计分数的分布
            # dropna=False 表示把空值 (NaN) 也统计出来
            score_counts = df[target_col].value_counts(dropna=False).sort_index()

            print("\n" + "=" * 40)
            print("FT Clinical UPDRS Score 分数统计结果")
            print("=" * 40)
            print(f"{'score':<10} | {'number':<10}")
            print("-" * 23)

            for score, count in score_counts.items():
                print(f"{str(score):<10} | {count:<10}")

            print("-" * 23)
            print(f"Total sample size: {len(df)}")

        else:
            print(f"\n错误: 在该 Sheet 中找不到列 '{target_col}'")
            print("存在的列名:", df.columns.tolist())

    except Exception as e:
        print(f"\n发生错误: {e}")


if __name__ == "__main__":
    analyze_updrs_scores(excel_path, sheet_name)