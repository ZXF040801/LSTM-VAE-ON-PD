# UPDRS 0 vs 1 特征分析（基于 FT Clinical UPDRS Score）

- 标签文件: `All-Ruijin-labels.xlsx`
- 原始符合条件记录数（标签=0或1）: **254**
- 成功匹配并提取特征的数据文件数: **254**
- 缺失数据文件数: **0**
- 0类样本数: **147**
- 1类样本数: **107**

## 区分能力最强的候选特征（按 |Cohen d| 排序）

| feature | mean(0) | mean(1) | change(1 vs 0) | cohen d |
|---|---:|---:|---:|---:|
| sample_count | 624.6259 | 755.0841 | 20.89% | 0.575 |
| duration_s | 10.3924 | 12.5669 | 20.92% | 0.575 |
| speed_mean | 16.6412 | 14.3089 | -14.02% | -0.445 |
| ang_speed_cv | 2.1897 | 2.5496 | 16.44% | 0.406 |
| acc_abs_mean | 722.0401 | 588.6410 | -18.48% | -0.294 |
| ang_speed_mean | 1040.7156 | 894.7212 | -14.03% | -0.271 |
| pos_std_z | 0.8954 | 0.8371 | -6.51% | -0.139 |
| pos_range_x | 3.0267 | 2.9240 | -3.39% | -0.110 |
| pos_std_x | 0.6860 | 0.6701 | -2.32% | -0.079 |
| speed_std | 20.2039 | 18.7723 | -7.09% | -0.070 |

## 特征解释建议

- `speed_mean`: 空间位置平均速度，反映敲击动作整体运动快慢。
- `speed_std` / `speed_cv`: 速度波动与稳定性，轻度帕金森常见节律不稳。
- `acc_abs_mean`: 速度变化率绝对值，可近似动作“顿挫感”或不平滑程度。
- `pos_range_z` / `pos_std_z`: z轴幅度和离散度，反映抬指高度与运动振幅。
- `ang_speed_mean` / `ang_speed_std`: 姿态角速度及其稳定性，反映手部旋转控制能力。

## 输出文件

- 详细统计: `analysis/updrs01_feature_summary.csv`
- 本报告: `analysis/updrs01_feature_report.md`
