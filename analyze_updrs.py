import pandas as pd
import numpy as np
import os
import json

# Read Excel file
df = pd.read_excel("D:/Final data/All-Ruijin-labels.xlsx")
print("Columns:", df.columns.tolist())
print("Shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head(5).to_string())

# Find UPDRS column
updrs_col = None
for col in df.columns:
    if 'UPDRS' in col or 'updrs' in col.lower():
        print(f"\nFound UPDRS column: '{col}'")
        print(df[col].value_counts())
        updrs_col = col

# Try exact column name
if 'FT Clinical UPDRS Score' in df.columns:
    updrs_col = 'FT Clinical UPDRS Score'

print(f"\nUsing UPDRS column: {updrs_col}")
mask = df[updrs_col].isin([0, 1])
filtered = df[mask]
print(f"UPDRS 0 or 1 count: {len(filtered)}")
print(filtered[updrs_col].value_counts())

# Find Data Filename column
print("\nAll columns:")
for i, col in enumerate(df.columns):
    print(f"  {i}: {col}")
