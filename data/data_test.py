# -*- coding: utf-8 -*-
"""
Check dataset integrity: NaN, inf, or abnormal values in each column
Target: /proj/home/ibs/spaceai_2025/ai2271056/data/simulation_data/train_imputed_dataset_final.csv
"""

import pandas as pd
import numpy as np
import csv
import os

CSV_PATH = "/proj/home/ibs/spaceai_2025/ai2271056/data/simulation_data/train_imputed_dataset_final.csv"
assert os.path.exists(CSV_PATH), f"File not found: {CSV_PATH}"

# ---------------------------------------------
# 1) Load dataset safely
# ---------------------------------------------
print(f"Loading CSV: {CSV_PATH}")
df = pd.read_csv(
    CSV_PATH,
    engine="python",
    sep=None,               # auto detect delimiter
    encoding="utf-8",
    quoting=csv.QUOTE_MINIMAL
)
df.columns = [c.strip().strip("\ufeff") for c in df.columns]
print(f"[info] Loaded shape = {df.shape}")
print(f"[info] Columns = {list(df.columns)}\n")

# ---------------------------------------------
# 2) Detect NaN, inf, and non-numeric columns
# ---------------------------------------------
summary = []

for col in df.columns:
    series = df[col]
    n_total = len(series)
    n_nan = series.isna().sum()
    n_inf = np.isinf(series).sum() if np.issubdtype(series.dtype, np.number) else 0

    # Try numeric conversion (for object columns)
    numeric_version = pd.to_numeric(series, errors="coerce")
    n_non_numeric = int((numeric_version.isna() & ~series.isna()).sum()) if series.dtype == "object" else 0

    # Extreme values
    if np.issubdtype(series.dtype, np.number):
        finite_values = series[np.isfinite(series)]
        if len(finite_values) > 0:
            max_val = finite_values.max()
            min_val = finite_values.min()
        else:
            max_val = min_val = np.nan
    else:
        max_val = min_val = None

    summary.append({
        "column": col,
        "dtype": str(series.dtype),
        "nan_count": int(n_nan),
        "inf_count": int(n_inf),
        "non_numeric_count": int(n_non_numeric),
        "min": float(min_val) if isinstance(min_val, (int, float, np.floating)) and np.isfinite(min_val) else None,
        "max": float(max_val) if isinstance(max_val, (int, float, np.floating)) and np.isfinite(max_val) else None,
    })

# ---------------------------------------------
# 3) Print summary neatly
# ---------------------------------------------
print("=== Column-wise Integrity Report ===")
bad_cols = []
for s in summary:
    line = (
        f"{s['column']:<25} | dtype={s['dtype']:<10} "
        f"| NaN={s['nan_count']:<5} inf={s['inf_count']:<5} non-num={s['non_numeric_count']:<5} "
        f"| min={s['min']} max={s['max']}"
    )
    print(line)
    if s["nan_count"] > 0 or s["inf_count"] > 0 or s["non_numeric_count"] > 0:
        bad_cols.append(s["column"])

# ---------------------------------------------
# 4) Final summary
# ---------------------------------------------
if bad_cols:
    print("\n⚠️  Abnormal columns detected:")
    for c in bad_cols:
        print(f" - {c}")
else:
    print("\n✅ No NaN / inf / non-numeric anomalies detected!")

