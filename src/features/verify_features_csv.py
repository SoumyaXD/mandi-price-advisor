"""Standalone CSV round-trip verification for mandi_prices_features.csv.

Deliberately starts fresh — no shared state with build_features.py.
Run:
    python -m src.features.verify_features_csv
"""

import sys
import math
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
from src.config import PROCESSED_DATA_DIR, COL_DATE

CSV = PROCESSED_DATA_DIR / "mandi_prices_features.csv"

EXPECTED_FLOAT = [
    "modal_price", "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30", "rolling_std_30",
    "month_sin", "month_cos", "price_vs_msp",
]
EXPECTED_INT   = ["day_of_week", "day_of_year"]
EXPECTED_OBJ   = ["state", "commodity"]

SEP = "-" * 60

# ---------------------------------------------------------------------------
# STEP 1 & 2 — naive load (no parse_dates)
# ---------------------------------------------------------------------------
print(SEP)
print("STEP 1+2: naive load (no parse_dates)")
print(SEP)
df_raw = pd.read_csv(CSV)
print(df_raw.dtypes.to_string())

# Assertions
issues = []

if str(df_raw[COL_DATE].dtype) != "object":
    issues.append(f"UNEXPECTED: {COL_DATE} dtype is {df_raw[COL_DATE].dtype}, expected object on naive load")
else:
    print(f"\n✓ {COL_DATE} is 'object' on naive load — CSV round-trip dtype loss confirmed as expected")

for col in EXPECTED_FLOAT:
    if str(df_raw[col].dtype) != "float64":
        issues.append(f"UNEXPECTED: {col} dtype is {df_raw[col].dtype}, expected float64")
    else:
        print(f"✓ {col}: float64")

for col in EXPECTED_INT:
    if str(df_raw[col].dtype) not in ("int64", "int32"):
        issues.append(f"UNEXPECTED: {col} dtype is {df_raw[col].dtype}, expected int64")
    else:
        print(f"✓ {col}: {df_raw[col].dtype}")

for col in EXPECTED_OBJ:
    if str(df_raw[col].dtype) != "object":
        issues.append(f"UNEXPECTED: {col} dtype is {df_raw[col].dtype}, expected object")
    else:
        print(f"✓ {col}: object (string — correct)")

# Check for any numeric column that loaded as object (stray non-numeric values)
print("\n--- Stray object columns check ---")
all_expected_numeric = EXPECTED_FLOAT + EXPECTED_INT
for col in all_expected_numeric:
    if df_raw[col].dtype == object:
        issues.append(f"STRAY NON-NUMERIC: {col} loaded as object — check for empty strings or text in column")
        print(f"⚠  {col}: loaded as object — investigate upstream")

if not issues:
    print("✓ No stray object-typed numeric columns")

# ---------------------------------------------------------------------------
# STEP 3 — reload with parse_dates
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 3: reload with parse_dates=[COL_DATE]")
print(SEP)
df = pd.read_csv(CSV, parse_dates=[COL_DATE])

date_dtype = str(df[COL_DATE].dtype)
if date_dtype.startswith("datetime64"):
    print(f"✓ {COL_DATE} dtype: {date_dtype}")
else:
    issues.append(f"UNEXPECTED: {COL_DATE} dtype is {date_dtype} after parse_dates — expected datetime64")

date_min, date_max = df[COL_DATE].min(), df[COL_DATE].max()
print(f"  Date range: {date_min.date()} → {date_max.date()}")
if date_min.year >= 2023 and date_max.year <= 2025:
    print("  ✓ Date range within expected 2023-2025 window")
else:
    issues.append(f"Date range {date_min.date()} – {date_max.date()} outside 2023-2025")

# ---------------------------------------------------------------------------
# STEP 4 — spot-check 3 random rows
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
print("STEP 4: spot-check 3 random rows (random_state=42)")
print(SEP)
sample = df.sample(3, random_state=42).sort_values(COL_DATE).reset_index(drop=True)
pd.set_option("display.float_format", "{:.4f}".format)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 140)
print(sample.to_string())

print("\n--- Manual verification of sample rows ---")
for i, row in sample.iterrows():
    month = row[COL_DATE].month
    expected_sin = math.sin(2 * math.pi * month / 12)
    expected_cos = math.cos(2 * math.pi * month / 12)
    sin_ok  = math.isclose(row["month_sin"], expected_sin, abs_tol=1e-6)
    cos_ok  = math.isclose(row["month_cos"], expected_cos, abs_tol=1e-6)
    is_wheat = str(row["commodity"]).lower() == "wheat"
    msp_ok = (not is_wheat and pd.isna(row["price_vs_msp"])) or \
             (is_wheat and not pd.isna(row["price_vs_msp"]))

    price_ok = pd.isna(row["lag_1"]) or (0 < row["lag_1"] < 1_000_000)
    lag7_ok  = pd.isna(row["lag_7"]) or (0 < row["lag_7"] < 1_000_000)

    print(f"\nRow {i} | {row['state']} / {row['commodity']} / {row[COL_DATE].date()}")
    print(f"  month={month}  expected sin={expected_sin:.4f}  stored={row['month_sin']:.4f}  {'✓' if sin_ok else '⚠ MISMATCH'}")
    print(f"  month={month}  expected cos={expected_cos:.4f}  stored={row['month_cos']:.4f}  {'✓' if cos_ok else '⚠ MISMATCH'}")
    print(f"  price_vs_msp={'NaN (non-Wheat ✓)' if (not is_wheat and pd.isna(row['price_vs_msp'])) else row['price_vs_msp']}")
    print(f"  lag_1={row['lag_1']}  {'✓ plausible' if price_ok else '⚠ IMPLAUSIBLE'}")
    print(f"  lag_7={row['lag_7']}  {'✓ plausible' if lag7_ok else '⚠ IMPLAUSIBLE'}")

    if not sin_ok:  issues.append(f"Row {i}: month_sin mismatch")
    if not cos_ok:  issues.append(f"Row {i}: month_cos mismatch")
    if not msp_ok:  issues.append(f"Row {i}: price_vs_msp wrong for commodity={row['commodity']}")
    if not price_ok: issues.append(f"Row {i}: lag_1 implausible value {row['lag_1']}")
    if not lag7_ok:  issues.append(f"Row {i}: lag_7 implausible value {row['lag_7']}")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------
print(f"\n{SEP}")
if issues:
    print("⚠  ISSUES FOUND:")
    for iss in issues:
        print(f"   - {iss}")
else:
    print("✓ All checks passed — no unexpected dtypes, no implausible values, no leakage indicators")
print(SEP)
