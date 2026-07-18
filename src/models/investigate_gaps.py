"""Investigate why Tomato and Wheat have no data after Nov 2023 / Feb 2024.

Run:
    python -m src.models.investigate_gaps
"""
import sys
from pathlib import Path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from src.config import PROCESSED_DATA_DIR, RAW_KAGGLE_FILE, COL_DATE, COL_COMMODITY, COL_STATE, COL_MARKET

SEP = "=" * 65
FOCUS = ["Tomato", "Wheat"]
ALL4  = ["Onion", "Potato", "Tomato", "Wheat"]

# ── 1. Date ranges in CLEANED CSV ────────────────────────────────
print(SEP)
print("1. Date ranges per commodity — CLEANED CSV (market-level)")
print(SEP)
cleaned = pd.read_csv(PROCESSED_DATA_DIR / "mandi_prices_cleaned.csv", parse_dates=[COL_DATE])
for c in ALL4:
    sub = cleaned[cleaned[COL_COMMODITY] == c]
    print(f"{c:10s}: {len(sub):7,d} rows  {sub[COL_DATE].min().date()} -> {sub[COL_DATE].max().date()}")

# ── 2. Date ranges in RAW CSV ─────────────────────────────────────
print()
print(SEP)
print("2. Date ranges per commodity — RAW CSV (before any cleaning)")
print(SEP)
raw = pd.read_csv(RAW_KAGGLE_FILE)
raw.columns = (raw.columns.str.strip().str.lower()
                .str.replace(" ", "_", regex=False)
                .str.replace(".", "", regex=False))
raw[COL_DATE] = pd.to_datetime(raw[COL_DATE], errors="coerce")
for c in ALL4:
    sub = raw[raw[COL_COMMODITY] == c]
    print(f"{c:10s}: {len(sub):7,d} rows  {sub[COL_DATE].min().date()} -> {sub[COL_DATE].max().date()}")

# ── 3. Monthly row counts for Tomato and Wheat ───────────────────
print()
print(SEP)
print("3. Monthly row counts — CLEANED CSV (market-level)")
print(SEP)
cleaned["year_month"] = cleaned[COL_DATE].dt.to_period("M")
for c in FOCUS:
    sub = cleaned[cleaned[COL_COMMODITY] == c]
    monthly = sub.groupby("year_month").size().reset_index(name="rows")
    print(f"\n{c} — monthly market-level row counts:")
    print(monthly.to_string(index=False))

# ── 4. Distinct states reporting per month ────────────────────────
print()
print(SEP)
print("4. Distinct states reporting per month — CLEANED CSV")
print(SEP)
for c in FOCUS:
    sub = cleaned[cleaned[COL_COMMODITY] == c]
    state_monthly = (sub.groupby("year_month")[COL_STATE]
                     .nunique()
                     .reset_index(name="distinct_states"))
    print(f"\n{c} — distinct states per month:")
    print(state_monthly.to_string(index=False))

# ── 5. Same check on RAW CSV ──────────────────────────────────────
print()
print(SEP)
print("5. Monthly row counts — RAW CSV (before cleaning)")
print(SEP)
raw["year_month"] = raw[COL_DATE].dt.to_period("M")
for c in FOCUS:
    sub = raw[raw[COL_COMMODITY] == c]
    monthly = sub.groupby("year_month").size().reset_index(name="rows")
    print(f"\n{c} — monthly raw row counts:")
    print(monthly.to_string(index=False))

# ── Verdict ───────────────────────────────────────────────────────
print()
print(SEP)
print("VERDICT")
print(SEP)
for c in FOCUS:
    raw_max  = raw[raw[COL_COMMODITY] == c][COL_DATE].max()
    cln_max  = cleaned[cleaned[COL_COMMODITY] == c][COL_DATE].max()
    if raw_max == cln_max:
        print(f"{c}: gap present in RAW data — cleaning pipeline did NOT introduce it.")
    else:
        print(f"{c}: raw ends {raw_max.date()}, cleaned ends {cln_max.date()} — PIPELINE may have caused the gap.")
