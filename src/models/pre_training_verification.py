"""Pre-training verification pass for agri-price-forecaster.

Runs a full integrity check BEFORE any real model is trained:
  1. Scope filter (Onion + Potato only, via config constant)
  2. Time-based 80/20 split integrity (no overlap / leakage)
  3. Confirm target column not present in feature inputs
  4. NaN audit of the modeling feature set (with an explicit handling decision)
  5. Categorical encoding plan + cardinality
  6. Target distribution sanity-check, train vs test, per commodity

This script DOES NOT train a model. It only reports findings.

Run:
    python -m src.models.pre_training_verification
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd

from src.config import (
    PROCESSED_DATA_DIR,
    COL_DATE,
    COL_STATE,
    COL_COMMODITY,
    COL_MODAL_PRICE,
    COMMODITIES_MODELING_V1,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FEATURES_CSV = PROCESSED_DATA_DIR / "mandi_prices_features.csv"

TRAIN_RATIO = 0.80

# Feature list that will actually be passed to the model. Pulled out as a
# constant so train AND verify reference the same source of truth.
#
# NOTE: 'price_vs_msp' is deliberately EXCLUDED. It is computed only for Wheat
# (the only MSP-covered crop in the dataset) in build_features.py, so for the
# Onion+Potato modeling scope it is 100% NaN — a column with zero signal.
# Including it would only add a uniformly-missing feature for XGBoost to
# learn a meaningless default direction on. See pre-training verification
# Finding F4.
MODEL_FEATURES = [
    "lag_1", "lag_7", "lag_14", "lag_30",
    "rolling_mean_7", "rolling_std_7", "rolling_mean_30", "rolling_std_30",
    "month_sin", "month_cos", "day_of_week", "day_of_year",
    "commodity", "state",
]

SEP_THICK = "=" * 78
SEP_THIN = "-" * 78


def section(title: str) -> None:
    print(f"\n{SEP_THICK}\n{title}\n{SEP_THICK}")


def load_filter_split():
    """Load features CSV, filter to COMMODITIES_MODELING_V1, and apply the
    time-based 80/20 split.

    Single source of truth for the load+filter+split logic. Both the
    pre-training verification pass and the XGBoost training script import
    this so they cannot drift apart.

    Returns:
        train, test, cutoff, df_filtered (pd.DataFrame each), where
        train = df_filtered[df_filtered[COL_DATE] <= cutoff]
        test  = df_filtered[df_filtered[COL_DATE] >  cutoff]
    """
    logger.info("Loading features CSV: %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV, parse_dates=[COL_DATE])
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Scope filter — config-driven, not hardcoded.
    df = df[df[COL_COMMODITY].isin(COMMODITIES_MODELING_V1)].reset_index(drop=True)
    logger.info("After scope filter (%s): %d rows", COMMODITIES_MODELING_V1, len(df))

    # Time-based 80/20 split (sort by date, cut at the 80th-percentile date).
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    date_min = df[COL_DATE].min()
    date_max = df[COL_DATE].max()
    cutoff = date_min + (date_max - date_min) * TRAIN_RATIO

    train = df[df[COL_DATE] <= cutoff].copy()
    test = df[df[COL_DATE] > cutoff].copy()
    logger.info(
        "Split @ %s | train %d rows (%s..%s) | test %d rows (%s..%s)",
        cutoff.date(), len(train), train[COL_DATE].min().date(), train[COL_DATE].max().date(),
        len(test), test[COL_DATE].min().date(), test[COL_DATE].max().date(),
    )
    return train, test, cutoff, df


def main():
    # =====================================================================
    # LOAD + FILTER + SPLIT  (reused by train_xgboost.py)
    # =====================================================================
    train, test, cutoff, df = load_filter_split()

    # =====================================================================
    # 1. RE-CONFIRM SCOPE FILTER
    # =====================================================================
    section("1. SCOPE FILTER — COMMODITIES_MODELING_V1")
    print(f"COMMODITIES_MODELING_V1 = {COMMODITIES_MODELING_V1}")
    print(f"\nValue counts AFTER filter ({len(df)} rows):")
    print(df[COL_COMMODITY].value_counts(dropna=False).to_string())

    leftover = set(df[COL_COMMODITY].unique()) - set(COMMODITIES_MODELING_V1)
    print(f"\nCommodities present after filter : {sorted(df[COL_COMMODITY].unique())}")
    if leftover:
        print(f"!! UNEXPECTED commodities slipped through: {leftover}")
    else:
        print("OK — only Onion and Potato remain; nothing else slipped through.")

    # =====================================================================
    # 2. RE-CONFIRM TRAIN/TEST SPLIT INTEGRITY
    # =====================================================================
    section("2. TIME-BASED 80/20 SPLIT INTEGRITY")
    date_min = df[COL_DATE].min()
    date_max = df[COL_DATE].max()
    span_days = (date_max - date_min).days

    print(f"Date range      : {date_min.date()}  ->  {date_max.date()}  ({span_days} days)")
    print(f"TRAIN_RATIO     : {TRAIN_RATIO}")
    print(f"Cutoff date     : {cutoff.date()}  (train <= cutoff, test > cutoff)")

    print(f"\nTrain rows      : {len(train)}")
    print(f"Test rows       : {len(test)}")
    print(f"Train ratio     : {len(train)/len(df):.4f}  (test {len(test)/len(df):.4f})")

    train_min, train_max = train[COL_DATE].min(), train[COL_DATE].max()
    test_min, test_max = test[COL_DATE].min(), test[COL_DATE].max()
    print(f"\nTrain date range: {train_min.date()}  ->  {train_max.date()}")
    print(f"Test  date range: {test_min.date()}  ->  {test_max.date()}")

    overlap = train_max <= test_min  # strict: train_max must be <= test_min boundary
    no_shared_dates = set(train[COL_DATE]) & set(test[COL_DATE])
    print(f"\nAll train dates strictly before test dates? "
          f"{train_max < test_min}  (train_max={train_max.date()}, test_min={test_min.date()})")
    print(f"Shared dates between train and test     : {len(no_shared_dates)}")
    if train_max < test_min and not no_shared_dates:
        print("OK — no overlap, no test-period rows leaked into train.")
    else:
        print("!! SPLIT LEAKAGE DETECTED.")

    # Per-commodity representation in train vs test
    rep = pd.DataFrame({
        "train": train[COL_COMMODITY].value_counts(),
        "test": test[COL_COMMODITY].value_counts(),
    }).fillna(0).astype(int)
    rep["test_share_%"] = (rep["test"] / rep["train"].replace(0, np.nan) * 100).round(1)
    print(f"\nPer-commodity row counts (train vs test):")
    print(rep.to_string())
    # Plausibility: both commodities should be present in test
    missing_in_test = set(COMMODITIES_MODELING_V1) - set(test[COL_COMMODITY].unique())
    if missing_in_test:
        print(f"!! These modeling commodities have ZERO test rows: {missing_in_test}")
    else:
        print("OK — both Onion and Potato have meaningful representation in test.")

    # =====================================================================
    # 3. RE-CONFIRM NO LEAKAGE IN FEATURE COLUMNS
    # =====================================================================
    section("3. TARGET-NOT-IN-FEATURES CHECK")
    print(f"Target column      : {COL_MODAL_PRICE!r}")
    print(f"MODEL_FEATURES     : {MODEL_FEATURES}")
    if COL_MODAL_PRICE in MODEL_FEATURES:
        print(f"!! LEAKAGE: target {COL_MODAL_PRICE!r} is in MODEL_FEATURES.")
    else:
        print(f"OK — target {COL_MODAL_PRICE!r} is NOT in MODEL_FEATURES.")
    # Also flag any feature column that is a trivial transform of the target
    # (only the lags/rollings are, and they were shifted to exclude the current row).
    print(f"\nAll columns in df : {list(df.columns)}")

    # =====================================================================
    # 4. NaN AUDIT OF MODELING FEATURE SET
    # =====================================================================
    section("4. NaN AUDIT (train and test, per feature)")
    nan_train = train[MODEL_FEATURES].isnull().sum()
    nan_test = test[MODEL_FEATURES].isnull().sum()
    nan_report = pd.DataFrame({
        "dtype": train[MODEL_FEATURES].dtypes.astype(str),
        "nan_train": nan_train,
        "nan_train_%": (nan_train / len(train) * 100).round(2),
        "nan_test": nan_test,
        "nan_test_%": (nan_test / len(test) * 100).round(2),
    })
    print(nan_report.to_string())

    print("\nDECISION — NaN handling strategy:")
    print("  XGBoost learns a default direction for missing values at each split")
    print("  and can ingest NaNs directly. We will therefore RELY on XGBoost's")
    print("  native NaN handling (enable it explicitly; do NOT impute).")
    print("  Rationale: the remaining NaNs are STRUCTURAL and informative —")
    print("    lag/rolling NaNs mark the start of each (state, commodity) group")
    print("    (too few prior reporting rows). Imputing would inject fabricated")
    print("    signal; letting XGB learn the default direction preserves the")
    print("    'not-enough-history' meaning.")
    print("  NOTE: price_vs_msp (100% NaN for Onion+Potato) has been REMOVED from")
    print("        MODEL_FEATURES — see Finding F4. It is not in the audit above.")

    # =====================================================================
    # 5. CATEGORICAL ENCODING PLAN + CARDINALITY
    # =====================================================================
    section("5. CATEGORICAL ENCODING PLAN")
    print("Plan: cast 'commodity' and 'state' to pandas `category` dtype and pass")
    print("      enable_categorical=True to XGBoost. XGBoost will treat them as")
    print("      categorical internally (no manual one-hot / label encoding).")
    print("      This keeps the column count small (no O(N states) blow-up) and")
    print("      avoids inventing an ordinal relationship via label encoding.")
    cat_cols = [COL_COMMODITY, COL_STATE]
    for c in cat_cols:
        card = df[c].nunique(dropna=False)
        print(f"\n  {c!r}: cardinality = {card}")
        print(df[c].value_counts(dropna=False).to_string())

    # =====================================================================
    # 6. TARGET DISTRIBUTION SANITY-CHECK (train vs test, per commodity)
    # =====================================================================
    section("6. TARGET DISTRIBUTION — modal_price (train vs test)")
    print("OVERALL:")
    overall = pd.DataFrame({
        "train": train[COL_MODAL_PRICE].describe().round(2),
        "test": test[COL_MODAL_PRICE].describe().round(2),
    })
    print(overall.to_string())

    for comm in COMMODITIES_MODELING_V1:
        tr = train[train[COL_COMMODITY] == comm][COL_MODAL_PRICE]
        te = test[test[COL_COMMODITY] == comm][COL_MODAL_PRICE]
        if len(tr) == 0 or len(te) == 0:
            print(f"\n[{comm}] empty in train or test — skipping.")
            continue
        comp = pd.DataFrame({
            "train": tr.describe().round(2),
            "test": te.describe().round(2),
        })
        # crude distribution-shift indicator
        mean_shift_pct = (te.mean() - tr.mean()) / tr.mean() * 100
        print(f"\n[{comm}]  (mean shift train->test: {mean_shift_pct:+.1f}%)")
        print(comp.to_string())

    # =====================================================================
    # FINDINGS SUMMARY
    # =====================================================================
    section("FINDINGS SUMMARY")
    print("F1 Scope        : filtered set contains only "
          f"{sorted(df[COL_COMMODITY].unique())}.")
    print(f"F2 Split        : cutoff {cutoff.date()}; train {len(train)} rows "
          f"({train_min.date()}..{train_max.date()}), test {len(test)} rows "
          f"({test_min.date()}..{test_max.date()}); "
          f"no overlap = {train_max < test_min and not no_shared_dates}.")
    print(f"F3 No leakage   : target {COL_MODAL_PRICE!r} absent from MODEL_FEATURES = "
          f"{COL_MODAL_PRICE not in MODEL_FEATURES}.")
    print("F4 NaN strategy : rely on XGBoost native NaN handling; "
          "price_vs_msp (100% NaN in scope) REMOVED from MODEL_FEATURES.")
    print("F5 Encoding     : category dtype + enable_categorical=True. "
          f"commodity card={df[COL_COMMODITY].nunique()}, "
          f"state card={df[COL_STATE].nunique()}.")
    print("F6 Distribution : per-commodity describe() printed above — review for")
    print("                   large mean shifts between train and test (structural")
    print("                   change / market shock the model has not learned).")
    print(f"\n{SEP_THIN}\nVerification complete. NO MODEL TRAINED.\n{SEP_THIN}")


if __name__ == "__main__":
    main()
