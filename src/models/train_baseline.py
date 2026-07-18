"""Naive lag-1 baseline model for agri-price-forecaster.

Establishes a hard-to-beat floor: predicted price = price from the
immediately prior reporting day (lag_1). Evaluated on a time-based
test split to avoid future leakage.

Run:
    python -m src.models.train_baseline
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

FEATURES_CSV    = PROCESSED_DATA_DIR / "mandi_prices_features.csv"
PREDICTIONS_CSV = PROCESSED_DATA_DIR / "baseline_predictions.csv"

TRAIN_RATIO = 0.80

# Commodities to break out individually in the per-commodity report.
# Driven by COMMODITIES_MODELING_V1 — Tomato/Wheat excluded due to hard
# reporting cutoffs before the test window (source data limitation, not a pipeline bug).
FOCUS_COMMODITIES = COMMODITIES_MODELING_V1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(actual: pd.Series, predicted: pd.Series) -> float:
    return (actual - predicted).abs().mean()


def rmse(actual: pd.Series, predicted: pd.Series) -> float:
    return np.sqrt(((actual - predicted) ** 2).mean())


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean Absolute Percentage Error — skips rows where actual == 0."""
    mask = actual != 0
    return ((actual[mask] - predicted[mask]).abs() / actual[mask]).mean() * 100


def metrics_row(label: str, actual: pd.Series, predicted: pd.Series) -> dict:
    return {
        "group":     label,
        "n":         len(actual),
        "MAE":       round(mae(actual, predicted), 2),
        "RMSE":      round(rmse(actual, predicted), 2),
        "MAPE (%)":  round(mape(actual, predicted), 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # 1. Load
    logger.info("Loading features CSV: %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV, parse_dates=[COL_DATE])
    logger.info("Loaded %d rows", len(df))

    # 2. Drop rows where lag_1 is NaN
    before = len(df)
    df = df.dropna(subset=["lag_1"]).reset_index(drop=True)
    dropped = before - len(df)
    logger.info("Dropped %d rows with NaN lag_1 — %d rows remaining", dropped, len(df))

    # 2b. Filter to modeling scope — Onion and Potato only
    # Tomato (ends 2023-11-06) and Wheat (ends 2024-02-06) fall entirely within
    # the training window and have no test-period data; see investigate_gaps.py.
    before_filter = len(df)
    df = df[df[COL_COMMODITY].isin(COMMODITIES_MODELING_V1)].reset_index(drop=True)
    logger.info(
        "Filtered to modeling commodities %s — %d rows remaining (dropped %d)",
        COMMODITIES_MODELING_V1, len(df), before_filter - len(df),
    )

    # 3. Time-based train/test split — sort by date, cut at 80th percentile date
    df = df.sort_values(COL_DATE).reset_index(drop=True)

    date_min   = df[COL_DATE].min()
    date_max   = df[COL_DATE].max()
    total_span  = (date_max - date_min).days
    # Use dateutil relativedelta-safe arithmetic: compute the cutoff date directly
    # from the date range to avoid numpy int dtype issues with pd.Timedelta
    cutoff = date_min + (date_max - date_min) * TRAIN_RATIO

    logger.info(
        "Date range: %s → %s  |  span: %d days  |  cutoff: %s",
        date_min.date(), date_max.date(), total_span, cutoff.date(),
    )

    train = df[df[COL_DATE] <= cutoff].copy()
    test  = df[df[COL_DATE] >  cutoff].copy()

    logger.info("Train rows: %d  |  Test rows: %d", len(train), len(test))

    # 4. Naive baseline: predicted = lag_1
    test = test.copy()
    test["predicted"] = test["lag_1"]

    actual    = test[COL_MODAL_PRICE]
    predicted = test["predicted"]

    # 5. Metrics — overall + per focus commodity
    results = [metrics_row("OVERALL", actual, predicted)]

    for commodity in FOCUS_COMMODITIES:
        mask = test[COL_COMMODITY].str.lower() == commodity.lower()
        subset = test[mask]
        if len(subset) == 0:
            # Not a bug — check if commodity exists at all in the dataset
            all_rows = df[df[COL_COMMODITY].str.lower() == commodity.lower()]
            if len(all_rows) == 0:
                logger.warning("Commodity '%s' not found in dataset at all", commodity)
            else:
                last_date = all_rows[COL_DATE].max().date()
                logger.info(
                    "Commodity '%s' has no test-period rows — last entry is %s "
                    "(entirely within training window, not a data error)",
                    commodity, last_date,
                )
            continue
        results.append(
            metrics_row(commodity, subset[COL_MODAL_PRICE], subset["predicted"])
        )

    metrics_df = pd.DataFrame(results).set_index("group")

    # 6. Save predictions
    out = test[[COL_STATE, COL_COMMODITY, COL_DATE, COL_MODAL_PRICE, "predicted"]].copy()
    out.to_csv(PREDICTIONS_CSV, index=False)
    logger.info("Baseline predictions saved to %s", PREDICTIONS_CSV)

    # 7. Print report
    SEP = "-" * 60
    print(f"\n{SEP}")
    print("NAIVE BASELINE REPORT (predicted = lag_1)")
    print(SEP)
    print(f"Rows with NaN lag_1 dropped : {dropped}")
    print(f"Rows remaining              : {len(df)}")
    print(f"Date range                  : {date_min.date()} → {date_max.date()}")
    print(f"Train/test cutoff           : {cutoff.date()}")
    print(f"Train rows                  : {len(train)}")
    print(f"Test rows                   : {len(test)}")
    print(f"\n{SEP}")
    print("METRICS (test set only)")
    print(SEP)
    print(metrics_df.to_string())
    print(SEP)


if __name__ == "__main__":
    main()
