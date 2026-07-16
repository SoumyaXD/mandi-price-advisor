"""Feature engineering pipeline for mandi-price-advisor.

Reads the cleaned CSV, aggregates to (state, commodity, date) daily level,
then builds lag, rolling, calendar, and MSP-relative features.

Run:
    python -m src.features.build_features
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
    RANDOM_SEED,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CLEANED_CSV  = PROCESSED_DATA_DIR / "mandi_prices_cleaned.csv"
FEATURES_CSV = PROCESSED_DATA_DIR / "mandi_prices_features.csv"

# ---------------------------------------------------------------------------
# MSP reference value for Wheat (2023-24 season, Government of India).
# Replace with a programmatically sourced value when integrating live MSP data.
# Source: https://pib.gov.in/PressReleasePage.aspx?PRID=1927051
WHEAT_MSP = 2425  # INR per quintal
# ---------------------------------------------------------------------------


def aggregate_to_state_commodity_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Group by (state, commodity, date) and take mean of modal_price.

    Collapses market-level rows to a single daily state+commodity observation,
    which is the granularity we confirmed is viable for lag/rolling features
    (~43% date coverage vs ~17% at market level).
    """
    logger.info("Aggregating to (state, commodity, date) — rows before: %d", len(df))
    agg = (
        df.groupby([COL_STATE, COL_COMMODITY, COL_DATE], as_index=False)[COL_MODAL_PRICE]
        .mean()
        .sort_values([COL_STATE, COL_COMMODITY, COL_DATE])
        .reset_index(drop=True)
    )
    logger.info("Rows after aggregation: %d", len(agg))
    return agg


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lag_1, lag_7, lag_14, lag_30 columns.

    NOTE: These are lags by REPORT COUNT (n prior reporting rows within the
    group), NOT literal calendar days. Reporting at (state, commodity) level
    is irregular — median ~316 distinct days out of 736 — so lag_7 means
    "7 reporting events ago", which may span more than 7 calendar days.
    """
    grp = df.groupby([COL_STATE, COL_COMMODITY])[COL_MODAL_PRICE]
    for n in [1, 7, 14, 30]:
        df[f"lag_{n}"] = grp.shift(n)
        logger.info("Added lag_%d", n)
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling_mean_{w} and rolling_std_{w} for windows [7, 30].

    LEAKAGE PREVENTION: the series is shifted by 1 before rolling so the
    current row's own modal_price is never included in its own statistic.
    Both shift and rolling are applied within each (state, commodity) group.
    """
    grp = df.groupby([COL_STATE, COL_COMMODITY])[COL_MODAL_PRICE]
    for w in [7, 30]:
        shifted = grp.shift(1)  # exclude current row
        df[f"rolling_mean_{w}"] = (
            shifted.groupby(df[COL_STATE].astype(str) + "__" + df[COL_COMMODITY])
            .transform(lambda s: s.rolling(w, min_periods=1).mean())
        )
        df[f"rolling_std_{w}"] = (
            shifted.groupby(df[COL_STATE].astype(str) + "__" + df[COL_COMMODITY])
            .transform(lambda s: s.rolling(w, min_periods=1).std())
        )
        logger.info("Added rolling_mean_%d and rolling_std_%d", w, w)
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cyclic month encoding, day_of_week, and day_of_year from COL_DATE."""
    month = df[COL_DATE].dt.month
    df["month_sin"]    = np.sin(2 * np.pi * month / 12)
    df["month_cos"]    = np.cos(2 * np.pi * month / 12)
    df["day_of_week"]  = df[COL_DATE].dt.dayofweek   # 0=Monday … 6=Sunday
    df["day_of_year"]  = df[COL_DATE].dt.dayofyear
    logger.info("Added calendar features: month_sin, month_cos, day_of_week, day_of_year")
    return df


def add_msp_relative_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add price_vs_msp = modal_price / MSP, for Wheat only.

    All other commodities get NaN (not 0) — a zero would be a nonsensical
    ratio and could corrupt model training if NaN-handling is bypassed.
    Rice was dropped in the cleaning pipeline so Wheat is the only
    MSP-covered crop remaining in the dataset.
    """
    df["price_vs_msp"] = np.where(
        df[COL_COMMODITY].str.lower() == "wheat",
        df[COL_MODAL_PRICE] / WHEAT_MSP,
        np.nan,
    )
    wheat_rows = (df[COL_COMMODITY].str.lower() == "wheat").sum()
    logger.info(
        "Added price_vs_msp (WHEAT_MSP=%d). Wheat rows: %d, non-Wheat set to NaN.",
        WHEAT_MSP, wheat_rows,
    )
    return df


def main():
    np.random.seed(RANDOM_SEED)

    # 1. Load
    logger.info("Loading cleaned CSV: %s", CLEANED_CSV)
    df = pd.read_csv(CLEANED_CSV, parse_dates=[COL_DATE])
    logger.info("Loaded %d rows, %d columns: %s", len(df), len(df.columns), list(df.columns))

    # 2. Aggregate
    df = aggregate_to_state_commodity_daily(df)
    logger.info("Post-aggregation columns: %s", list(df.columns))

    # 3. Lag features
    df = add_lag_features(df)
    logger.info("Columns after lag features (%d total): %s", len(df.columns), list(df.columns))

    # 4. Rolling features
    df = add_rolling_features(df)
    logger.info("Columns after rolling features (%d total): %s", len(df.columns), list(df.columns))

    # 5. Calendar features
    df = add_calendar_features(df)
    logger.info("Columns after calendar features (%d total): %s", len(df.columns), list(df.columns))

    # 6. MSP-relative feature
    df = add_msp_relative_feature(df)
    logger.info("Columns after MSP feature (%d total): %s", len(df.columns), list(df.columns))

    # 7. Save
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    logger.info("Feature CSV saved to %s  (%d rows x %d cols)", FEATURES_CSV, len(df), len(df.columns))

    # Summary
    print("\n--- Feature Build Summary ---")
    print(f"Rows       : {len(df)}")
    print(f"Columns    : {len(df.columns)}")
    print(f"Columns    : {list(df.columns)}")
    print(f"NaN counts :\n{df.isnull().sum().to_string()}")


if __name__ == "__main__":
    main()
