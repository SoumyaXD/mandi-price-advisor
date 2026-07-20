"""Seasonal check: is the train->test mean-price drop a recurring seasonal
pattern, or specific to the Jan-Jun 2025 test period?

Approach:
  * Restrict to the modeling scope (Onion + Potato) and to TRAINING rows only
    (dates <= the 80% time-based cutoff), so we judge seasonality from data
    the model actually sees.
  * For each commodity, compute mean modal_price by (year, month).
  * Compare the Jan-Jun window of each available prior year against the
    Jan-Jun 2025 TEST window side by side.

The test window (2025-01-15 .. 2025-06-11) is by definition NOT in training,
so its monthly means come from the TEST split — that is intentional: we want
to know whether the test-period prices look like a normal seasonal dip.

Run:
    python -m src.models.seasonal_check
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
TEST_WINDOW_MONTHS = list(range(1, 7))  # Jan-Jun

SEP_THICK = "=" * 78
SEP_THIN = "-" * 78


def section(title: str) -> None:
    print(f"\n{SEP_THICK}\n{title}\n{SEP_THICK}")


def main():
    logger.info("Loading features CSV: %s", FEATURES_CSV)
    df = pd.read_csv(FEATURES_CSV, parse_dates=[COL_DATE])

    # Scope to modeling commodities
    df = df[df[COL_COMMODITY].isin(COMMODITIES_MODELING_V1)].reset_index(drop=True)

    # Same time-based cutoff as train_baseline.py / pre_training_verification.py
    df = df.sort_values(COL_DATE).reset_index(drop=True)
    date_min, date_max = df[COL_DATE].min(), df[COL_DATE].max()
    cutoff = date_min + (date_max - date_min) * TRAIN_RATIO
    train = df[df[COL_DATE] <= cutoff].copy()
    test = df[df[COL_DATE] > cutoff].copy()
    logger.info("Cutoff %s | train %d rows | test %d rows", cutoff.date(), len(train), len(test))

    # Calendar columns for grouping
    for d in (train, test):
        d["year"] = d[COL_DATE].dt.year
        d["month"] = d[COL_DATE].dt.month

    section("SEASONAL ANALYSIS — mean modal_price by (year, month)")
    print(f"Modeling scope : {COMMODITIES_MODELING_V1}")
    print(f"Train cutoff   : {cutoff.date()}")
    print(f"Test window    : Jan-Jun 2025  ({test[COL_DATE].min().date()} -> "
          f"{test[COL_DATE].max().date()})")
    print(f"Test-window months evaluated: {TEST_WINDOW_MONTHS}")

    for comm in COMMODITIES_MODELING_V1:
        section(f"[{comm}]  mean modal_price by (year, month) — TRAINING rows only")
        tr_sub = train[train[COL_COMMODITY] == comm]
        if tr_sub.empty:
            print("  (no training rows)")
            continue

        # Pivot: rows = year, cols = month, values = mean modal_price
        pivot = (
            tr_sub.groupby(["year", "month"])[COL_MODAL_PRICE]
            .mean()
            .unstack("month")
            .reindex(columns=range(1, 13))
        )
        pivot.columns = [pd.Timestamp(2000, m, 1).strftime("%b") for m in pivot.columns]
        print("TRAINING means (INR/quintal):")
        print(pivot.round(1).to_string())

        # Test-period monthly means (Jan-Jun 2025)
        te_sub = test[test[COL_COMMODITY] == comm]
        te_monthly = (
            te_sub[te_sub["month"].isin(TEST_WINDOW_MONTHS)]
            .groupby("month")[COL_MODAL_PRICE]
            .mean()
            .reindex(TEST_WINDOW_MONTHS)
        )
        te_monthly.index = [pd.Timestamp(2000, m, 1).strftime("%b") for m in te_monthly.index]

        # Year-over-year comparison for Jan-Jun across all years present.
        #
        # IMPORTANT: only FULL Jan-Jun windows are comparable. The train split
        # contains a partial Jan 2025 (Jan 1-14, before the cutoff); that sliver
        # is NOT a Jan-Jun window and must be excluded so we don't compare the
        # test against 14 days of a single month. We detect "full window" by
        # requiring all 6 months (Jan-Jun) to be present for that year.
        print(f"\n[{comm}]  Jan-Jun window, year-over-year:")
        comp_rows = []
        comparable_prior_means = []  # full Jan-Jun windows from train, by year
        for y in sorted(train["year"].unique()):
            yr_sub = tr_sub[(tr_sub["year"] == y) & (tr_sub["month"].isin(TEST_WINDOW_MONTHS))]
            if yr_sub.empty:
                continue
            months_present = yr_sub["month"].nunique()
            is_full = months_present == len(TEST_WINDOW_MONTHS)
            comp_rows.append({
                "period": f"Jan-Jun {y} (train)" + ("" if is_full else f"  [partial: {months_present}/6 months]"),
                "n_rows": len(yr_sub),
                "mean": round(yr_sub[COL_MODAL_PRICE].mean(), 1),
                "median": round(yr_sub[COL_MODAL_PRICE].median(), 1),
                "std": round(yr_sub[COL_MODAL_PRICE].std(), 1),
            })
            if is_full:
                comparable_prior_means.append((y, yr_sub[COL_MODAL_PRICE].mean()))
        # Test row
        if not te_sub.empty:
            te_jj = te_sub[te_sub["month"].isin(TEST_WINDOW_MONTHS)]
            comp_rows.append({
                "period": "Jan-Jun 2025 (TEST)",
                "n_rows": len(te_jj),
                "mean": round(te_jj[COL_MODAL_PRICE].mean(), 1),
                "median": round(te_jj[COL_MODAL_PRICE].median(), 1),
                "std": round(te_jj[COL_MODAL_PRICE].std(), 1),
            })
        comp = pd.DataFrame(comp_rows).set_index("period")

        # YoY delta vs the most recent FULL prior Jan-Jun window in training
        if comparable_prior_means and "Jan-Jun 2025 (TEST)" in comp.index:
            prior_y, prior_mean = comparable_prior_means[-1]
            test_mean = comp.loc["Jan-Jun 2025 (TEST)", "mean"]
            delta_pct = (test_mean - prior_mean) / prior_mean * 100
            prior_label = f"Jan-Jun {prior_y}"
        else:
            delta_pct = float("nan")
            prior_label = "N/A"

        print(comp.to_string())
        print(f"\n  Jan-Jun 2025 (TEST) vs most recent FULL prior window "
              f"({prior_label}, train): {delta_pct:+.1f}%")

        # Seasonality signal: was the prior full Jan-Jun also below that year's
        # own full-train mean? If yes, the test dip is consistent with seasonality.
        if comparable_prior_means:
            full_train_mean = tr_sub[COL_MODAL_PRICE].mean()
            prior_y2, prior_jj_mean = comparable_prior_means[-1]
            prior_dip_vs_full = (prior_jj_mean - full_train_mean) / full_train_mean * 100
            print(f"  {prior_label} was {prior_dip_vs_full:+.1f}% vs its own "
                  f"full-train mean ({full_train_mean:.1f}).")

    section("INTERPRETATION")
    print("If Jan-Jun of prior year(s) shows a similar dip relative to their own")
    print("annual mean, the test-period drop is consistent with recurring")
    print("seasonality (Indian rabi/harvest cycle). If only 2025 dips while prior")
    print("years' Jan-Jun is in line with their annual mean, the shift is")
    print("test-period-specific (e.g. a 2025 supply/market shock) and the model's")
    print("test metrics should be interpreted with that caveat.")


if __name__ == "__main__":
    main()
