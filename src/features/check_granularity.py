"""Granularity analysis for mandi_prices_cleaned.csv.

Checks how densely each (market, commodity) and (state, commodity) group
covers the overall date span — a prerequisite for deciding whether
lag/rolling features are viable at each granularity level.

Run:
    python -m src.features.check_granularity
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
from src.config import PROCESSED_DATA_DIR, COL_DATE, COL_STATE, COL_MARKET, COL_COMMODITY

CLEANED_CSV = PROCESSED_DATA_DIR / "mandi_prices_cleaned.csv"


def coverage_stats(df: pd.DataFrame, group_cols: list[str], date_col: str, total_days: int) -> dict:
    """Return median distinct-date count and coverage % for a grouping."""
    distinct_dates = (
        df.groupby(group_cols)[date_col]
        .nunique()
    )
    median_dates = distinct_dates.median()
    pct = (median_dates / total_days) * 100
    return {
        "median_dates": median_dates,
        "total_days": total_days,
        "pct": pct,
        "group_counts": len(distinct_dates),
        "series": distinct_dates,
    }


def main():
    # --- Load ------------------------------------------------------------------
    # IMPORTANT: parse_dates is required. A plain read_csv() loads price_date as
    # dtype=object (strings), silently breaking every date arithmetic call below.
    df = pd.read_csv(CLEANED_CSV, parse_dates=[COL_DATE])

    if df[COL_DATE].dtype == object:
        raise RuntimeError(
            f"'{COL_DATE}' loaded as object — parse_dates failed. "
            "Check the column name in config.py matches the CSV header."
        )

    # --- Overall date span -----------------------------------------------------
    date_min = df[COL_DATE].min()
    date_max = df[COL_DATE].max()
    total_days = (date_max - date_min).days

    print(f"Dataset date range : {date_min.date()} → {date_max.date()}")
    print(f"Total span (days)  : {total_days}\n")

    # --- Granularity 1: (market_name, commodity) ------------------------------------
    mc_stats = coverage_stats(df, [COL_MARKET, COL_COMMODITY], COL_DATE, total_days)

    # --- Granularity 2: (state, commodity) -------------------------------------
    sc_stats = coverage_stats(df, [COL_STATE, COL_COMMODITY], COL_DATE, total_days)

    # --- Report ----------------------------------------------------------------
    print(
        f"Market+commodity granularity : median {mc_stats['median_dates']:.0f} distinct dates "
        f"out of {total_days} total days ({mc_stats['pct']:.1f}%)"
        f"  [{mc_stats['group_counts']} groups]"
    )
    print(
        f"State+commodity  granularity : median {sc_stats['median_dates']:.0f} distinct dates "
        f"out of {total_days} total days ({sc_stats['pct']:.1f}%)"
        f"  [{sc_stats['group_counts']} groups]"
    )

    # --- Ballpark check --------------------------------------------------------
    print()
    MC_BALLPARK, SC_BALLPARK = 17.0, 40.0
    for label, stats, ballpark in [
        ("Market+commodity", mc_stats, MC_BALLPARK),
        ("State+commodity",  sc_stats, SC_BALLPARK),
    ]:
        diff = abs(stats["pct"] - ballpark)
        flag = "⚠  FLAGGED — meaningfully different from ballpark" if diff > 10 else "✓  within ballpark"
        print(f"{label}: {stats['pct']:.1f}% vs ballpark ~{ballpark}%  →  {flag}")

    # --- Viability verdict -----------------------------------------------------
    print()
    print("--- Lag/Rolling Feature Viability ---")

    mc_pct = mc_stats["pct"]
    sc_pct = sc_stats["pct"]

    # Decision thresholds (conservative but practical)
    VIABLE_THRESHOLD = 30.0   # below this → lag features are unreliable

    for label, pct in [("Market+commodity", mc_pct), ("State+commodity", sc_pct)]:
        if pct >= VIABLE_THRESHOLD:
            verdict = "VIABLE"
            note = (
                f"~{pct:.0f}% coverage means most groups have enough observations "
                "for lag/rolling features to be meaningful."
            )
        else:
            verdict = "RISKY / NOT RECOMMENDED"
            note = (
                f"~{pct:.0f}% coverage means the median group only has a price entry "
                f"roughly every {100/pct:.0f} days. "
                "A 'lag_7' feature would not represent 7 actual calendar days ago — "
                "it would be 7 observations ago, which could span weeks or months "
                "depending on the group's density. Dense groups would learn short-term "
                "momentum while sparse groups would accidentally learn long-term trends "
                "from the same feature, making the model inconsistent and hard to validate."
            )
        print(f"\n{label}: {verdict}")
        print(f"  {note}")

    print()
    if sc_pct >= VIABLE_THRESHOLD and mc_pct < VIABLE_THRESHOLD:
        print(
            "Recommendation: build lag/rolling features at (state, commodity) granularity. "
            "Market-level data is too sparse for reliable time-series features without "
            "heavy imputation or resampling first."
        )
    elif mc_pct >= VIABLE_THRESHOLD:
        print(
            "Recommendation: market-level granularity is dense enough — "
            "prefer it for lag/rolling features to capture local price dynamics."
        )
    else:
        print(
            "Recommendation: both granularities are sparse. "
            "Consider weekly resampling or using state-level features as a proxy."
        )


if __name__ == "__main__":
    main()
