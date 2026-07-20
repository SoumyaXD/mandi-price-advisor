"""Data cleaning pipeline for the agri-price-forecaster project.

This script reads the raw CSV extracted from the zip archive, applies a series of cleaning
steps, and writes the cleaned data to `data/processed/mandi_prices_cleaned.csv`.
It also prints a short summary (row counts and basic statistics) for verification.
"""

import sys
from pathlib import Path
import logging
# Ensure the project root (containing the 'src' package) is in sys.path for module imports
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd

# Import the path constant from the central config
from src.config import RAW_KAGGLE_FILE, PROCESSED_DATA_DIR, RANDOM_SEED, COL_DATE, COL_MODAL_PRICE

# Setup a basic logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

def compute_and_print_iqr_stats(df: pd.DataFrame, column: str = "modal_price") -> None:
    """Compute and print per-commodity IQR statistics.

    Prints the upper bound (Q3 + 3*IQR), median, and number of outlier rows
    (values greater than the upper bound) for each commodity.
    """
    if "commodity" not in df.columns:
        logger.warning("'commodity' column not found; cannot compute IQR stats.")
        return
    print("\n--- Per-Commodity IQR Statistics ---")
    for commodity, group in df.groupby("commodity"):
        q1 = group[column].quantile(0.25)
        q3 = group[column].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 3 * iqr
        median = group[column].median()
        outlier_count = (group[column] > upper).sum()
        print(f"Commodity: {commodity}")
        print(f"  Upper bound (Q3 + 3*IQR): {upper:.2f}")
        print(f"  Median: {median:.2f}")
        print(f"  Outlier rows (>{upper:.2f}): {outlier_count}")


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names lower‑case and replace spaces with underscores."""
    df = df.rename(columns=lambda x: x.strip().lower().replace(" ", "_").replace(".", ""))
    return df


def parse_date(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    """Parse the date column to datetime, coercing errors to NaT."""
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    else:
        logger.warning("Date column '%s' not found.", col)
    return df


def drop_invalid_price_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Apply price‑related validation rules.
    - min_price <= max_price
    - modal_price within [min_price, max_price]
    - All price columns > 0
    """
    price_cols = ["min_price", "max_price", "modal_price"]
    for col in price_cols:
        if col not in df.columns:
            logger.error("Expected price column '%s' not found.", col)
            raise KeyError(f"Missing column {col}")

    # Ensure numeric types
    df[price_cols] = df[price_cols].apply(pd.to_numeric, errors="coerce")

    # Drop rows where any price is NaN
    df = df.dropna(subset=price_cols)

    # Rule 1: min <= max
    df = df[df["min_price"] <= df["max_price"]]

    # Rule 2: modal within bounds
    df = df[(df["modal_price"] >= df["min_price"]) & (df["modal_price"] <= df["max_price"])]

    # Rule 3: positive prices only
    df = df[(df[price_cols] > 0).all(axis=1)]
    return df


def drop_commodity(df: pd.DataFrame, commodity: str = "rice") -> pd.DataFrame:
    """Remove rows for a specific commodity (case‑insensitive)."""
    if "commodity" in df.columns:
        df = df[~df["commodity"].str.lower().eq(commodity.lower())]
    else:
        logger.warning("'commodity' column not found; cannot drop %s.", commodity)
    return df


# Canonical state names — fixes whitespace and spelling duplicates confirmed
# by inspecting the raw cleaned data. Built from the ACTUAL 30 unique values,
# not a guess. Merges + spelling normalizations:
#   ' Punjab'             (whitespace)            -> 'Punjab'
#   'Chattisgarh'         (spelling, 2576 rows)   -> 'Chhattisgarh'
#   'Jammu and Kashmir'   (spelling, 926 rows)    -> 'Jammu & Kashmir'
#   'Tamilnadu'           (spelling, 64626 rows)  -> 'Tamil Nadu'
#   'Gao'                 (typo, 4 rows)          -> 'Goa'
#   'Uttrakhand'          (typo, 38 rows)         -> 'Uttarakhand'
# Result: 30 raw values -> 26 canonical states/UTs.
STATE_MAPPING = {
    " Punjab": "Punjab",
    "Chattisgarh": "Chhattisgarh",
    "Jammu and Kashmir": "Jammu & Kashmir",
    "Tamilnadu": "Tamil Nadu",
    "Gao": "Goa",
    "Uttrakhand": "Uttarakhand",
}


def standardize_state_names(df: pd.DataFrame, col: str = "state") -> pd.DataFrame:
    """Normalize state names: strip whitespace, then merge known duplicates.

    Pure relabeling — never drops rows. Applied so that downstream grouping
    (aggregation, lag/rolling by state) doesn't treat 'Punjab'/' Punjab'
    or 'Tamilnadu'/'Tamil Nadu' as separate entities.

    Any value not in STATE_MAPPING is left unchanged (after stripping).
    """
    if col not in df.columns:
        logger.warning("'%s' column not found; skipping state standardization.", col)
        return df

    before_card = df[col].nunique(dropna=False)
    before_rows = len(df)

    # Defensive strip first (catches any future leading/trailing whitespace),
    # then apply the explicit merge mapping.
    df[col] = df[col].astype(str).str.strip().replace(STATE_MAPPING)

    after_card = df[col].nunique(dropna=False)
    after_rows = len(df)

    logger.info(
        "State standardization: %d -> %d unique values, %d -> %d rows (relabeling only)",
        before_card, after_card, before_rows, after_rows,
    )
    return df


def remove_iqr_outliers(df: pd.DataFrame, column: str = "modal_price") -> pd.DataFrame:
    """Remove outliers per commodity using the IQR method on `column`."""
    if "commodity" not in df.columns:
        logger.warning("'commodity' column not found; skipping IQR outlier removal.")
        return df

    def iqr_filter(group):
        q1 = group[column].quantile(0.25)
        q3 = group[column].quantile(0.75)
        iqr = q3 - q1
        # Use 3 * IQR for the upper bound as requested
        lower = q1 - 1.5 * iqr  # lower bound used in filter to keep values above Q1 - 1.5*IQR
        upper = q3 + 3 * iqr
        return group[(group[column] >= lower) & (group[column] <= upper)]

    df = df.groupby("commodity", group_keys=False).apply(iqr_filter)
    return df


def main():
    logger.info("Starting cleaning pipeline")
    # 1. Load raw CSV
    logger.info("Loading raw CSV from %s", RAW_KAGGLE_FILE)
    df = pd.read_csv(RAW_KAGGLE_FILE)
    original_rows = len(df)
    logger.info("Original row count: %d", original_rows)

    # 2. Standardize column names
    df = standardize_columns(df)
    logger.info("After standardizing columns: %d rows", len(df))

    # 3. Parse date column (assumed named 'date')
    df = parse_date(df, col=COL_DATE)
    logger.info("After parsing dates: %d rows", len(df))

    # 4. Drop invalid price rows
    df = drop_invalid_price_rows(df)
    logger.info("After price validation: %d rows", len(df))

    # 5. Drop Rice commodity
    df = drop_commodity(df, commodity="Rice")
    logger.info("After dropping Rice: %d rows", len(df))

    # 5b. Standardize state names (strip whitespace + merge spelling duplicates)
    df = standardize_state_names(df, col="state")

    # 6. Compute and print per-commodity IQR stats before outlier removal
    compute_and_print_iqr_stats(df, column="modal_price")
    # 7. Remove IQR outliers per commodity on modal_price
    df = remove_iqr_outliers(df, column="modal_price")
    logger.info("After IQR outlier removal: %d rows", len(df))

    # 8. Drop the min_price and max_price columns (no longer needed)
    df = df.drop(columns=["min_price", "max_price"], errors="ignore")
    logger.info("After dropping min/max price columns: %d rows", len(df))

    # 8. Save cleaned data
    output_path = PROCESSED_DATA_DIR / "mandi_prices_cleaned.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Cleaned data saved to %s", output_path)

    # 9. Verification prints (as requested)
    print("--- Verification Summary ---")
    print(f"Original rows: {original_rows}")
    print(f"Final rows   : {len(df)}")
    if "commodity" in df.columns:
        print("Commodity value counts (top 10):")
        print(df["commodity"].value_counts().head(10))
    else:
        print("No 'commodity' column to show counts.")
    if "modal_price" in df.columns:
        print("modal_price describe():")
        print(df["modal_price"].describe())
    else:
        print("No 'modal_price' column to describe.")
    if "state" in df.columns:
        print(f"State cardinality: {df['state'].nunique()} unique values")
        print("State value counts:")
        print(df["state"].value_counts().to_string())
    else:
        print("No 'state' column to summarize.")

    # 10. CSV round-trip date dtype verification
    # CSV has no type metadata — to_csv() writes dates as plain text strings, so a
    # plain read_csv() will load price_date back as dtype=object.  We need to pass
    # parse_dates=[COL_DATE] (or call pd.to_datetime after loading) to restore datetime64.
    print("\n--- CSV Round-Trip Date Verification ---")
    df_raw_reload  = pd.read_csv(output_path)                                   # no parse_dates
    df_parsed_reload = pd.read_csv(output_path, parse_dates=[COL_DATE])         # with parse_dates

    raw_dtype    = df_raw_reload[COL_DATE].dtype
    parsed_dtype = df_parsed_reload[COL_DATE].dtype

    print(f"dtype WITHOUT parse_dates : {raw_dtype}")
    if str(raw_dtype) == "object":
        print(
            "  ⚠  As expected: CSV round-trip loses type info. "
            "price_date is read back as plain strings (object). "
            "Always use read_csv(..., parse_dates=['price_date']) or "
            "pd.to_datetime() after loading."
        )
    else:
        print("  ✓  Unexpectedly already parsed — pandas may have inferred the type.")

    print(f"dtype WITH  parse_dates   : {parsed_dtype}")
    if str(parsed_dtype).startswith("datetime64"):
        print("  ✓  Correct: datetime64 dtype restored via parse_dates.")
    else:
        print(f"  ⚠  Unexpected dtype after parse_dates: {parsed_dtype}")

    date_min = df_parsed_reload[COL_DATE].min()
    date_max = df_parsed_reload[COL_DATE].max()
    print(f"Date range (parsed reload): {date_min.date()} → {date_max.date()}")
    if date_min.year >= 2023 and date_max.year <= 2025:
        print("  ✓  Date range is within expected 2023-2025 window.")
    else:
        print(f"  ⚠  Date range falls outside 2023-2025 — check the source data.")


if __name__ == "__main__":
    # Ensure reproducibility where applicable
    try:
        import numpy as np
        np.random.seed(RANDOM_SEED)
    except Exception:
        pass
    main()
