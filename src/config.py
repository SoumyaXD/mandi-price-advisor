"""Centralized configuration for the agri-price-forecaster project.

All paths and constants used throughout the codebase should be imported from this module to avoid hard‑coded values.
"""
import os
from pathlib import Path

# Base directory of the project (assuming this file is located at <repo>/src/config.py)
BASE_DIR = Path(__file__).resolve().parent.parent

# Data directories
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed"

# Model storage directory
MODEL_STORE_DIR = BASE_DIR / "models_store"

# Path to the extracted Kaggle CSV file
RAW_KAGGLE_FILE = BASE_DIR / "data" / "raw" / "extracted" / "Agriculture_price_dataset.csv"
# Column names after standardization (lowercase + underscores applied in clean.py)
COL_DATE        = "price_date"
COL_STATE       = "state"
COL_DISTRICT    = "district_name"
COL_MARKET      = "market_name"
COL_COMMODITY   = "commodity"
COL_VARIETY     = "variety"
COL_GRADE       = "grade"
COL_MODAL_PRICE = "modal_price"

# Other constants
RANDOM_SEED = 42

# Commodity scope
# All 4 commodities present in the cleaned/feature data — kept for completeness
# (Tomato/Wheat may be useful for CNN or RAG portfolio pieces, or a v2 forecast).
COMMODITIES_V1 = ["Onion", "Potato", "Tomato", "Wheat"]

# Modeling scope is narrower: Tomato has a hard reporting cutoff at 2023-11-06
# and Wheat at 2024-02-06 in the source data (both confirmed present in raw CSV,
# not introduced by our pipeline). Only Onion and Potato have continuous coverage
# through June 2025 and are suitable for a time-based train/test split.
COMMODITIES_MODELING_V1 = ["Onion", "Potato"]
