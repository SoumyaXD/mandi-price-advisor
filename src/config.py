"""Centralized configuration for the mandi-price-advisor project.

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
# Column name for the date after standardization
COL_DATE = "price_date"

# Other constants
RANDOM_SEED = 42
