# mandi-price-advisor

A machine learning pipeline to predict agricultural commodity prices across Indian mandi (market) locations. Built to help farmers, traders, and analysts make informed decisions based on historical price trends.

## Problem Statement

Agricultural commodity prices in Indian mandis fluctuate significantly based on season, location, and market dynamics. This project aims to build a price prediction system that ingests historical mandi price data, cleans and engineers features from it, trains forecasting models, and exposes predictions via an API.

## Project Structure

```
mandi-price-advisor/
├─ data/
│  ├─ raw/
│  │  └─ extracted/              # Raw Kaggle CSV (Agriculture_price_dataset.csv)
│  └─ processed/
│     ├─ mandi_prices_cleaned.csv    # Output of Phase 1
│     └─ mandi_prices_features.csv  # Output of Phase 2
├─ src/
│  ├─ data/
│  │  └─ clean.py                # Data cleaning pipeline (Phase 1)
│  ├─ features/
│  │  ├─ check_granularity.py    # Granularity analysis (Phase 2)
│  │  └─ build_features.py       # Feature engineering (Phase 2)
│  ├─ models/                    # Model training & evaluation (upcoming)
│  ├─ api/                       # FastAPI prediction service (upcoming)
│  └─ config.py                  # Centralized paths and column name constants
├─ models_store/                 # Saved model artifacts
├─ notebooks/                    # Exploratory analysis
├─ tests/
├─ requirements.txt
└─ .gitignore
```

## Phases

### Phase 1 — Data Ingestion & Cleaning (complete)

- Source: Kaggle Agriculture Price Dataset (`data/raw/extracted/Agriculture_price_dataset.csv`)
- Cleaning steps in `src/data/clean.py`:
  - Column name standardization (lowercase + underscores)
  - Date parsing with error coercion → `price_date`
  - Price validation: min ≤ modal ≤ max, all prices > 0
  - Rice commodity removed (out of scope)
  - Per-commodity IQR outlier removal on `modal_price` (upper bound: Q3 + 3×IQR)
  - `min_price` and `max_price` dropped post-validation
  - CSV round-trip date dtype verification (confirms `parse_dates` is required on reload)
- Output: `data/processed/mandi_prices_cleaned.csv` (~722k rows, 8 columns)
- Actual columns: `state`, `district_name`, `market_name`, `commodity`, `variety`, `grade`, `modal_price`, `price_date`

### Phase 2 — Feature Engineering (complete)

#### Granularity analysis (`src/features/check_granularity.py`)

Confirmed viable aggregation level before building time-series features:

| Granularity | Median distinct dates | Coverage | Groups |
|---|---|---|---|
| market_name + commodity | 125 / 736 days | 17.0% | 3,392 |
| state + commodity | 316 / 736 days | 42.9% | 98 |

Market-level is too sparse for reliable lag features (a `lag_7` would span weeks, not days, in sparse groups). State+commodity at ~43% coverage is viable.

#### Feature build (`src/features/build_features.py`)

Aggregates to `(state, commodity, price_date)` daily level (~36k rows), then adds:

- Lag features: `lag_1`, `lag_7`, `lag_14`, `lag_30` (lags by report count within group, not calendar days)
- Rolling features: `rolling_mean_7/30`, `rolling_std_7/30` — shift(1) applied before rolling to prevent leakage
- Calendar features: `month_sin`, `month_cos` (cyclic), `day_of_week`, `day_of_year`
- MSP-relative: `price_vs_msp = modal_price / 2425` for Wheat only; NaN for all other commodities
- Output: `data/processed/mandi_prices_features.csv` (36,068 rows × 17 columns)

### Phase 3 — Model Training & Evaluation (upcoming)

### Phase 4 — API & Serving (upcoming)

## Setup

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/macOS
source venv/bin/activate

pip install -r requirements.txt
```

## Running the Pipeline

```bash
# Phase 1 — clean raw data
python -m src.data.clean

# Phase 2 — check granularity (optional diagnostic)
python -m src.features.check_granularity

# Phase 2 — build features
python -m src.features.build_features
```

## Column Name Reference

All column names are defined in `src/config.py` — never hardcoded in scripts.

| Constant | Value |
|---|---|
| `COL_DATE` | `price_date` |
| `COL_STATE` | `state` |
| `COL_DISTRICT` | `district_name` |
| `COL_MARKET` | `market_name` |
| `COL_COMMODITY` | `commodity` |
| `COL_MODAL_PRICE` | `modal_price` |

## Stack

- Data: pandas, numpy
- Models: scikit-learn, XGBoost, LightGBM
- Experiment tracking: MLflow
- API: FastAPI + Uvicorn
- Testing: pytest
