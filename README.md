# agri-price-forecaster

A machine learning pipeline to predict agricultural commodity prices across Indian mandi (market) locations. Built to help farmers, traders, and analysts make informed decisions based on historical price trends.

## Problem Statement

Agricultural commodity prices in Indian mandis fluctuate significantly based on season, location, and market dynamics. This project aims to build a price prediction system that ingests historical mandi price data, cleans and engineers features from it, trains forecasting models, and exposes predictions via an API.

## Project Structure

```
agri-price-forecaster/
├─ data/
│  ├─ raw/
│  │  └─ extracted/              # Raw Kaggle CSV (Agriculture_price_dataset.csv)
│  └─ processed/
│     ├─ mandi_prices_cleaned.csv    # Output of cleaning pipeline
│     ├─ mandi_prices_features.csv   # Output of feature engineering
│     └─ baseline_predictions.csv    # Naive lag-1 baseline predictions
├─ src/
│  ├─ data/
│  │  └─ clean.py                # Data cleaning + state-name deduplication
│  ├─ features/
│  │  ├─ check_granularity.py    # Granularity analysis
│  │  └─ build_features.py       # Feature engineering
│  ├─ models/
│  │  ├─ train_baseline.py       # Naive lag-1 baseline
│  │  ├─ investigate_gaps.py     # Commodity coverage gap analysis
│  │  ├─ pre_training_verification.py  # Pre-training integrity checks
│  │  ├─ seasonal_check.py       # Train/test seasonality analysis
│  │  └─ train_xgboost.py        # XGBoost training + baseline comparison
│  ├─ api/                       # FastAPI prediction service (upcoming)
│  └─ config.py                  # Centralized paths and column name constants
├─ models_store/                 # Saved model artifacts + run logs
├─ notebooks/                    # Exploratory analysis
├─ tests/
├─ requirements.txt
└─ .gitignore
```

## Data & Modeling Scope

**Original dataset** (`data/raw/extracted/Agriculture_price_dataset.csv`): **737,392 rows**, 5 commodities, 2023-06-06 → 2025-06-11:

| Commodity | Rows | Share |
|---|---:|---:|
| Potato | 327,332 | 44.4% |
| Onion | 298,658 | 40.5% |
| Wheat | 76,976 | 10.4% |
| Tomato | 26,644 | 3.6% |
| Rice | 7,782 | 1.1% |

**Rice dropped early** during cleaning (~1% of data, insufficient volume for state+commodity coverage).

**Cleaning pipeline** (`src/data/clean.py`):
- Column name standardization (lowercase + underscores)
- Date parsing with error coercion → `price_date`
- Price validation: `min_price ≤ modal_price ≤ max_price`, all prices > 0
- Per-commodity IQR outlier removal on `modal_price` (upper bound Q3 + 3×IQR)
- **State name deduplication** (added after pre-training verification surfaced duplicate state labels): 30 raw state variants → 26 canonical states/UTs. Merges: `' Punjab'→'Punjab'`, `'Chattisgarh'→'Chhattisgarh'`, `'Jammu and Kashmir'→'Jammu & Kashmir'`, `'Tamilnadu'→'Tamil Nadu'`, `'Gao'→'Goa'`, `'Uttrakhand'→'Uttarakhand'`.
- Output: `data/processed/mandi_prices_cleaned.csv` (**722,909 rows × 8 columns**)

**Feature granularity: state + commodity, not market.** Empirical coverage analysis (`src/features/check_granularity.py`) showed market-level grouping is too sparse for time-series features — a `lag_7` at market level would span weeks of calendar time, not days:

| Granularity | Median distinct dates | Coverage of 736-day span | Groups |
|---|---|---|---|
| market_name + commodity | 125 | **17.0%** | 3,392 |
| state + commodity | 316 | **42.9%** | 98 |

State+commodity was chosen as the modeling granularity.

**Tomato and Wheat excluded from the v1 MODELING scope (not the cleaning/feature scope).** A hard national reporting cutoff was discovered in the source data: Tomato stops at **2023-11-06**, Wheat at **2024-02-06**. This was confirmed as a genuine source-data seasonal-tracking gap, not a pipeline bug, via (a) raw-vs-cleaned comparison showing identical cutoffs and (b) state-count analysis showing a simultaneous national drop rather than a gradual coverage decline. Both commodities fall entirely inside the training window and have zero test-period rows, making them unusable for a time-based train/test split. They remain cleaned and feature-engineered for future use.

**v1 modeling scope: Onion + Potato only** (`COMMODITIES_MODELING_V1` in `src/config.py`). After scope filter + state+commodity aggregation: **32,870 rows × 17 columns** in `data/processed/mandi_prices_features.csv`.

## Feature Engineering

Final model input: **14 features** (`MODEL_FEATURES` in `src/models/pre_training_verification.py`, shared by import with `train_xgboost.py`).

| Group | Features | Notes |
|---|---|---|
| Lag | `lag_1`, `lag_7`, `lag_14`, `lag_30` | Lags by **report count** within each (state, commodity) group, not calendar days (reporting is irregular). |
| Rolling | `rolling_mean_7`, `rolling_std_7`, `rolling_mean_30`, `rolling_std_30` | Window over prior reporting rows. |
| Calendar | `month_sin`, `month_cos`, `day_of_week`, `day_of_year` | Cyclic month encoding (sin/cos) avoids a false Jan→Dec jump. |
| Categorical | `commodity`, `state` | Cast to pandas `category` dtype; passed to XGBoost with `enable_categorical=True`. Train and test share the exact same category set. |

**Leakage prevention:** every rolling statistic is computed on a `.shift(1)`'d series within group, so a row's own `modal_price` is never included in its own rolling mean/std. Lags are by definition prior rows. The target `modal_price` is verified absent from the feature list (pre-training check F3).

**NaN handling — explicit decision:** no imputation. Lag/rolling NaNs occur at the start of each (state, commodity) group (too few prior reporting rows) and are structurally informative. XGBoost learns a default split direction for missing values natively, preserving the "not-enough-history" signal rather than fabricating values.

**`price_vs_msp` deliberately excluded.** It is computed only for Wheat (the only MSP-covered crop) in `build_features.py`, so it is 100% NaN in the Onion+Potato scope. Including it would add a uniformly-missing column with zero signal.

## Modeling Results

Time-based 80/20 split, cutoff **2025-01-14** (train ≤ cutoff, test > cutoff). Train: **22,277 rows** (2023-06-06 → 2025-01-14). Test: **5,599 rows** (2025-01-15 → 2025-06-11). No overlap, no leakage.

**XGBoost (v1)**: `n_estimators=300, max_depth=6, learning_rate=0.05`, `objective=reg:squarederror`, `enable_categorical=True`, `tree_method=hist`, `random_state=42`. Native NaN handling, no imputation.

**Baseline (naive lag-1)**: `predicted = lag_1`, evaluated on the same 5,599-row test set as XGBoost for an apples-to-apples comparison.

| Group | n | MAE base | MAE xgb | MAE Δ% | RMSE base | RMSE xgb | RMSE Δ% | MAPE base | MAPE xgb | MAPE Δ% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| OVERALL | 5,599 | 109.16 | 122.20 | **−11.9** | 244.76 | 225.92 | **+7.7** | 6.10 | 6.93 | **−13.6** |
| Onion | 2,499 | 110.48 | 137.45 | −24.4 | 251.04 | 243.31 | +3.1 | 5.69 | 7.00 | −23.0 |
| Potato | 3,100 | 108.09 | 109.90 | −1.7 | 239.57 | 210.87 | +12.0 | 6.44 | 6.87 | −6.7 |

Δ% positive = XGBoost better (all metrics are lower-is-better).

**Honest headline: XGBoost does NOT beat the naive lag-1 baseline on MAE or MAPE, but does beat it on RMSE.** Overall MAPE goes 6.10% → 6.93% (13.6% worse), while overall RMSE improves 244.76 → 225.92 (7.7% better). The pattern holds per-commodity: XGBoost shrinks the largest errors (RMSE) at the cost of adding noise to typical small moves (MAE), most pronounced on Onion (MAE −24.4% / RMSE +3.1%).

**Feature importance (gain-based, XGBoost v1):**

| Rank | Feature | gain % |
|---|---|---:|
| 1 | `lag_1` | **58.72** |
| 2 | `rolling_mean_7` | **37.18** |
| 3 | `state` | 0.49 |
| 4 | `commodity` | 0.45 |
| 5 | `month_sin` | 0.42 |
| 6 | `day_of_year` | 0.35 |
| 7 | `month_cos` | 0.34 |
| 8 | `lag_14` | 0.34 |
| 9–14 | `rolling_std_7`, `rolling_mean_30`, `day_of_week`, `rolling_std_30`, `lag_7`, `lag_30` | 0.25–0.32 |

`lag_1` + `rolling_mean_7` together account for **~96% of total gain**. Every calendar feature, both categorical features, and all longer lags contribute <1% combined.

**Interpretation.** At daily state-level granularity, mandi modal prices behave close to a random walk: the best predictor of tomorrow's price is today's price (`lag_1`), and a 7-period rolling mean (`rolling_mean_7`) is the only feature that adds meaningful signal on top. Calendar effects and state/commodity identity carry negligible predictive weight once recent price is known. Beating a naive lag baseline here is genuinely difficult without structural features — arrival volume, weather, and policy shocks — that are not present in this dataset. The RMSE improvement shows the model does extract *some* signal (it meaningfully reduces the largest errors), but not enough to overcome the noise it adds on typical small day-to-day moves.

**This is a real, evidenced finding, not a failed experiment.** The pipeline correctly verifies its scope, prevents leakage, evaluates apples-to-apples, and reports an honest null result against a strong baseline.

**Next planned experiment:** predict the **residual** (`modal_price − lag_1`, i.e. the price *change*) instead of the raw price, then add `lag_1` back at inference. This forces the model to learn what the naive baseline gets wrong rather than relearning `lag_1` from scratch — directly targeting the MAE weakness surfaced by feature importance.

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
# Phase 1 — clean raw data (includes state-name deduplication)
python -m src.data.clean

# Phase 2 — check granularity (optional diagnostic)
python -m src.features.check_granularity

# Phase 2 — build features
python -m src.features.build_features

# Phase 3 — pre-training integrity verification (no model trained)
python -m src.models.pre_training_verification

# Phase 3 — train/test seasonality analysis
python -m src.models.seasonal_check

# Phase 3 — naive lag-1 baseline
python -m src.models.train_baseline

# Phase 3 — train XGBoost, score vs baseline, save model + log run
python -m src.models.train_xgboost
```

## Known Limitations

- **State-level, not mandi-level.** v1 forecasts state+commodity daily averages, not individual-mandi recommendations. This is a data-density limitation: market-level coverage is ~17% of the date span (too sparse for lag features), documented in the granularity analysis above.
- **v1 scope is Onion + Potato only.** Tomato and Wheat have a hard national reporting cutoff in the source data (Nov 2023 / Feb 2024) and need a supplementary or updated data source to become forecastable. They remain cleaned and feature-engineered for when such data is available.
- **No structural demand/supply features.** Arrivals (market arrival volume), weather, MSP (where in scope), and policy events are not currently included. Feature importance shows the price-history-only signal is near-saturated by `lag_1` + `rolling_mean_7`, so structural features are the most promising direction for future improvement.

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
