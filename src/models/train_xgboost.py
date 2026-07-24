"""Train the real XGBoost price-forecasting model on the verified setup.

Scope / split / feature decisions are NOT re-derived here — they are imported
from pre_training_verification.load_filter_split() and train_baseline's metric
helpers, so the training script is guaranteed consistent with the verified
configuration:

  * Scope     : COMMODITIES_MODELING_V1 = ['Onion', 'Potato']
  * Split     : time-based 80/20, cutoff = 2025-01-14 (train <= cutoff, test >)
  * Features  : 14 columns (lag/rolling/calendar + commodity, state as category)
  * Excluded  : price_vs_msp (100% NaN for Onion+Potato — see pre_training F4)
  * NaNs      : no imputation; XGBoost native NaN handling
  * Encoding  : pandas 'category' dtype + enable_categorical=True, with train
                and test sharing the EXACT same category set (categories fit on
                the full filtered dataset before splitting)

Run:
    python -m src.models.train_xgboost
"""

import sys
import json
import logging
import datetime as _dt
from pathlib import Path

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import numpy as np
import pandas as pd
import xgboost as xgb

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_STORE_DIR,
    COL_DATE,
    COL_STATE,
    COL_COMMODITY,
    COL_MODAL_PRICE,
    RANDOM_SEED,
    COMMODITIES_MODELING_V1,
)
# Reuse the VERIFIED load/filter/split logic — single source of truth.
from src.models.pre_training_verification import load_filter_split, MODEL_FEATURES
# Reuse the EXACT metric functions used for the naive baseline, so the
# baseline-vs-XGBoost comparison is apples-to-apples.
from src.models.train_baseline import mae, rmse, mape, metrics_row, FOCUS_COMMODITIES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BASELINE_PREDICTIONS_CSV = PROCESSED_DATA_DIR / "baseline_predictions.csv"
TEST_PREDICTIONS_CSV = PROCESSED_DATA_DIR / "direct_xgboost_test_predictions.csv"
MODEL_PATH = MODEL_STORE_DIR / "xgboost_v1.json"
RUN_LOG_PATH = MODEL_STORE_DIR / "xgboost_v1_run_log.json"

# Categorical columns (must be a subset of MODEL_FEATURES).
CAT_COLS = [COL_COMMODITY, "state"]

# First-pass XGBoost hyperparameters — baseline-beating attempt, not final tuning.
XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    objective="reg:squarederror",
    enable_categorical=True,
    tree_method="hist",     # required combination with enable_categorical=True
    random_state=RANDOM_SEED,
    n_jobs=-1,
)

SEP_THICK = "=" * 78
SEP_THIN = "-" * 78


def section(title: str) -> None:
    print(f"\n{SEP_THICK}\n{title}\n{SEP_THICK}")


# ---------------------------------------------------------------------------
# Category alignment
# ---------------------------------------------------------------------------
def align_categories(train: pd.DataFrame, test: pd.DataFrame, cols):
    """Cast `cols` to pandas 'category' dtype with a SHARED category set on
    train and test.

    XGBoost encodes categorical columns by the position of each value in the
    pandas category list. If train and test have different category orderings
    (or different members), the same string would map to different integer
    codes across the two frames, silently corrupting predictions.

    We therefore fit categories on train and reindex test's categories to
    match exactly. Any test value not seen in train becomes NaN (a missing
    categorical), which XGBoost handles via its default-direction mechanism —
    the safe, explicit behavior. (In this dataset both splits see the same
    commodity set; states may differ if a state appears only in the test
    window.)
    """
    train = train.copy()
    test = test.copy()
    for c in cols:
        train[c] = train[c].astype("category")
        # Force test to use train's exact category list. .CategoricalIndex
        # built from train's categories guarantees identical codes.
        train_cats = train[c].cat.categories
        test[c] = pd.Categorical(test[c], categories=train_cats)
        # Report any unseen-in-train values that became NaN.
        n_new_na = test[c].isna().sum()
        if n_new_na:
            logger.warning(
                "Column %r: %d test rows had values unseen in train -> set to NaN "
                "(XGBoost will route them via default split direction).",
                c, n_new_na,
            )
        logger.info(
            "Aligned categories for %r: %d categories (train==test set).",
            c, len(train_cats),
        )
    return train, test


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics_table(test: pd.DataFrame, predicted: pd.Series, label: str):
    """Return a list of metric-row dicts: OVERALL then per FOCUS_COMMODITY.

    Uses the same metrics_row() / mae / rmse / mape as train_baseline.py.
    """
    rows = [metrics_row(label + "_OVERALL", test[COL_MODAL_PRICE], predicted)]
    for comm in FOCUS_COMMODITIES:
        mask = test[COL_COMMODITY].astype(str).str.lower() == comm.lower()
        sub = test[mask]
        if len(sub) == 0:
            logger.warning("Commodity %s missing in test set — skipping.", comm)
            continue
        rows.append(metrics_row(f"{label}_{comm}", sub[COL_MODAL_PRICE], predicted[mask]))
    return rows


# ---------------------------------------------------------------------------
# Baseline comparison
# ---------------------------------------------------------------------------
def build_comparison_table(xgb_rows, baseline_rows):
    """Match rows by commodity and compute % improvement.

    Both inputs are lists of dicts with keys: group, n, MAE, RMSE, MAPE (%).
    'group' values look like 'XGB_OVERALL', 'BASELINE_Onion', etc. We strip
    the prefix to match on commodity (or 'OVERALL').
    """
    def key(g):
        return g.split("_", 1)[1]

    base = {key(r["group"]): r for r in baseline_rows}
    rows = []
    for xr in xgb_rows:
        k = key(xr["group"])
        br = base.get(k)
        if br is None:
            logger.warning("No baseline row to compare against for %s", k)
            continue
        # % improvement = (baseline_metric - xgb_metric) / baseline_metric * 100
        # Positive = XGBoost better. MAE/RMSE/MAPE are all lower-is-better.
        def imp(metric):
            if br[metric] == 0:
                return float("nan")
            return (br[metric] - xr[metric]) / br[metric] * 100
        rows.append({
            "group":   k,
            "n":       xr["n"],
            "MAE_base":     br["MAE"],    "MAE_xgb":     xr["MAE"],    "MAE_imp_%":   round(imp("MAE"), 1),
            "RMSE_base":    br["RMSE"],   "RMSE_xgb":    xr["RMSE"],   "RMSE_imp_%":  round(imp("RMSE"), 1),
            "MAPE_base_%":  br["MAPE (%)"], "MAPE_xgb_%":  xr["MAPE (%)"], "MAPE_imp_%": round(imp("MAPE (%)"), 1),
        })
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------------------
# Logging (MLflow with JSON fallback)
# ---------------------------------------------------------------------------
def log_run(params, metrics_dict, feature_importance, extra=None):
    """Log the run to MLflow if it's available and a tracking URI is set;
    otherwise (or on failure) write a JSON run log to RUN_LOG_PATH.

    Never raises — logging is best-effort and must not block training.
    """
    run_record = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": "xgboost_v1",
        "params": {k: (v if isinstance(v, (int, float, str, bool, type(None))) else str(v))
                   for k, v in params.items()},
        "metrics": metrics_dict,
        "feature_importance_gain": feature_importance,
        **(extra or {}),
    }

    # --- Try MLflow ---
    mlflow_logged = False
    try:
        import mlflow
        tracking_uri = mlflow.get_tracking_uri()
        # Only treat as "wired up" if something is actually configured.
        if tracking_uri:
            mlflow.set_experiment("agri-price-forecaster")
            with mlflow.start_run(run_name="xgboost_v1") as run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                # MLflow wants flat metric values; flatten per-commodity dict.
                for k, v in metrics_dict.items():
                    if isinstance(v, (int, float)):
                        mlflow.log_metric(k, v)
                mlflow.log_dict(feature_importance, "feature_importance_gain.json")
                mlflow.log_dict(run_record, "run_record.json")
            mlflow_logged = True
            logger.info("Logged run to MLflow (tracking_uri=%s, run_id=%s).",
                        tracking_uri, run.info.run_id)
    except Exception as e:
        logger.warning("MLflow logging skipped (%s). Falling back to JSON run log.", e)

    # --- JSON fallback (always written, even alongside MLflow, for portability) ---
    try:
        MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
        with open(RUN_LOG_PATH, "w") as f:
            json.dump(run_record, f, indent=2)
        logger.info("Wrote JSON run log to %s (mlflow_logged=%s).",
                    RUN_LOG_PATH, mlflow_logged)
    except Exception as e:
        logger.error("Failed to write JSON run log: %s", e)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)

    # 1. Load + filter + split (REUSED from verification pass)
    section("1. LOAD / FILTER / SPLIT (reused from pre_training_verification)")
    train, test, cutoff, _ = load_filter_split()
    print(f"MODEL_FEATURES ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
    print(f"CAT_COLS          : {CAT_COLS}")
    print(f"Cutoff            : {cutoff.date()}")

    # 2. Align categorical dtypes across train and test (shared category set)
    section("2. CATEGORICAL ALIGNMENT (train and test share category set)")
    # Sanity: every CAT_COL is in MODEL_FEATURES.
    bad = [c for c in CAT_COLS if c not in MODEL_FEATURES]
    if bad:
        raise ValueError(f"CAT_COLS not in MODEL_FEATURES: {bad}")
    train, test = align_categories(train, test, CAT_COLS)

    # 3. Build X/y matrices — NO imputation. XGBoost handles NaNs natively.
    X_train = train[MODEL_FEATURES]
    y_train = train[COL_MODAL_PRICE]
    X_test = test[MODEL_FEATURES]
    y_test = test[COL_MODAL_PRICE]
    logger.info(
        "X_train %s, y_train %s | X_test %s, y_test %s",
        X_train.shape, y_train.shape, X_test.shape, y_test.shape,
    )

    # 4. Train
    section("3. TRAIN XGBOOST")
    print(f"XGB_PARAMS: {XGB_PARAMS}")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 5. Predict + metrics (using the SAME metric fns as the baseline)
    section("4. PREDICTIONS + METRICS")
    test = test.copy()
    test["predicted"] = model.predict(X_test)
    xgb_rows = compute_metrics_table(test, test["predicted"], label="XGB")
    xgb_metrics_df = pd.DataFrame(xgb_rows).set_index("group")
    print("XGBoost metrics (test set):")
    print(xgb_metrics_df.to_string())

    # Save test predictions to CSV for visualization
    predictions_df = test[[COL_DATE, COL_STATE, COL_COMMODITY, COL_MODAL_PRICE, "predicted"]].copy()
    predictions_df = predictions_df.rename(columns={COL_MODAL_PRICE: "actual_modal_price", "predicted": "predicted_modal_price"})
    predictions_df.to_csv(TEST_PREDICTIONS_CSV, index=False)
    logger.info("Saved test predictions to %s", TEST_PREDICTIONS_CSV)

    # 6. Baseline comparison
    section("5. BASELINE vs XGBOOST (side-by-side)")
    # IMPORTANT — apples-to-apples comparison.
    # train_baseline.py drops NaN-lag_1 rows BEFORE computing its time cutoff,
    # which shifts its cutoff by ~1 day vs ours (we split on all rows, since
    # XGBoost handles NaN natively). That means baseline_predictions.csv is
    # evaluated on a slightly different test set (5,575 rows) than our XGBoost
    # test set (5,599 rows). Comparing metrics computed on different row sets
    # would be misleading.
    # -> We therefore RE-COMPUTE the baseline (predicted = lag_1) on OUR exact
    #    test set, so both models are scored on identical rows. We still load
    #    baseline_predictions.csv below for reference / cross-check.
    logger.info("Computing naive baseline (predicted = lag_1) on the SAME test set "
                "as XGBoost for an apples-to-apples comparison.")
    test["baseline_predicted"] = test["lag_1"]
    baseline_rows = compute_metrics_table(test, test["baseline_predicted"], label="BASELINE")
    # Cross-check against the saved baseline CSV ( informational only ).
    if BASELINE_PREDICTIONS_CSV.exists():
        logger.info("Reference: saved baseline predictions at %s (generated by "
                    "train_baseline.py on its own split).", BASELINE_PREDICTIONS_CSV)
        saved_bp = pd.read_csv(BASELINE_PREDICTIONS_CSV, parse_dates=[COL_DATE])
        print(f"  [ref] saved baseline_predictions.csv: {len(saved_bp)} rows "
              f"(its own split; NOT used for the comparison table).")
    comparison = build_comparison_table(xgb_rows, baseline_rows)
    print(comparison.to_string())

    # Plain-language verdict on the OVERALL row.
    if "OVERALL" in comparison.index:
        mape_imp = comparison.loc["OVERALL", "MAPE_imp_%"]
        if pd.notna(mape_imp) and mape_imp > 0:
            verdict = (f"XGBoost BEATS the naive baseline on overall MAPE by "
                       f"{mape_imp:.1f}% relative improvement.")
        else:
            verdict = (f"XGBoost does NOT beat the naive baseline on overall MAPE "
                       f"(relative improvement {mape_imp:.1f}%).")
        print(f"\nVERDICT: {verdict}")

    # 7. Feature importance (gain-based)
    section("6. FEATURE IMPORTANCE (gain, descending)")
    importance = (pd.Series(model.get_booster().get_score(importance_type="gain"),
                            name="gain")
                  .sort_values(ascending=False))
    # get_score only returns features that were actually used in splits; show
    # the full MODEL_FEATURES list with 0 for unused ones so nothing is hidden.
    importance = importance.reindex(MODEL_FEATURES).fillna(0.0).sort_values(ascending=False)
    imp_pct = (importance / importance.sum() * 100).round(2)
    fi_df = pd.DataFrame({"gain": importance.round(2), "gain_%": imp_pct})
    print(fi_df.to_string())
    unused = fi_df[fi_df["gain"] == 0.0].index.tolist()
    if unused:
        print(f"\nFeatures with ZERO gain (not used in any split): {unused}")

    # 8. Save model + log run
    section("7. SAVE MODEL + LOG RUN")
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    # Flatten metrics for the run log: overall + per-commodity, prefixed.
    flat_metrics = {}
    for r in xgb_rows:
        prefix = r["group"]
        for metric in ("MAE", "RMSE", "MAPE (%)"):
            flat_metrics[f"{prefix}_{metric.replace(' (%)','_pct').replace(' ','_')}"] = r[metric]
    flat_metrics["n_test_overall"] = int(len(test))

    log_run(
        params=XGB_PARAMS,
        metrics_dict=flat_metrics,
        feature_importance=imp_pct.to_dict(),
        extra={
            "model_path": str(MODEL_PATH),
            "cutoff": str(cutoff.date()),
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "model_features": MODEL_FEATURES,
            "excluded_features": ["price_vs_msp"],
        },
    )

    print(f"\n{SEP_THIN}\nTraining complete.\n  model    : {MODEL_PATH}\n  run log  : {RUN_LOG_PATH}\n{SEP_THIN}")


if __name__ == "__main__":
    main()
