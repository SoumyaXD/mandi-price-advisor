"""Train an XGBoost model on the price RESIDUAL (modal_price - lag_1), then
reconstruct the price prediction at inference time as

    predicted_price = lag_1 + predicted_residual

Motivation
----------
The direct XGBoost v1 run showed `lag_1` dominates feature importance (~59% of
gain) and `rolling_mean_7` accounts for most of the rest — i.e. the direct model
mostly re-learns the price LEVEL, which the naive lag-1 baseline already
captures. On MAE and MAPE the direct model actually LOSES to the naive baseline
(it adds noise to typical small day-over-day moves). By asking the model to
predict only the PRICE CHANGE (the residual), we remove the level-learning
problem and let the model spend its capacity on what the naive baseline gets
wrong.

Scope / split are NOT re-derived here — identical to train_xgboost.py via the
shared load_filter_split() import, so the residual model is scored on the same
post-cutoff test window as the direct model and the naive baseline.

Row-set note: a residual target cannot be computed without lag_1, so rows with
NaN lag_1 are dropped (train 22,277 -> slightly fewer; test 5,599 -> 5,575).
For a fair comparison ALL THREE models (naive baseline, direct XGBoost loaded
from xgboost_v1.json, and this residual model) are re-scored on the identical
post-drop test subset.

Run:
    python -m src.models.train_xgboost_residual
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
    MODEL_STORE_DIR,
    COL_COMMODITY,
    COL_MODAL_PRICE,
    RANDOM_SEED,
)
# Reuse the VERIFIED load/filter/split + feature list — single source of truth.
from src.models.pre_training_verification import load_filter_split, MODEL_FEATURES
# Reuse the EXACT metric functions used for the naive baseline and direct
# XGBoost, so the three-way comparison is apples-to-apples-to-apples.
from src.models.train_baseline import mae, rmse, mape, metrics_row, FOCUS_COMMODITIES
# Reuse helpers from the direct run so categorical encoding + reporting stay
# identical (and the saved direct model can be re-scored on our test subset
# with the same category codes it was trained on).
from src.models.train_xgboost import (
    align_categories,
    compute_metrics_table,
    section,
    SEP_THICK,
    SEP_THIN,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Saved artifacts from the direct (price-level) run — re-scored on our row subset.
DIRECT_MODEL_PATH = MODEL_STORE_DIR / "xgboost_v1.json"
DIRECT_RUN_LOG = MODEL_STORE_DIR / "xgboost_v1_run_log.json"

# Output artifacts for THIS run.
MODEL_PATH = MODEL_STORE_DIR / "xgboost_residual_v1.json"
RUN_LOG_PATH = MODEL_STORE_DIR / "xgboost_residual_v1_run_log.json"

# Categorical columns (subset of MODEL_FEATURES).
CAT_COLS = [COL_COMMODITY, "state"]

# lag_1 is EXCLUDED from the residual feature set.
# Reasoning: the residual target is `modal_price - lag_1`, so lag_1 is already
# baked into the target's construction. Including it as a feature would NOT be
# leakage (the target uses the SAME row's lag_1, which is by definition a PRIOR
# row's price), but it IS redundant — the model would largely learn to invert
# lag_1 to recover a near-zero residual, eating capacity for no signal gain.
# Excluding it forces the remaining features to explain the price CHANGE
# directly, which is the actual question of interest.
# To test the include-variant, just add "lag_1" back to this list.
RESIDUAL_FEATURES = [f for f in MODEL_FEATURES if f != "lag_1"]
assert "lag_1" in MODEL_FEATURES and len(RESIDUAL_FEATURES) == len(MODEL_FEATURES) - 1, \
    "expected to drop exactly lag_1 from MODEL_FEATURES"

# Same hyperparameters as the direct run — only the target changes.
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


# ---------------------------------------------------------------------------
# Three-way comparison table
# ---------------------------------------------------------------------------
def build_three_way_table(baseline_rows, direct_rows, resid_rows):
    """Match rows by commodity across naive baseline / direct XGB / residual XGB.

    Each input is a list of metrics_row dicts with a 'group' key like
    'BASELINE_OVERALL', 'XGB_Onion', 'XGBRESID_Potato'. We strip the prefix to
    match on commodity (or 'OVERALL').

    The focal delta is residual-vs-baseline (the question this experiment is
    designed to answer). Direct-vs-baseline is already documented in the README
    and is included for reference, not re-computed as a delta here.
    """
    def key(g):
        return g.split("_", 1)[1]

    base = {key(r["group"]): r for r in baseline_rows}
    direct = {key(r["group"]): r for r in direct_rows}
    rows = []
    for rr in resid_rows:
        k = key(rr["group"])
        br = base.get(k)
        dr = direct.get(k)
        if br is None or dr is None:
            logger.warning("Missing baseline/direct row for %s "
                           "(base=%s, direct=%s)", k, br is not None, dr is not None)
            continue

        def imp(metric_key, baseline_row, model_row):
            denom = baseline_row[metric_key]
            if denom == 0 or pd.isna(denom):
                return float("nan")
            return (denom - model_row[metric_key]) / denom * 100

        rows.append({
            "group":      k,
            "n":          rr["n"],
            "MAE_base":     br["MAE"],     "MAE_direct":    dr["MAE"],     "MAE_resid":    rr["MAE"],
            "MAE_resid_vs_base_%":  round(imp("MAE", br, rr), 1),
            "RMSE_base":    br["RMSE"],    "RMSE_direct":   dr["RMSE"],    "RMSE_resid":   rr["RMSE"],
            "RMSE_resid_vs_base_%": round(imp("RMSE", br, rr), 1),
            "MAPE_base_%":   br["MAPE (%)"], "MAPE_direct_%": dr["MAPE (%)"], "MAPE_resid_%":  rr["MAPE (%)"],
            "MAPE_resid_vs_base_%": round(imp("MAPE (%)", br, rr), 1),
        })
    return pd.DataFrame(rows).set_index("group")


# ---------------------------------------------------------------------------
# Logging (MLflow with JSON fallback) — mirrors train_xgboost.log_run
# ---------------------------------------------------------------------------
def log_run(params, metrics_dict, feature_importance, extra=None):
    """Log the run to MLflow if a tracking URI is configured; otherwise (or on
    failure) write a JSON run log to RUN_LOG_PATH. Best-effort, never raises.
    """
    run_record = {
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "model": "xgboost_residual_v1",
        "target": "residual = modal_price - lag_1",
        "inference": "predicted_price = lag_1 + predicted_residual",
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
        if tracking_uri:
            mlflow.set_experiment("agri-price-forecaster")
            with mlflow.start_run(run_name="xgboost_residual_v1") as run:
                mlflow.log_params({k: str(v) for k, v in params.items()})
                mlflow.log_param("target", "residual = modal_price - lag_1")
                mlflow.log_param("inference", "predicted_price = lag_1 + predicted_residual")
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


def _load_logged_direct_metrics():
    """Read the direct run's saved metrics (xgboost_v1_run_log.json) so we can
    cross-check our re-scored subset metrics against the logged 5,599-row run.
    """
    if not DIRECT_RUN_LOG.exists():
        return None
    try:
        with open(DIRECT_RUN_LOG) as f:
            return json.load(f).get("metrics", {})
    except Exception as e:
        logger.warning("Could not read %s: %s", DIRECT_RUN_LOG, e)
        return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    np.random.seed(RANDOM_SEED)

    # 1. Load + filter + split (REUSED from verification pass — identical to
    #    train_xgboost.py: Onion+Potato, time-based 80/20, cutoff 2025-01-14).
    section("1. LOAD / FILTER / SPLIT (reused from pre_training_verification)")
    train, test, cutoff, _ = load_filter_split()
    print(f"MODEL_FEATURES    ({len(MODEL_FEATURES)}): {MODEL_FEATURES}")
    print(f"RESIDUAL_FEATURES ({len(RESIDUAL_FEATURES)}): {RESIDUAL_FEATURES}")
    print(f"CAT_COLS            : {CAT_COLS}")
    print(f"Cutoff              : {cutoff.date()}")
    print(f"Rows before NaN-lag_1 drop: train={len(train)}, test={len(test)}")

    # 2. Align categories on the FULL (pre-drop) train — same category set the
    #    saved direct model (xgboost_v1.json) was trained on, so we can re-score
    #    that model on our test subset without an encoding mismatch. Dropping
    #    rows afterward preserves the category dtype (a row-level operation).
    section("2. CATEGORICAL ALIGNMENT (shared category set, as in direct run)")
    bad = [c for c in CAT_COLS if c not in MODEL_FEATURES]
    if bad:
        raise ValueError(f"CAT_COLS not in MODEL_FEATURES: {bad}")
    train, test = align_categories(train, test, CAT_COLS)

    # 3. Construct the RESIDUAL target, then drop rows where lag_1 is NaN.
    #    residual = modal_price - lag_1 (the day-over-day price change).
    #    A NaN lag_1 yields a NaN residual — unusable as a train/eval target.
    #    This drop is the ONLY deviation from the direct run's row set.
    section("3. RESIDUAL TARGET + NaN-lag_1 DROP")
    train = train.copy()
    test = test.copy()
    train["residual"] = train[COL_MODAL_PRICE] - train["lag_1"]
    test["residual"] = test[COL_MODAL_PRICE] - test["lag_1"]

    train_before, test_before = len(train), len(test)
    train = train.dropna(subset=["lag_1"]).reset_index(drop=True)
    test = test.dropna(subset=["lag_1"]).reset_index(drop=True)
    print(f"Train: {train_before} -> {len(train)}  (dropped {train_before - len(train)} NaN-lag_1 rows)")
    print(f"Test : {test_before} -> {len(test)}  (dropped {test_before - len(test)} NaN-lag_1 rows)")
    print(f"\nAll three models below are scored on the SAME {len(test)}-row test "
          f"subset (lag_1 non-NaN; required to construct the residual target).")

    # Residual target sanity-check.
    print(f"\nResidual target (train) describe:")
    print(train["residual"].describe().round(2).to_string())

    # 4. Build X/y matrices — target is the RESIDUAL, features exclude lag_1.
    X_train = train[RESIDUAL_FEATURES]
    y_train = train["residual"]
    X_test = test[RESIDUAL_FEATURES]
    # NOTE: evaluation target is the TRUE modal_price — we compare the
    # RECONSTRUCTED prediction (lag_1 + pred_residual) against it, not the raw
    # residual output.
    logger.info(
        "X_train %s, y_train(residual) %s | X_test %s",
        X_train.shape, y_train.shape, X_test.shape,
    )

    # 5. Train XGBoost on the residual.
    section("4. TRAIN XGBOOST (target = residual)")
    print(f"XGB_PARAMS: {XGB_PARAMS}")
    model = xgb.XGBRegressor(**XGB_PARAMS)
    model.fit(X_train, y_train)
    logger.info("Training complete.")

    # 6. Predict residual + RECONSTRUCT the price.
    section("5. PREDICT + RECONSTRUCT PRICE (predicted_price = lag_1 + pred_residual)")
    test = test.copy()
    test["pred_residual"] = model.predict(X_test)
    test["predicted_price"] = test["lag_1"] + test["pred_residual"]
    print("Reconstructed predicted_price (test) describe:")
    print(test["predicted_price"].describe().round(2).to_string())

    # Residual-model metrics — RECONSTRUCTED price vs true modal_price.
    resid_rows = compute_metrics_table(test, test["predicted_price"], label="XGBRESID")
    resid_metrics_df = pd.DataFrame(resid_rows).set_index("group")
    print("\nResidual-XGBoost metrics (RECONSTRUCTED price vs true modal_price):")
    print(resid_metrics_df.to_string())

    # 7. Re-score naive baseline + direct (previous) XGBoost on the SAME row
    #    subset, for a fair three-way comparison.
    section("6. THREE-WAY COMPARISON (baseline vs direct vs residual, same rows)")

    # (a) Naive baseline: predicted = lag_1
    test["baseline_predicted"] = test["lag_1"]
    baseline_rows = compute_metrics_table(test, test["baseline_predicted"], label="BASELINE")

    # (b) Direct XGBoost (previous run): load saved model, predict on the same
    #     test subset using ALL 14 MODEL_FEATURES (categories already aligned to
    #     match what this model was trained on).
    if DIRECT_MODEL_PATH.exists():
        logger.info("Loading saved direct model %s to re-score on the %d-row subset.",
                    DIRECT_MODEL_PATH, len(test))
        direct_model = xgb.XGBRegressor()
        direct_model.load_model(DIRECT_MODEL_PATH)
        test["direct_predicted"] = direct_model.predict(test[MODEL_FEATURES])
        direct_rows = compute_metrics_table(test, test["direct_predicted"], label="XGB")

        # Cross-check: re-scored subset metrics vs the direct run's logged
        # (5,599-row) metrics. They should be very close — the direct run's
        # metrics also silently skipped NaN-prediction rows for MAE/RMSE/MAPE,
        # and we only removed 24 NaN-lag_1 rows here.
        logged = _load_logged_direct_metrics()
        if logged:
            print("\nCross-check — direct model re-scored on subset vs its logged (5,599-row) run:")
            for metric_key, logged_key in [("MAE", "XGB_OVERALL_MAE"),
                                           ("RMSE", "XGB_OVERALL_RMSE"),
                                           ("MAPE (%)", "XGB_OVERALL_MAPE_pct")]:
                subset_val = next(
                    (r[metric_key] for r in direct_rows if r["group"] == "XGB_OVERALL"), None
                )
                print(f"  OVERALL {metric_key:8s}: subset={subset_val}  logged={logged.get(logged_key)}")
    else:
        logger.warning("Saved direct model not found at %s — direct column will be missing.",
                       DIRECT_MODEL_PATH)
        direct_rows = []

    comparison = build_three_way_table(baseline_rows, direct_rows, resid_rows)
    print()
    print(comparison.to_string())

    # Plain-language verdict: does the residual model beat the naive baseline?
    if "OVERALL" in comparison.index:
        mae_imp = comparison.loc["OVERALL", "MAE_resid_vs_base_%"]
        mape_imp = comparison.loc["OVERALL", "MAPE_resid_vs_base_%"]
        rmse_imp = comparison.loc["OVERALL", "RMSE_resid_vs_base_%"]
        beats_mae = pd.notna(mae_imp) and mae_imp > 0
        beats_mape = pd.notna(mape_imp) and mape_imp > 0
        print("\nVERDICT (residual XGBoost vs naive lag-1 baseline, OVERALL):")
        print(f"  MAE  : {'BEATS' if beats_mae else 'does NOT beat'} baseline "
              f"(relative change {mae_imp:+.1f}% — positive = residual better).")
        print(f"  MAPE : {'BEATS' if beats_mape else 'does NOT beat'} baseline "
              f"(relative change {mape_imp:+.1f}%).")
        print(f"  RMSE : relative change {rmse_imp:+.1f}% vs baseline.")
        succeeded = beats_mae and beats_mape
        print(f"  -> Experiment {'SUCCEEDED' if succeeded else 'did NOT succeed'} "
              f"at beating the naive baseline on BOTH MAE and MAPE.")

    # 8. Feature importance (gain). lag_1 is NOT a feature here, so this ranking
    #    shows what actually drives price CHANGES — a more interesting question
    #    than what drives price LEVEL.
    section("7. FEATURE IMPORTANCE (gain, residual model — lag_1 excluded)")
    importance = (pd.Series(model.get_booster().get_score(importance_type="gain"),
                            name="gain")
                  .sort_values(ascending=False))
    # Reindex over RESIDUAL_FEATURES so unused features show as 0 (nothing hidden).
    importance = importance.reindex(RESIDUAL_FEATURES).fillna(0.0).sort_values(ascending=False)
    imp_pct = (importance / importance.sum() * 100).round(2)
    fi_df = pd.DataFrame({"gain": importance.round(2), "gain_%": imp_pct})
    print(fi_df.to_string())
    unused = fi_df[fi_df["gain"] == 0.0].index.tolist()
    if unused:
        print(f"\nFeatures with ZERO gain (not used in any split): {unused}")

    # 9. Save model + log run.
    section("8. SAVE MODEL + LOG RUN")
    MODEL_STORE_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    logger.info("Saved model to %s", MODEL_PATH)

    # Flatten metrics: overall + per-commodity, prefixed.
    flat_metrics = {}
    for r in resid_rows:
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
            "model_features": RESIDUAL_FEATURES,
            "excluded_features": ["lag_1", "price_vs_msp"],
        },
    )

    print(f"\n{SEP_THIN}\nTraining complete.\n  model    : {MODEL_PATH}\n  "
          f"run log  : {RUN_LOG_PATH}\n{SEP_THIN}")


if __name__ == "__main__":
    main()
