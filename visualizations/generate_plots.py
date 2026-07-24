"""Generate all project visualizations.

This script creates 5 key visualizations for the agri-price-forecaster project:
1. Price trends over time (daily averages per commodity)
2. Monthly seasonality patterns
3. Train/test split visualization
4. Predicted vs actual scatter plots
5. Feature importance comparison (direct vs residual XGBoost)

Run:
    python -m visualizations.generate_plots
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import json

from src.config import (
    PROCESSED_DATA_DIR,
    MODEL_STORE_DIR,
    COL_DATE,
    COL_COMMODITY,
    COL_MODAL_PRICE,
    COMMODITIES_V1,
    COMMODITIES_MODELING_V1,
)
from src.models.pre_training_verification import load_filter_split

# Output directory for all plots
VIZ_DIR = Path(__file__).resolve().parent

# Data file paths
CLEANED_CSV = PROCESSED_DATA_DIR / "mandi_prices_cleaned.csv"
FEATURES_CSV = PROCESSED_DATA_DIR / "mandi_prices_features.csv"
PREDICTIONS_CSV = PROCESSED_DATA_DIR / "direct_xgboost_test_predictions.csv"
DIRECT_RUN_LOG = MODEL_STORE_DIR / "xgboost_v1_run_log.json"
RESIDUAL_RUN_LOG = MODEL_STORE_DIR / "xgboost_residual_v1_run_log.json"


def plot_price_trends():
    """Plot daily average modal_price over time, one subplot per commodity.
    
    Source: data/processed/mandi_prices_cleaned.csv
    All 4 commodities: Onion, Tomato, Potato, Wheat
    """
    print("Generating price_trends.png...")
    
    df = pd.read_csv(CLEANED_CSV, parse_dates=[COL_DATE])
    
    # Filter to the 4 commodities in COMMODITIES_V1
    df = df[df[COL_COMMODITY].isin(COMMODITIES_V1)]
    
    # Aggregate by (price_date, commodity) mean
    daily_avg = df.groupby([COL_DATE, COL_COMMODITY])[COL_MODAL_PRICE].mean().reset_index()
    
    # Create subplots - one per commodity
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    axes = axes.flatten()
    
    for i, comm in enumerate(COMMODITIES_V1):
        comm_data = daily_avg[daily_avg[COL_COMMODITY] == comm].sort_values(COL_DATE)
        axes[i].plot(comm_data[COL_DATE], comm_data[COL_MODAL_PRICE], linewidth=1)
        axes[i].set_ylabel('Price (₹/quintal)')
        axes[i].set_title(f'{comm} - Daily Average Modal Price')
        axes[i].grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Date')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    output_path = VIZ_DIR / "price_trends.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_monthly_seasonality():
    """Plot average modal_price by calendar month, one line per commodity.
    
    Source: data/processed/mandi_prices_cleaned.csv
    All 4 commodities on a single shared plot with legend.
    """
    print("Generating monthly_seasonality.png...")
    
    df = pd.read_csv(CLEANED_CSV, parse_dates=[COL_DATE])
    df = df[df[COL_COMMODITY].isin(COMMODITIES_V1)]
    
    # Extract month
    df['month'] = df[COL_DATE].dt.month
    
    # Average modal_price by month and commodity
    monthly_avg = df.groupby(['month', COL_COMMODITY])[COL_MODAL_PRICE].mean().reset_index()
    
    # Plot
    plt.figure(figsize=(12, 6))
    for comm in COMMODITIES_V1:
        comm_data = monthly_avg[monthly_avg[COL_COMMODITY] == comm].sort_values('month')
        plt.plot(comm_data['month'], comm_data[COL_MODAL_PRICE], 
                marker='o', linewidth=2, label=comm)
    
    plt.xlabel('Month')
    plt.ylabel('Average Modal Price (₹/quintal)')
    plt.title('Monthly Seasonality - Average Price by Month')
    plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    output_path = VIZ_DIR / "monthly_seasonality.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_train_test_split():
    """Plot modal_price over time with train/test split visualization.
    
    Source: data/processed/mandi_prices_features.csv
    Filtered to COMMODITIES_MODELING_V1 (Onion, Potato)
    Uses the exact split logic from pre_training_verification.load_filter_split()
    """
    print("Generating train_test_split.png...")
    
    # Load and split using the verified function
    train, test, cutoff, df_filtered = load_filter_split()
    
    # Aggregate by date (mean across all states/commodities in scope)
    train_daily = train.groupby(COL_DATE)[COL_MODAL_PRICE].mean().reset_index().sort_values(COL_DATE)
    test_daily = test.groupby(COL_DATE)[COL_MODAL_PRICE].mean().reset_index().sort_values(COL_DATE)
    
    # Plot
    plt.figure(figsize=(14, 6))
    
    # Plot train period in blue
    plt.plot(train_daily[COL_DATE], train_daily[COL_MODAL_PRICE], 
             color='blue', linewidth=1.5, label='Train period')
    
    # Plot test period in orange
    plt.plot(test_daily[COL_DATE], test_daily[COL_MODAL_PRICE], 
             color='orange', linewidth=1.5, label='Test period')
    
    # Add vertical dashed line at cutoff
    plt.axvline(x=cutoff, color='red', linestyle='--', linewidth=2, 
                label=f'Train/Test cutoff: {cutoff.date()}')
    
    # Shade the periods
    plt.axvspan(train_daily[COL_DATE].min(), cutoff, alpha=0.1, color='blue')
    plt.axvspan(cutoff, test_daily[COL_DATE].max(), alpha=0.1, color='orange')
    
    plt.xlabel('Date')
    plt.ylabel('Average Modal Price (₹/quintal)')
    plt.title(f'Train/Test Split - {COMMODITIES_MODELING_V1} (aggregated)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Format x-axis dates
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    
    output_path = VIZ_DIR / "train_test_split.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_predicted_vs_actual():
    """Plot predicted vs actual modal_price from XGBoost test predictions.
    
    Source: data/processed/direct_xgboost_test_predictions.csv
    Scatter plot with y=x reference line, one subplot per commodity (Onion, Potato)
    """
    print("Generating predicted_vs_actual.png...")
    
    df = pd.read_csv(PREDICTIONS_CSV, parse_dates=[COL_DATE])
    
    # Create subplots for Onion and Potato
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    for i, comm in enumerate(COMMODITIES_MODELING_V1):
        comm_data = df[df[COL_COMMODITY] == comm]
        
        actual = comm_data['actual_modal_price']
        predicted = comm_data['predicted_modal_price']
        
        # Scatter plot
        axes[i].scatter(actual, predicted, alpha=0.5, s=20)
        
        # y=x reference line
        min_val = min(actual.min(), predicted.min())
        max_val = max(actual.max(), predicted.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], 
                    'r--', linewidth=2, label='y = x')
        
        axes[i].set_xlabel('Actual Modal Price (₹/quintal)')
        axes[i].set_ylabel('Predicted Modal Price (₹/quintal)')
        axes[i].set_title(f'{comm} - Predicted vs Actual')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = VIZ_DIR / "predicted_vs_actual.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def plot_feature_importance_comparison():
    """Plot feature importance comparison between direct and residual XGBoost.
    
    Source: models_store/xgboost_v1_run_log.json and 
             models_store/xgboost_residual_v1_run_log.json
    
    Two horizontal bar charts side by side with SHARED x-axis scale to show
    the contrast between "one feature dominates" (direct) vs 
    "no feature dominates" (residual).
    """
    print("Generating feature_importance.png...")
    
    # Load run logs
    with open(DIRECT_RUN_LOG, 'r') as f:
        direct_log = json.load(f)
    with open(RESIDUAL_RUN_LOG, 'r') as f:
        residual_log = json.load(f)
    
    # Extract feature importance (gain percentages)
    direct_imp = pd.Series(direct_log['feature_importance_gain']).sort_values(ascending=True)
    residual_imp = pd.Series(residual_log['feature_importance_gain']).sort_values(ascending=True)
    
    # Create side-by-side horizontal bar plots with SHARED x-axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), sharex=True)
    
    # Direct XGBoost
    direct_imp.plot(kind='barh', ax=ax1, color='steelblue')
    ax1.set_xlabel('Feature Importance (%)')
    ax1.set_ylabel('Feature')
    ax1.set_title('Direct XGBoost Feature Importance\n(lag_1 + rolling_mean_7 dominate)')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Residual XGBoost
    residual_imp.plot(kind='barh', ax=ax2, color='coral')
    ax2.set_xlabel('Feature Importance (%)')
    ax2.set_ylabel('Feature')
    ax2.set_title('Residual XGBoost Feature Importance\n(flat ~6-9.5% spread)')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    output_path = VIZ_DIR / "feature_importance.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved to {output_path}")


def main():
    """Generate all 5 visualizations."""
    print("=" * 60)
    print("Generating all project visualizations")
    print("=" * 60)
    
    plot_price_trends()
    plot_monthly_seasonality()
    plot_train_test_split()
    plot_predicted_vs_actual()
    plot_feature_importance_comparison()
    
    print("=" * 60)
    print("All visualizations generated successfully!")
    print(f"Output directory: {VIZ_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
