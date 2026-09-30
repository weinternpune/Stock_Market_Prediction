"""
project_config.py
-----------------
Centralized project configuration for NIFTY 50 Stock Price Prediction & Analytics.
Defines parameters for data validation, cleaning, feature engineering, modeling,
evaluation, backtesting, and dashboard styling.
"""

from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEANED_DATA_DIR = DATA_DIR / "cleaned"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PREDICTIONS_DATA_DIR = DATA_DIR / "predictions"
METADATA_DIR = DATA_DIR / "metadata"

MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CHARTS_DIR = RESULTS_DIR / "charts"
REPORTS_DIR = RESULTS_DIR / "reports"

# Primary raw master data file
RAW_MASTER_CSV = RAW_DATA_DIR / "master_merged_stock_data_all_50_stocks.csv"
METADATA_CONSTITUENTS_CSV = METADATA_DIR / "nifty50_constituents.csv"

CLEANED_DATA_PARQUET = CLEANED_DATA_DIR / "cleaned_nifty50_data.parquet"
CLEANED_DATA_CSV = CLEANED_DATA_DIR / "cleaned_nifty50_data.csv"

PROCESSED_DATA_PARQUET = PROCESSED_DATA_DIR / "feature_engineered_data.parquet"
PROCESSED_DATA_CSV = PROCESSED_DATA_DIR / "feature_engineered_data.csv"

PREDICTIONS_CSV = PREDICTIONS_DATA_DIR / "stock_predictions.csv"
MODEL_COMPARISON_CSV = PREDICTIONS_DATA_DIR / "model_comparison.csv"
STOCK_RANKING_CSV = PREDICTIONS_DATA_DIR / "stock_ranking.csv"
BACKTEST_PREDICTIONS_CSV = PREDICTIONS_DATA_DIR / "backtest_predictions.csv"

# Modeling Parameters
DEFAULT_PREDICTION_HORIZON = 21  # 21 trading days (~1 month ahead)
SUPPORTED_HORIZONS = [1, 5, 10, 21]

LOOKBACK_WINDOW = 20  # Lookback sequence length for LSTM
TEST_SIZE_SESSIONS = 120  # Holdout test sessions (~6 months unseen future)
VAL_SIZE_SESSIONS = 60    # Validation sessions
RANDOM_SEED = 42

# Technical Indicator Parameters
SMA_WINDOWS = [10, 20, 50, 100, 200]
EMA_WINDOWS = [10, 20, 50, 100]
RSI_WINDOW = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_WINDOW = 20
BOLLINGER_STD = 2.0
VOLATILITY_WINDOWS = [10, 20, 50]
RETURN_WINDOWS = [1, 3, 5, 10, 20]
LAG_DAYS = [1, 2, 3, 5]

# Corporate Continuity Mappings (Symbol transitions)
SYMBOL_ALIASES = {
    'SRTRANSFIN': 'SHRIRAMFIN',
    'ETERNAL': 'ZOMATO',
    'TMPV': 'TATAMOTORS'
}

# Reverse mapping for documentation/logging
PRIMARY_TO_ALIASES = {
    'SHRIRAMFIN': ['SRTRANSFIN'],
    'ZOMATO': ['ETERNAL'],
    'TATAMOTORS': ['TMPV']
}

# Models Enabled
MODELS_ENABLED = ["Baseline", "ARIMA", "XGBoost", "LSTM"]
