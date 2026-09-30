"""
target_builder.py
-----------------
Target Variable Creation & Chronological Train/Val/Test Split.
Enforces strict zero-leakage constraints per Section 12, 13, and 14 of specification.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import (
    DEFAULT_PREDICTION_HORIZON,
    TEST_SIZE_SESSIONS,
    VAL_SIZE_SESSIONS
)
from src.utilities.logger import get_logger

logger = get_logger("TargetBuilder")

# Feature columns to exclude from model input X (metadata, future targets, unscaled identifiers)
NON_FEATURE_COLS = {
    'Date', 'Symbol', 'Company', 'Industry', 'Sector', 'Original_Symbol',
    'Split_Factor', 'Target_Close_21D', 'Target_Close_10D', 'Target_Close_5D', 'Target_Close_1D',
    'Future_Close_21D', 'Future_Close_10D', 'Future_Close_5D', 'Future_Close_1D'
}

def add_future_targets(df: pd.DataFrame, horizon: int = DEFAULT_PREDICTION_HORIZON) -> pd.DataFrame:
    """
    Creates target variable strictly shifted forward in time:
    Future_Close_{horizon}D = Close(t + horizon).
    """
    df = df.sort_values('Date').copy()
    target_col = f'Future_Close_{horizon}D'
    # Use Adj_Close if present, otherwise Close
    price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
    df[target_col] = df[price_col].shift(-horizon)
    return df

def get_model_features(df: pd.DataFrame) -> List[str]:
    """Returns the list of valid numeric predictor feature column names (excluding targets/metadata)."""
    cols = [
        c for c in df.columns
        if c not in NON_FEATURE_COLS
        and not c.startswith('Future_Close')
        and not c.startswith('Target_')
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    return sorted(cols)

def chronological_split(
    stock_df: pd.DataFrame,
    horizon: int = DEFAULT_PREDICTION_HORIZON,
    test_sessions: int = TEST_SIZE_SESSIONS,
    val_sessions: int = VAL_SIZE_SESSIONS
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits an individual stock's time-series chronologically into:
    1. train_df: Earliest historical data (fitted scalers, model training)
    2. val_df: Validation data (early stopping, hyperparameter tuning)
    3. test_df: Most recent historical sessions (unseen out-of-sample evaluation)
    4. latest_row: The most recent single session (t=latest) used to generate the final future forecast
    
    Zero data leakage guarantee:
    - No rows are shuffled.
    - Test set is strictly from the end of the timeline.
    - Future targets with NaN at the tail are properly handled.
    """
    df = stock_df.sort_values('Date').copy().reset_index(drop=True)
    df = add_future_targets(df, horizon=horizon)
    
    target_col = f'Future_Close_{horizon}D'
    
    # The latest session row (to forecast future 21D from today's market close)
    latest_row = df.iloc[[-1]].copy()
    
    # Rows with valid target (excluding the last horizon rows which haven't occurred yet)
    labeled_df = df.dropna(subset=[target_col]).copy().reset_index(drop=True)
    
    # Clean initial warm-up NaNs from rolling indicators (e.g. SMA 200)
    # Require at least non-null Return_1D and RSI
    labeled_df = labeled_df.dropna(subset=['Return_1D', 'RSI_14']).copy().reset_index(drop=True)
    
    n_total = len(labeled_df)
    if n_total <= (test_sessions + val_sessions + 50):
        # Fallback proportional split if history is shorter (e.g. JIOFIN)
        n_test = max(30, int(n_total * 0.15))
        n_val = max(20, int(n_total * 0.10))
    else:
        n_test = test_sessions
        n_val = val_sessions
        
    train_end = n_total - n_test - n_val
    val_end = n_total - n_test
    
    train_df = labeled_df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = labeled_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = labeled_df.iloc[val_end:].copy().reset_index(drop=True)
    
    return train_df, val_df, test_df, latest_row

if __name__ == "__main__":
    from src.utilities.io_helpers import load_dataframe
    from config.project_config import PROCESSED_DATA_PARQUET
    
    if PROCESSED_DATA_PARQUET.exists():
        df = load_dataframe(PROCESSED_DATA_PARQUET)
        sample_stock = df[df['Symbol'] == 'TCS'].copy()
        tr, val, te, latest = chronological_split(sample_stock)
        print(f"TCS Chronological Split: Train={len(tr)}, Val={len(val)}, Test={len(te)}, Latest={len(latest)}")
        print(f"Features available: {len(get_model_features(tr))}")
