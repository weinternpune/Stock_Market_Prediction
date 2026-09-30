"""
technical_indicators.py
-----------------------
Reusable, stock-by-stock Feature Engineering Pipeline.
Computes technical indicators, price features, return features, volatility,
volume metrics, and historical lag features strictly using causal historical data.
"""

from pathlib import Path
from typing import List
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import (
    CLEANED_DATA_PARQUET,
    PROCESSED_DATA_PARQUET,
    SMA_WINDOWS,
    EMA_WINDOWS,
    RSI_WINDOW,
    MACD_FAST,
    MACD_SLOW,
    MACD_SIGNAL,
    BOLLINGER_WINDOW,
    BOLLINGER_STD,
    VOLATILITY_WINDOWS,
    RETURN_WINDOWS,
    LAG_DAYS
)
from src.utilities.logger import get_logger
from src.utilities.io_helpers import load_dataframe, save_dataframe

logger = get_logger("FeatureEngineering")

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes Relative Strength Index (RSI) using exponential moving average."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def engineer_stock_features(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies comprehensive feature engineering to a single stock's time series.
    Assumes stock_df is sorted chronologically by Date.
    """
    df = stock_df.sort_values('Date').copy().reset_index(drop=True)
    
    # Use split-adjusted prices where available for continuity
    p_close = df['Adj_Close'] if 'Adj_Close' in df.columns else df['Close']
    p_open = df['Adj_Open'] if 'Adj_Open' in df.columns else df['Open']
    p_high = df['Adj_High'] if 'Adj_High' in df.columns else df['High']
    p_low = df['Adj_Low'] if 'Adj_Low' in df.columns else df['Low']
    vol = df['Adj_Volume'] if 'Adj_Volume' in df.columns else df['Volume']
    
    # 1. Price Features
    df['Daily_Range'] = p_high - p_low
    df['High_Low_Pct'] = (p_high - p_low) / (p_low + 1e-9)
    df['Open_Close_Pct'] = (p_close - p_open) / (p_open + 1e-9)
    
    # 2. Return Features (pct change over N sessions)
    for n in RETURN_WINDOWS:
        df[f'Return_{n}D'] = p_close.pct_change(n)
        
    # 3. Simple Moving Averages (SMA)
    for w in SMA_WINDOWS:
        df[f'SMA_{w}'] = p_close.rolling(window=w, min_periods=1).mean()
        
    # 4. Exponential Moving Averages (EMA)
    for w in EMA_WINDOWS:
        df[f'EMA_{w}'] = p_close.ewm(span=w, adjust=False).mean()
        
    # 5. Momentum: RSI
    df[f'RSI_{RSI_WINDOW}'] = compute_rsi(p_close, period=RSI_WINDOW)
    
    # 6. Momentum: MACD
    ema_fast = p_close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = p_close.ewm(span=MACD_SLOW, adjust=False).mean()
    df['MACD'] = ema_fast - ema_slow
    df['MACD_Signal'] = df['MACD'].ewm(span=MACD_SIGNAL, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 7. Bollinger Bands (20-day, 2 std)
    bb_mid = p_close.rolling(window=BOLLINGER_WINDOW, min_periods=1).mean()
    bb_std = p_close.rolling(window=BOLLINGER_WINDOW, min_periods=1).std().fillna(0)
    df['BB_Middle'] = bb_mid
    df['BB_Upper'] = bb_mid + (BOLLINGER_STD * bb_std)
    df['BB_Lower'] = bb_mid - (BOLLINGER_STD * bb_std)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / (df['BB_Middle'] + 1e-9)
    df['BB_PctB'] = (p_close - df['BB_Lower']) / ((df['BB_Upper'] - df['BB_Lower']) + 1e-9)
    
    # 8. Volatility Features (annualized std of daily returns)
    daily_ret = df['Return_1D']
    for v in VOLATILITY_WINDOWS:
        df[f'Volatility_{v}D'] = daily_ret.rolling(window=v, min_periods=2).std() * np.sqrt(252)
        
    # 9. Volume Features
    df['Volume_Change'] = vol.pct_change()
    df['Volume_SMA_20'] = vol.rolling(window=20, min_periods=1).mean()
    df['Volume_Ratio'] = vol / (df['Volume_SMA_20'] + 1e-9)
    
    # 10. Historical Lag Features
    lag_base_cols = [
        'Close',
        'Return_1D',
        'Volume',
        f'RSI_{RSI_WINDOW}',
        f'Volatility_{VOLATILITY_WINDOWS[1]}D'
    ]
    for col in lag_base_cols:
        target_series = p_close if col == 'Close' else (vol if col == 'Volume' else df[col])
        for lag in LAG_DAYS:
            df[f'{col}_lag_{lag}'] = target_series.shift(lag)
            
    return df

def run_feature_engineering(input_path: Path = CLEANED_DATA_PARQUET) -> pd.DataFrame:
    """Processes all stocks through the feature engineering pipeline."""
    logger.info("=" * 65)
    logger.info("PHASE 6: GENERATING TECHNICAL & STATISTICAL FEATURES")
    logger.info("=" * 65)
    
    clean_df = load_dataframe(input_path)
    clean_df['Date'] = pd.to_datetime(clean_df['Date'])
    
    processed_dfs = []
    symbols = sorted(clean_df['Symbol'].unique())
    logger.info(f"Processing features stock-by-stock across {len(symbols)} symbols...")
    
    for sym in symbols:
        stock_data = clean_df[clean_df['Symbol'] == sym].copy()
        feat_df = engineer_stock_features(stock_data)
        processed_dfs.append(feat_df)
        
    combined_features = pd.concat(processed_dfs, ignore_index=True)
    save_dataframe(combined_features, PROCESSED_DATA_PARQUET)
    
    logger.info(f"✔ Feature engineering complete: {len(combined_features)} rows, {len(combined_features.columns)} features.")
    logger.info(f"✔ Saved to Parquet: {PROCESSED_DATA_PARQUET}")
    return combined_features

if __name__ == "__main__":
    feat_df = run_feature_engineering()
    print("Feature columns count:", len(feat_df.columns))
    print("Feature sample columns:", [c for c in feat_df.columns if 'RSI' in c or 'SMA' in c or 'MACD' in c][:10])
