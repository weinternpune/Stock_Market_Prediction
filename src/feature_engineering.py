"""
Feature Engineering Module for Nifty 500 Market Data.
Computes technical indicators, statistical volatility metrics, momentum signals,
lagged returns, and future target variables.
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DATA_DIR = DATA_DIR / "features"


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / (loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Calculates MACD line, Signal line, and MACD Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist


def calculate_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series, pd.Series]:
    """Calculates Bollinger Bands: Middle, Upper, Lower, Width, and %B."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    width = (upper - lower) / (middle + 1e-9)
    pct_b = (series - lower) / (upper - lower + 1e-9)
    return middle, upper, lower, width, pct_b


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes all technical indicators, statistical features, and forward target values.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    close = df["close"]
    volume = df["volume"]

    # 1. Trend Indicators (Moving Averages)
    df["sma_20"] = close.rolling(window=20).mean()
    df["sma_50"] = close.rolling(window=50).mean()
    df["sma_200"] = close.rolling(window=200).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()
    df["ema_50"] = close.ewm(span=50, adjust=False).mean()

    # Price to MA ratios
    df["ratio_close_sma20"] = close / df["sma_20"]
    df["ratio_close_sma50"] = close / df["sma_50"]
    df["ratio_close_sma200"] = close / df["sma_200"]

    # 2. Momentum Indicators
    df["rsi_14"] = calculate_rsi(close, period=14)
    macd_line, signal_line, macd_hist = calculate_macd(close)
    df["macd_line"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = macd_hist

    # 3. Volatility Indicators
    bb_mid, bb_up, bb_low, bb_w, bb_pctb = calculate_bollinger_bands(close, period=20, num_std=2.0)
    df["bb_upper"] = bb_up
    df["bb_lower"] = bb_low
    df["bb_width"] = bb_w
    df["bb_pct_b"] = bb_pctb

    # Rolling Volatility (annualized by multiplying by sqrt(252))
    df["volatility_20d"] = df["daily_return"].rolling(window=20).std() * np.sqrt(252)
    df["volatility_50d"] = df["daily_return"].rolling(window=50).std() * np.sqrt(252)

    # 4. Lagged Returns & Price Lags
    df["return_1d"] = df["daily_return"]
    df["return_5d"] = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)

    df["lag_close_1"] = close.shift(1)
    df["lag_close_2"] = close.shift(2)
    df["lag_close_5"] = close.shift(5)

    # 5. Volume Indicators
    df["volume_sma_20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = volume / (df["volume_sma_20"] + 1e-9)

    # 6. Target Variables (Future Target Price per PRD FR6)
    # Target 1: Next-day Close price (T+1)
    df["target_close"] = close.shift(-1)
    # Target 2: Next-day return (T+1)
    df["target_return"] = df["target_close"].pct_change()

    # Drop warmup rows where 200-day moving average is NaN
    # Drop the very last row where target_close is NaN
    initial_len = len(df)
    features_df = df.dropna().reset_index(drop=True)
    print(f"[FEATURE ENGINEERING] Created features. Filtered warmup period from {initial_len} to {len(features_df)} rows.")

    return features_df


def run_feature_pipeline() -> pd.DataFrame:
    """Executes feature engineering and saves feature dataset."""
    FEATURES_DATA_DIR.mkdir(parents=True, exist_ok=True)
    cleaned_path = PROCESSED_DATA_DIR / "nifty_500_cleaned.csv"

    if not cleaned_path.exists():
        from src.data_preprocessing import preprocess_pipeline
        df_clean, _, _ = preprocess_pipeline()
    else:
        df_clean = pd.read_csv(cleaned_path)

    features_df = engineer_features(df_clean)
    output_path = FEATURES_DATA_DIR / "nifty_500_features.csv"
    features_df.to_csv(output_path, index=False)
    print(f"[FEATURE ENGINEERING] Saved engineered features dataset to: {output_path}")

    return features_df


if __name__ == "__main__":
    run_feature_pipeline()
