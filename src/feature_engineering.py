"""
feature_engineering.py
----------------------
Implements Phases 17 through 27 of the Project Roadmap:
- Phase 17 (Step 21): Moving averages (SMA 10, 20, 50, 200; EMA 12, 26).
- Phase 18 (Step 22): RSI (14-day Wilder smoothing).
- Phase 19 (Step 23): MACD line, signal line, MACD histogram.
- Phase 20 (Step 24): Bollinger Bands (Middle, Upper, Lower, %B, Bandwidth).
- Phase 21 (Step 25): Price and Return lag features (t-1, t-2, t-3, t-5).
- Phase 22 (Step 26): Rolling volatility (10-day, 20-day annualized).
- Phase 23 (Step 27): Final modeling table.
- Phase 24 (Step 28): Target definition (Target = Close_{t+1}, zero leakage).
- Phase 25 (Step 29): Indicator warm-up handling (drop early NaNs).
- Phase 26 (Step 30): Strict time-based train/test split (no random shuffling).
- Phase 27 (Step 31): Time-series cross-validation strategy.
"""

from pathlib import Path
import pandas as pd
import numpy as np


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Computes RSI using Wilder's exponential smoothing."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(clean_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Engineers all technical and statistical features on the clean Nifty 500 series.
    Returns:
      1. features_df: Complete historical feature dataset with Target column.
      2. forecast_row: The latest available day (31-Aug-2026) for forward projection.
    """
    print("=" * 70)
    print("PHASES 17-25: FEATURE ENGINEERING & TARGET CREATION")
    print("=" * 70)
    
    df = clean_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 1. Moving Averages (Phase 17)
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_20'] = df['Close'].rolling(20).mean()
    df['SMA_50'] = df['Close'].rolling(50).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()
    
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # Trend distance features
    df['Dist_SMA_50'] = (df['Close'] - df['SMA_50']) / df['SMA_50']
    df['Dist_SMA_200'] = (df['Close'] - df['SMA_200']) / df['SMA_200']
    
    # 2. RSI (Phase 18)
    df['RSI_14'] = compute_rsi(df['Close'], 14)
    
    # 3. MACD (Phase 19)
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # 4. Bollinger Bands (Phase 20)
    rolling_std_20 = df['Close'].rolling(20).std()
    df['BB_Middle'] = df['SMA_20']
    df['BB_Upper'] = df['BB_Middle'] + (2 * rolling_std_20)
    df['BB_Lower'] = df['BB_Middle'] - (2 * rolling_std_20)
    df['BB_Bandwidth'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_PctB'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'] + 1e-9)
    
    # 5. Price & Return Lags (Phase 21)
    df['Return'] = df['Close'].pct_change()
    df['Return_Lag1'] = df['Return'].shift(1)
    df['Return_Lag2'] = df['Return'].shift(2)
    df['Return_Lag3'] = df['Return'].shift(3)
    df['Return_Lag5'] = df['Return'].shift(5)
    
    df['Close_Lag1'] = df['Close'].shift(1)
    df['Close_Lag2'] = df['Close'].shift(2)
    
    # 6. Rolling Volatility (Phase 22)
    df['Vol_10'] = df['Return'].rolling(10).std() * np.sqrt(250)
    df['Vol_20'] = df['Return'].rolling(20).std() * np.sqrt(250)
    
    # 7. Intraday Dynamics
    df['High_Low_Ratio'] = df['High'] / (df['Low'] + 1e-9)
    df['Close_Open_Ratio'] = df['Close'] / (df['Open'] + 1e-9)
    
    # 8. Prediction Target (Phase 24)
    # Today's features at t -> Tomorrow's closing price Close_{t+1}
    df['Target'] = df['Close'].shift(-1)
    
    print(f"Total features computed: {len(df.columns) - 2} indicators")
    
    # The last row contains today's latest features but Target is NaN (for tomorrow's real prediction!)
    forecast_row = df.iloc[-1:].copy()
    
    # Drop indicator warm-up rows (first 200 rows due to SMA_200) and the final row where Target is NaN
    model_df = df.dropna(subset=['SMA_200', 'Target']).reset_index(drop=True)
    
    print(f"Total valid modeling rows after warm-up removal: {len(model_df)}")
    print(f"Modeling Date Range: {model_df['Date'].min().strftime('%Y-%m-%d')} to {model_df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"Forward forecast row date: {forecast_row['Date'].values[0]}")
    
    return model_df, forecast_row


def get_train_test_split(model_df: pd.DataFrame, test_sessions: int = 208) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Phase 26 (Step 30): Strict time-based train/test split.
    Last 208 trading sessions (~10 months) held out for out-of-sample testing.
    Zero random shuffling to prevent lookahead data leakage.
    """
    print("\n" + "=" * 70)
    print("PHASE 26: TIME-BASED TRAIN / TEST SPLIT (NO SHUFFLING)")
    print("=" * 70)
    
    split_idx = len(model_df) - test_sessions
    train_df = model_df.iloc[:split_idx].copy().reset_index(drop=True)
    test_df = model_df.iloc[split_idx:].copy().reset_index(drop=True)
    
    print(f"Training set:   {len(train_df)} sessions ({train_df['Date'].min().strftime('%Y-%m-%d')} to {train_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Test set:       {len(test_df)} sessions ({test_df['Date'].min().strftime('%Y-%m-%d')} to {test_df['Date'].max().strftime('%Y-%m-%d')})")
    print(f"Test proportion: {len(test_df) / len(model_df) * 100:.1f}%")
    
    return train_df, test_df


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    nse_clean = base_dir / "data" / "processed" / "NIFTY500_clean.csv"
    features_dir = base_dir / "data" / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    df = pd.read_csv(nse_clean)
    model_df, forecast_row = engineer_features(df)
    
    features_path = features_dir / "nifty_500_features.csv"
    model_df.to_csv(features_path, index=False)
    print(f"\n>> Saved feature dataset to: {features_path}")
    
    train_df, test_df = get_train_test_split(model_df)
