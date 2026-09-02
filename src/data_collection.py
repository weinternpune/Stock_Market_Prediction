"""
Data Collection Module for Nifty 500 and BSE 500 Index Data.
Collects 5 years of daily historical OHLCV market data.
"""

import os
from pathlib import Path
import pandas as pd
import yfinance as yf

# Standard ticker symbols
NSE_NIFTY500_TICKER = "^CRSLDX"      # S&P CNX 500 / NIFTY 500 on NSE
BSE_500_TICKER = "BSE-500.BO"        # S&P BSE 500 on BSE

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"


def fetch_historical_data(ticker: str, period: str = "5y") -> pd.DataFrame:
    """
    Fetch historical daily OHLCV data using yfinance.

    Args:
        ticker (str): Ticker symbol (e.g., '^CRSLDX', 'BSE-500.BO').
        period (str): Lookback period (default: '5y').

    Returns:
        pd.DataFrame: Cleaned dataframe with standard OHLCV columns.
    """
    print(f"[DATA COLLECTION] Downloading {ticker} data for period: {period}...")
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        raise ValueError(f"Failed to download data for ticker '{ticker}'. Result is empty.")

    # Flatten MultiIndex columns if present (yfinance >= 0.2.36 often returns MultiIndex)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    # Reset index to make 'Date' a column
    df = df.reset_index()

    # Normalize column names
    col_mapping = {
        "Date": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Volume": "volume"
    }
    df = df.rename(columns={k: v for k, v in col_mapping.items() if k in df.columns})

    # If adj_close is missing, mirror close
    if "adj_close" not in df.columns and "close" in df.columns:
        df["adj_close"] = df["close"]

    # Ensure date column is datetime
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    print(f"[DATA COLLECTION] Fetched {len(df)} trading records for {ticker} "
          f"({df['date'].min().date()} to {df['date'].max().date()})")
    return df


def collect_and_save_data(output_dir: Path = RAW_DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch both NSE Nifty 500 and BSE 500 datasets and persist to raw data directory.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_nse, df_bse)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    nse_path = output_dir / "nifty_500_nse_raw.csv"
    bse_path = output_dir / "bse_500_raw.csv"

    df_nse = fetch_historical_data(NSE_NIFTY500_TICKER, period="5y")
    df_bse = fetch_historical_data(BSE_500_TICKER, period="5y")

    df_nse.to_csv(nse_path, index=False)
    df_bse.to_csv(bse_path, index=False)

    print(f"[DATA COLLECTION] Saved NSE raw data to: {nse_path}")
    print(f"[DATA COLLECTION] Saved BSE raw data to: {bse_path}")

    return df_nse, df_bse


if __name__ == "__main__":
    collect_and_save_data()
