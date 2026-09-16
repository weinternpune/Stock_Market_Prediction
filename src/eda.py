"""
eda.py
------
Exploratory Data Analysis module for NIFTY 50 Stock Universe.
Implements Phase 5 dataset-level, individual stock, and cross-stock analytics.
"""

from pathlib import Path
from typing import Dict, Any
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.project_config import (
    CLEANED_DATA_PARQUET,
    METRICS_DIR,
    CHARTS_DIR
)
from src.utilities.logger import get_logger
from src.utilities.io_helpers import load_dataframe, save_json, ensure_dir

logger = get_logger("EDA")

def compute_stock_eda_metrics(group: pd.DataFrame) -> Dict[str, Any]:
    """Computes key exploratory metrics for an individual stock."""
    grp = group.sort_values('Date').copy()
    close = grp['Adj_Close'].values
    dates = grp['Date']
    
    daily_returns = np.diff(close) / close[:-1]
    
    # Cumulative return
    cum_return = (close[-1] - close[0]) / close[0]
    
    # Annualized Volatility (assuming 252 trading days)
    ann_vol = float(np.std(daily_returns) * np.sqrt(252)) if len(daily_returns) > 1 else 0.0
    
    # Max Drawdown
    cum_max = np.maximum.accumulate(close)
    drawdowns = (close - cum_max) / cum_max
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) > 0 else 0.0
    
    # 52-week High and Low (last 252 sessions)
    recent_window = close[-252:] if len(close) >= 252 else close
    high_52w = float(np.max(recent_window))
    low_52w = float(np.min(recent_window))
    
    return {
        "symbol": grp['Symbol'].iloc[0],
        "company": grp['Company'].iloc[0],
        "industry": grp['Industry'].iloc[0],
        "records": len(grp),
        "start_date": dates.min().strftime('%Y-%m-%d'),
        "end_date": dates.max().strftime('%Y-%m-%d'),
        "latest_close": float(close[-1]),
        "first_close": float(close[0]),
        "cumulative_return_pct": round(float(cum_return * 100), 2),
        "annualized_volatility_pct": round(ann_vol * 100, 2),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "high_52w": round(high_52w, 2),
        "low_52w": round(low_52w, 2),
        "avg_daily_volume": float(grp['Volume'].mean())
    }

def run_eda(cleaned_path: Path = CLEANED_DATA_PARQUET) -> Dict[str, Any]:
    logger.info("=" * 65)
    logger.info("PHASE 5: RUNNING EXPLORATORY DATA ANALYSIS (EDA)")
    logger.info("=" * 65)
    
    df = load_dataframe(cleaned_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    symbols = sorted(df['Symbol'].unique().tolist())
    logger.info(f"Loaded {len(df)} records across {len(symbols)} stocks for EDA.")
    
    # 1. Individual Stock Profiles
    stock_summaries = {}
    returns_dict = {}
    
    for sym, grp in df.groupby('Symbol'):
        grp_sorted = grp.sort_values('Date')
        stock_summaries[sym] = compute_stock_eda_metrics(grp_sorted)
        # Store daily returns aligned by date for correlation
        ret_series = grp_sorted.set_index('Date')['Adj_Close'].pct_change().dropna()
        returns_dict[sym] = ret_series
        
    # 2. Cross-Stock Correlation Matrix
    returns_df = pd.DataFrame(returns_dict)
    corr_matrix = returns_df.corr().round(4)
    corr_csv_path = METRICS_DIR / "correlation_matrix.csv"
    ensure_dir(METRICS_DIR)
    corr_matrix.to_csv(corr_csv_path)
    logger.info(f"✔ Cross-stock correlation matrix saved to {corr_csv_path}")
    
    # 3. Overall Market-level summary
    summary_df = pd.DataFrame(list(stock_summaries.values()))
    
    eda_report = {
        "total_stocks": len(symbols),
        "total_records": len(df),
        "universe_start_date": df['Date'].min().strftime('%Y-%m-%d'),
        "universe_end_date": df['Date'].max().strftime('%Y-%m-%d'),
        "total_trading_days": int(df['Date'].nunique()),
        "average_cumulative_return_pct": round(float(summary_df['cumulative_return_pct'].mean()), 2),
        "median_cumulative_return_pct": round(float(summary_df['cumulative_return_pct'].median()), 2),
        "average_annualized_volatility_pct": round(float(summary_df['annualized_volatility_pct'].mean()), 2),
        "top_5_cumulative_gainers": summary_df.sort_values('cumulative_return_pct', ascending=False)[['symbol', 'company', 'cumulative_return_pct']].head(5).to_dict(orient='records'),
        "bottom_5_cumulative_performers": summary_df.sort_values('cumulative_return_pct', ascending=True)[['symbol', 'company', 'cumulative_return_pct']].head(5).to_dict(orient='records'),
        "stock_profiles": stock_summaries
    }
    
    save_json(eda_report, METRICS_DIR / "eda_summary.json")
    logger.info("✔ EDA summary report successfully generated and saved to results/metrics/eda_summary.json")
    return eda_report

if __name__ == "__main__":
    report = run_eda()
    print("EDA Complete. Total stocks analyzed:", report["total_stocks"])
    print("Avg 5-Year Cumulative Return:", report["average_cumulative_return_pct"], "%")
