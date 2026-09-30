"""
baseline.py
-----------
Baseline models for 21-trading-day future closing price prediction:
1. Naive Persistence Baseline: Future Close = Latest Observed Close
2. 20-Day Moving Average Baseline: Future Close = 20-Day SMA of Close
"""

import numpy as np
import pandas as pd

class BaselinePredictor:
    """Computes naive statistical benchmarks for the 21-trading-day forecast horizon."""
    
    def __init__(self):
        self.name = "Baseline"
        
    def predict_persistence(self, df: pd.DataFrame) -> np.ndarray:
        """Predicts future price as current closing price (persistence)."""
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        return df[price_col].to_numpy(dtype=np.float64)
        
    def predict_sma(self, df: pd.DataFrame, window: int = 20) -> np.ndarray:
        """Predicts future price as current rolling 20-day SMA."""
        price_col = 'Adj_Close' if 'Adj_Close' in df.columns else 'Close'
        sma = df[price_col].rolling(window=window, min_periods=1).mean()
        return sma.to_numpy(dtype=np.float64)
        
    def forecast_future(self, latest_row: pd.DataFrame) -> float:
        """Forecasts future 21-day price from the latest market close."""
        price_col = 'Adj_Close' if 'Adj_Close' in latest_row.columns else 'Close'
        return float(latest_row[price_col].iloc[-1])
