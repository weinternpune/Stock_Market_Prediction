"""
baseline.py
-----------
Implements Phases 28 and 29 of the Project Roadmap:
- Phase 28 (Step 32): Naive Persistence Baseline (Tomorrow's price = Today's close).
- Phase 29 (Step 33): 5-Day Simple Moving Average Baseline.
"""

import pandas as pd
import numpy as np


class NaivePersistenceBaseline:
    """Predicts next day's closing price as today's closing price: P_{t+1} = P_t."""
    def __init__(self):
        self.name = "Naive Persistence"
        
    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        # Today's closing price is in column 'Close'
        return test_df['Close'].to_numpy(dtype=np.float64)


class MovingAverageBaseline:
    """Predicts next day's closing price as the rolling 5-day SMA."""
    def __init__(self, window: int = 5):
        self.window = window
        self.name = f"Moving Average ({window}-Day SMA)"
        
    def predict(self, full_df: pd.DataFrame, n_test: int) -> np.ndarray:
        # 5-day SMA of past prices leading up to today
        sma = full_df['Close'].rolling(self.window).mean()
        return sma.iloc[-n_test:].to_numpy(dtype=np.float64)
