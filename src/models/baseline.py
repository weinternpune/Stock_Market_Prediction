"""
Baseline Models for Stock Price Prediction.
Implements Naive Persistence and Rolling Moving Average baselines.
"""

import numpy as np
import pandas as pd


class NaiveBaselineModel:
    """
    Naive Persistence Model (Random Walk):
    Predicts that tomorrow's closing price will equal today's closing price:
    P_{t+1} = P_t
    """
    def __init__(self):
        self.name = "Naive Baseline (Persistence)"

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """No training required for persistence baseline."""
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Predicts close price of the current row as the forecast for target_close.
        """
        if "close" not in df.columns:
            raise KeyError("Dataframe must contain 'close' column.")
        return df["close"].values


class MovingAverageBaselineModel:
    """
    Moving Average Baseline:
    Predicts that tomorrow's closing price will equal the average of the last k days.
    """
    def __init__(self, window: int = 5):
        self.window = window
        self.name = f"MA Baseline ({window}-day SMA)"

    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Calculates k-day moving average of close prices.
        """
        if "close" not in df.columns:
            raise KeyError("Dataframe must contain 'close' column.")
        return df["close"].rolling(window=self.window, min_periods=1).mean().values
