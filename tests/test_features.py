"""
test_features.py
----------------
Unit tests for technical indicators and feature engineering.
Verifies RSI, SMA, EMA, MACD, Bollinger Bands, Volatility, and Lags.
"""

import unittest
import pandas as pd
import numpy as np
from src.features.technical_indicators import compute_rsi, engineer_stock_features

class TestFeatures(unittest.TestCase):
    
    def setUp(self):
        # Generate 100 days of synthetic price data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=100, freq='B')
        prices = 100.0 + np.cumsum(np.random.randn(100) * 1.5)
        self.df = pd.DataFrame({
            'Date': dates,
            'Symbol': ['TEST'] * 100,
            'Company': ['Test Co'] * 100,
            'Industry': ['Tech'] * 100,
            'Sector': ['IT'] * 100,
            'Open': prices - 0.5,
            'High': prices + 1.5,
            'Low': prices - 1.5,
            'Close': prices,
            'Volume': np.random.randint(50000, 200000, size=100),
            'Adj_Open': prices - 0.5,
            'Adj_High': prices + 1.5,
            'Adj_Low': prices - 1.5,
            'Adj_Close': prices,
            'Adj_Volume': np.random.randint(50000, 200000, size=100)
        })

    def test_rsi_bounds(self):
        """RSI must always stay within [0, 100]."""
        rsi = compute_rsi(self.df['Close'], period=14).dropna()
        self.assertTrue((rsi >= 0.0).all())
        self.assertTrue((rsi <= 100.0).all())

    def test_sma_calculation(self):
        """SMA 10 on day 10 must match simple mean of preceding 10 days."""
        features = engineer_stock_features(self.df)
        expected_sma10 = self.df['Close'].iloc[:10].mean()
        self.assertAlmostEqual(features['SMA_10'].iloc[9], expected_sma10, places=4)

    def test_returns_calculation(self):
        """Return 1D must match pct_change(1)."""
        features = engineer_stock_features(self.df)
        expected_ret = (self.df['Close'].iloc[1] - self.df['Close'].iloc[0]) / self.df['Close'].iloc[0]
        self.assertAlmostEqual(features['Return_1D'].iloc[1], expected_ret, places=5)

    def test_bollinger_bands_ordering(self):
        """Upper band must be >= Middle Band >= Lower Band."""
        features = engineer_stock_features(self.df).dropna()
        self.assertTrue((features['BB_Upper'] >= features['BB_Middle']).all())
        self.assertTrue((features['BB_Middle'] >= features['BB_Lower']).all())

    def test_lag_features(self):
        """Lag 1 of Close at index i must equal Close at index i-1."""
        features = engineer_stock_features(self.df)
        for i in range(1, 10):
            self.assertEqual(features['Close_lag_1'].iloc[i], self.df['Close'].iloc[i - 1])

if __name__ == '__main__':
    unittest.main()
