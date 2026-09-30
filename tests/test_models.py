"""
test_models.py
--------------
Unit tests verifying training, inference, and predictions across all 4 architectures:
Baseline, ARIMA, XGBoost, and PyTorch LSTM.
"""

import unittest
import pandas as pd
import numpy as np

from src.models.baseline import BaselinePredictor
from src.models.arima_model import ARIMAPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMPredictor
from src.features.technical_indicators import engineer_stock_features
from src.features.target_builder import chronological_split, get_model_features

class TestModels(unittest.TestCase):
    
    def setUp(self):
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=150, freq='B')
        prices = 200.0 + np.cumsum(np.random.randn(150) * 2.0)
        raw_df = pd.DataFrame({
            'Date': dates,
            'Symbol': ['TEST'] * 150,
            'Company': ['Test Co'] * 150,
            'Industry': ['Tech'] * 150,
            'Sector': ['IT'] * 150,
            'Open': prices - 0.5,
            'High': prices + 2.0,
            'Low': prices - 2.0,
            'Close': prices,
            'Volume': np.random.randint(20000, 80000, size=150),
            'Adj_Open': prices - 0.5,
            'Adj_High': prices + 2.0,
            'Adj_Low': prices - 2.0,
            'Adj_Close': prices,
            'Adj_Volume': np.random.randint(20000, 80000, size=150)
        })
        self.feat_df = engineer_stock_features(raw_df)
        self.train_df, self.val_df, self.test_df, self.latest_row = chronological_split(self.feat_df, horizon=21)
        self.features = get_model_features(self.train_df)

    def test_baseline_predictions(self):
        """Baseline should predict current close for test sessions."""
        model = BaselinePredictor()
        preds = model.predict_persistence(self.test_df)
        self.assertEqual(len(preds), len(self.test_df))
        self.assertAlmostEqual(preds[0], self.test_df['Adj_Close'].iloc[0])

    def test_arima_predictions(self):
        """ARIMA should fit on train series and produce valid out-of-sample predictions."""
        model = ARIMAPredictor(order=(1, 1, 1))
        model.fit(self.train_df['Adj_Close'])
        preds = model.predict_test(self.train_df['Adj_Close'], self.test_df['Adj_Close'], horizon=21)
        self.assertEqual(len(preds), len(self.test_df))
        self.assertFalse(np.isnan(preds).any())

    def test_xgboost_predictions(self):
        """XGBoost should train and predict on feature columns."""
        model = XGBoostPredictor()
        target_col = 'Future_Close_21D'
        model.fit(
            self.train_df[self.features],
            self.train_df[target_col],
            self.val_df[self.features],
            self.val_df[target_col]
        )
        preds = model.predict(self.test_df[self.features])
        self.assertEqual(len(preds), len(self.test_df))
        self.assertFalse(np.isnan(preds).any())
        
        future_pred = model.forecast_future(self.latest_row[self.features])
        self.assertGreater(future_pred, 0.0)

    def test_lstm_predictions(self):
        """PyTorch LSTM should train and generate valid inverse-scaled predictions."""
        target_col = 'Future_Close_21D'
        model = LSTMPredictor(lookback=20, epochs=5)
        model.fit(self.train_df[self.features], self.train_df[target_col])
        
        test_start = len(self.train_df) + len(self.val_df)
        comb = pd.concat([self.train_df, self.val_df, self.test_df], ignore_index=True)
        preds = model.predict_test(comb, test_start_idx=test_start, test_len=len(self.test_df))
        self.assertEqual(len(preds), len(self.test_df))
        self.assertFalse(np.isnan(preds).any())

if __name__ == '__main__':
    unittest.main()
