"""
test_predictions.py
-------------------
Unit tests verifying generated prediction artifacts and ranking integrity.
"""

import unittest
import pandas as pd
from pathlib import Path

from config.project_config import (
    PREDICTIONS_CSV,
    STOCK_RANKING_CSV,
    MODEL_COMPARISON_CSV,
    BACKTEST_PREDICTIONS_CSV
)

class TestPredictions(unittest.TestCase):
    
    def test_prediction_files_exist(self):
        """All 4 master result artifacts must exist on disk."""
        self.assertTrue(PREDICTIONS_CSV.exists(), "stock_predictions.csv missing")
        self.assertTrue(STOCK_RANKING_CSV.exists(), "stock_ranking.csv missing")
        self.assertTrue(MODEL_COMPARISON_CSV.exists(), "model_comparison.csv missing")
        self.assertTrue(BACKTEST_PREDICTIONS_CSV.exists(), "backtest_predictions.csv missing")

    def test_stock_predictions_schema(self):
        """stock_predictions.csv must have all required fields and no NaNs in key fields."""
        df = pd.read_csv(PREDICTIONS_CSV)
        required_cols = [
            'Symbol', 'Company', 'Industry', 'Current_Price', 'Predicted_21D_Price',
            'Expected_Change', 'Expected_Return_Pct', 'Best_Model', 'Lower_Bound_90',
            'Upper_Bound_90', 'RMSE', 'MAE', 'MAPE', 'Directional_Accuracy'
        ]
        for col in required_cols:
            self.assertIn(col, df.columns, f"Missing column in predictions: {col}")
        self.assertEqual(df['Current_Price'].isna().sum(), 0)
        self.assertEqual(df['Predicted_21D_Price'].isna().sum(), 0)
        self.assertEqual(df['Best_Model'].isna().sum(), 0)

    def test_stock_ranking_order(self):
        """stock_ranking.csv must be strictly sorted by Expected_Return_Pct descending."""
        df = pd.read_csv(STOCK_RANKING_CSV)
        self.assertIn("Rank", df.columns)
        self.assertEqual(df['Rank'].iloc[0], 1)
        is_sorted_desc = df['Expected_Return_Pct'].is_monotonic_decreasing
        self.assertTrue(is_sorted_desc, "Stock ranking is not sorted by expected return descending!")

    def test_model_comparison_coverage(self):
        """All 4 architectures must be present in model_comparison.csv."""
        df = pd.read_csv(MODEL_COMPARISON_CSV)
        models = set(df['Model'].unique())
        expected_models = {'Baseline', 'ARIMA', 'XGBoost', 'LSTM'}
        self.assertTrue(expected_models.issubset(models), f"Missing models in comparison: {expected_models - models}")

if __name__ == '__main__':
    unittest.main()
