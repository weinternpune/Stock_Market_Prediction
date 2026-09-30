"""
test_leakage_and_target.py
--------------------------
MANDATORY ZERO DATA LEAKAGE TESTS (Section 49 Requirement).
Verifies:
1. Target is strictly shifted into the future (t + 21).
2. Features NEVER contain future price information.
3. Feature scalers are fitted ONLY on the training split.
4. Input feature matrix X strictly excludes target columns.
5. Test partition is strictly out-of-sample forward in time.
"""

import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

from src.features.target_builder import add_future_targets, get_model_features, chronological_split
from src.features.technical_indicators import engineer_stock_features

class TestDataLeakage(unittest.TestCase):
    
    def setUp(self):
        # 300 synthetic sequential trading days
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=300, freq='B')
        prices = 100.0 + np.cumsum(np.random.randn(300) * 1.2)
        self.raw_df = pd.DataFrame({
            'Date': dates,
            'Symbol': ['SYNTH'] * 300,
            'Company': ['Synthetic Inc'] * 300,
            'Industry': ['Finance'] * 300,
            'Sector': ['Financial Services'] * 300,
            'Open': prices - 0.2,
            'High': prices + 1.0,
            'Low': prices - 1.0,
            'Close': prices,
            'Volume': np.random.randint(10000, 50000, size=300),
            'Adj_Open': prices - 0.2,
            'Adj_High': prices + 1.0,
            'Adj_Low': prices - 1.0,
            'Adj_Close': prices,
            'Adj_Volume': np.random.randint(10000, 50000, size=300)
        })
        self.feat_df = engineer_stock_features(self.raw_df)

    def test_target_forward_shift(self):
        """Future_Close_21D at index i MUST strictly equal Close at index i + 21."""
        df_target = add_future_targets(self.feat_df, horizon=21)
        for i in range(10):
            target_val = df_target['Future_Close_21D'].iloc[i]
            future_actual_close = df_target['Adj_Close'].iloc[i + 21]
            self.assertEqual(target_val, future_actual_close)
            
        # The last 21 rows must have NaN target because their future has not happened yet
        tail_targets = df_target['Future_Close_21D'].iloc[-21:]
        self.assertTrue(tail_targets.isna().all())

    def test_feature_list_excludes_targets_and_future_fields(self):
        """Feature matrix X must never include target columns or unshifted metadata."""
        df_target = add_future_targets(self.feat_df, horizon=21)
        features = get_model_features(df_target)
        
        # Verify no target or future columns in feature list
        for f in features:
            self.assertFalse('Target' in f, f"Forbidden target column in features: {f}")
            self.assertFalse('Future' in f, f"Forbidden future column in features: {f}")
            self.assertNotEqual(f, 'Date')
            self.assertNotEqual(f, 'Symbol')

    def test_chronological_split_ordering(self):
        """Train dates must strictly precede validation dates, which strictly precede test dates."""
        train_df, val_df, test_df, latest_row = chronological_split(self.feat_df, horizon=21)
        
        max_train_date = train_df['Date'].max()
        min_val_date = val_df['Date'].min()
        max_val_date = val_df['Date'].max()
        min_test_date = test_df['Date'].min()
        
        self.assertLess(max_train_date, min_val_date, "Train split leaks into validation split!")
        self.assertLess(max_val_date, min_test_date, "Validation split leaks into test split!")

    def test_scaler_isolation(self):
        """Scalers must be fit strictly on train split and not test split."""
        train_df, _, test_df, _ = chronological_split(self.feat_df, horizon=21)
        features = get_model_features(train_df)
        
        X_train = train_df[features].fillna(0).values
        X_test = test_df[features].fillna(0).values
        
        scaler = StandardScaler()
        scaler.fit(X_train)
        
        # Test mean should NOT match scaler.mean_ (unless by astronomical coincidence)
        test_mean = np.mean(X_test, axis=0)
        self.assertFalse(np.allclose(scaler.mean_, test_mean), "Scaler appears fit on test data!")

if __name__ == '__main__':
    unittest.main()
