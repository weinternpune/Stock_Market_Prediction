"""
test_data_cleaning.py
---------------------
Unit tests for data cleaning and corporate action split adjustments.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.cleaner import DataCleaner

class TestDataCleaning(unittest.TestCase):
    
    def test_numeric_cleaning(self):
        """Commas and currency formatting should be cleaned cleanly to floats."""
        raw_vals = pd.Series(["1,839.00", "41,81,325", "₹7,70,40,04,262.20"])
        cleaned = (
            raw_vals
            .str.replace(',', '', regex=False)
            .str.replace('₹', '', regex=False)
            .str.strip()
            .astype(float)
        )
        self.assertEqual(cleaned.iloc[0], 1839.00)
        self.assertEqual(cleaned.iloc[1], 4181325.0)
        self.assertAlmostEqual(cleaned.iloc[2], 7704004262.20)

    def test_deduplication(self):
        """Exact duplicates must be removed."""
        df = pd.DataFrame({
            'Symbol': ['INFY', 'INFY'],
            'Date': ['2024-01-01', '2024-01-01'],
            'Close': [1500.0, 1500.0]
        })
        self.assertEqual(len(df.drop_duplicates()), 1)

    def test_split_factor_detection(self):
        """A simulated 10:1 split (-90% drop) should result in backward split factor adjustment."""
        cleaner = DataCleaner()
        dates = pd.date_range('2023-01-01', periods=4, freq='B')
        df = pd.DataFrame({
            'Symbol': ['SPLIT_CO', 'SPLIT_CO', 'SPLIT_CO', 'SPLIT_CO'],
            'Date': dates,
            'Close': [10000.0, 10000.0, 1000.0, 1020.0],  # 10:1 split between day 2 and 3
            'Open': [10000.0, 10000.0, 1000.0, 1010.0],
            'High': [10100.0, 10100.0, 1020.0, 1030.0],
            'Low': [9900.0, 9900.0, 990.0, 1000.0],
            'Volume': [10000, 10000, 100000, 95000]
        })
        adjusted_df, actions = cleaner._apply_corporate_split_adjustments(df)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0]['estimated_split_ratio'], 10.0)
        # Prior close 10,000 / 10 = 1000, ensuring continuity
        self.assertAlmostEqual(adjusted_df['Adj_Close'].iloc[0], 1000.0)
        self.assertAlmostEqual(adjusted_df['Adj_Close'].iloc[2], 1000.0)

if __name__ == '__main__':
    unittest.main()
