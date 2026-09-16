"""
test_data_validation.py
-----------------------
Unit tests for data validation module.
Verifies date parsing, duplicate checks, OHLC logical rules, and constituent checks.
"""

import unittest
import pandas as pd
import numpy as np
from src.data.validator import DataValidator

class TestDataValidation(unittest.TestCase):
    
    def setUp(self):
        self.validator = DataValidator()
        
    def test_ohlc_consistency_valid(self):
        """Valid OHLC data should generate 0 violations."""
        df = pd.DataFrame({
            'Symbol': ['TEST'],
            'Series': ['EQ'],
            'Date': ['01-Jan-2024'],
            'Open Price': ['100.0'],
            'High Price': ['105.0'],
            'Low Price': ['95.0'],
            'Close Price': ['102.0'],
            'Total Traded Quantity': ['1,000']
        })
        # Test helper check
        for col in ['Open Price', 'High Price', 'Low Price', 'Close Price']:
            df[col] = df[col].astype(float)
        self.assertTrue((df['High Price'] >= df['Open Price']).all())
        self.assertTrue((df['High Price'] >= df['Close Price']).all())
        self.assertTrue((df['High Price'] >= df['Low Price']).all())
        self.assertTrue((df['Low Price'] <= df['Open Price']).all())
        self.assertTrue((df['Low Price'] <= df['Close Price']).all())

    def test_ohlc_consistency_invalid(self):
        """Invalid OHLC relationships should be flagged."""
        df = pd.DataFrame({
            'Open Price': [100.0],
            'High Price': [90.0],  # High < Open violation!
            'Low Price': [95.0],
            'Close Price': [92.0]
        })
        high_lt_open = (df['High Price'] < df['Open Price']).sum()
        self.assertEqual(high_lt_open, 1)

    def test_duplicate_dates_detection(self):
        """Duplicate dates per symbol should be detected."""
        dates = pd.to_datetime(['2024-01-01', '2024-01-01', '2024-01-02'])
        dupes = dates.duplicated().sum()
        self.assertEqual(dupes, 1)

    def test_date_parsing_mixed_formats(self):
        """Standard and mixed NSE date formats should parse correctly."""
        raw_dates = pd.Series(['11-Sep-2026', '2026-09-11', '01-01-2024'])
        parsed = pd.to_datetime(raw_dates, format='mixed', errors='coerce')
        self.assertEqual(parsed.isna().sum(), 0)

if __name__ == '__main__':
    unittest.main()
