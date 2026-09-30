"""
test_dashboard.py
-----------------
Unit tests verifying dashboard files and syntax across all 10 Streamlit pages.
"""

import unittest
import importlib.util
from pathlib import Path

class TestDashboard(unittest.TestCase):
    
    def setUp(self):
        self.app_dir = Path(__file__).resolve().parent.parent / "app"
        self.pages_dir = self.app_dir / "pages"
        
    def test_app_entrypoint_exists(self):
        """app/app.py must exist."""
        self.assertTrue((self.app_dir / "app.py").exists())

    def test_all_pages_exist(self):
        """All dual-universe multipage files must exist."""
        expected_pages = [
            "01_overview.py",
            "02_stock_analysis.py",
            "03_stock_ranking.py",
            "04_model_performance.py",
            "05_sector_analysis.py",
            "06_correlation_analysis.py",
            "07_backtesting.py",
            "08_future_predictions.py",
            "09_data_quality.py",
            "11_nifty500_overview.py",
            "12_nifty500_explorer.py",
            "13_nifty500_eda.py",
            "14_nifty500_features.py",
            "15_nifty500_models.py",
            "16_nifty500_backtest.py",
            "17_nifty500_forecast.py",
            "18_nifty500_deck.py"
        ]
        for p in expected_pages:
            p_path = self.pages_dir / p
            self.assertTrue(p_path.exists(), f"Page missing: {p}")

    def test_pages_python_syntax(self):
        """All page files should compile into valid Python code without syntax errors."""
        for py_file in self.pages_dir.glob("*.py"):
            with open(py_file, 'r', encoding='utf-8') as f:
                code = f.read()
            try:
                compile(code, str(py_file), 'exec')
            except SyntaxError as e:
                self.fail(f"SyntaxError in {py_file.name}: {e}")

if __name__ == '__main__':
    unittest.main()
