"""
03_eda.py
---------
Notebook runner for Phase 5: Exploratory Data Analysis.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eda import run_eda

if __name__ == "__main__":
    print(">>> Executing Phase 5: Exploratory Data Analysis...")
    report = run_eda()
    print(f"EDA completed across {report['total_stocks']} stocks.")
    print(f"Average 5-Year Cumulative Return: {report['average_cumulative_return_pct']}%")
