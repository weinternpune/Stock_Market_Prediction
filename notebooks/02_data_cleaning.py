"""
02_data_cleaning.py
-------------------
Notebook runner for Phase 3 & 4: Data Cleaning & Corporate Actions.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.cleaner import clean_data

if __name__ == "__main__":
    print(">>> Executing Phase 3 & 4: Data Cleaning & Corporate Split Adjustments...")
    clean_df, report = clean_data()
    print(f"Cleaned records: {len(clean_df)}")
    print(f"Corporate actions adjusted: {len(report['corporate_actions_detected'])}")
