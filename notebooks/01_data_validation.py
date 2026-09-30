"""
01_data_validation.py
---------------------
Notebook runner for Phase 2: Master Dataset Validation.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.validator import validate_master_data

if __name__ == "__main__":
    print(">>> Executing Phase 2: Master Dataset Validation...")
    report, _ = validate_master_data()
    print(f"Status: {report['validation_status']}")
    print(f"Total rows: {report['total_rows']}, Unique symbols: {report['unique_symbols_count']}")
