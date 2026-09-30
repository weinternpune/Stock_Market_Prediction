"""
validator.py
------------
Data Validation module for NIFTY 50 Stock Price Prediction System.
Performs Phase 2 comprehensive dataset-level, symbol-level, and OHLC-level validation.
"""

from pathlib import Path
from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import (
    METADATA_CONSTITUENTS_CSV,
    METRICS_DIR,
    SYMBOL_ALIASES
)
from src.utilities.logger import get_logger
from src.utilities.io_helpers import save_json

logger = get_logger("DataValidator")

class DataValidator:
    """Automated validator for the NIFTY 50 historical master dataset."""
    
    def __init__(self, metadata_path: Path = METADATA_CONSTITUENTS_CSV):
        self.metadata_path = Path(metadata_path)
        self.expected_symbols = set()
        if self.metadata_path.exists():
            meta_df = pd.read_csv(self.metadata_path)
            self.expected_symbols = set(meta_df['Symbol'].str.strip().tolist())
            
    def validate_raw_dataset(self, file_path: Path) -> Tuple[Dict[str, Any], pd.DataFrame]:
        logger.info("=" * 65)
        logger.info(f"PHASE 2: VALIDATING MASTER DATASET: {file_path}")
        logger.info("=" * 65)
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"Master dataset not found at {file_path}")
            
        # Read raw CSV
        df = pd.read_csv(file_path)
        report: Dict[str, Any] = {
            "file_name": file_path.name,
            "total_rows": len(df),
            "total_columns": len(df.columns),
            "columns": list(df.columns),
            "warnings": [],
            "errors": [],
            "checks": {}
        }
        
        # 1. Dataset-level checks
        unique_symbols = sorted(df['Symbol'].dropna().unique().tolist())
        report["unique_symbols_count"] = len(unique_symbols)
        report["unique_symbols"] = unique_symbols
        
        # 2. Date checks
        parsed_dates = pd.to_datetime(df['Date'], format='mixed', errors='coerce')
        invalid_dates = int(parsed_dates.isna().sum())
        report["checks"]["invalid_dates_count"] = invalid_dates
        if invalid_dates > 0:
            report["errors"].append(f"Found {invalid_dates} unparseable dates in dataset.")
            
        min_date = parsed_dates.min().strftime('%Y-%m-%d')
        max_date = parsed_dates.max().strftime('%Y-%m-%d')
        report["date_range"] = {"min_date": min_date, "max_date": max_date}
        
        # Unique trading dates across universe
        unique_trading_days = parsed_dates.nunique()
        report["unique_trading_days"] = unique_trading_days
        
        # 3. Exact Duplicate Rows
        exact_duplicates = int(df.duplicated().sum())
        report["checks"]["exact_duplicate_rows"] = exact_duplicates
        if exact_duplicates > 0:
            report["warnings"].append(f"Found {exact_duplicates} exact duplicate rows. These will be dropped during cleaning.")
            
        # 4. Duplicate Dates per Symbol
        dupes_per_sym = df.groupby('Symbol')['Date'].apply(lambda s: s.duplicated().sum()).to_dict()
        flagged_sym_dupes = {s: int(c) for s, c in dupes_per_sym.items() if c > 0}
        report["checks"]["duplicate_dates_per_symbol"] = flagged_sym_dupes
        if flagged_sym_dupes:
            report["warnings"].append(f"Symbols with duplicate dates: {flagged_sym_dupes}")
            
        # 5. Missing Values Check
        missing_counts = {col: int(cnt) for col, cnt in df.isna().sum().items() if cnt > 0}
        report["checks"]["missing_values"] = missing_counts
        if missing_counts:
            report["warnings"].append(f"Columns with missing values: {missing_counts}")
        else:
            report["checks"]["missing_values"] = "0 missing values across all raw columns."
            
        # 6. Expected 50 NIFTY 50 Symbols Check (Section 5 Requirement)
        raw_symbol_set = set(unique_symbols)
        
        # Canonicalize with aliases (e.g. SRTRANSFIN -> SHRIRAMFIN, ETERNAL -> ZOMATO, TMPV -> TATAMOTORS)
        mapped_symbols = set(SYMBOL_ALIASES.get(s, s) for s in raw_symbol_set)
        
        matched_symbols = self.expected_symbols.intersection(mapped_symbols)
        missing_symbols = sorted(list(self.expected_symbols - mapped_symbols))
        extra_symbols = sorted(list(raw_symbol_set - self.expected_symbols - set(SYMBOL_ALIASES.keys())))
        
        report["expected_symbols_count"] = len(self.expected_symbols)
        report["matched_symbols_count"] = len(matched_symbols)
        report["missing_symbols"] = missing_symbols
        report["extra_symbols"] = extra_symbols
        report["corporate_continuity_aliases_detected"] = {
            s: SYMBOL_ALIASES[s] for s in raw_symbol_set if s in SYMBOL_ALIASES
        }
        
        if missing_symbols:
            msg = f"Constituent check: {len(missing_symbols)} expected NIFTY 50 symbols not in dataset: {missing_symbols}."
            report["warnings"].append(msg)
            logger.warning(msg)
            
        if extra_symbols:
            msg = f"Constituent check: {len(extra_symbols)} extra symbols present in raw dataset: {extra_symbols}."
            report["warnings"].append(msg)
            logger.warning(msg)
            
        # 7. Numeric Price & OHLC Consistency Checks (sampling cleaned numbers)
        numeric_cols = ['Open Price', 'High Price', 'Low Price', 'Close Price']
        clean_num_df = df.copy()
        for col in numeric_cols:
            clean_num_df[col] = (
                clean_num_df[col]
                .astype(str)
                .str.replace(',', '')
                .str.strip()
                .astype(float)
            )
            
        # Negative or zero prices
        non_positive_prices = {}
        for col in numeric_cols:
            cnt = int((clean_num_df[col] <= 0).sum())
            if cnt > 0:
                non_positive_prices[col] = cnt
                report["errors"].append(f"Column '{col}' has {cnt} non-positive values.")
        report["checks"]["non_positive_prices"] = non_positive_prices
        
        # OHLC logic check
        high_lt_open = int((clean_num_df['High Price'] < clean_num_df['Open Price']).sum())
        high_lt_close = int((clean_num_df['High Price'] < clean_num_df['Close Price']).sum())
        high_lt_low = int((clean_num_df['High Price'] < clean_num_df['Low Price']).sum())
        low_gt_open = int((clean_num_df['Low Price'] > clean_num_df['Open Price']).sum())
        low_gt_close = int((clean_num_df['Low Price'] > clean_num_df['Close Price']).sum())
        
        ohlc_violations = {
            "High < Open": high_lt_open,
            "High < Close": high_lt_close,
            "High < Low": high_lt_low,
            "Low > Open": low_gt_open,
            "Low > Close": low_gt_close
        }
        total_violations = sum(ohlc_violations.values())
        report["checks"]["ohlc_logical_violations"] = ohlc_violations
        report["checks"]["ohlc_total_violations"] = total_violations
        
        if total_violations > 0:
            report["errors"].append(f"Found {total_violations} OHLC logical consistency violations.")
            logger.warning(f"OHLC logical violations: {ohlc_violations}")
        else:
            report["checks"]["ohlc_status"] = "PASS: All OHLC records logically consistent."
            logger.info("✔ OHLC logical consistency: 100% Valid (0 violations).")
            
        # Status determination
        if not report["errors"]:
            report["validation_status"] = "PASS"
            logger.info("✔ Data Validation Status: PASS")
        else:
            report["validation_status"] = "WARNINGS_PRESENT"
            logger.warning(f"Data Validation completed with {len(report['errors'])} errors and {len(report['warnings'])} warnings.")
            
        # Save validation report
        save_json(report, METRICS_DIR / "validation_report.json")
        return report, df

def validate_master_data(file_path: Path = None):
    from config.project_config import RAW_MASTER_CSV
    target_path = file_path or RAW_MASTER_CSV
    validator = DataValidator()
    return validator.validate_raw_dataset(target_path)

if __name__ == "__main__":
    report, _ = validate_master_data()
    print("Validation Report Status:", report["validation_status"])
    print(f"Total rows: {report['total_rows']}, Unique symbols: {report['unique_symbols_count']}")
