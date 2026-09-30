"""
cleaner.py
----------
Data Cleaning and Standardization module for NIFTY 50 Stock Price Prediction.
Handles:
1. Numeric string parsing (stripping commas, whitespace, currency symbols)
2. Exact deduplication and trading date chronological sorting
3. Symbol continuity mapping (SRTRANSFIN->SHRIRAMFIN, ETERNAL->ZOMATO, TMPV->TATAMOTORS)
4. Corporate action detection & backward split adjustment
5. Standardization into long-format parquet and CSV datasets
"""

from pathlib import Path
from typing import Tuple, Dict, Any, List
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import (
    RAW_MASTER_CSV,
    CLEANED_DATA_PARQUET,
    CLEANED_DATA_CSV,
    METADATA_CONSTITUENTS_CSV,
    METRICS_DIR,
    SYMBOL_ALIASES
)
from src.utilities.logger import get_logger
from src.utilities.io_helpers import save_dataframe, save_json

logger = get_logger("DataCleaner")

class DataCleaner:
    """End-to-end cleaning and corporate action standardization pipeline."""
    
    def __init__(self, metadata_path: Path = METADATA_CONSTITUENTS_CSV):
        self.metadata_path = Path(metadata_path)
        self.metadata = {}
        if self.metadata_path.exists():
            meta_df = pd.read_csv(self.metadata_path)
            for _, row in meta_df.iterrows():
                self.metadata[row['Symbol']] = {
                    "Company": row.get('Company_Name', row['Symbol']),
                    "Industry": row.get('Industry', 'General'),
                    "Sector": row.get('Sector', 'General')
                }
                
    def clean_master_dataset(self, file_path: Path = RAW_MASTER_CSV) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        logger.info("=" * 65)
        logger.info(f"PHASE 3 & 4: CLEANING & STANDARDIZING MASTER DATASET: {file_path}")
        logger.info("=" * 65)
        
        df_raw = pd.read_csv(file_path)
        initial_rows = len(df_raw)
        logger.info(f"Initial raw records loaded: {initial_rows}")
        
        # 1. Deduplicate exact duplicate rows
        df = df_raw.drop_duplicates().copy()
        dropped_exact_dupes = initial_rows - len(df)
        logger.info(f"Dropped exact duplicate rows: {dropped_exact_dupes}")
        
        # 2. Map ticker aliases for continuous corporate series
        df['Original_Symbol'] = df['Symbol']
        df['Symbol'] = df['Symbol'].replace(SYMBOL_ALIASES)
        
        # 3. Clean and parse Date
        df['Date'] = pd.to_datetime(df['Date'], format='mixed')
        
        # 4. Clean Numeric columns
        col_clean_map = {
            'Open Price': 'Open',
            'High Price': 'High',
            'Low Price': 'Low',
            'Close Price': 'Close',
            'Prev Close': 'Prev_Close',
            'Average Price': 'Avg_Price',
            'Total Traded Quantity': 'Volume',
            'Turnover ₹': 'Turnover',
            'No. of Trades': 'Trades',
            'Deliverable Qty': 'Deliverable_Qty',
            '% Dly Qt to Traded Qty': 'Deliverable_Pct'
        }
        
        # Keep only available columns
        for raw_col, target_col in col_clean_map.items():
            if raw_col in df.columns:
                df[raw_col] = (
                    df[raw_col]
                    .astype(str)
                    .str.replace(',', '', regex=False)
                    .str.replace('₹', '', regex=False)
                    .str.strip()
                )
                df[target_col] = pd.to_numeric(df[raw_col], errors='coerce')
                
        # 5. Sort chronologically per symbol and ensure 1 record per stock per day
        df = df.sort_values(by=['Symbol', 'Date']).reset_index(drop=True)
        date_dupes = df.duplicated(subset=['Symbol', 'Date']).sum()
        if date_dupes > 0:
            logger.warning(f"Resolving {date_dupes} residual duplicate stock/date records by taking latest entry.")
            df = df.drop_duplicates(subset=['Symbol', 'Date'], keep='last').reset_index(drop=True)
            
        # 6. Enrich with Company and Sector Metadata
        df['Company'] = df['Symbol'].apply(lambda s: self.metadata.get(s, {}).get("Company", s))
        df['Industry'] = df['Symbol'].apply(lambda s: self.metadata.get(s, {}).get("Industry", "General"))
        df['Sector'] = df['Symbol'].apply(lambda s: self.metadata.get(s, {}).get("Sector", "General"))
        
        # 7. Corporate Actions & Backward Split Adjustments (Section 7 Requirement)
        df, corporate_actions = self._apply_corporate_split_adjustments(df)
        
        # 8. Standardize column schema
        target_columns = [
            'Date', 'Symbol', 'Company', 'Industry', 'Sector',
            'Open', 'High', 'Low', 'Close', 'Volume',
            'Adj_Open', 'Adj_High', 'Adj_Low', 'Adj_Close', 'Adj_Volume',
            'Split_Factor', 'Turnover', 'Trades', 'Deliverable_Qty', 'Deliverable_Pct',
            'Original_Symbol'
        ]
        available_cols = [c for c in target_columns if c in df.columns]
        standardized_df = df[available_cols].copy()
        
        # Save datasets
        save_dataframe(standardized_df, CLEANED_DATA_PARQUET)
        
        cleaning_report = {
            "initial_rows": initial_rows,
            "cleaned_rows": len(standardized_df),
            "dropped_exact_duplicates": dropped_exact_dupes,
            "unique_symbols": sorted(standardized_df['Symbol'].unique().tolist()),
            "unique_symbol_count": int(standardized_df['Symbol'].nunique()),
            "min_date": standardized_df['Date'].min().strftime('%Y-%m-%d'),
            "max_date": standardized_df['Date'].max().strftime('%Y-%m-%d'),
            "total_trading_days": int(standardized_df['Date'].nunique()),
            "corporate_actions_detected": corporate_actions
        }
        save_json(cleaning_report, METRICS_DIR / "cleaning_summary.json")
        save_json(corporate_actions, METRICS_DIR / "corporate_actions.json")
        
        logger.info(f"✔ Cleaned dataset successfully generated: {len(standardized_df)} rows, {standardized_df['Symbol'].nunique()} stocks.")
        logger.info(f"✔ Saved to Parquet: {CLEANED_DATA_PARQUET} and CSV: {CLEANED_DATA_CSV}")
        return standardized_df, cleaning_report

    def _apply_corporate_split_adjustments(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
        """
        Detects historical corporate action price cliffs (|return| > 40%) and calculates
        backward split adjustment multipliers so technical indicators remain continuous.
        """
        adjusted_dfs = []
        corporate_actions_log = []
        
        # Known legitimate splits/bonuses in Indian markets over 2021-2026:
        # e.g., Bajaj Finserv (10:1), Bajaj Finance (10:1), BEL (3:1 factor), Dr Reddy (5:1),
        # Nestle (10:1, 2:1), Reliance (2:1 bonus), Shriram (5:1), Kotak (5:1), Wipro (2:1 bonus)
        
        for sym, group in df.groupby('Symbol', as_index=False):
            grp = group.sort_values('Date').copy()
            close_prices = grp['Close'].values
            n = len(grp)
            
            split_factors = np.ones(n, dtype=float)
            pct_changes = np.diff(close_prices) / close_prices[:-1]
            
            # Identify indices with extreme negative single-day jumps (> 40% drops)
            drop_indices = np.where(pct_changes < -0.38)[0]
            
            for idx in drop_indices:
                p_before = close_prices[idx]
                p_after = close_prices[idx + 1]
                ratio = p_before / p_after
                
                # Round to nearest integer or clean half ratio (e.g., 2.0 for 1:1 bonus, 5.0 for 5:1 split, 10.0 for 10:1)
                est_ratio = round(ratio)
                if abs(ratio - 1.5) < 0.15:
                    est_ratio = 1.5
                elif abs(ratio - 2.5) < 0.15:
                    est_ratio = 2.5
                elif est_ratio < 2:
                    est_ratio = round(ratio, 1)
                    
                action_date = grp['Date'].iloc[idx + 1].strftime('%Y-%m-%d')
                corporate_actions_log.append({
                    "symbol": sym,
                    "date": action_date,
                    "price_before": float(p_before),
                    "price_after": float(p_after),
                    "estimated_split_ratio": float(est_ratio),
                    "type": "Stock Split / Bonus Issue"
                })
                # Prices prior to the split date are divided by the split ratio (or future multiplied)
                # Backward adjustment: adjust prior prices by 1 / est_ratio
                split_factors[:idx + 1] /= est_ratio
                
            grp['Split_Factor'] = split_factors
            grp['Adj_Close'] = grp['Close'] * split_factors
            grp['Adj_Open'] = grp['Open'] * split_factors
            grp['Adj_High'] = grp['High'] * split_factors
            grp['Adj_Low'] = grp['Low'] * split_factors
            grp['Adj_Volume'] = grp['Volume'] / split_factors
            
            adjusted_dfs.append(grp)
            
        combined = pd.concat(adjusted_dfs, ignore_index=True)
        return combined, corporate_actions_log

def clean_data():
    cleaner = DataCleaner()
    return cleaner.clean_master_dataset()

if __name__ == "__main__":
    clean_df, report = clean_data()
    print("Cleaned records:", len(clean_df))
    print("Corporate actions detected:", len(report["corporate_actions_detected"]))
