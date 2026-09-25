"""
storage.py
----------
Storage and persistence component for Live Market Data.
Persists real-time snapshots (latest quote per stock) and maintains an
append-only deduplicated historical quote archive.
"""

from datetime import datetime
from pathlib import Path
from typing import List, Optional, Dict, Any
import pandas as pd

from src.live.models import MarketQuote
from config.live_config import (
    LIVE_DATA_DIR,
    LIVE_SNAPSHOT_CSV,
    LIVE_HISTORY_PARQUET,
    LIVE_HISTORY_CSV
)
from src.utilities.logger import get_logger

logger = get_logger("LiveQuoteStorage")


class LiveQuoteStorage:
    """
    Manages persistence of live market quotes:
    1. Latest Snapshot CSV: One latest row per stock for instant UI/API rendering.
    2. Historical Quotes Parquet/CSV: Append-only time-series archive with deduplication.
    """

    def __init__(
        self,
        storage_dir: Path = LIVE_DATA_DIR,
        snapshot_file: Path = LIVE_SNAPSHOT_CSV,
        history_parquet: Path = LIVE_HISTORY_PARQUET,
        history_csv: Path = LIVE_HISTORY_CSV
    ):
        self.storage_dir = storage_dir
        self.snapshot_file = snapshot_file
        self.history_parquet = history_parquet
        self.history_csv = history_csv

        # Ensure directory exists
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save_snapshot(self, quotes: List[MarketQuote]) -> Path:
        """
        Saves the current batch of quotes as the latest market snapshot.
        Replaces previous snapshot file atomically.
        """
        if not quotes:
            logger.warning("Empty quote list provided to save_snapshot.")
            return self.snapshot_file

        records = [q.to_dict() for q in quotes]
        df = pd.DataFrame(records)

        # Standard column order
        cols_order = [
            "Symbol", "Company", "Timestamp", "Current Price", "Change", "Change %",
            "Open", "High", "Low", "Previous Close", "Volume", "Day High", "Day Low",
            "Market Status", "is_demo", "data_source"
        ]
        available_cols = [c for c in cols_order if c in df.columns]
        df = df[available_cols]

        temp_path = self.snapshot_file.with_suffix(".tmp")
        df.to_csv(temp_path, index=False)
        temp_path.replace(self.snapshot_file)

        logger.info(f"Saved latest live snapshot ({len(df)} stocks) to {self.snapshot_file}")
        return self.snapshot_file

    def append_history(self, quotes: List[MarketQuote]) -> Path:
        """
        Appends new quotes to the historical log, enforcing deduplication on (Symbol, Timestamp).
        """
        if not quotes:
            return self.history_parquet

        new_records = [q.to_dict() for q in quotes]
        new_df = pd.DataFrame(new_records)

        if self.history_parquet.exists():
            try:
                existing_df = pd.read_parquet(self.history_parquet)
                combined = pd.concat([existing_df, new_df], ignore_index=True)
                # Deduplicate by Symbol and timestamp
                combined = combined.drop_duplicates(subset=["Symbol", "Timestamp"], keep="last")
            except Exception as e:
                logger.warning(f"Failed to read existing history parquet ({e}), writing fresh.")
                combined = new_df
        else:
            combined = new_df

        # Save to parquet (efficient compression) and CSV mirror
        combined.to_parquet(self.history_parquet, index=False)
        combined.to_csv(self.history_csv, index=False)

        logger.info(f"Appended {len(quotes)} quotes to history archive. Total records: {len(combined)}")
        return self.history_parquet

    def save_quotes(self, quotes: List[MarketQuote]) -> Dict[str, Path]:
        """Convenience method to save both the latest snapshot and historical log."""
        s_path = self.save_snapshot(quotes)
        h_path = self.append_history(quotes)
        return {"snapshot": s_path, "history": h_path}

    def load_latest_snapshot(self) -> Optional[pd.DataFrame]:
        """Loads the most recent live quotes snapshot dataframe."""
        if not self.snapshot_file.exists():
            logger.info(f"No live snapshot found at {self.snapshot_file}")
            return None
        return pd.read_csv(self.snapshot_file)

    def load_history(self, symbol: Optional[str] = None) -> Optional[pd.DataFrame]:
        """Loads historical live quote ticks, optionally filtered by symbol."""
        if not self.history_parquet.exists():
            return None
        df = pd.read_parquet(self.history_parquet)
        if symbol:
            df = df[df["Symbol"] == symbol.strip().upper()]
        return df
