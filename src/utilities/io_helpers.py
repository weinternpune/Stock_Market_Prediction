"""
io_helpers.py
-------------
File I/O helpers for Parquet, CSV, JSON, and model artifact serialization.
"""

from pathlib import Path
import json
import pandas as pd
import joblib

def ensure_dir(path: Path) -> Path:
    """Ensures directory exists and returns it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_dataframe(df: pd.DataFrame, file_base_path: Path):
    """Saves DataFrame as Parquet (if pyarrow/fastparquet available) and CSV."""
    file_base_path = Path(file_base_path)
    ensure_dir(file_base_path.parent)
    
    csv_path = file_base_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)
    
    parquet_path = file_base_path.with_suffix('.parquet')
    try:
        df.to_parquet(parquet_path, index=False, engine='pyarrow')
    except Exception:
        pass

def load_dataframe(file_path: Path) -> pd.DataFrame:
    """Loads DataFrame from Parquet if available, otherwise CSV."""
    file_path = Path(file_path)
    parquet_path = file_path.with_suffix('.parquet')
    csv_path = file_path.with_suffix('.csv')
    
    if parquet_path.exists():
        try:
            return pd.read_parquet(parquet_path)
        except Exception:
            pass
    if csv_path.exists():
        return pd.read_csv(csv_path)
    elif file_path.exists():
        if file_path.suffix == '.parquet':
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)
    raise FileNotFoundError(f"Neither {parquet_path} nor {csv_path} found.")

def save_json(data: dict, file_path: Path):
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, default=str)

def load_json(file_path: Path) -> dict:
    file_path = Path(file_path)
    if not file_path.exists():
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_joblib(obj, file_path: Path):
    file_path = Path(file_path)
    ensure_dir(file_path.parent)
    joblib.dump(obj, file_path)

def load_joblib(file_path: Path):
    file_path = Path(file_path)
    return joblib.load(file_path)
