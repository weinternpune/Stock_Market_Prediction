"""
05_baseline_model.py
--------------------
Notebook runner for Phase 9: Naive Baselines.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config.project_config import PROCESSED_DATA_PARQUET
from src.features.target_builder import chronological_split
from src.models.baseline import BaselinePredictor
from src.evaluation.evaluator import compute_evaluation_metrics
from src.utilities.io_helpers import load_dataframe

if __name__ == "__main__":
    print(">>> Executing Phase 9: Naive Baseline Evaluation...")
    df = load_dataframe(PROCESSED_DATA_PARQUET)
    sample_stock = df[df['Symbol'] == 'RELIANCE'].copy()
    train_df, val_df, test_df, latest_row = chronological_split(sample_stock, horizon=21)
    
    baseline = BaselinePredictor()
    preds = baseline.predict_persistence(test_df)
    y_test = test_df['Future_Close_21D'].values
    metrics = compute_evaluation_metrics(y_test, preds)
    print("RELIANCE Baseline Test Metrics:", metrics)
