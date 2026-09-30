"""
08_lstm_model.py
----------------
Notebook runner for Phase 12: Deep Learning PyTorch LSTM Model.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
from config.project_config import PROCESSED_DATA_PARQUET
from src.features.target_builder import chronological_split, get_model_features
from src.models.lstm_model import LSTMPredictor
from src.evaluation.evaluator import compute_evaluation_metrics
from src.utilities.io_helpers import load_dataframe

if __name__ == "__main__":
    print(">>> Executing Phase 12: PyTorch LSTM Model Evaluation...")
    df = load_dataframe(PROCESSED_DATA_PARQUET)
    sample_stock = df[df['Symbol'] == 'HDFCBANK'].copy()
    train_df, val_df, test_df, latest_row = chronological_split(sample_stock, horizon=21)
    features = get_model_features(train_df)
    
    lstm = LSTMPredictor(lookback=20, epochs=5)
    lstm.fit(train_df[features], train_df['Future_Close_21D'])
    
    test_start = len(train_df) + len(val_df)
    comb = pd.concat([train_df, val_df, test_df], ignore_index=True)
    preds = lstm.predict_test(comb, test_start_idx=test_start, test_len=len(test_df))
    y_test = test_df['Future_Close_21D'].values
    metrics = compute_evaluation_metrics(y_test, preds)
    print("HDFCBANK LSTM Test Metrics:", metrics)
