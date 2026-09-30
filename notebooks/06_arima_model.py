"""
06_arima_model.py
-----------------
Notebook runner for Phase 10: Statistical ARIMA Model.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import PROCESSED_DATA_PARQUET
from src.features.target_builder import chronological_split
from src.models.arima_model import ARIMAPredictor
from src.evaluation.evaluator import compute_evaluation_metrics
from src.utilities.io_helpers import load_dataframe

if __name__ == "__main__":
    print(">>> Executing Phase 10: ARIMA Model Evaluation...")
    df = load_dataframe(PROCESSED_DATA_PARQUET)
    sample_stock = df[df['Symbol'] == 'TCS'].copy()
    train_df, val_df, test_df, latest_row = chronological_split(sample_stock, horizon=21)
    
    arima = ARIMAPredictor(order=(1, 1, 1))
    arima.fit(train_df['Adj_Close'])
    preds = arima.predict_test(train_df['Adj_Close'], test_df['Adj_Close'], horizon=21)
    y_test = test_df['Future_Close_21D'].values
    metrics = compute_evaluation_metrics(y_test, preds)
    print("TCS ARIMA Test Metrics:", metrics)
