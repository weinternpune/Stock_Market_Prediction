"""
07_xgboost_model.py
-------------------
Notebook runner for Phase 11: Classical ML XGBoost Model.
"""

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config.project_config import PROCESSED_DATA_PARQUET
from src.features.target_builder import chronological_split, get_model_features
from src.models.xgboost_model import XGBoostPredictor
from src.evaluation.evaluator import compute_evaluation_metrics
from src.utilities.io_helpers import load_dataframe

if __name__ == "__main__":
    print(">>> Executing Phase 11: XGBoost Model Evaluation...")
    df = load_dataframe(PROCESSED_DATA_PARQUET)
    sample_stock = df[df['Symbol'] == 'INFY'].copy()
    train_df, val_df, test_df, latest_row = chronological_split(sample_stock, horizon=21)
    features = get_model_features(train_df)
    
    xgb = XGBoostPredictor()
    xgb.fit(train_df[features], train_df['Future_Close_21D'], val_df[features], val_df['Future_Close_21D'])
    preds = xgb.predict(test_df[features])
    y_test = test_df['Future_Close_21D'].values
    metrics = compute_evaluation_metrics(y_test, preds)
    print("INFY XGBoost Test Metrics:", metrics)
    print("Top 5 Features:", list(xgb.feature_importances.items())[:5])
