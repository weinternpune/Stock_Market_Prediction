"""
pipeline.py
-----------
Master End-to-End Execution Pipeline for Nifty 500 Stock Market Prediction.
Orchestrates Phases 1 through 47 in a single, reproducible script:
Raw Data -> Validation -> Cleaning -> Outliers -> Features -> Models -> Backtesting -> Scorecard -> Forecast.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure src/ is on sys.path
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_validation import validate_nse_dataset, validate_bse_dataset, reconcile_dates
from data_cleaning import clean_nse_dataset, clean_bse_dataset, investigate_outliers, save_master_datasets
from eda import perform_eda
from feature_engineering import engineer_features, get_train_test_split
from models.baseline import NaivePersistenceBaseline, MovingAverageBaseline
from models.arima_model import ARIMAPredictor
from models.ml_models import ClassicalMLManager
from models.lstm_model import LSTMPredictorManager
from models.forecast_service import ForecastService
from evaluate import evaluate_all_models


def run_complete_pipeline():
    print("\n" + "=" * 75)
    print(">>> NIFTY 500 STOCK MARKET PREDICTION: END-TO-END AUTOMATED PIPELINE")
    print("=" * 75 + "\n")
    
    # Paths
    raw_dir = PROJECT_ROOT / "data" / "raw"
    processed_dir = PROJECT_ROOT / "data" / "processed"
    features_dir = PROJECT_ROOT / "data" / "features"
    models_dir = PROJECT_ROOT / "models"
    saved_models_dir = models_dir / "saved_models"
    reports_dir = PROJECT_ROOT / "reports"
    
    nse_raw_path = raw_dir / "nse_nifty500_raw.csv"
    bse_raw_path = raw_dir / "bse_500_raw.csv"
    
    # -------------------------------------------------------------
    # Step 1: Validation (Phases 6-8)
    # -------------------------------------------------------------
    nse_val_report, nse_raw_df = validate_nse_dataset(nse_raw_path)
    bse_val_report, bse_raw_df = validate_bse_dataset(bse_raw_path)
    rec_report = reconcile_dates(nse_raw_df, bse_raw_df)
    
    # -------------------------------------------------------------
    # Step 2: Cleaning & Outlier Audit (Phases 9-11)
    # -------------------------------------------------------------
    nse_clean = clean_nse_dataset(nse_raw_path)
    bse_clean = clean_bse_dataset(bse_raw_path)
    outlier_df = investigate_outliers(nse_clean, bse_clean, models_dir)
    save_master_datasets(nse_clean, bse_clean, processed_dir)
    
    # -------------------------------------------------------------
    # Step 3: Exploratory Data Analysis (Phases 12-16)
    # -------------------------------------------------------------
    eda_summary = perform_eda(processed_dir / "NIFTY500_clean.csv", 
                              processed_dir / "BSE500_clean.csv", 
                              reports_dir)
    
    # -------------------------------------------------------------
    # Step 4: Feature Engineering & Target Split (Phases 17-27)
    # -------------------------------------------------------------
    model_df, forecast_row = engineer_features(nse_clean)
    model_df.to_csv(features_dir / "nifty_500_features.csv", index=False)
    train_df, test_df = get_train_test_split(model_df, test_sessions=208)
    
    # -------------------------------------------------------------
    # Step 5: Model Training & Prediction (Phases 28-40)
    # -------------------------------------------------------------
    predictions = {}
    
    # 5.1 Naive Persistence Baseline (Phase 28)
    naive_model = NaivePersistenceBaseline()
    predictions["Naive Persistence"] = naive_model.predict(test_df)
    
    # 5.2 5-Day Moving Average Baseline (Phase 29)
    ma_model = MovingAverageBaseline(window=5)
    predictions["Moving Average (5-Day SMA)"] = ma_model.predict(model_df, len(test_df))
    
    # 5.3 ARIMA(1, 1, 1) Walk-Forward (Phase 30-31)
    arima = ARIMAPredictor(order=(1, 1, 1))
    predictions["ARIMA(1, 1, 1) Walk-Forward"] = arima.fit_and_predict(train_df['Close'], test_df['Close'])
    
    # 5.4 Classical ML: Random Forest & XGBoost (Phase 32-35)
    ml_mgr = ClassicalMLManager(saved_models_dir)
    ml_results = ml_mgr.train_and_predict(train_df, test_df)
    predictions["Random Forest"] = ml_results["rf_preds"]
    predictions["XGBoost"] = ml_results["xgb_preds"]
    
    # 5.5 Deep Learning: PyTorch LSTM (Phase 36-40)
    lstm_mgr = LSTMPredictorManager(saved_models_dir, lookback=20)
    predictions["LSTM Network"] = lstm_mgr.train_and_predict(model_df, train_df, test_df)
    
    # -------------------------------------------------------------
    # Step 6: Evaluation & Master Scorecard (Phases 41-46)
    # -------------------------------------------------------------
    scorecard_df = evaluate_all_models(test_df, predictions, models_dir, processed_dir)
    
    # -------------------------------------------------------------
    # Step 7: Forward Forecasting (Phase 47)
    # -------------------------------------------------------------
    forecast_srv = ForecastService(models_dir)
    latest_feat = forecast_row.iloc[0]
    last_close = float(nse_clean['Close'].iloc[-1])
    forward_df = forecast_srv.generate_forecast(latest_feat, last_close, horizon_days=30)
    forward_df.to_csv(processed_dir / "future_forecast_t30.csv", index=False)
    print("\n" + "=" * 70)
    print("PHASE 47: 30-DAY FORWARD FORECAST GENERATED (2026-09-01 to 2026-10-12)")
    print("=" * 70)
    print(forward_df.head(5).to_string(index=False))
    print(f">> Future forecast saved to: {processed_dir / 'future_forecast_t30.csv'}")
    
    print("\n" + "=" * 75)
    print(">>> END-TO-END PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 75)


if __name__ == "__main__":
    run_complete_pipeline()
