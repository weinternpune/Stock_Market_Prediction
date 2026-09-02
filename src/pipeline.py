"""
End-to-End Execution Pipeline for Nifty 500 Stock Prediction.
Orchestrates data ingestion, cleaning, feature engineering, modeling, and evaluation.
"""

from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from src.data_collection import collect_and_save_data
from src.data_preprocessing import preprocess_pipeline
from src.feature_engineering import run_feature_pipeline
from src.models.baseline import NaiveBaselineModel, MovingAverageBaselineModel
from src.models.arima_model import ArimaForecaster, check_stationarity
from src.models.ml_models import MLForecastingSuite
from src.models.lstm_model import LSTMForecaster
from src.evaluate import calculate_metrics, generate_comparison_table, save_metrics_summary

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"


def run_complete_pipeline():
    print("\n" + "="*70)
    print("STARTING NIFTY 500 PREDICTION PIPELINE (PRD v1.1)")
    print("="*70 + "\n")

    # Step 1: Data Collection & Preprocessing
    print(">>> STEP 1 & 2: Ingesting and Preprocessing 5Y Market Data...")
    df_clean, df_bse_clean, recon_summary = preprocess_pipeline()

    # Step 2: Feature Engineering
    print("\n>>> STEP 3: Engineering Financial & Technical Indicators...")
    features_df = run_feature_pipeline()

    # Step 3: Chronological Train-Test Split (80% Train, 20% Test)
    print("\n>>> STEP 4: Chronological Train-Test Split...")
    n_samples = len(features_df)
    train_size = int(n_samples * 0.80)
    train_df = features_df.iloc[:train_size].copy()
    test_df = features_df.iloc[train_size:].copy()

    print(f"Total Samples: {n_samples}")
    print(f"Train Set: {len(train_df)} samples ({train_df['date'].iloc[0]} to {train_df['date'].iloc[-1]})")
    print(f"Test Set:  {len(test_df)} samples ({test_df['date'].iloc[0]} to {test_df['date'].iloc[-1]})")

    # Step 4: Baseline Models
    print("\n>>> STEP 5: Benchmarking Baseline Models...")
    naive_model = NaiveBaselineModel()
    pred_naive = naive_model.predict(test_df)

    ma_model = MovingAverageBaselineModel(window=5)
    pred_ma5 = ma_model.predict(test_df)

    # Step 5: Statistical Model (ARIMA)
    print("\n>>> STEP 6: Fitting Statistical Model (ARIMA)...")
    adf_res = check_stationarity(train_df["close"], "Nifty 500 Train Close")
    print(f"ADF Test on Level Prices: ADF Stat={adf_res['adf_statistic']:.3f}, p-value={adf_res['p_value']:.4f} -> {adf_res['conclusion']}")

    adf_diff = check_stationarity(train_df["close"].diff().dropna(), "Nifty 500 1st Difference")
    print(f"ADF Test on 1st Difference: ADF Stat={adf_diff['adf_statistic']:.3f}, p-value={adf_diff['p_value']:.4e} -> {adf_diff['conclusion']}")

    arima = ArimaForecaster(order=(1, 1, 1))
    arima.fit(train_df["close"])
    pred_arima = arima.predict_test(test_df["close"])

    # Step 6: Classical Machine Learning (Random Forest & XGBoost)
    print("\n>>> STEP 7: Training Classical ML Models (Random Forest & XGBoost)...")
    ml_suite = MLForecastingSuite(random_state=42)
    X_train, y_train = ml_suite.prepare_data(train_df, target_col="target_close")
    X_test, y_test = ml_suite.prepare_data(test_df, target_col="target_close")

    ml_suite.fit(X_train, y_train)
    pred_rf = ml_suite.predict_rf(X_test)
    pred_xgb = ml_suite.predict_xgb(X_test)

    ml_suite.save_models(SAVED_MODELS_DIR)

    fi_df = ml_suite.get_feature_importances()
    fi_path = MODELS_DIR / "feature_importances.csv"
    fi_df.to_csv(fi_path, index=False)
    print(f"Top 5 Most Important Features:\n{fi_df.head(5)[['feature', 'mean_importance']]}")

    # Step 7: Deep Learning (PyTorch LSTM)
    print("\n>>> STEP 8: Training Deep Learning Sequence Model (PyTorch LSTM)...")
    # Features for LSTM: key technical and price indicators
    lstm_feature_cols = [
        "close", "open", "high", "low", "volume",
        "sma_20", "sma_50", "ema_20", "rsi_14", "macd_line",
        "bb_upper", "bb_lower", "volatility_20d", "return_1d"
    ]
    full_X = features_df[lstm_feature_cols].values
    full_y = features_df["target_close"].values

    X_train_lstm = full_X[:train_size]
    y_train_lstm = full_y[:train_size]

    lstm_forecaster = LSTMForecaster(
        seq_len=20,
        hidden_size=64,
        num_layers=2,
        lr=0.002,
        epochs=50,
        batch_size=32
    )

    # Use tail of training set for validation
    val_split = int(len(X_train_lstm) * 0.85)
    lstm_forecaster.fit(
        X_train_lstm[:val_split], y_train_lstm[:val_split],
        X_val=X_train_lstm[val_split:], y_val=y_train_lstm[val_split:]
    )

    test_indices = np.arange(train_size, n_samples)
    pred_lstm = lstm_forecaster.predict(full_X, test_indices)
    lstm_forecaster.save_model(SAVED_MODELS_DIR)

    # Step 8: Evaluation & Comparison
    print("\n>>> STEP 9: Computing Evaluation Metrics & Scorecard...")
    y_true = test_df["target_close"].values
    y_current = test_df["close"].values

    results = {
        "Naive Baseline (Persistence)": calculate_metrics(y_true, pred_naive, y_current),
        "Moving Average (5-day SMA)": calculate_metrics(y_true, pred_ma5, y_current),
        "ARIMA(1,1,1)": calculate_metrics(y_true, pred_arima, y_current),
        "Random Forest Regressor": calculate_metrics(y_true, pred_rf, y_current),
        "XGBoost Regressor": calculate_metrics(y_true, pred_xgb, y_current),
        "LSTM Neural Network": calculate_metrics(y_true, pred_lstm, y_current),
    }

    scorecard_df = generate_comparison_table(results, baseline_name="Naive Baseline (Persistence)")
    print("\n" + "="*70)
    print("MODEL BENCHMARK SCORECARD")
    print("="*70)
    print(scorecard_df.to_string(index=False))
    print("="*70 + "\n")

    # Save metrics summary
    save_metrics_summary(results, MODELS_DIR / "metrics_summary.json")

    # Save test predictions for visualization in Streamlit and Notebook
    pred_df = pd.DataFrame({
        "date": test_df["date"].values,
        "actual_target": y_true,
        "current_close": y_current,
        "pred_naive": pred_naive,
        "pred_ma5": pred_ma5,
        "pred_arima": pred_arima,
        "pred_rf": pred_rf,
        "pred_xgb": pred_xgb,
        "pred_lstm": pred_lstm
    })
    pred_path = DATA_DIR / "processed" / "test_predictions.csv"
    pred_df.to_csv(pred_path, index=False)
    print(f"[PIPELINE] Saved backtest predictions to: {pred_path}")

    # Save pipeline metadata
    meta = {
        "total_samples": n_samples,
        "train_samples": len(train_df),
        "test_samples": len(test_df),
        "train_start_date": str(train_df["date"].iloc[0]),
        "train_end_date": str(train_df["date"].iloc[-1]),
        "test_start_date": str(test_df["date"].iloc[0]),
        "test_end_date": str(test_df["date"].iloc[-1]),
        "reconciliation": recon_summary,
        "features_count": len(ml_suite.FEATURE_COLS)
    }
    with open(MODELS_DIR / "pipeline_metadata.json", "w") as f:
        json.dump(meta, f, indent=4)

    print("\n[SUCCESS] PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    return results, scorecard_df


if __name__ == "__main__":
    run_complete_pipeline()
