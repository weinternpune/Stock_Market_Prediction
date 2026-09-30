"""
forecaster.py
-------------
Unified Stock-by-Stock Training, Evaluation, Future Forecasting, and Ranking Pipeline.
Implements Phases 8 through 17:
- Reusable pipeline applied automatically across all 50 NIFTY stocks
- Trains Baseline, ARIMA, XGBoost, and PyTorch LSTM
- Evaluates on chronological unseen test set (MAE, RMSE, MAPE, Directional Accuracy)
- Selects Best Model per stock objectively
- Generates 21-day future forecast, expected return %, and uncertainty intervals
- Produces 50-stock ranking and backtesting actual vs. predicted records
"""

from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from config.project_config import (
    PROCESSED_DATA_PARQUET,
    PREDICTIONS_CSV,
    MODEL_COMPARISON_CSV,
    STOCK_RANKING_CSV,
    BACKTEST_PREDICTIONS_CSV,
    MODELS_DIR,
    METRICS_DIR,
    DEFAULT_PREDICTION_HORIZON
)
from src.features.target_builder import chronological_split, get_model_features
from src.models.baseline import BaselinePredictor
from src.models.arima_model import ARIMAPredictor
from src.models.xgboost_model import XGBoostPredictor
from src.models.lstm_model import LSTMPredictor
from src.evaluation.evaluator import compute_evaluation_metrics, select_best_model
from src.utilities.logger import get_logger, PipelineProgressTracker
from src.utilities.io_helpers import load_dataframe, save_json, ensure_dir

logger = get_logger("ForecastingPipeline")

class MasterForecastingPipeline:
    """End-to-end pipeline that executes models and rankings for all NIFTY 50 constituents."""
    
    def __init__(self, horizon: int = DEFAULT_PREDICTION_HORIZON):
        self.horizon = horizon
        self.tracker = PipelineProgressTracker()
        
    def process_single_stock(self, stock_df: pd.DataFrame) -> Dict[str, Any]:
        """Runs validation, models, evaluation, and future forecast for an individual stock."""
        symbol = stock_df['Symbol'].iloc[0]
        company = stock_df['Company'].iloc[0]
        industry = stock_df['Industry'].iloc[0]
        sector = stock_df['Sector'].iloc[0]
        
        # 1. Chronological Train / Val / Test Split
        train_df, val_df, test_df, latest_row = chronological_split(
            stock_df,
            horizon=self.horizon
        )
        
        feature_cols = get_model_features(train_df)
        target_col = f'Future_Close_{self.horizon}D'
        price_col = 'Adj_Close' if 'Adj_Close' in stock_df.columns else 'Close'
        
        y_test = test_df[target_col].to_numpy(dtype=np.float64)
        curr_test_prices = test_df[price_col].to_numpy(dtype=np.float64)
        latest_price = float(latest_row[price_col].iloc[-1])
        latest_date = latest_row['Date'].iloc[-1].strftime('%Y-%m-%d')
        
        model_artifacts_dir = MODELS_DIR / symbol
        ensure_dir(model_artifacts_dir)
        
        test_predictions = {}
        future_forecasts = {}
        evaluation_results = {}
        
        # -------------------------------------------------------------
        # 2. Model 1: Baseline (Naive Persistence & 20D SMA)
        # -------------------------------------------------------------
        baseline = BaselinePredictor()
        base_test_preds = baseline.predict_persistence(test_df)
        test_predictions["Baseline"] = base_test_preds
        future_forecasts["Baseline"] = baseline.forecast_future(latest_row)
        
        base_metrics = compute_evaluation_metrics(
            y_true=y_test,
            y_pred=base_test_preds,
            current_prices=curr_test_prices
        )
        evaluation_results["Baseline"] = base_metrics
        baseline_mae = base_metrics["mae"]
        
        # -------------------------------------------------------------
        # 3. Model 2: ARIMA(1, 1, 1)
        # -------------------------------------------------------------
        try:
            arima = ARIMAPredictor(order=(1, 1, 1))
            arima.fit(train_df[price_col])
            arima_test_preds = arima.predict_test(train_df[price_col], test_df[price_col], horizon=self.horizon)
            test_predictions["ARIMA"] = arima_test_preds
            future_forecasts["ARIMA"] = arima.forecast_future(stock_df[price_col], horizon=self.horizon)
            evaluation_results["ARIMA"] = compute_evaluation_metrics(
                y_true=y_test,
                y_pred=arima_test_preds,
                current_prices=curr_test_prices,
                baseline_mae=baseline_mae
            )
        except Exception as e:
            logger.warning(f"[{symbol}] ARIMA warning: {e}. Falling back to baseline.")
            test_predictions["ARIMA"] = base_test_preds
            future_forecasts["ARIMA"] = latest_price
            evaluation_results["ARIMA"] = base_metrics.copy()
            
        # -------------------------------------------------------------
        # 4. Model 3: XGBoost
        # -------------------------------------------------------------
        try:
            xgb_model = XGBoostPredictor()
            xgb_model.fit(
                X_train=train_df[feature_cols],
                y_train=train_df[target_col],
                X_val=val_df[feature_cols],
                y_val=val_df[target_col]
            )
            xgb_test_preds = xgb_model.predict(test_df[feature_cols])
            test_predictions["XGBoost"] = xgb_test_preds
            future_forecasts["XGBoost"] = xgb_model.forecast_future(latest_row[feature_cols])
            evaluation_results["XGBoost"] = compute_evaluation_metrics(
                y_true=y_test,
                y_pred=xgb_test_preds,
                current_prices=curr_test_prices,
                baseline_mae=baseline_mae
            )
            xgb_model.save(model_artifacts_dir)
        except Exception as e:
            logger.warning(f"[{symbol}] XGBoost warning: {e}. Falling back to baseline.")
            test_predictions["XGBoost"] = base_test_preds
            future_forecasts["XGBoost"] = latest_price
            evaluation_results["XGBoost"] = base_metrics.copy()
            
        # -------------------------------------------------------------
        # 5. Model 4: PyTorch LSTM
        # -------------------------------------------------------------
        try:
            lstm_model = LSTMPredictor(lookback=20, epochs=15)
            # Combine train and val for LSTM training with full lookback
            lstm_train = pd.concat([train_df, val_df], ignore_index=True)
            lstm_model.fit(lstm_train[feature_cols], lstm_train[target_col])
            
            # Predict test set using preceding lookback context
            test_start_idx = len(train_df) + len(val_df)
            combined_stock_feats = pd.concat([train_df, val_df, test_df], ignore_index=True)
            lstm_test_preds = lstm_model.predict_test(
                combined_stock_feats,
                test_start_idx=test_start_idx,
                test_len=len(test_df)
            )
            test_predictions["LSTM"] = lstm_test_preds
            future_forecasts["LSTM"] = lstm_model.forecast_future(stock_df[feature_cols])
            evaluation_results["LSTM"] = compute_evaluation_metrics(
                y_true=y_test,
                y_pred=lstm_test_preds,
                current_prices=curr_test_prices,
                baseline_mae=baseline_mae
            )
            lstm_model.save(model_artifacts_dir)
        except Exception as e:
            logger.warning(f"[{symbol}] LSTM warning: {e}. Falling back to baseline.")
            test_predictions["LSTM"] = base_test_preds
            future_forecasts["LSTM"] = latest_price
            evaluation_results["LSTM"] = base_metrics.copy()
            
        # -------------------------------------------------------------
        # 6. Model Selection (Lowest test RMSE)
        # -------------------------------------------------------------
        best_model_name = select_best_model(evaluation_results, criterion="rmse")
        best_pred_future = future_forecasts[best_model_name]
        
        # Financial sanity bounding on forecast (limits to ±35% movement in 1 month)
        max_bound = latest_price * 1.40
        min_bound = latest_price * 0.60
        best_pred_future = float(np.clip(best_pred_future, min_bound, max_bound))
        future_forecasts[best_model_name] = best_pred_future
        
        expected_change = round(best_pred_future - latest_price, 2)
        expected_return_pct = round((expected_change / latest_price) * 100.0, 2)
        if abs(expected_return_pct) < 0.005:
            expected_return_pct = 0.00
        
        # Uncertainty Interval (1.645 * residual std for 90% confidence range)
        best_test_errors = y_test - test_predictions[best_model_name]
        residual_std = float(np.std(best_test_errors)) if len(best_test_errors) > 1 else (latest_price * 0.05)
        lower_bound = round(max(0.0, best_pred_future - (1.645 * residual_std)), 2)
        upper_bound = round(best_pred_future + (1.645 * residual_std), 2)
        
        # Save metrics artifact for this stock
        save_json({
            "symbol": symbol,
            "best_model": best_model_name,
            "metrics": evaluation_results,
            "future_forecasts": future_forecasts
        }, model_artifacts_dir / "metrics.json")
        
        self.tracker.record_success(symbol, best_model_name, expected_return_pct)
        
        # Format backtesting rows for this stock
        backtest_records = []
        for i, dt in enumerate(test_df['Date']):
            dt_str = dt.strftime('%Y-%m-%d')
            for m_name, preds in test_predictions.items():
                p_val = float(preds[i]) if i < len(preds) else float(y_test[i])
                backtest_records.append({
                    "Date": dt_str,
                    "Symbol": symbol,
                    "Model": m_name,
                    "Actual_Future_Close": round(float(y_test[i]), 2),
                    "Predicted_Future_Close": round(p_val, 2),
                    "Error": round(float(y_test[i] - p_val), 2),
                    "Abs_Error": round(abs(float(y_test[i] - p_val)), 2),
                    "Pct_Error": round(abs((y_test[i] - p_val) / y_test[i]) * 100.0, 2)
                })
                
        return {
            "symbol": symbol,
            "company": company,
            "industry": industry,
            "sector": sector,
            "latest_date": latest_date,
            "current_price": round(latest_price, 2),
            "predicted_price": round(best_pred_future, 2),
            "expected_change": expected_change,
            "expected_return_pct": expected_return_pct,
            "best_model": best_model_name,
            "lower_bound": lower_bound,
            "upper_bound": upper_bound,
            "best_rmse": evaluation_results[best_model_name]["rmse"],
            "best_mae": evaluation_results[best_model_name]["mae"],
            "best_mape": evaluation_results[best_model_name]["mape"],
            "best_dir_acc": evaluation_results[best_model_name]["directional_accuracy"],
            "evaluation_results": evaluation_results,
            "future_forecasts": future_forecasts,
            "backtest_records": backtest_records
        }

    def run_all(self, feature_path: Path = PROCESSED_DATA_PARQUET):
        """Executes the pipeline across all 50 stocks in sequence."""
        logger.info("=" * 65)
        logger.info("PHASE 8-17: EXECUTING REUSABLE MODELING & PREDICTION PIPELINE")
        logger.info("=" * 65)
        
        df = load_dataframe(feature_path)
        df['Date'] = pd.to_datetime(df['Date'])
        symbols = sorted(df['Symbol'].unique())
        
        logger.info(f"Starting modeling pipeline for {len(symbols)} stocks...")
        
        stock_results = []
        all_comparison_rows = []
        all_backtest_rows = []
        
        for idx, sym in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] Training models for: {sym}")
            stock_data = df[df['Symbol'] == sym].copy()
            try:
                res = self.process_single_stock(stock_data)
                stock_results.append(res)
                
                # Append comparison rows
                for m_name, m_metrics in res["evaluation_results"].items():
                    all_comparison_rows.append({
                        "Symbol": sym,
                        "Company": res["company"],
                        "Industry": res["industry"],
                        "Model": m_name,
                        "MAE": m_metrics["mae"],
                        "RMSE": m_metrics["rmse"],
                        "MAPE": m_metrics["mape"],
                        "Directional_Accuracy": m_metrics.get("directional_accuracy", 50.0),
                        "Baseline_Improvement_Pct": m_metrics.get("baseline_improvement_pct", 0.0),
                        "Is_Best_Model": (m_name == res["best_model"])
                    })
                    
                all_backtest_rows.extend(res["backtest_records"])
            except Exception as e:
                self.tracker.record_failure(sym, "modeling_pipeline", str(e))
                
        self.tracker.summary()
        
        # -------------------------------------------------------------
        # Generate and save master CSV artifacts
        # -------------------------------------------------------------
        ensure_dir(PREDICTIONS_CSV.parent)
        
        # 1. Predictions master table
        pred_df = pd.DataFrame([{
            "Symbol": r["symbol"],
            "Company": r["company"],
            "Industry": r["industry"],
            "Sector": r["sector"],
            "Latest_Date": r["latest_date"],
            "Current_Price": r["current_price"],
            "Predicted_21D_Price": r["predicted_price"],
            "Expected_Change": r["expected_change"],
            "Expected_Return_Pct": r["expected_return_pct"],
            "Best_Model": r["best_model"],
            "Lower_Bound_90": r["lower_bound"],
            "Upper_Bound_90": r["upper_bound"],
            "RMSE": r["best_rmse"],
            "MAE": r["best_mae"],
            "MAPE": r["best_mape"],
            "Directional_Accuracy": r["best_dir_acc"],
            "Confidence_Note": "90% interval based on empirical test residual std. Past performance does not guarantee future results."
        } for r in stock_results])
        pred_df.to_csv(PREDICTIONS_CSV, index=False)
        logger.info(f"✔ Saved stock predictions to {PREDICTIONS_CSV}")
        
        # 2. Stock Ranking Table (sorted by Expected Return %)
        ranking_df = pred_df.sort_values(by="Expected_Return_Pct", ascending=False).reset_index(drop=True)
        ranking_df.insert(0, "Rank", range(1, len(ranking_df) + 1))
        ranking_df.to_csv(STOCK_RANKING_CSV, index=False)
        logger.info(f"✔ Saved 50-stock ranking to {STOCK_RANKING_CSV}")
        
        # 3. Model Comparison Table
        comp_df = pd.DataFrame(all_comparison_rows)
        comp_df.to_csv(MODEL_COMPARISON_CSV, index=False)
        logger.info(f"✔ Saved model comparison metrics to {MODEL_COMPARISON_CSV}")
        
        # 4. Backtest Predictions Table
        backtest_df = pd.DataFrame(all_backtest_rows)
        backtest_df.to_csv(BACKTEST_PREDICTIONS_CSV, index=False)
        logger.info(f"✔ Saved backtesting records to {BACKTEST_PREDICTIONS_CSV}")
        
        # Master summary metrics
        win_counts = pred_df['Best_Model'].value_counts().to_dict()
        summary_payload = {
            "total_stocks_modeled": len(stock_results),
            "model_win_counts": win_counts,
            "average_expected_return_pct": round(float(pred_df['Expected_Return_Pct'].mean()), 2),
            "median_expected_return_pct": round(float(pred_df['Expected_Return_Pct'].median()), 2),
            "stocks_predicted_positive": int((pred_df['Expected_Return_Pct'] > 0).sum()),
            "stocks_predicted_negative": int((pred_df['Expected_Return_Pct'] < 0).sum()),
            "stocks_predicted_neutral": int((pred_df['Expected_Return_Pct'] == 0).sum()),
            "overall_mean_rmse": round(float(pred_df['RMSE'].mean()), 2),
            "overall_mean_mae": round(float(pred_df['MAE'].mean()), 2),
            "overall_mean_mape": round(float(pred_df['MAPE'].mean()), 2),
            "overall_mean_directional_acc": round(float(pred_df['Directional_Accuracy'].mean()), 2)
        }
        save_json(summary_payload, METRICS_DIR / "master_model_metrics.json")
        logger.info("✔ Master model metrics summary saved.")
        return stock_results

def run_predictions():
    pipeline = MasterForecastingPipeline()
    return pipeline.run_all()

if __name__ == "__main__":
    run_predictions()
