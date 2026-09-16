"""
evaluator.py
------------
Model Evaluation module implementing Phase 13 & 14.
Computes MAE, RMSE, MAPE, Directional Accuracy, and determines best model per stock.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

def compute_evaluation_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    current_prices: np.ndarray = None,
    baseline_mae: float = None
) -> Dict[str, float]:
    """
    Computes standard time-series regression metrics:
    - MAE: Mean Absolute Error
    - RMSE: Root Mean Squared Error
    - MAPE: Mean Absolute Percentage Error (%)
    - Directional Accuracy (% of correct direction of change vs current price)
    - Baseline Improvement (% MAE reduction vs baseline)
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    
    # Filter valid pairs
    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[valid_mask]
    y_pred = y_pred[valid_mask]
    
    if len(y_true) == 0:
        return {"mae": 999999.0, "rmse": 999999.0, "mape": 100.0, "directional_accuracy": 0.0}
        
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    
    # Avoid division by zero in MAPE
    denom = np.where(y_true == 0, 1e-9, y_true)
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    
    metrics = {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2)
    }
    
    # Directional Accuracy
    if current_prices is not None:
        curr = np.asarray(current_prices, dtype=np.float64)[valid_mask]
        actual_dir = np.sign(y_true - curr)
        pred_dir = np.sign(y_pred - curr)
        dir_acc = float(np.mean(actual_dir == pred_dir) * 100.0)
        metrics["directional_accuracy"] = round(dir_acc, 2)
    else:
        metrics["directional_accuracy"] = 50.0
        
    if baseline_mae and baseline_mae > 0:
        improvement = ((baseline_mae - mae) / baseline_mae) * 100.0
        metrics["baseline_improvement_pct"] = round(float(improvement), 2)
    else:
        metrics["baseline_improvement_pct"] = 0.0
        
    return metrics

def select_best_model(model_metrics: Dict[str, Dict[str, float]], criterion: str = "rmse") -> str:
    """
    Objectively selects the best performing model based on holdout validation/test metric.
    Does not assume LSTM is best.
    """
    best_model = None
    lowest_error = float('inf')
    
    for model_name, metrics in model_metrics.items():
        err = metrics.get(criterion, float('inf'))
        if err < lowest_error:
            lowest_error = err
            best_model = model_name
            
    return best_model or "Baseline"
