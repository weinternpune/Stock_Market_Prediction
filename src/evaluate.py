"""
Evaluation & Metrics Module for Model Benchmarking.
Computes RMSE, MAE, MAPE, Directional Accuracy (Hit Rate),
Binomial Statistical Significance Testing, and TimeSeriesSplit Cross-Validation.
Adheres strictly to PRD v1.1 and econometric best practices.
"""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def perform_binomial_directional_test(y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray) -> dict:
    """
    Performs a formal Binomial hypothesis test for directional market accuracy:
    H0: p = 0.50 (directional accuracy equals random coin-toss)
    H1: p > 0.50 (model directional accuracy exceeds random chance)

    Returns:
        dict: hits, total_observations, hit_rate_pct, p_value_one_sided,
              p_value_two_sided, ci_95_wilson, is_significant_5pct
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_current = np.asarray(y_current)

    actual_direction = np.sign(y_true - y_current)
    predicted_direction = np.sign(y_pred - y_current)

    # Valid trading days where actual price moved
    valid_mask = actual_direction != 0
    actual_dir_valid = actual_direction[valid_mask]
    pred_dir_valid = predicted_direction[valid_mask]

    n = len(actual_dir_valid)
    if n == 0:
        return {}

    hits = int((actual_dir_valid == pred_dir_valid).sum())
    hit_rate = (hits / n) * 100.0

    # Formal Binomial Test against 0.50 null hypothesis
    binom_res_greater = stats.binomtest(hits, n, p=0.5, alternative="greater")
    binom_res_two_sided = stats.binomtest(hits, n, p=0.5, alternative="two-sided")
    ci = binom_res_two_sided.proportion_ci(confidence_level=0.95, method="wilson")

    return {
        "hits": hits,
        "total_valid_days": n,
        "hit_rate_pct": float(round(hit_rate, 2)),
        "p_value_one_sided": float(round(binom_res_greater.pvalue, 4)),
        "p_value_two_sided": float(round(binom_res_two_sided.pvalue, 4)),
        "ci_95_low_pct": float(round(ci.low * 100, 2)),
        "ci_95_high_pct": float(round(ci.high * 100, 2)),
        "is_significant_5pct": bool(binom_res_greater.pvalue < 0.05)
    }


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_current: np.ndarray = None, is_naive: bool = False) -> dict:
    """
    Computes key financial regression performance metrics:
    - Root Mean Squared Error (RMSE)
    - Mean Absolute Error (MAE)
    - Mean Absolute Percentage Error (MAPE %)
    - Directional Accuracy / Hit Ratio (%) with formal Binomial Hypothesis Test

    NOTE on Naive Persistence Baseline:
    Since the naive model predicts Pt+1 = Pt (zero price change), it does not predict
    a directional movement. Its directional accuracy is accurately reported as 'N/A'
    rather than 0% (which would falsely imply it predicted the opposite direction).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    mape = float(mean_absolute_percentage_error(y_true, y_pred) * 100)

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE (%)": mape,
    }

    if is_naive or (y_current is not None and np.allclose(y_pred, y_current)):
        metrics["Directional Accuracy (%)"] = "N/A"
        metrics["Directional Hits"] = "N/A (Neutral Model)"
        metrics["Binomial p-value (1-sided)"] = "N/A"
        metrics["Binomial p-value (2-sided)"] = "N/A"
        metrics["95% Wilson CI (%)"] = "N/A"
        metrics["Statistically Significant (p < 0.05)"] = "N/A"
        return metrics

    if y_current is not None:
        binom_test = perform_binomial_directional_test(y_true, y_pred, y_current)
        if binom_test:
            metrics["Directional Accuracy (%)"] = binom_test["hit_rate_pct"]
            metrics["Directional Hits"] = f"{binom_test['hits']}/{binom_test['total_valid_days']}"
            metrics["Binomial p-value (1-sided)"] = binom_test["p_value_one_sided"]
            metrics["Binomial p-value (2-sided)"] = binom_test["p_value_two_sided"]
            metrics["95% Wilson CI (%)"] = f"[{binom_test['ci_95_low_pct']}%, {binom_test['ci_95_high_pct']}%]"
            metrics["Statistically Significant (p < 0.05)"] = binom_test["is_significant_5pct"]

    return metrics


def perform_time_series_cv(model_name: str, model_factory, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """
    Performs expanding-window TimeSeriesSplit cross-validation per PRD Section 12 (Risks & Limitations).
    Evaluates out-of-sample stability across rolling periods to mitigate overfitting risks.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rmse_list = []
    mae_list = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        model = model_factory()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        rmse = np.sqrt(mean_squared_error(y_va, preds))
        mae = mean_absolute_error(y_va, preds)
        rmse_list.append(rmse)
        mae_list.append(mae)

    return {
        "model": model_name,
        "n_splits": n_splits,
        "cv_rmse_mean": float(round(np.mean(rmse_list), 2)),
        "cv_rmse_std": float(round(np.std(rmse_list), 2)),
        "cv_mae_mean": float(round(np.mean(mae_list), 2)),
        "cv_mae_std": float(round(np.std(mae_list), 2)),
    }


def generate_comparison_table(results_dict: dict[str, dict], baseline_name: str = "Naive Baseline (Persistence)") -> pd.DataFrame:
    """
    Generates a structured comparison table benchmarked against the naive baseline.
    Computes % difference in RMSE over baseline and reports binomial significance.
    """
    rows = []
    baseline_rmse = None
    if baseline_name in results_dict:
        baseline_rmse = results_dict[baseline_name].get("RMSE")

    for model_name, metrics in results_dict.items():
        dir_acc = metrics.get("Directional Accuracy (%)", "N/A")
        if isinstance(dir_acc, (int, float)):
            dir_str = f"{dir_acc:.2f}%"
        else:
            dir_str = str(dir_acc)

        row = {
            "Model": model_name,
            "RMSE": round(metrics.get("RMSE", 0), 2),
            "MAE": round(metrics.get("MAE", 0), 2),
            "MAPE (%)": round(metrics.get("MAPE (%)", 0), 3),
            "Directional Acc": dir_str,
            "Binomial p-val": metrics.get("Binomial p-value (1-sided)", "N/A"),
            "95% Wilson CI": metrics.get("95% Wilson CI (%)", "N/A"),
        }
        if baseline_rmse and baseline_rmse > 0:
            improvement = ((baseline_rmse - metrics["RMSE"]) / baseline_rmse) * 100
            row["vs Baseline RMSE (%)"] = f"{improvement:+.2f}%"
        else:
            row["vs Baseline RMSE (%)"] = "0.00%"

        rows.append(row)

    table = pd.DataFrame(rows)
    return table


def save_metrics_summary(results_dict: dict[str, dict], cv_dict: dict = None, output_path: Path = MODELS_DIR / "metrics_summary.json"):
    """
    Saves metrics summary as a JSON file with transparent PRD status reporting.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "prd_goal_status": "PRD functionality implemented; naive-baseline performance target not achieved.",
        "best_rmse_model": "Naive Baseline (Persistence)",
        "models": results_dict,
        "cross_validation_5fold": cv_dict if cv_dict else {}
    }
    with open(output_path, "w") as f:
        json.dump(summary_payload, f, indent=4)
    print(f"[EVALUATION] Saved metrics summary to: {output_path}")


def load_metrics_summary(input_path: Path = MODELS_DIR / "metrics_summary.json") -> dict:
    """Loads metrics summary JSON."""
    if not input_path.exists():
        return {}
    with open(input_path, "r") as f:
        return json.load(f)
