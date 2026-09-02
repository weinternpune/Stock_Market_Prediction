"""
Multi-Step Forecasting Service for Nifty 500 Index.
Implements real trained model forward projections using:
1. Statistical ARIMA (via statsmodels get_forecast with parametric confidence intervals)
2. Recursive Autoregressive XGBoost & Random Forest (dynamic indicator updating)
3. Autoregressive PyTorch LSTM Sequence Rollouts
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import torch

from src.feature_engineering import calculate_rsi, calculate_macd, calculate_bollinger_bands
from src.models.ml_models import MLForecastingSuite
from src.models.lstm_model import LSTMNetwork

warnings.filterwarnings("ignore")

MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved_models"


def forecast_arima(price_series: pd.Series, steps: int = 15, confidence_level: int = 95) -> dict:
    """
    Generates genuine multi-step forecasts using fitted ARIMA(1, 1, 1).
    Returns predicted mean and parametric confidence intervals.
    """
    model = ARIMA(price_series.dropna(), order=(1, 1, 1))
    fitted = model.fit()

    alpha = 1.0 - (confidence_level / 100.0)
    forecast_res = fitted.get_forecast(steps=steps)
    preds = forecast_res.predicted_mean.values
    conf_int = forecast_res.conf_int(alpha=alpha).values

    lower_bounds = conf_int[:, 0]
    upper_bounds = conf_int[:, 1]

    return {
        "model_name": "ARIMA(1,1,1)",
        "projected_prices": preds.tolist(),
        "lower_bounds": lower_bounds.tolist(),
        "upper_bounds": upper_bounds.tolist()
    }


def forecast_recursive_ml(
    clean_df: pd.DataFrame,
    ml_suite: MLForecastingSuite,
    model_type: str = "XGBoost",
    steps: int = 15,
    confidence_level: int = 95,
    test_rmse: float = 226.93
) -> dict:
    """
    Generates genuine multi-step forward forecasts using recursive autoregressive rollout.
    At each step:
      1. Recalculates technical indicators from the updated price history
      2. Predicts next day price using trained XGBoost or Random Forest
      3. Appends forecast to history and steps forward
    """
    history_prices = clean_df["close"].dropna().tolist()
    projected = []
    upper_bounds = []
    lower_bounds = []

    z_score = stats.norm.ppf(0.5 + confidence_level / 200.0)

    for h in range(1, steps + 1):
        s = pd.Series(history_prices)
        c = float(s.iloc[-1])

        # Recalculate indicators with latest forecasted price history
        sma20 = float(s.rolling(20).mean().iloc[-1])
        sma50 = float(s.rolling(50).mean().iloc[-1])
        sma200 = float(s.rolling(200).mean().iloc[-1])
        ema20 = float(s.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(s.ewm(span=50, adjust=False).mean().iloc[-1])

        rsi = float(calculate_rsi(s).iloc[-1])
        macd_l, macd_s, macd_h = calculate_macd(s)
        bb_mid, bb_u, bb_l, bb_w, bb_b = calculate_bollinger_bands(s)

        vol20 = float(s.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252))
        vol50 = float(s.pct_change().rolling(50).std().iloc[-1] * np.sqrt(252))

        # Build feature vector
        row_features = {
            "close": c,
            "open": c,
            "high": c * 1.002,
            "low": c * 0.998,
            "volume": 20000000.0,
            "sma_20": sma20,
            "sma_50": sma50,
            "sma_200": sma200,
            "ema_20": ema20,
            "ema_50": ema50,
            "ratio_close_sma20": c / (sma20 + 1e-9),
            "ratio_close_sma50": c / (sma50 + 1e-9),
            "ratio_close_sma200": c / (sma200 + 1e-9),
            "rsi_14": rsi,
            "macd_line": float(macd_l.iloc[-1]),
            "macd_signal": float(macd_s.iloc[-1]),
            "macd_hist": float(macd_h.iloc[-1]),
            "bb_upper": float(bb_u.iloc[-1]),
            "bb_lower": float(bb_l.iloc[-1]),
            "bb_width": float(bb_w.iloc[-1]),
            "bb_pct_b": float(bb_b.iloc[-1]),
            "volatility_20d": vol20 if not np.isnan(vol20) else 0.12,
            "volatility_50d": vol50 if not np.isnan(vol50) else 0.12,
            "return_1d": float(s.pct_change().iloc[-1]),
            "return_5d": float(s.pct_change(5).iloc[-1]),
            "return_20d": float(s.pct_change(20).iloc[-1]),
            "lag_close_1": float(s.iloc[-1]),
            "lag_close_2": float(s.iloc[-2]),
            "lag_close_5": float(s.iloc[-5]),
            "volume_sma_20": 20000000.0,
            "volume_ratio": 1.0
        }

        vec = np.array([[row_features[k] for k in ml_suite.feature_names]])

        if "xgboost" in model_type.lower():
            pred_next = float(ml_suite.predict_xgb(vec)[0])
        else:
            pred_next = float(ml_suite.predict_rf(vec)[0])

        projected.append(pred_next)
        history_prices.append(pred_next)

        # Compounding uncertainty interval across multi-step horizon
        spread = z_score * test_rmse * np.sqrt(h)
        upper_bounds.append(pred_next + spread)
        lower_bounds.append(pred_next - spread)

    return {
        "model_name": f"Recursive {model_type}",
        "projected_prices": projected,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }


def forecast_recursive_lstm(
    features_df: pd.DataFrame,
    steps: int = 15,
    confidence_level: int = 95,
    test_rmse: float = 610.91,
    saved_dir: Path = SAVED_MODELS_DIR
) -> dict:
    """
    Generates genuine multi-step forecasts using autoregressive PyTorch LSTM sequence rollouts.
    """
    import joblib

    config = joblib.load(saved_dir / "lstm_config.joblib")
    feature_scaler = joblib.load(saved_dir / "lstm_feature_scaler.joblib")
    target_scaler = joblib.load(saved_dir / "lstm_target_scaler.joblib")

    lstm_feature_cols = [
        "close", "open", "high", "low", "volume",
        "sma_20", "sma_50", "ema_20", "rsi_14", "macd_line",
        "bb_upper", "bb_lower", "volatility_20d", "return_1d"
    ]

    model = LSTMNetwork(
        input_size=config["input_size"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=0.0
    )
    model.load_state_dict(torch.load(saved_dir / "lstm_weights.pt", map_location="cpu", weights_only=True))
    model.eval()

    # Get last seq_len rows
    seq_len = config["seq_len"]
    recent_features = features_df[lstm_feature_cols].iloc[-seq_len:].values.copy()
    recent_scaled = feature_scaler.transform(recent_features)

    projected = []
    upper_bounds = []
    lower_bounds = []
    z_score = stats.norm.ppf(0.5 + confidence_level / 200.0)

    curr_window = recent_scaled.copy()

    with torch.no_grad():
        for h in range(1, steps + 1):
            window_tensor = torch.tensor(curr_window, dtype=torch.float32).unsqueeze(0)
            pred_scaled = model(window_tensor).numpy().flatten()[0]
            pred_price = float(target_scaler.inverse_transform([[pred_scaled]])[0][0])

            projected.append(pred_price)

            # Update rolling sequence window
            new_row = curr_window[-1].copy()
            new_row[0] = pred_scaled  # close feature
            curr_window = np.vstack([curr_window[1:], new_row])

            spread = z_score * test_rmse * np.sqrt(h)
            upper_bounds.append(pred_price + spread)
            lower_bounds.append(pred_price - spread)

    return {
        "model_name": "Autoregressive PyTorch LSTM",
        "projected_prices": projected,
        "lower_bounds": lower_bounds,
        "upper_bounds": upper_bounds
    }
