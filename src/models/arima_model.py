"""
Statistical Modeling Module: ARIMA / SARIMAX for Time Series Forecasting.
Includes ADF Stationarity Testing, Auto-order Selection, and Rolling Backtesting.
"""

import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")


def check_stationarity(series: pd.Series, name: str = "Price Series") -> dict:
    """
    Augmented Dickey-Fuller (ADF) test for stationarity.
    H0: The series has a unit root (non-stationary).
    H1: The series is stationary.
    """
    result = adfuller(series.dropna())
    adf_stat, p_value, lags, nobs, crit_values, _ = result

    is_stationary = p_value < 0.05
    summary = {
        "series_name": name,
        "adf_statistic": float(adf_stat),
        "p_value": float(p_value),
        "used_lags": int(lags),
        "n_obs": int(nobs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "is_stationary": bool(is_stationary),
        "conclusion": "Stationary (Reject H0)" if is_stationary else "Non-Stationary (Fail to Reject H0)"
    }
    return summary


class ArimaForecaster:
    """
    ARIMA Forecaster for Nifty 500 Price Series.
    Uses ARIMA(p, d, q) where d=1 handles non-stationarity of asset prices.
    """
    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self.name = f"Statistical (ARIMA{order})"
        self.model_fit = None
        self.train_history = None

    def fit(self, train_series: pd.Series):
        """Fits ARIMA model on training series."""
        self.train_history = train_series.copy()
        model = ARIMA(train_series, order=self.order)
        self.model_fit = model.fit()
        return self

    def predict_test(self, test_series: pd.Series) -> np.ndarray:
        """
        Generates genuine 1-step ahead forecasts for the test set using a strict
        walk-forward rolling process with ZERO future information leakage:
          1. Start with model parameters fitted strictly on training data.
          2. At each test day t, forecast 1-step ahead (T+1) based only on data up to t.
          3. Append the newly observed actual test value via .extend([val]).
          4. Advance to the next day without re-smoothing future observations.
        """
        predictions = []
        curr_res = self.model_fit

        for val in test_series.values:
            p = float(curr_res.forecast(steps=1).iloc[0])
            predictions.append(p)
            curr_res = curr_res.extend([val])

        return np.array(predictions)

    def forecast_future(self, steps: int = 15) -> np.ndarray:
        """Forecasts multi-step into the future."""
        if self.model_fit is None:
            raise ValueError("Model must be fitted before forecasting.")
        forecast = self.model_fit.forecast(steps=steps)
        return forecast.values
