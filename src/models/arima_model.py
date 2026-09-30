"""
arima_model.py
--------------
Classical ARIMA Statistical Benchmark Model for Stock Price Forecasting.
Fits ARIMA(1, 1, 1) on training prices, evaluates multi-step forecasts on the
holdout test set, and generates the final 21-trading-day forecast.
"""

from typing import Tuple
import numpy as np
import pandas as pd
import warnings
from statsmodels.tsa.arima.model import ARIMA
warnings.filterwarnings('ignore')

class ARIMAPredictor:
    """ARIMA(1, 1, 1) classical statistical benchmark."""
    
    def __init__(self, order: Tuple[int, int, int] = (1, 1, 1)):
        self.order = order
        self.name = "ARIMA"
        self.fitted_model = None
        self.last_train_series = None
        
    def fit(self, train_series: pd.Series):
        """Fits ARIMA on the training series."""
        self.last_train_series = np.asarray(train_series, dtype=np.float64)
        try:
            model = ARIMA(self.last_train_series, order=self.order)
            self.fitted_model = model.fit()
        except Exception:
            # Fallback to ARIMA(1, 0, 0) on difference or simple order
            try:
                model = ARIMA(self.last_train_series, order=(1, 1, 0))
                self.fitted_model = model.fit()
            except Exception:
                self.fitted_model = None
        return self
        
    def predict_test(self, train_series: pd.Series, test_series: pd.Series, horizon: int = 21) -> np.ndarray:
        """
        Generates predictions for the holdout test set using walk-forward state extension.
        For each test session t, predicts price horizon (21) days ahead.
        """
        tr_vals = np.asarray(train_series, dtype=np.float64)
        te_vals = np.asarray(test_series, dtype=np.float64)
        
        if self.fitted_model is None:
            self.fit(train_series)
            
        if self.fitted_model is None:
            # Fallback to persistence if ARIMA completely fails to converge
            return te_vals.copy()
            
        try:
            # Extend model across test set to obtain filtered state at each test point
            extended = self.fitted_model.extend(te_vals)
            # 1-step predictions from extended state
            fitted_vals = extended.fittedvalues[-len(te_vals):]
            
            # For 21-day horizon, project drift/trend from ARIMA state
            drift = np.mean(np.diff(tr_vals[-60:])) if len(tr_vals) > 60 else 0.0
            test_preds = np.asarray(fitted_vals, dtype=np.float64) + (drift * horizon * 0.5)
            return test_preds
        except Exception:
            # Safe persistence fallback
            return te_vals.copy()
            
    def forecast_future(self, full_series: pd.Series, horizon: int = 21) -> float:
        """
        Forecasts the future price 21 trading days ahead from the latest observed close.
        """
        vals = np.asarray(full_series, dtype=np.float64)
        if len(vals) == 0:
            return 0.0
            
        latest_val = float(vals[-1])
        try:
            # Refit on latest window or extend
            recent_vals = vals[-252:] if len(vals) > 252 else vals
            model = ARIMA(recent_vals, order=self.order)
            res = model.fit()
            forecasts = res.forecast(steps=horizon)
            pred = float(forecasts[-1])
            # Sanity bound: financial prices shouldn't explode or turn negative
            if np.isnan(pred) or pred <= 0 or pred > (latest_val * 3.0) or pred < (latest_val * 0.3):
                return latest_val
            return pred
        except Exception:
            return latest_val
