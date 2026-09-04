"""
arima_model.py
--------------
Implements Phases 30 and 31 of the Project Roadmap:
- Phase 30 (Steps 34-36): Prepare data, fit ARIMA(1, 1, 1), generate walk-forward 1-step test predictions.
- Phase 31 (Step 37): Statistical model evaluation without lookahead leakage.
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')


class ARIMAPredictor:
    """Walk-forward ARIMA(p, d, q) for 1-step-ahead level price forecasting."""
    def __init__(self, order: tuple = (1, 1, 1)):
        self.order = order
        self.name = f"ARIMA{order} Walk-Forward"
        self.fitted_model = None
        
    def fit_and_predict(self, train_series: pd.Series, test_series: pd.Series) -> np.ndarray:
        """
        Fits ARIMA on the training series and produces 1-step-ahead walk-forward
        predictions across the test series using .extend().
        """
        print(f"Fitting {self.name} on {len(train_series)} training observations...")
        model = ARIMA(train_series.to_numpy(dtype=np.float64), order=self.order)
        self.fitted_model = model.fit()
        
        # Pure walk-forward rolling extend across test data (zero leakage)
        full_series = np.concatenate([train_series.to_numpy(dtype=np.float64), test_series.to_numpy(dtype=np.float64)])
        extended_res = self.fitted_model.extend(test_series.to_numpy(dtype=np.float64))
        
        # Forecasts for each test step t: predict step t+1 from info up to step t
        predictions = extended_res.fittedvalues[-len(test_series):]
        return np.array(predictions, dtype=np.float64)
