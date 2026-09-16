"""
xgboost_model.py
----------------
Classical Machine Learning Model for Stock Price Forecasting.
Trains an XGBoost Regressor on technical indicators, momentum, volatility, and lags.
Strict zero-leakage scaling and validation.
"""

from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib

class XGBoostPredictor:
    """XGBoost Regressor for multi-feature financial price prediction."""
    
    def __init__(self, random_state: int = 42):
        self.name = "XGBoost"
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method='hist'
        )
        self.feature_cols: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fits scaler strictly on X_train and trains XGBoost model."""
        self.feature_cols = list(X_train.columns)
        X_tr = X_train.values.astype(np.float64)
        y_tr = y_train.values.astype(np.float64)
        
        # Clean any remaining NaNs or Infs
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Fit scaler ONLY on training data
        X_tr_scaled = self.scaler.fit_transform(X_tr)
        
        eval_set = None
        if X_val is not None and y_val is not None:
            X_v = X_val.values.astype(np.float64)
            X_v = np.nan_to_num(X_v, nan=0.0, posinf=0.0, neginf=0.0)
            X_v_scaled = self.scaler.transform(X_v)
            eval_set = [(X_v_scaled, y_val.values.astype(np.float64))]
            
        self.model.fit(
            X_tr_scaled,
            y_tr,
            eval_set=eval_set,
            verbose=False
        )
        
        # Record feature importances
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            self.feature_importances = {
                col: float(imp)
                for col, imp in sorted(zip(self.feature_cols, importances), key=lambda x: x[1], reverse=True)
            }
        return self
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predicts on unseen test features using the fitted scaler."""
        X_arr = X[self.feature_cols].values.astype(np.float64)
        X_arr = np.nan_to_num(X_arr, nan=0.0, posinf=0.0, neginf=0.0)
        X_scaled = self.scaler.transform(X_arr)
        preds = self.model.predict(X_scaled)
        return np.asarray(preds, dtype=np.float64)
        
    def forecast_future(self, latest_feature_row: pd.DataFrame) -> float:
        """Forecasts future price from the latest feature row."""
        preds = self.predict(latest_feature_row)
        return float(preds[0])
        
    def save(self, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_dir / "xgboost_model.joblib")
        joblib.dump(self.scaler, model_dir / "xgboost_scaler.joblib")
        joblib.dump(self.feature_cols, model_dir / "xgboost_features.joblib")
        joblib.dump(self.feature_importances, model_dir / "xgboost_importances.joblib")
        
    def load(self, model_dir: Path):
        model_dir = Path(model_dir)
        self.model = joblib.load(model_dir / "xgboost_model.joblib")
        self.scaler = joblib.load(model_dir / "xgboost_scaler.joblib")
        self.feature_cols = joblib.load(model_dir / "xgboost_features.joblib")
        self.feature_importances = joblib.load(model_dir / "xgboost_importances.joblib")
        return self
