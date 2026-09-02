"""
Classical Machine Learning Models for Stock Price Prediction.
Implements Random Forest Regressor and XGBoost Regressor with Feature Importance Ranking.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved_models"


class MLForecastingSuite:
    """
    Suite containing Random Forest and XGBoost forecasting models
    with automated preprocessing pipelines and feature importance tracking.
    """
    FEATURE_COLS = [
        "close", "open", "high", "low", "volume",
        "sma_20", "sma_50", "sma_200", "ema_20", "ema_50",
        "ratio_close_sma20", "ratio_close_sma50", "ratio_close_sma200",
        "rsi_14", "macd_line", "macd_signal", "macd_hist",
        "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
        "volatility_20d", "volatility_50d",
        "return_1d", "return_5d", "return_20d",
        "lag_close_1", "lag_close_2", "lag_close_5",
        "volume_sma_20", "volume_ratio"
    ]

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=4,
            random_state=random_state,
            n_jobs=-1
        )
        self.xgb_model = XGBRegressor(
            n_estimators=150,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1
        )
        self.is_fitted = False
        self.feature_names = []

    def prepare_data(self, df: pd.DataFrame, target_col: str = "target_close"):
        """Prepares feature matrix X and target vector y."""
        available_features = [col for col in self.FEATURE_COLS if col in df.columns]
        self.feature_names = available_features
        X = df[available_features].values
        y = df[target_col].values
        return X, y

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        """Fits scaler, Random Forest, and XGBoost models on training set."""
        X_train_scaled = self.scaler.fit_transform(X_train)
        print("[ML MODELS] Training Random Forest Regressor...")
        self.rf_model.fit(X_train_scaled, y_train)

        print("[ML MODELS] Training XGBoost Regressor...")
        self.xgb_model.fit(X_train_scaled, y_train)

        self.is_fitted = True
        return self

    def predict_rf(self, X: np.ndarray) -> np.ndarray:
        """Predicts using Random Forest."""
        X_scaled = self.scaler.transform(X)
        return self.rf_model.predict(X_scaled)

    def predict_xgb(self, X: np.ndarray) -> np.ndarray:
        """Predicts using XGBoost."""
        X_scaled = self.scaler.transform(X)
        return self.xgb_model.predict(X_scaled)

    def get_feature_importances(self) -> pd.DataFrame:
        """Returns feature importances for both Random Forest and XGBoost."""
        if not self.is_fitted:
            raise ValueError("Models must be fitted before obtaining feature importances.")

        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "rf_importance": self.rf_model.feature_importances_,
            "xgb_importance": self.xgb_model.feature_importances_
        })
        fi_df["mean_importance"] = (fi_df["rf_importance"] + fi_df["xgb_importance"]) / 2
        return fi_df.sort_values("mean_importance", ascending=False).reset_index(drop=True)

    def save_models(self, output_dir: Path = MODEL_SAVE_DIR):
        """Saves models and scaler to disk."""
        output_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, output_dir / "ml_scaler.joblib")
        joblib.dump(self.rf_model, output_dir / "random_forest_model.joblib")
        joblib.dump(self.xgb_model, output_dir / "xgboost_model.joblib")
        joblib.dump(self.feature_names, output_dir / "feature_names.joblib")
        print(f"[ML MODELS] Saved trained models and scaler to: {output_dir}")

    def load_models(self, input_dir: Path = MODEL_SAVE_DIR):
        """Loads models and scaler from disk."""
        self.scaler = joblib.load(input_dir / "ml_scaler.joblib")
        self.rf_model = joblib.load(input_dir / "random_forest_model.joblib")
        self.xgb_model = joblib.load(input_dir / "xgboost_model.joblib")
        self.feature_names = joblib.load(input_dir / "feature_names.joblib")
        self.is_fitted = True
        return self
