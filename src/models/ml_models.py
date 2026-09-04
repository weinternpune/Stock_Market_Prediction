"""
ml_models.py
------------
Implements Phases 32, 33, 34, and 35 of the Project Roadmap:
- Phase 32 (Step 38): Prepare ML feature matrices and standardize inputs.
- Phase 33 (Step 39): Train Random Forest Regressor with 5-fold TimeSeriesSplit.
- Phase 34 (Step 40): Evaluate Random Forest on test set.
- Phase 35 (Step 41): Train and evaluate XGBoost Regressor; compare RF vs. XGBoost.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import xgboost as xgb


class ClassicalMLManager:
    """Manages training, cross-validation, and prediction for Random Forest and XGBoost."""
    def __init__(self, saved_models_dir: Path):
        self.saved_models_dir = saved_models_dir
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.feature_cols = []
        self.rf_model = None
        self.xgb_model = None
        
    def get_feature_matrix(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Separates feature matrix X and target vector y."""
        exclude_cols = ['Date', 'Target']
        self.feature_cols = [c for c in df.columns if c not in exclude_cols]
        X = df[self.feature_cols].to_numpy(dtype=np.float64)
        y = df['Target'].to_numpy(dtype=np.float64)
        return X, y
        
    def run_cv(self, X_train: np.ndarray, y_train: np.ndarray, model, model_name: str, n_splits: int = 5) -> dict:
        """Evaluates model stability using 5-Fold expanding window TimeSeriesSplit."""
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_rmses = []
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
            X_tr, y_tr = X_train[train_idx], y_train[train_idx]
            X_val, y_val = X_train[val_idx], y_train[val_idx]
            
            scaler_fold = StandardScaler()
            X_tr_scaled = scaler_fold.fit_transform(X_tr)
            X_val_scaled = scaler_fold.transform(X_val)
            
            model.fit(X_tr_scaled, y_tr)
            preds = model.predict(X_val_scaled)
            
            rmse = root_mean_squared_error(y_val, preds)
            mae = mean_absolute_error(y_val, preds)
            fold_rmses.append(rmse)
            fold_maes.append(mae)
            
        cv_summary = {
            "cv_mean_rmse": round(float(np.mean(fold_rmses)), 2),
            "cv_std_rmse": round(float(np.std(fold_rmses)), 2),
            "cv_mean_mae": round(float(np.mean(fold_maes)), 2),
            "cv_std_mae": round(float(np.std(fold_maes)), 2)
        }
        print(f"5-Fold CV for {model_name}: Mean RMSE = {cv_summary['cv_mean_rmse']} ± {cv_summary['cv_std_rmse']}")
        return cv_summary

    def train_and_predict(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
        """Trains both Random Forest and XGBoost, evaluates CV, predicts on test set."""
        X_train, y_train = self.get_feature_matrix(train_df)
        X_test, y_test = self.get_feature_matrix(test_df)
        
        # Fit scaler on training set strictly
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # 1. Random Forest Regressor (Phase 33-34)
        print("Training Random Forest Regressor...")
        self.rf_model = RandomForestRegressor(
            n_estimators=150,
            max_depth=8,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        rf_cv = self.run_cv(X_train, y_train, self.rf_model, "Random Forest")
        
        # Fit on full training set
        self.rf_model.fit(X_train_scaled, y_train)
        rf_preds = self.rf_model.predict(X_test_scaled)
        
        # 2. XGBoost Regressor (Phase 35)
        print("Training XGBoost Regressor...")
        self.xgb_model = xgb.XGBRegressor(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        xgb_cv = self.run_cv(X_train, y_train, self.xgb_model, "XGBoost")
        
        # Fit on full training set
        self.xgb_model.fit(X_train_scaled, y_train)
        xgb_preds = self.xgb_model.predict(X_test_scaled)
        
        # Feature importances
        importances_df = pd.DataFrame({
            'Feature': self.feature_cols,
            'RF_Importance': self.rf_model.feature_importances_,
            'XGB_Importance': self.xgb_model.feature_importances_
        }).sort_values('RF_Importance', ascending=False)
        
        importances_path = self.saved_models_dir.parent / "feature_importances.csv"
        importances_df.to_csv(importances_path, index=False)
        print(f"Feature importances saved to: {importances_path}")
        
        # Save model artifacts
        joblib.dump(self.rf_model, self.saved_models_dir / "random_forest_model.joblib")
        joblib.dump(self.xgb_model, self.saved_models_dir / "xgboost_model.joblib")
        joblib.dump(self.scaler, self.saved_models_dir / "ml_scaler.joblib")
        joblib.dump(self.feature_cols, self.saved_models_dir / "feature_names.joblib")
        print("Saved RF & XGBoost models and scaler to models/saved_models/")
        
        return {
            "rf_preds": rf_preds,
            "xgb_preds": xgb_preds,
            "rf_cv": rf_cv,
            "xgb_cv": xgb_cv,
            "feature_importances": importances_df
        }
