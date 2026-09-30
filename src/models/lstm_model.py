"""
lstm_model.py
-------------
Deep Learning Model for Stock Price Forecasting using PyTorch LSTM.
Constructs multi-feature sliding window sequences (lookback=20), scales inputs strictly
on training data, trains an LSTM network with Adam optimizer, and generates inverse-scaled
predictions and future 21-day forecasts.
"""

from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib

class PyTorchLSTM(nn.Module):
    """Stacked LSTM with linear projection head for continuous price forecasting."""
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_layers: int = 1, dropout: float = 0.1):
        super(PyTorchLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
        
    def forward(self, x):
        out, _ = self.lstm(x)
        last_step = out[:, -1, :]
        return self.head(last_step)

class LSTMPredictor:
    """Manages PyTorch LSTM sequence dataset creation, training, and inference."""
    
    def __init__(self, lookback: int = 20, hidden_dim: int = 32, epochs: int = 15, lr: float = 0.003):
        self.name = "LSTM"
        self.lookback = lookback
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.lr = lr
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_cols: List[str] = []
        
    def _create_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray = None):
        """Creates sliding sequences of shape (num_samples, lookback, num_features)."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(X_scaled) + 1):
            X_seq.append(X_scaled[i - self.lookback:i])
            if y_scaled is not None and (i - 1) < len(y_scaled):
                y_seq.append(y_scaled[i - 1])
                
        X_arr = np.array(X_seq, dtype=np.float32)
        if y_scaled is not None:
            y_arr = np.array(y_seq, dtype=np.float32)
            return X_arr, y_arr
        return X_arr
        
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series):
        """Fits scalers strictly on training data and trains LSTM network."""
        self.feature_cols = list(X_train.columns)
        
        # Clean NaNs
        X_raw = np.nan_to_num(X_train.values.astype(np.float64), nan=0.0)
        y_raw = np.nan_to_num(y_train.values.reshape(-1, 1).astype(np.float64), nan=0.0)
        
        # Fit scalers STRICTLY on training split
        X_scaled = self.feature_scaler.fit_transform(X_raw)
        y_scaled = self.target_scaler.fit_transform(y_raw)
        
        X_seq, y_seq = self._create_sequences(X_scaled, y_scaled)
        if len(X_seq) < 10:
            # Fallback for very small sequences
            self.model = None
            return self
            
        dataset = TensorDataset(torch.tensor(X_seq), torch.tensor(y_seq))
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        input_dim = X_seq.shape[2]
        self.model = PyTorchLSTM(input_dim=input_dim, hidden_dim=self.hidden_dim).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                out = self.model(batch_x)
                loss = criterion(out, batch_y)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
        return self
        
    def predict_test(self, full_feature_df: pd.DataFrame, test_start_idx: int, test_len: int) -> np.ndarray:
        """
        Generates sequence-based predictions for the holdout test set using preceding lookback context.
        """
        if self.model is None:
            return np.zeros(test_len, dtype=np.float64)
            
        self.model.eval()
        X_full = np.nan_to_num(full_feature_df[self.feature_cols].values.astype(np.float64), nan=0.0)
        X_scaled = self.feature_scaler.transform(X_full)
        
        preds_scaled = []
        with torch.no_grad():
            for i in range(test_start_idx, test_start_idx + test_len):
                seq = X_scaled[i - self.lookback + 1:i + 1]
                if len(seq) < self.lookback:
                    pad = np.zeros((self.lookback - len(seq), seq.shape[1]))
                    seq = np.vstack([pad, seq])
                seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
                p = self.model(seq_tensor).cpu().item()
                preds_scaled.append([p])
                
        preds_scaled = np.array(preds_scaled, dtype=np.float64)
        preds_unscaled = self.target_scaler.inverse_transform(preds_scaled).flatten()
        return preds_unscaled
        
    def forecast_future(self, full_feature_df: pd.DataFrame) -> float:
        """Forecasts future 21-day price using the latest lookback window of features."""
        if self.model is None:
            return 0.0
            
        self.model.eval()
        X_full = np.nan_to_num(full_feature_df[self.feature_cols].values.astype(np.float64), nan=0.0)
        X_scaled = self.feature_scaler.transform(X_full)
        seq = X_scaled[-self.lookback:]
        
        with torch.no_grad():
            seq_tensor = torch.tensor(seq, dtype=torch.float32).unsqueeze(0).to(self.device)
            p_scaled = self.model(seq_tensor).cpu().numpy()
            p_unscaled = self.target_scaler.inverse_transform(p_scaled).item()
            return float(p_unscaled)
            
    def save(self, model_dir: Path):
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        if self.model is not None:
            torch.save(self.model.state_dict(), model_dir / "lstm_weights.pt")
        joblib.dump(self.feature_scaler, model_dir / "lstm_feature_scaler.joblib")
        joblib.dump(self.target_scaler, model_dir / "lstm_target_scaler.joblib")
        joblib.dump(self.feature_cols, model_dir / "lstm_features.joblib")
