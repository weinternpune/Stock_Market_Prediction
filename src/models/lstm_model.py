"""
lstm_model.py
-------------
Implements Phases 36, 37, 38, 39, and 40 of the Project Roadmap:
- Phase 36 (Step 42): Convert data into sliding sequences (lookback=20).
- Phase 37 (Step 43): Normalize/scale inputs (MinMaxScaler fitted strictly on training data).
- Phase 38 (Step 44): Design PyTorch LSTM architecture.
- Phase 39 (Step 45): Train LSTM with validation loss tracking and early stopping.
- Phase 40 (Step 46): Generate inverse-scaled test predictions and metrics.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import joblib


class StockSequenceDataset(Dataset):
    """PyTorch Dataset for multi-feature sliding window time-series sequences."""
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(1)
        
    def __len__(self):
        return len(self.sequences)
        
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class LSTMRegressor(nn.Module):
    """Stacked LSTM with Dropout and Dense output for price level regression."""
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMRegressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


class LSTMPredictorManager:
    """Manages scaling, sequence generation, training, and inference for PyTorch LSTM."""
    def __init__(self, saved_models_dir: Path, lookback: int = 20):
        self.saved_models_dir = saved_models_dir
        self.saved_models_dir.mkdir(parents=True, exist_ok=True)
        self.lookback = lookback
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.feature_cols = []
        
    def create_sequences(self, features: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Creates sliding sequences of shape (num_samples, lookback, num_features)."""
        X_seq, y_seq = [], []
        for i in range(self.lookback, len(features)):
            X_seq.append(features[i - self.lookback:i])
            y_seq.append(targets[i])
        return np.array(X_seq), np.array(y_seq)
        
    def train_and_predict(self, full_features_df: pd.DataFrame, train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
        """
        Prepares sequences, fits scalers on train set only, trains LSTM with early stopping,
        and produces inverse-scaled test predictions.
        """
        print(f"\nPhase 36-40: Initializing PyTorch LSTM on device: {self.device}")
        
        exclude_cols = ['Date', 'Target']
        self.feature_cols = [c for c in train_df.columns if c not in exclude_cols]
        
        X_train_raw = train_df[self.feature_cols].to_numpy(dtype=np.float64)
        y_train_raw = train_df[['Target']].to_numpy(dtype=np.float64)
        
        # Fit scalers STRICTLY on training split
        X_train_scaled = self.feature_scaler.fit_transform(X_train_raw)
        y_train_scaled = self.target_scaler.fit_transform(y_train_raw)
        
        # Create training sequences
        X_tr_seq, y_tr_seq = self.create_sequences(X_train_scaled, y_train_scaled)
        
        # Train/Validation split within training set (last 15% of train for early stopping)
        val_size = int(len(X_tr_seq) * 0.15)
        train_ds = StockSequenceDataset(X_tr_seq[:-val_size], y_tr_seq[:-val_size])
        val_ds = StockSequenceDataset(X_tr_seq[-val_size:], y_tr_seq[-val_size:])
        
        train_loader = DataLoader(train_ds, batch_size=32, shuffle=False)
        val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
        
        # Initialize model
        input_dim = len(self.feature_cols)
        self.model = LSTMRegressor(input_dim=input_dim, hidden_dim=64, num_layers=2, dropout=0.2).to(self.device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-5)
        
        print("Training LSTM network (Phase 39)...")
        epochs = 60
        best_loss = float('inf')
        patience = 12
        patience_counter = 0
        best_weights = None
        
        self.model.train()
        for epoch in range(1, epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                out = self.model(X_batch)
                loss = criterion(out, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)
                
            train_loss = epoch_loss / len(train_ds)
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for X_v, y_v in val_loader:
                    X_v, y_v = X_v.to(self.device), y_v.to(self.device)
                    out_v = self.model(X_v)
                    v_loss = criterion(out_v, y_v)
                    val_loss += v_loss.item() * len(X_v)
            val_loss /= len(val_ds)
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_weights = self.model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
                
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:2d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}")
                
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}. Best Val Loss: {best_loss:.6f}")
                break
                
        # Load best model weights
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
            
        # -------------------------------------------------------------
        # Test Set Prediction (Phase 40)
        # -------------------------------------------------------------
        # To evaluate on test set of length N, we need the preceding lookback rows
        test_start_idx = len(train_df)
        eval_window = full_features_df.iloc[test_start_idx - self.lookback : test_start_idx + len(test_df)]
        
        X_eval_raw = eval_window[self.feature_cols].to_numpy(dtype=np.float64)
        y_eval_raw = eval_window[['Target']].to_numpy(dtype=np.float64)
        
        # Scale test features using fitted training scaler (NO DATA LEAKAGE)
        X_eval_scaled = self.feature_scaler.transform(X_eval_raw)
        y_eval_scaled = self.target_scaler.transform(y_eval_raw)
        
        X_test_seq, _ = self.create_sequences(X_eval_scaled, y_eval_scaled)
        
        self.model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_seq, dtype=torch.float32).to(self.device)
            lstm_scaled_preds = self.model(X_test_t).cpu().numpy()
            
        # Inverse transform to get rupee price predictions
        lstm_preds = self.target_scaler.inverse_transform(lstm_scaled_preds).flatten()
        
        # Save artifacts
        torch.save(self.model.state_dict(), self.saved_models_dir / "lstm_weights.pt")
        joblib.dump(self.feature_scaler, self.saved_models_dir / "lstm_feature_scaler.joblib")
        joblib.dump(self.target_scaler, self.saved_models_dir / "lstm_target_scaler.joblib")
        joblib.dump({
            "input_dim": input_dim,
            "hidden_dim": 64,
            "num_layers": 2,
            "dropout": 0.2,
            "lookback": self.lookback
        }, self.saved_models_dir / "lstm_config.joblib")
        print("Saved LSTM weights, scalers, and config to models/saved_models/")
        
        return lstm_preds
