"""
Deep Learning Module: PyTorch LSTM for Stock Price Sequence Forecasting.
Implements multi-layer LSTM neural network with sequence windowing,
min-max normalization, early stopping, and checkpointing.
"""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

MODEL_SAVE_DIR = Path(__file__).resolve().parent.parent.parent / "models" / "saved_models"


class LSTMNetwork(nn.Module):
    """PyTorch 2-layer LSTM with Dropout and Dense Prediction Head."""
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the output of the final time step
        last_step_out = out[:, -1, :]
        h = self.relu(self.fc1(last_step_out))
        prediction = self.fc2(h)
        return prediction


def create_sequences(data: np.ndarray, targets: np.ndarray, seq_len: int = 30):
    """
    Creates sliding sequence windows:
    X[i] = data[i : i + seq_len]
    y[i] = targets[i + seq_len - 1]
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(targets[i + seq_len])
    return np.array(X), np.array(y)


class LSTMForecaster:
    """
    High-level LSTM forecaster wrapper handling scaling, sequence formation,
    training, evaluation, and inference.
    """
    def __init__(self, seq_len: int = 30, hidden_size: int = 64, num_layers: int = 2,
                 lr: float = 0.001, epochs: int = 60, batch_size: int = 32):
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.name = f"Deep Learning (LSTM - {seq_len}d Window)"

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray = None, y_val: np.ndarray = None):
        """
        Fits LSTM model using sequence windows.
        Scalers are fitted STRICTLY on training data to prevent lookahead leakage.
        """
        # Fit scalers
        X_train_scaled = self.feature_scaler.fit_transform(X_train)
        y_train_scaled = self.target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()

        # Create training sequences
        X_seq_train, y_seq_train = create_sequences(X_train_scaled, y_train_scaled, self.seq_len)

        # Convert to PyTorch tensors
        train_dataset = TensorDataset(
            torch.tensor(X_seq_train, dtype=torch.float32),
            torch.tensor(y_seq_train, dtype=torch.float32).unsqueeze(1)
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=False)

        val_loader = None
        if X_val is not None and y_val is not None:
            # Transform validation set with fitted scalers
            X_val_scaled = self.feature_scaler.transform(X_val)
            y_val_scaled = self.target_scaler.transform(y_val.reshape(-1, 1)).flatten()
            X_seq_val, y_seq_val = create_sequences(X_val_scaled, y_val_scaled, self.seq_len)
            if len(X_seq_val) > 0:
                val_dataset = TensorDataset(
                    torch.tensor(X_seq_val, dtype=torch.float32),
                    torch.tensor(y_seq_val, dtype=torch.float32).unsqueeze(1)
                )
                val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Initialize network
        input_size = X_train.shape[1]
        self.model = LSTMNetwork(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=0.2
        ).to(self.device)

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)

        print(f"[LSTM] Training LSTM on {self.device} for {self.epochs} epochs (Batch size: {self.batch_size})...")
        best_val_loss = float("inf")
        patience = 12
        patience_counter = 0

        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                preds = self.model(batch_x)
                loss = criterion(preds, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(batch_x)

            epoch_train_loss = total_loss / len(train_dataset)

            # Validation check
            if val_loader:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(self.device), vy.to(self.device)
                        v_preds = self.model(vx)
                        v_loss = criterion(v_preds, vy)
                        val_loss += v_loss.item() * len(vx)
                epoch_val_loss = val_loss / len(val_dataset)
                self.model.train()

                if epoch_val_loss < best_val_loss:
                    best_val_loss = epoch_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{self.epochs}] Train Loss: {epoch_train_loss:.6f} | Val Loss: {epoch_val_loss:.6f}")

                if patience_counter >= patience:
                    print(f"  Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"  Epoch [{epoch+1}/{self.epochs}] Train Loss: {epoch_train_loss:.6f}")

        return self

    def predict(self, full_X: np.ndarray, test_indices: np.ndarray) -> np.ndarray:
        """
        Generates predictions for test samples using sliding sequences from full_X.
        Inverse-transforms scaled predictions to actual Nifty 500 price levels.
        """
        self.model.eval()
        X_scaled = self.feature_scaler.transform(full_X)

        predictions = []
        with torch.no_grad():
            for idx in test_indices:
                if idx < self.seq_len:
                    # Not enough history for full window; pad with first available row
                    pad_len = self.seq_len - idx
                    window = np.vstack([np.repeat(X_scaled[[0]], pad_len, axis=0), X_scaled[:idx]])
                else:
                    window = X_scaled[idx - self.seq_len : idx]

                window_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(self.device)
                pred_scaled = self.model(window_tensor).cpu().numpy().flatten()[0]
                predictions.append(pred_scaled)

        pred_scaled_arr = np.array(predictions).reshape(-1, 1)
        pred_prices = self.target_scaler.inverse_transform(pred_scaled_arr).flatten()
        return pred_prices

    def save_model(self, output_dir: Path = MODEL_SAVE_DIR):
        """Saves model weights and scalers."""
        output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_dir / "lstm_weights.pt")
        joblib.dump(self.feature_scaler, output_dir / "lstm_feature_scaler.joblib")
        joblib.dump(self.target_scaler, output_dir / "lstm_target_scaler.joblib")
        joblib.dump({
            "seq_len": self.seq_len,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "input_size": self.feature_scaler.n_features_in_
        }, output_dir / "lstm_config.joblib")
        print(f"[LSTM] Saved LSTM artifacts to: {output_dir}")
