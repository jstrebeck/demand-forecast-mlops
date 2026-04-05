"""
LSTM-based sequence model for demand forecasting.

Takes a sequence of weekly feature vectors and predicts the next week's
order quantity per SKU.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from src.models.baseline import FEATURE_COLS, TARGET_COL, compute_metrics


class DemandDataset(Dataset):
    """Sliding-window dataset that produces (sequence, target) pairs per SKU."""

    def __init__(self, df: pd.DataFrame, seq_len: int = 8):
        self.seq_len = seq_len
        self.samples = []

        for _, group in df.groupby("sku"):
            group = group.sort_values("date")
            features = group[FEATURE_COLS].values.astype(np.float32)
            targets = group[TARGET_COL].values.astype(np.float32)

            for i in range(len(group) - seq_len):
                x = features[i : i + seq_len]
                y = targets[i + seq_len]
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x), torch.tensor(y)


class DemandLSTM(nn.Module):
    """LSTM model for weekly demand prediction."""

    def __init__(
        self,
        input_size: int = len(FEATURE_COLS),
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use the last time step's output
        last = lstm_out[:, -1, :]
        return self.fc(last).squeeze(-1)


def train_lstm(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    seq_len: int = 8,
    hidden_size: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 1e-3,
    epochs: int = 30,
    batch_size: int = 256,
):
    """Train the LSTM model and return model, predictions, metrics, and loss history."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = DemandDataset(train_df, seq_len=seq_len)
    test_ds = DemandDataset(test_df, seq_len=seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    model = DemandLSTM(
        input_size=len(FEATURE_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    loss_history = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            preds = model(x_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / n_batches
        loss_history.append(avg_loss)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    # Evaluate
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    metrics = compute_metrics(y_true, y_pred)

    return model, y_pred, metrics, loss_history
