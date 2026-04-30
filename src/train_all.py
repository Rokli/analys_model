import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import os
from pathlib import Path

from config import DATA_DIR, LABELS_PATH, DATASETS, SEQ_LEN, HIDDEN_DIM, LATENT_DIM, EPOCHS, SMOOTH_WINDOW


# =========================
# Utils
# =========================

def load_nab_timeseries(csv_rel_path):
    csv_path = os.path.join(DATA_DIR, csv_rel_path)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    return df.index, df['value'].values.astype(np.float32)


def create_sequences(series, seq_len):
    return np.array([series[i:i+seq_len] for i in range(len(series) - seq_len + 1)])


def smooth(y, w):
    return np.convolve(y, np.ones(w)/w, mode='same')


# =========================
# Model
# =========================

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(0.2)
        self.decoder = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)
        z = self.dropout(self.latent(hidden.squeeze(0)))
        z = z.unsqueeze(1).repeat(1, x.size(1), 1)
        out, _ = self.decoder(z)
        return self.output(out)


# =========================
# Train
# =========================

def train_on_dataset(dataset_key):
    print(f"\n=== Training on {dataset_key} ===")

    timestamps, series = load_nab_timeseries(dataset_key)

    with open(LABELS_PATH) as f:
        labels_dict = json.load(f)

    anomaly_windows = [
        (pd.to_datetime(s), pd.to_datetime(e))
        for s, e in labels_dict.get(dataset_key, [])
    ]

    X_all = create_sequences(series, SEQ_LEN)
    t_all = timestamps[SEQ_LEN - 1:]

    first_anom = min(w[0] for w in anomaly_windows) if anomaly_windows else t_all[-1]
    train_mask = t_all < first_anom

    if not np.any(train_mask):
        split = int(0.7 * len(X_all))
        train_mask = np.zeros(len(X_all), dtype=bool)
        train_mask[:split] = True

    X_train = X_all[train_mask]

    # =========================
    # Нормализация (z-score)
    # =========================
    mean = np.mean(X_train)
    std = np.std(X_train) + 1e-8

    X_train = (X_train - mean) / std
    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)

    model = LSTMAutoencoder(1, HIDDEN_DIM, LATENT_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        out = model(X_train_t)
        loss = criterion(out, X_train_t)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    print(f"Final loss: {loss.item():.6f}")

    # =========================
    # Threshold
    # =========================
    model.eval()
    with torch.no_grad():
        recon = model(X_train_t)

    mse = torch.mean((X_train_t - recon) ** 2, dim=(1, 2)).numpy()
    mse_smooth = smooth(mse, SMOOTH_WINDOW)

    mean_err = np.mean(mse_smooth)
    std_err = np.std(mse_smooth)

    threshold = mean_err + 2.5 * std_err

    print(f"Threshold: {threshold:.6f}")

    # =========================
    # Save
    # =========================
    safe_name = dataset_key.replace('/', '_')
    model_dir = os.path.join("model", safe_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(model_dir, "model.pt"))
    np.save(os.path.join(model_dir, "norm.npy"), np.array([mean, std]))

    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    print(f"Saved to {model_dir}")


if __name__ == "__main__":
    for ds in DATASETS:
        try:
            train_on_dataset(ds)
        except Exception as e:
            print(f"Ошибка {ds}: {e}")