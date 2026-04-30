import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import (
    PROM_QUERY,
    PROM_MODEL_DIR,
    PROM_TRAIN_MINUTES,
    PROM_STEP,
    SEQ_LEN,
    HIDDEN_DIM,
    LATENT_DIM,
    EPOCHS,
    SMOOTH_WINDOW
)

from prometheus_client import query_range
from train_all import LSTMAutoencoder, create_sequences, smooth


def train_prometheus_model():
    print("Collecting Prometheus metrics...")

    df = query_range(
        PROM_QUERY,
        minutes=PROM_TRAIN_MINUTES,
        step=PROM_STEP
    )

    print(df.head())
    print(df.describe())

    series = df["value"].values.astype(np.float32)

    if len(series) < SEQ_LEN:
        raise RuntimeError("Недостаточно точек для обучения")

    X = create_sequences(series, SEQ_LEN)

    mean = np.mean(X)
    std = np.std(X) + 1e-8

    X_norm = (X - mean) / std
    X_t = torch.tensor(X_norm, dtype=torch.float32).unsqueeze(-1)

    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()

        out = model(X_t)
        loss = criterion(out, X_t)

        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    model.eval()

    with torch.no_grad():
        recon = model(X_t)

    mse = torch.mean((X_t - recon) ** 2, dim=(1, 2)).numpy()
    mse_smooth = smooth(mse, SMOOTH_WINDOW)

    #threshold = np.mean(mse_smooth) + 1.5 * np.std(mse_smooth)
    threshold = np.percentile(mse_smooth, 85)
    Path(PROM_MODEL_DIR).mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(PROM_MODEL_DIR, "model.pt"))
    np.save(os.path.join(PROM_MODEL_DIR, "norm.npy"), np.array([mean, std]))

    with open(os.path.join(PROM_MODEL_DIR, "threshold.txt"), "w") as f:
        f.write(str(threshold))

    print(f"Model saved to {PROM_MODEL_DIR}")
    print(f"Threshold: {threshold:.6f}")


if __name__ == "__main__":
    train_prometheus_model()