import os
import time
from collections import deque

import numpy as np
import torch

from config import (
    PROM_QUERY,
    PROM_MODEL_DIR,
    SEQ_LEN,
    HIDDEN_DIM,
    LATENT_DIM,
    REALTIME_INTERVAL
)

from prometheus_client import query_instant
from train_all import LSTMAutoencoder


def main():
    model = LSTMAutoencoder(
        input_dim=1,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM
    )

    model.load_state_dict(torch.load(os.path.join(PROM_MODEL_DIR, "model.pt")))
    model.eval()

    mean, std = np.load(os.path.join(PROM_MODEL_DIR, "norm.npy"))

    with open(os.path.join(PROM_MODEL_DIR, "threshold.txt")) as f:
        threshold = float(f.read().strip())

    window = deque(maxlen=SEQ_LEN)

    print("Real-time anomaly detection started")
    print(f"Threshold: {threshold:.6f}")

    while True:
        value = query_instant(PROM_QUERY)

        if value is None:
            print("No metric value from Prometheus")
            time.sleep(REALTIME_INTERVAL)
            continue

        window.append(value)

        if len(window) < SEQ_LEN:
            print(f"Collecting window: {len(window)}/{SEQ_LEN}, value={value:.2f}")
            time.sleep(REALTIME_INTERVAL)
            continue

        x = np.array(window, dtype=np.float32)
        x_norm = (x - mean) / std

        x_t = torch.tensor(x_norm, dtype=torch.float32).view(1, SEQ_LEN, 1)

        with torch.no_grad():
            recon = model(x_t)

        mse = torch.mean((x_t - recon) ** 2).item()

        status = "ANOMALY" if mse > threshold else "normal"

        print(
            f"value={value:.2f}, "
            f"mse={mse:.6f}, "
            f"threshold={threshold:.6f}, "
            f"status={status}"
        )

        time.sleep(REALTIME_INTERVAL)


if __name__ == "__main__":
    main()