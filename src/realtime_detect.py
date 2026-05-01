import os
import time
import csv
from datetime import datetime
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
from state_store import ensure_storage, save_state, add_alert


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

    os.makedirs("model", exist_ok=True)
    ensure_storage()

    log_file = "model/realtime_log.csv"

    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp",
            "value",
            "delta",
            "mse",
            "threshold",
            "raw_status",
            "status"
        ])

    window = deque(maxlen=SEQ_LEN)

    current_status = "normal"
    last_status = "normal"

    anomaly_counter = 0
    normal_counter = 0
    prev_value = None

    print("Real-time anomaly detection started")
    print(f"Threshold: {threshold:.6f}")
    print(f"Log file: {log_file}")

    while True:
        value = query_instant(PROM_QUERY)

        if value is None:
            print("No metric value from Prometheus")
            time.sleep(REALTIME_INTERVAL)
            continue

        if prev_value is None:
            delta = 0.0
        else:
            delta = value - prev_value

        prev_value = value

        window.append(value)

        if len(window) < SEQ_LEN:
            print(
                f"Collecting window: {len(window)}/{SEQ_LEN}, "
                f"value={value:.2f}, delta={delta:.2f}"
            )

            state = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "value": value,
                "delta": delta,
                "mse": None,
                "threshold": threshold,
                "raw_status": "collecting",
                "status": "collecting"
            }

            save_state(state)

            time.sleep(REALTIME_INTERVAL)
            continue

        x = np.array(window, dtype=np.float32)
        x_norm = (x - mean) / std

        x_t = torch.tensor(x_norm, dtype=torch.float32).view(1, SEQ_LEN, 1)

        with torch.no_grad():
            recon = model(x_t)

        mse = torch.mean((x_t - recon) ** 2).item()

        raw_status = "ANOMALY" if mse > threshold else "normal"

        if raw_status == "ANOMALY":
            anomaly_counter += 1
            normal_counter = 0
        else:
            normal_counter += 1
            anomaly_counter = 0

        # =========================
        # 4 состояния системы
        # =========================
        # normal      — нормальная работа
        # load_start  — появляется нагрузка
        # anomaly     — аномальное состояние
        # recovery    — разгрузка после нагрузки

        if current_status == "normal":
            if delta > 5:
                current_status = "load_start"

        elif current_status == "load_start":
            if anomaly_counter >= 2:
                current_status = "anomaly"
            elif delta < -2 and raw_status == "normal":
                current_status = "normal"

        elif current_status == "anomaly":
            if delta < -5:
                current_status = "recovery"

        elif current_status == "recovery":
            if anomaly_counter >= 2:
                current_status = "anomaly"
            elif normal_counter >= 3:
                current_status = "normal"

        status = current_status

        timestamp = datetime.now().isoformat(timespec="seconds")

        state = {
            "timestamp": timestamp,
            "value": value,
            "delta": delta,
            "mse": mse,
            "threshold": threshold,
            "raw_status": raw_status,
            "status": status
        }

        save_state(state)

        if status != last_status:
            add_alert(
                old_status=last_status,
                new_status=status,
                value=value,
                mse=mse,
                threshold=threshold
            )
            last_status = status

        print(
            f"value={value:.2f}, "
            f"delta={delta:.2f}, "
            f"mse={mse:.6f}, "
            f"threshold={threshold:.6f}, "
            f"raw={raw_status}, "
            f"status={status}"
        )

        with open(log_file, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                timestamp,
                value,
                delta,
                mse,
                threshold,
                raw_status,
                status
            ])

        time.sleep(REALTIME_INTERVAL)


if __name__ == "__main__":
    main()