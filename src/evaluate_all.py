import torch
import numpy as np
import pandas as pd
import json
import os
import matplotlib.pyplot as plt

from config import DATA_DIR, LABELS_PATH, DATASETS, SEQ_LEN, HIDDEN_DIM, LATENT_DIM, SMOOTH_WINDOW
from train_all import LSTMAutoencoder, load_nab_timeseries, create_sequences, smooth


# =========================
# Window-based metrics
# =========================

def compute_window_metrics(pred, t_test, anomaly_windows):
    detected = []
    in_block = False

    for i, p in enumerate(pred):
        if p and not in_block:
            start = t_test[i]
            in_block = True
        elif not p and in_block:
            detected.append((start, t_test[i]))
            in_block = False

    if in_block:
        detected.append((start, t_test[-1]))

    TP = 0
    for real_start, real_end in anomaly_windows:
        for pred_start, pred_end in detected:
            if pred_start <= real_end and pred_end >= real_start:
                TP += 1
                break

    FP = len(detected) - TP
    FN = len(anomaly_windows) - TP

    precision = TP / (TP + FP) if (TP + FP) else 0
    recall = TP / (TP + FN) if (TP + FN) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return precision, recall, f1, TP, FP, FN


# =========================
# Evaluate
# =========================

def evaluate_dataset(dataset_key):
    print(f"\nEvaluating {dataset_key}")

    safe_name = dataset_key.replace('/', '_')
    model_dir = os.path.join("model", safe_name)

    if not os.path.exists(model_dir):
        return None

    threshold = float(open(os.path.join(model_dir, "threshold.txt")).read())
    mean, std = np.load(os.path.join(model_dir, "norm.npy"))

    timestamps, series = load_nab_timeseries(dataset_key)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    anomaly_windows = [
        (pd.to_datetime(s), pd.to_datetime(e))
        for s, e in labels.get(dataset_key, [])
    ]

    X = create_sequences(series, SEQ_LEN)
    t = timestamps[SEQ_LEN - 1:]

    X = (X - mean) / std
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)

    model = LSTMAutoencoder(1, HIDDEN_DIM, LATENT_DIM)
    model.load_state_dict(torch.load(os.path.join(model_dir, "model.pt")))
    model.eval()

    with torch.no_grad():
        recon = model(X_t)

    mse = torch.mean((X_t - recon) ** 2, dim=(1, 2)).numpy()
    mse = smooth(mse, SMOOTH_WINDOW)

    pred = mse > threshold
    pred = pd.Series(pred).rolling(5, center=True).max().fillna(0).astype(bool).values

    precision, recall, f1, TP, FP, FN = compute_window_metrics(pred, t, anomaly_windows)

    print(f"TP={TP}, FP={FP}, FN={FN}")
    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return {
        "Dataset": dataset_key,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


if __name__ == "__main__":
    results = []

    for ds in DATASETS:
        res = evaluate_dataset(ds)
        if res:
            results.append(res)

    df = pd.DataFrame(results)

    print("\n===== RESULT =====")
    print(df)

    print("\nMean:")
    print(df.mean(numeric_only=True))