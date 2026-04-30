import numpy as np
import pandas as pd
import json
import os

from sklearn.ensemble import IsolationForest

from config import DATA_DIR, LABELS_PATH, DATASETS, SEQ_LEN
from train_all import load_nab_timeseries, create_sequences


# =========================
# Window metrics (тот же)
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

    return precision, recall, f1


# =========================
# Evaluate IF
# =========================
def evaluate_iforest(dataset_key):
    print(f"\nIForest: {dataset_key}")

    timestamps, series = load_nab_timeseries(dataset_key)

    with open(LABELS_PATH) as f:
        labels = json.load(f)

    anomaly_windows = [
        (pd.to_datetime(s), pd.to_datetime(e))
        for s, e in labels.get(dataset_key, [])
    ]

    X = create_sequences(series, SEQ_LEN)
    t = timestamps[SEQ_LEN - 1:]

    # 👉 flatten окна
    X_flat = X.reshape(len(X), -1)

    # =========================
    # MODEL
    # =========================
    model = IsolationForest(
        contamination=0.05,  # % аномалий
        random_state=42
    )

    model.fit(X_flat)

    scores = model.decision_function(X_flat)
    pred = scores < 0  # аномалии

    # =========================
    # smoothing (как в LSTM)
    # =========================
    pred = pd.Series(pred).rolling(15, center=True).max().fillna(0).astype(bool).values

    precision, recall, f1 = compute_window_metrics(pred, t, anomaly_windows)

    print(f"Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

    return {
        "Dataset": dataset_key,
        "Precision_IF": precision,
        "Recall_IF": recall,
        "F1_IF": f1
    }


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    results = []

    for ds in DATASETS:
        res = evaluate_iforest(ds)
        results.append(res)

    df = pd.DataFrame(results)

    print("\n===== Isolation Forest =====")
    print(df)

    print("\nMean:")
    print(df.mean(numeric_only=True))

    df.to_csv("model/results_iforest.csv", index=False)