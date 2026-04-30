import torch
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import joblib
import matplotlib.pyplot as plt
from train_torch import Autoencoder, load_nab  # чтобы не дублировать загрузку

if __name__ == "__main__":
    # Загружаем данные повторно (можно вынести в общую функцию, но для понятности дублируем)
    csv_path = "data/ec2_cpu_utilization_5f5533.csv"
    labels_path = "labels/combined_windows.json"
    data, timestamps, anomaly_windows = load_nab(csv_path, labels_path)

    split_idx = int(0.7 * len(data))
    train_data = data[:split_idx]   # не используется, но нужно для scaler
    test_data = data[split_idx:]
    test_timestamps = timestamps[split_idx:]

    scaler = joblib.load("model/scaler.pkl")
    test_scaled = scaler.transform(test_data)

    model = Autoencoder(input_dim=2)
    model.load_state_dict(torch.load("model/autoencoder.pt"))
    model.eval()

    X_test = torch.tensor(test_scaled, dtype=torch.float32)
    with torch.no_grad():
        recon = model(X_test)
    mse = torch.mean((X_test - recon) ** 2, dim=1).numpy()

    # Загружаем порог
    with open("model/threshold.txt") as f:
        threshold = float(f.read())

    # Предсказанные аномалии: любая точка с mse > threshold
    pred_anomaly = mse > threshold

    # Создаём истинные метки: для каждой временной метки test определяем, попадает ли она в окно аномалий
    true_labels = np.zeros(len(test_timestamps), dtype=bool)
    for start, end in anomaly_windows:
        mask = (test_timestamps >= start) & (test_timestamps <= end)
        true_labels |= mask

    # Оценка
    TP = np.sum(pred_anomaly & true_labels)
    FP = np.sum(pred_anomaly & ~true_labels)
    FN = np.sum(~pred_anomaly & true_labels)
    precision = TP / (TP + FP) if (TP+FP) > 0 else 0
    recall = TP / (TP + FN) if (TP+FN) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision+recall) > 0 else 0

    print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

    # График
    plt.figure(figsize=(12, 6))
    plt.plot(test_timestamps, mse, label='MSE')
    plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
    # выделим истинные аномалии
    in_anomaly = np.where(true_labels)[0]
    if len(in_anomaly) > 0:
        plt.fill_between(test_timestamps, 0, mse.max(),
                         where=true_labels, alpha=0.3, color='orange', label='True anomalies')
    plt.legend()
    plt.title("MSE and anomalies")
    plt.xlabel("Time")
    plt.ylabel("Reconstruction MSE")
    plt.tight_layout()
    plt.savefig("model/nab_anomalies.png")
    plt.show()