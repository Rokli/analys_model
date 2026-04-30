import torch
import numpy as np
import pandas as pd
import json
import joblib
import matplotlib.pyplot as plt
from train_torch_new import WindowAutoencoder, load_nab_timeseries, create_sequences


def smooth(y, w):
    return np.convolve(y, np.ones(w)/w, mode='same')

# Параметры сохранены в файле threshold.txt
with open("model/window_threshold.txt") as f:
    threshold = float(f.readline().strip())
    SEQ_LEN = int(f.readline().strip())

print(f"Используем seq_len={SEQ_LEN}, threshold={threshold:.6f}")

CSV = "data/ec2_cpu_utilization_5f5533.csv"
LABELS = "labels/combined_windows.json"
KEY = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

series, timestamps, anomaly_windows = load_nab_timeseries(CSV, LABELS, KEY)
X = create_sequences(series, SEQ_LEN)
t_windows = timestamps[SEQ_LEN-1:]

# Загружаем scaler
scaler = joblib.load("model/window_scaler.pkl")
X_scaled = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)

model = WindowAutoencoder(input_dim=SEQ_LEN)
model.load_state_dict(torch.load("model/window_ae.pt"))
model.eval()

X_t = torch.tensor(X_scaled, dtype=torch.float32)
with torch.no_grad():
    recon = model(X_t)
mse = torch.mean((X_t - recon) ** 2, dim=1).numpy()
mse_smooth = smooth(mse, 5)

# Предсказанные аномалии
pred_anomaly = mse_smooth > threshold

# Истинные метки: окно считается аномальным, если хоть одна точка внутри окна попадает в размеченный интервал
true_anomaly = np.zeros(len(t_windows), dtype=bool)
for start, end in anomaly_windows:
    mask = (t_windows >= start) & (t_windows <= end)
    true_anomaly |= mask

TP = np.sum(pred_anomaly & true_anomaly)
FP = np.sum(pred_anomaly & ~true_anomaly)
FN = np.sum(~pred_anomaly & true_anomaly)
precision = TP / (TP+FP) if (TP+FP) else 0
recall = TP / (TP+FN) if (TP+FN) else 0
f1 = 2 * precision * recall / (precision+recall) if (precision+recall) else 0

print(f"Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")

# Визуализация
plt.figure(figsize=(12, 5))
plt.plot(t_windows, mse_smooth, label='Smoothed MSE (window=5)')
plt.axhline(threshold, color='r', linestyle='--', label='Threshold')
# выделяем истинные аномалии
in_anomaly = np.where(true_anomaly)[0]
if len(in_anomaly) > 0:
    plt.fill_between(t_windows, 0, mse_smooth.max(), where=true_anomaly,
                     alpha=0.3, color='orange', label='True anomalies')
plt.legend()
plt.title("Window Autoencoder – Smoothed Reconstruction Error")
plt.xlabel("Time")
plt.ylabel("MSE (smoothed)")
plt.tight_layout()
plt.savefig("model/window_ae_result.png")
plt.show()