import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from sklearn.preprocessing import MinMaxScaler
import joblib

# ------------------------------
# 1. Загрузка NAB и разметка окон
# ------------------------------
def load_nab_timeseries(csv_path, labels_path, key):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    series = df['value'].values.astype(np.float32)

    with open(labels_path) as f:
        labels_dict = json.load(f)
    windows = []
    for start_str, end_str in labels_dict[key]:
        windows.append((pd.to_datetime(start_str), pd.to_datetime(end_str)))
    return series, df.index, windows

# ------------------------------
# 2. Формирование окон (seq_len)
# ------------------------------
def create_sequences(series, seq_len):
    xs = []
    for i in range(len(series) - seq_len + 1):   # +1
        xs.append(series[i:i+seq_len])
    return np.array(xs)

# ------------------------------
# 3. Автоэнкодер (полносвязный, для развёрнутого окна)
# ------------------------------
class WindowAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 8)      # узкое место
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# ------------------------------
# 4. Обучение и сохранение
# ------------------------------
if __name__ == "__main__":
    SEQUENCE_LENGTH = 32
    CSV = "data/ec2_cpu_utilization_5f5533.csv"
    LABELS = "labels/combined_windows.json"
    KEY = "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv"

    series, timestamps, anomaly_windows = load_nab_timeseries(CSV, LABELS, KEY)

    # Создаём окна
    X_all = create_sequences(series, SEQUENCE_LENGTH)
    # Временные метки для окон берём начало окна (можно середину, не принципиально)
    t_all = timestamps[SEQUENCE_LENGTH-1:]  # последний индекс окна

    # Определяем train без аномалий: берём непрерывный участок до первого аномального окна
    first_anom = min(w[0] for w in anomaly_windows) if anomaly_windows else t_all[-1]
    train_mask = t_all < first_anom
    X_train = X_all[train_mask]
    # test – всё, что после (содержит аномалии)
    test_mask = ~train_mask
    X_test = X_all[test_mask]
    t_test = t_all[test_mask]

    # Масштабирование (по всем точкам train, применяем к test)
    # Для окон преобразуем форму: (samples, seq_len) -> (samples * seq_len,) для fit, но проще:
    scaler = MinMaxScaler()
    # fit по всем пикселям train
    scaler.fit(X_train.reshape(-1, 1))
    # transform
    X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Тензоры
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    X_test_t  = torch.tensor(X_test_scaled, dtype=torch.float32)

    # Модель
    model = WindowAutoencoder(input_dim=SEQUENCE_LENGTH)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение (можно больше эпох, добавить early stopping)
    EPOCHS = 100
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, X_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Ошибка восстановления на train (для порога)
    model.eval()
    with torch.no_grad():
        train_recon = model(X_train_t)
    train_mse = torch.mean((X_train_t - train_recon) ** 2, dim=1).numpy()

    # Сглаживание ошибки (окно 5)
    def smooth(y, w):
        return np.convolve(y, np.ones(w)/w, mode='same')

    train_mse_smooth = smooth(train_mse, 5)
    threshold = np.percentile(train_mse_smooth, 95)

    # Сохраняем всё
    torch.save(model.state_dict(), "model/window_ae.pt")
    joblib.dump(scaler, "model/window_scaler.pkl")
    with open("model/window_threshold.txt", "w") as f:
        f.write(f"{threshold}\n{SEQUENCE_LENGTH}\n")

    print(f"Порог (95% перцентиль сглаженной MSE): {threshold:.6f}")
    print("Обучение завершено, модель, scaler и порог сохранены.")