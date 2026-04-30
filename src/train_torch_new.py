import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import json
from sklearn.preprocessing import MinMaxScaler
import joblib

class Autoencoder(nn.Module):
    def __init__(self, input_dim=2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# === Загрузка NAB ===
def load_nab(csv_path, labels_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    series = df['value'].values.astype(float)

    # создаём второй признак: lag1
    lagged = np.roll(series, 1)
    lagged[0] = lagged[1]
    data = np.column_stack([series, lagged])

    with open(labels_path) as f:
        labels_dict = json.load(f)
    # ключ в json — относительный путь от data/
    key = "data/ec2_cpu_utilization_5f5533.csv"
    windows = []
    for start_str, end_str in labels_dict[key]:
        windows.append((pd.to_datetime(start_str), pd.to_datetime(end_str)))
    return data, df.index, windows

if __name__ == "__main__":
    csv_path = "data/ec2_cpu_utilization_5f5533.csv"
    labels_path = "labels/combined_windows.json"

    data, timestamps, anomaly_windows = load_nab(csv_path, labels_path)
    print("Загружено точек:", len(data), "аномальных окон:", len(anomaly_windows))

    # Разделение 70/30 (первые 70% тренировка)
    split_idx = int(0.7 * len(data))
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    test_timestamps = timestamps[split_idx:]

    # Масштабирование
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)

    # Тензоры
    X_train = torch.tensor(train_scaled, dtype=torch.float32)

    # Модель
    model = Autoencoder(input_dim=2)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Обучение
    print("Обучение...")
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, X_train)
        loss.backward()
        optimizer.step()
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

    # Вычисляем порог на train
    model.eval()
    with torch.no_grad():
        recon_train = model(X_train)
    train_mse = torch.mean((X_train - recon_train) ** 2, dim=1).numpy()
    threshold = np.percentile(train_mse, 95)
    print(f"Порог (95-й перцентиль MSE) = {threshold:.6f}")

    # Сохраняем модель, scaler, threshold
    torch.save(model.state_dict(), "model/autoencoder.pt")
    joblib.dump(scaler, "model/scaler.pkl")
    with open("model/threshold.txt", "w") as f:
        f.write(str(threshold))

    print("Модель, scaler и порог сохранены.")