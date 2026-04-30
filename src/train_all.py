import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
import joblib
import os
from sklearn.preprocessing import MinMaxScaler
from config import DATA_DIR, LABELS_PATH, DATASETS, SEQ_LEN, HIDDEN_DIM, LATENT_DIM, EPOCHS, SMOOTH_WINDOW, THRESHOLD_PERCENTILE
from pathlib import Path

# Импортируем общие функции (можно вынести в utils.py)
def load_nab_timeseries(csv_rel_path):
    csv_path = os.path.join(DATA_DIR, csv_rel_path)
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    series = df['value'].values.astype(np.float32)
    return df.index, series

def create_sequences(series, seq_len):
    xs = []
    for i in range(len(series) - seq_len + 1):   # +1 обязательно
        xs.append(series[i:i+seq_len])
    return np.array(xs)

def smooth(y, w):
    return np.convolve(y, np.ones(w)/w, mode='same')

class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.latent = nn.Linear(hidden_dim, latent_dim)
        # Декодер принимает latent_dim, выдает hidden_dim
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        # Восстанавливаем исходную размерность (input_dim)
        self.output_layer = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        _, (hidden, _) = self.encoder(x)          # hidden: (1, batch, hidden_dim)
        latent = self.latent(hidden.squeeze(0))    # (batch, latent_dim)
        repeated = latent.unsqueeze(1).repeat(1, x.size(1), 1)  # (batch, seq_len, latent_dim)
        out, _ = self.decoder_lstm(repeated)       # (batch, seq_len, hidden_dim)
        out = self.output_layer(out)               # (batch, seq_len, input_dim)
        return out

def train_on_dataset(dataset_key):
    print(f"\n=== Training on {dataset_key} ===")
    timestamps, series = load_nab_timeseries(dataset_key)

    # Загружаем метки для этого датасета
    with open(LABELS_PATH) as f:
        labels_dict = json.load(f)
    anomaly_windows_raw = labels_dict.get(dataset_key, [])
    anomaly_windows = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in anomaly_windows_raw]

    # Создание окон
    X_all = create_sequences(series, SEQ_LEN)
    t_all = timestamps[SEQ_LEN-1:]

    # Разделение: train до первого аномального окна (если окон нет — весь ряд)
    first_anom = min(w[0] for w in anomaly_windows) if anomaly_windows else t_all[-1] + pd.Timedelta(1)
    train_mask = t_all < first_anom
    if not np.any(train_mask):
        # Если первая же точка аномальна, берём первые 70% как train (компромисс)
        split = int(0.7 * len(X_all))
        train_mask = np.zeros(len(X_all), dtype=bool)
        train_mask[:split] = True
        print("  Внимание: train содержит аномалии (первая точка уже аномальна). Использую 70% разбиение.")

    X_train = X_all[train_mask]
    X_test = X_all[~train_mask]
    t_test = t_all[~train_mask]

    # Масштабирование
    scaler = MinMaxScaler()
    scaler.fit(X_train.reshape(-1, 1))
    X_train_scaled = scaler.transform(X_train.reshape(-1, 1)).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

    # Тензоры 3D
    X_train_t = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)

    model = LSTMAutoencoder(input_dim=1, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train_t)
        loss = criterion(output, X_train_t)
        loss.backward()
        optimizer.step()
    print(f"  Final loss: {loss.item():.6f}")

    # Порог на train
    model.eval()
    with torch.no_grad():
        train_recon = model(X_train_t)
    train_mse = torch.mean((X_train_t - train_recon) ** 2, dim=(1,2)).numpy()
    train_mse_smooth = smooth(train_mse, SMOOTH_WINDOW)
    threshold = np.percentile(train_mse_smooth, THRESHOLD_PERCENTILE)

    # Сохранение
    safe_name = dataset_key.replace('/', '_')
    model_dir = os.path.join("model", safe_name)
    Path(model_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, "lstm_ae.pt"))
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    with open(os.path.join(model_dir, "threshold.txt"), "w") as f:
        f.write(f"{threshold}\n{SEQ_LEN}\n")
    
    print(f"  Saved to {model_dir}")
    return True

if __name__ == "__main__":
    for ds in DATASETS:
        try:
            train_on_dataset(ds)
        except Exception as e:
            print(f"  Ошибка при обработке {ds}: {e}")