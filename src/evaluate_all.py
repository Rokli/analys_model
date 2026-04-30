import torch
import numpy as np
import pandas as pd
import json
import joblib
import os
from config import DATA_DIR, LABELS_PATH, DATASETS, SEQ_LEN, HIDDEN_DIM, LATENT_DIM, SMOOTH_WINDOW
from train_all import LSTMAutoencoder, load_nab_timeseries, create_sequences, smooth

def evaluate_dataset(dataset_key):
    safe_name = dataset_key.replace('/', '_')
    model_dir = os.path.join("model", safe_name)
    threshold_file = os.path.join(model_dir, "threshold.txt")
    if not os.path.exists(threshold_file):
        print(f"  Нет модели для {dataset_key}, пропускаем")
        return None

    with open(threshold_file) as f:
        threshold = float(f.readline().strip())
        # seq_len из файла (вторая строка) – мы и так знаем из конфига, можно не читать

    timestamps, series = load_nab_timeseries(dataset_key)
    with open(LABELS_PATH) as f:
        labels_dict = json.load(f)
    anomaly_windows_raw = labels_dict.get(dataset_key, [])
    anomaly_windows = [(pd.to_datetime(s), pd.to_datetime(e)) for s, e in anomaly_windows_raw]

    X = create_sequences(series, SEQ_LEN)
    t_windows = timestamps[SEQ_LEN-1:]

    # Разделение должно совпадать с train (понадобится заново определить train_mask)
    # Чтобы не усложнять, можно сохранять маску, но проще повторить логику из train.
    first_anom = min(w[0] for w in anomaly_windows) if anomaly_windows else t_windows[-1] + pd.Timedelta(1)
    train_mask = t_windows < first_anom
    if not np.any(train_mask):
        split = int(0.7 * len(X))
        train_mask = np.zeros(len(X), dtype=bool)
        train_mask[:split] = True
    X_test = X[~train_mask]
    t_test = t_windows[~train_mask]

    # Масштабирование с сохранённым scaler
    scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
    X_test_scaled = scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)
    X_test_t = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)

    model = LSTMAutoencoder(input_dim=1, hidden_dim=HIDDEN_DIM, latent_dim=LATENT_DIM)
    model.load_state_dict(torch.load(os.path.join(model_dir, "lstm_ae.pt")))
    model.eval()

    with torch.no_grad():
        recon = model(X_test_t)
    mse = torch.mean((X_test_t - recon) ** 2, dim=(1,2)).numpy()
    mse_smooth = smooth(mse, SMOOTH_WINDOW)

    pred = mse_smooth > threshold
    true = np.zeros(len(t_test), dtype=bool)
    for start, end in anomaly_windows:
        true |= (t_test >= start) & (t_test <= end)

    TP = np.sum(pred & true)
    FP = np.sum(pred & ~true)
    FN = np.sum(~pred & true)
    precision = TP / (TP+FP) if (TP+FP) else 0
    recall = TP / (TP+FN) if (TP+FN) else 0
    f1 = 2 * precision * recall / (precision+recall) if (precision+recall) else 0
    return {"Dataset": dataset_key, "Precision": precision, "Recall": recall, "F1": f1}

if __name__ == "__main__":
    results = []
    for ds in DATASETS:
        print(f"Evaluating {ds}...")
        res = evaluate_dataset(ds)
        if res:
            results.append(res)
    df = pd.DataFrame(results)
    print("\n===== Сводная таблица =====")
    print(df.to_string(index=False))
    print("\nСредние метрики:")
    print(f"Precision: {df['Precision'].mean():.3f} ± {df['Precision'].std():.3f}")
    print(f"Recall:    {df['Recall'].mean():.3f} ± {df['Recall'].std():.3f}")
    print(f"F1:        {df['F1'].mean():.3f} ± {df['F1'].std():.3f}")
    df.to_csv("model/results_lstm_ae.csv", index=False)