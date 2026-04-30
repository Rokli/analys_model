import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_nab_data_and_labels(csv_path, labels_json_path):
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp').sort_index()
    series = df['value']

    # Загружаем метки
    import json
    with open(labels_json_path) as f:
        all_labels = json.load(f)

    # Ключ для этого файла
    relative_key = "data/ec2_cpu_utilization_5f5533.csv"
    anomaly_windows = all_labels.get(relative_key, [])

    # Преобразуем окна в список интервалов
    windows = []
    for w in anomaly_windows:
        start = pd.to_datetime(w[0])
        end = pd.to_datetime(w[1])
        windows.append((start, end))

    # Создаём 2D-массив: [value, lag1]
    vals = series.values.astype(float).reshape(-1, 1)
    lagged = np.roll(vals, shift=1)
    lagged[0] = lagged[1]   # заполним первую точку
    data = np.hstack([vals, lagged])

    return data, series.index, windows