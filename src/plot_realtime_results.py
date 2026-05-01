import os
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "model/realtime_log.csv"
OUT_PATH = "model/realtime_result_plot.png"

STATUS_COLORS = {
    "normal": "green",
    "load_start": "orange",
    "anomaly": "red",
    "recovery": "blue",
}


def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Не найден файл: {LOG_PATH}")

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        raise RuntimeError("Файл realtime_log.csv пустой")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # CPU
    ax1.plot(df["timestamp"], df["value"], label="CPU usage, %")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("CPU usage, %")

    # Цветные точки статусов на CPU
    for status, color in STATUS_COLORS.items():
        part = df[df["status"] == status]
        if not part.empty:
            ax1.scatter(
                part["timestamp"],
                part["value"],
                label=status,
                s=25,
                color=color
            )

    # MSE на второй оси
    ax2 = ax1.twinx()
    ax2.plot(df["timestamp"], df["mse"], label="MSE", linestyle="--")
    ax2.plot(df["timestamp"], df["threshold"], label="Threshold", linestyle=":")
    ax2.set_ylabel("Reconstruction error")

    # Легенды
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()

    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.title("Real-time anomaly detection result")
    plt.xticks(rotation=30)
    plt.tight_layout()

    os.makedirs("model", exist_ok=True)
    plt.savefig(OUT_PATH, dpi=300)
    plt.close()

    print(f"График сохранён: {OUT_PATH}")


if __name__ == "__main__":
    main()