import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

LOG_PATH = "model/realtime_log.csv"
LSTM_PATH = "model/results_lstm_ae.csv"
IF_PATH = "model/results_iforest.csv"

st.set_page_config(page_title="VM Anomaly Detection", layout="wide")

# автообновление каждые 5 секунд
st_autorefresh(interval=5000, key="dashboard_refresh")

st.title("Мониторинг аномалий виртуальной машины")

if not os.path.exists(LOG_PATH):
    st.warning("Нет model/realtime_log.csv. Сначала запусти realtime_detect.py")
    st.stop()

df = pd.read_csv(LOG_PATH)

if df.empty:
    st.warning("Лог пока пустой")
    st.stop()

df["timestamp"] = pd.to_datetime(df["timestamp"])

latest = df.iloc[-1]
status = latest["status"]

# =========================
# STATUS CARD
# =========================

if status == "normal":
    st.success("🟢 Нормальное состояние")
elif status == "load_start":
    st.warning("🟠 Появляется нагрузка")
elif status == "anomaly":
    st.error("🔴 Аномалия")
elif status == "recovery":
    st.info("🔵 Разгрузка")
else:
    st.warning(f"Неизвестный статус: {status}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CPU, %", f"{latest['value']:.2f}")
col2.metric("Delta", f"{latest['delta']:.2f}")
col3.metric("MSE", f"{latest['mse']:.4f}")
col4.metric("Threshold", f"{latest['threshold']:.4f}")

# =========================
# COLOR MAP
# =========================

status_colors = {
    "normal": "green",
    "load_start": "orange",
    "anomaly": "red",
    "recovery": "blue",
}

# =========================
# CPU GRAPH
# =========================

st.subheader("CPU usage")

fig_cpu = go.Figure()

fig_cpu.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["value"],
    mode="lines",
    name="CPU"
))

for s, color in status_colors.items():
    part = df[df["status"] == s]
    if not part.empty:
        fig_cpu.add_trace(go.Scatter(
            x=part["timestamp"],
            y=part["value"],
            mode="markers",
            name=s,
            marker=dict(color=color, size=8)
        ))

fig_cpu.update_layout(
    xaxis_title="Время",
    yaxis_title="CPU, %",
    height=400
)

st.plotly_chart(fig_cpu, use_container_width=True)

# =========================
# MSE GRAPH
# =========================

st.subheader("Ошибка реконструкции модели")

fig_mse = go.Figure()

fig_mse.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["mse"],
    mode="lines",
    name="MSE"
))

fig_mse.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["threshold"],
    mode="lines",
    name="Threshold",
    line=dict(dash="dash")
))

for s, color in status_colors.items():
    part = df[df["status"] == s]
    if not part.empty:
        fig_mse.add_trace(go.Scatter(
            x=part["timestamp"],
            y=part["mse"],
            mode="markers",
            name=f"{s} points",
            marker=dict(color=color, size=8),
            showlegend=False
        ))

fig_mse.update_layout(
    xaxis_title="Время",
    yaxis_title="MSE",
    height=400
)

st.plotly_chart(fig_mse, use_container_width=True)

# =========================
# LOG TABLE
# =========================

with st.expander("Последние записи лога"):
    st.dataframe(df.tail(30), use_container_width=True)

# =========================
# MODEL COMPARISON
# =========================

st.subheader("Сравнение моделей на NAB")

if os.path.exists(LSTM_PATH) and os.path.exists(IF_PATH):
    lstm = pd.read_csv(LSTM_PATH)
    iforest = pd.read_csv(IF_PATH)

    merged = lstm.merge(iforest, on="Dataset")

    col_a, col_b = st.columns(2)

    col_a.metric("Средний F1 LSTM Autoencoder", f"{merged['F1'].mean():.3f}")
    col_b.metric("Средний F1 Isolation Forest", f"{merged['F1_IF'].mean():.3f}")

    st.dataframe(merged, use_container_width=True)
else:
    st.info("Файлы сравнения моделей пока не найдены")