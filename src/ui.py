import os
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

LOG_PATH = "model/realtime_log.csv"
LSTM_PATH = "model/results_lstm_ae.csv"
IF_PATH = "model/results_iforest.csv"

st.set_page_config(page_title="Anomaly Detection", layout="wide")

st.title("🚨 Anomaly Detection Dashboard")

# =========================
# LOAD DATA
# =========================
if not os.path.exists(LOG_PATH):
    st.warning("Нет realtime_log.csv — сначала запусти realtime_detect.py")
    st.stop()

df = pd.read_csv(LOG_PATH)
df["timestamp"] = pd.to_datetime(df["timestamp"])

# =========================
# CURRENT STATUS
# =========================
latest = df.iloc[-1]

status = latest["status"]

if status == "ANOMALY":
    st.error(f"🔴 ANOMALY detected")
else:
    st.success("🟢 System normal")

col1, col2, col3 = st.columns(3)

col1.metric("CPU", f"{latest['value']:.2f}")
col2.metric("MSE", f"{latest['mse']:.4f}")
col3.metric("Threshold", f"{latest['threshold']:.4f}")

# =========================
# CPU GRAPH
# =========================
st.subheader("CPU usage")

fig_cpu = go.Figure()

fig_cpu.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["value"],
    name="CPU"
))

# аномалии
anomaly_df = df[df["status"] == "ANOMALY"]

fig_cpu.add_trace(go.Scatter(
    x=anomaly_df["timestamp"],
    y=anomaly_df["value"],
    mode="markers",
    name="Anomaly",
))

st.plotly_chart(fig_cpu, use_container_width=True)

# =========================
# MSE GRAPH
# =========================
st.subheader("Reconstruction error (MSE)")

fig_mse = go.Figure()

fig_mse.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["mse"],
    name="MSE"
))

fig_mse.add_trace(go.Scatter(
    x=df["timestamp"],
    y=df["threshold"],
    name="Threshold",
    line=dict(dash="dash")
))

st.plotly_chart(fig_mse, use_container_width=True)

# =========================
# MODEL COMPARISON
# =========================
st.subheader("Model comparison")

if os.path.exists(LSTM_PATH) and os.path.exists(IF_PATH):
    lstm = pd.read_csv(LSTM_PATH)
    iforest = pd.read_csv(IF_PATH)

    merged = lstm.merge(iforest, on="Dataset")

    st.dataframe(merged)

    st.write("Средние значения:")
    st.write({
        "LSTM F1": merged["F1"].mean(),
        "IForest F1": merged["F1_IF"].mean()
    })
else:
    st.info("Нет файлов результатов моделей")