import requests
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from config import API_URL

st.set_page_config(page_title="VM Monitoring Client", layout="wide")

st_autorefresh(interval=5000, key="refresh")

st.title("Клиент мониторинга виртуальной машины")

def get_json(path):
    r = requests.get(f"{API_URL}{path}", timeout=5)
    r.raise_for_status()
    return r.json()

try:
    status = get_json("/status")
    metrics = get_json("/metrics?limit=200")
    alerts = get_json("/alerts")
except Exception as e:
    st.error(f"Не удалось подключиться к API: {e}")
    st.stop()

current_status = status.get("status", "unknown")

if current_status == "normal":
    st.success("🟢 Нормальное состояние")
elif current_status == "load_start":
    st.warning("🟠 Появляется нагрузка")
elif current_status == "anomaly":
    st.error("🔴 Аномалия")
elif current_status == "recovery":
    st.info("🔵 Разгрузка")
elif current_status == "collecting":
    st.info("⏳ Сбор начального окна")
else:
    st.warning(f"Неизвестный статус: {current_status}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("CPU, %", f"{status.get('value') or 0:.2f}")
col2.metric("Delta", f"{status.get('delta') or 0:.2f}")
col3.metric("MSE", f"{status.get('mse') or 0:.4f}")
col4.metric("Threshold", f"{status.get('threshold') or 0:.4f}")

if metrics:
    df = pd.DataFrame(metrics)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    st.subheader("CPU usage")

    fig_cpu = go.Figure()
    fig_cpu.add_trace(go.Scatter(
        x=df["timestamp"],
        y=df["value"],
        mode="lines",
        name="CPU"
    ))

    st.plotly_chart(fig_cpu, use_container_width=True)

    st.subheader("Ошибка реконструкции")

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
        name="Threshold"
    ))

    st.plotly_chart(fig_mse, use_container_width=True)

st.subheader("Алерты")

if alerts:
    alerts_df = pd.DataFrame(alerts)
    st.dataframe(alerts_df.tail(20), use_container_width=True)
else:
    st.info("Алертов пока нет")