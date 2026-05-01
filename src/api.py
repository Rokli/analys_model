import os
import pandas as pd
from fastapi import FastAPI

from src.state_store import load_state, load_alerts

LOG_PATH = "model/realtime_log.csv"

app = FastAPI(title="VM Anomaly Detection API")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/status")
def get_status():
    return load_state()


@app.get("/alerts")
def get_alerts():
    return load_alerts()


@app.get("/metrics")
def get_metrics(limit: int = 100):
    if not os.path.exists(LOG_PATH):
        return []

    df = pd.read_csv(LOG_PATH)

    if df.empty:
        return []

    return df.tail(limit).to_dict(orient="records")