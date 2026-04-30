import time
import requests
import pandas as pd
from config import PROM_URL


def query_range(query, minutes=30, step="15s"):
    end = time.time()
    start = end - minutes * 60

    url = f"{PROM_URL}/api/v1/query_range"

    response = requests.get(url, params={
        "query": query,
        "start": start,
        "end": end,
        "step": step
    })

    response.raise_for_status()
    data = response.json()

    if data["status"] != "success":
        raise RuntimeError(data)

    result = data["data"]["result"]

    if not result:
        raise RuntimeError("Prometheus вернул пустой результат")

    values = result[0]["values"]

    df = pd.DataFrame(values, columns=["timestamp", "value"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df["value"] = df["value"].astype(float)

    return df


def query_instant(query):
    url = f"{PROM_URL}/api/v1/query"

    response = requests.get(url, params={"query": query})
    response.raise_for_status()

    data = response.json()
    result = data["data"]["result"]

    if not result:
        return None

    return float(result[0]["value"][1])