import requests
import pandas as pd

PROM_URL = "http://localhost:9090"

def query_range(query, start, end, step="5s"):
    url = f"{PROM_URL}/api/v1/query_range"
    response = requests.get(url, params={
        "query": query,
        "start": start,
        "end": end,
        "step": step
    })

    data = response.json()

    if data["status"] != "success":
        return None

    result = data["data"]["result"]

    if not result:
        return None

    values = result[0]["values"]
    return [float(v[1]) for v in values]


def collect_metrics():
    import time

    end = time.time()
    start = end - 600  # последние 10 минут

    cpu = query_range(
        '100 - (avg by(instance)(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)',
        start, end
    )

    memory = query_range(
        '(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100',
        start, end
    )

    if cpu is None or memory is None:
        raise Exception("No data from Prometheus")

    df = pd.DataFrame({
        "cpu": cpu,
        "memory": memory
    })

    return df