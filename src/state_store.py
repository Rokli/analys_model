import json
import os
from datetime import datetime

STATE_PATH = "model/state.json"
ALERTS_PATH = "model/alerts.json"


def ensure_storage():
    os.makedirs("model", exist_ok=True)

    if not os.path.exists(STATE_PATH):
        save_state({
            "timestamp": None,
            "value": None,
            "delta": None,
            "mse": None,
            "threshold": None,
            "raw_status": None,
            "status": "normal"
        })

    if not os.path.exists(ALERTS_PATH):
        save_alerts([])


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=4)


def load_state():
    ensure_storage()
    with open(STATE_PATH) as f:
        return json.load(f)


def save_alerts(alerts):
    with open(ALERTS_PATH, "w") as f:
        json.dump(alerts, f, indent=4)


def load_alerts():
    ensure_storage()
    with open(ALERTS_PATH) as f:
        return json.load(f)


def add_alert(old_status, new_status, value, mse, threshold):
    alerts = load_alerts()

    alert = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "old_status": old_status,
        "new_status": new_status,
        "value": value,
        "mse": mse,
        "threshold": threshold,
        "message": f"Status changed: {old_status} -> {new_status}"
    }

    alerts.append(alert)
    save_alerts(alerts[-100:])

    return alert