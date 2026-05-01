import sys
import requests
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QLabel,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QTableWidget,
    QTableWidgetItem,
    QGroupBox,
    QMessageBox,
)
import pyqtgraph as pg

from config import API_URL


class MonitoringClient(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("VM Anomaly Monitoring Client")
        self.resize(1100, 750)

        self.status_label = QLabel("Статус: unknown")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.status_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                padding: 16px;
                border-radius: 8px;
                background-color: #444;
                color: white;
            }
        """)

        self.cpu_label = QLabel("CPU: —")
        self.delta_label = QLabel("Delta: —")
        self.mse_label = QLabel("MSE: —")
        self.threshold_label = QLabel("Threshold: —")

        for label in [
            self.cpu_label,
            self.delta_label,
            self.mse_label,
            self.threshold_label,
        ]:
            label.setStyleSheet("font-size: 18px; padding: 8px;")

        metrics_box = QGroupBox("Текущие метрики")
        metrics_layout = QGridLayout()
        metrics_layout.addWidget(self.cpu_label, 0, 0)
        metrics_layout.addWidget(self.delta_label, 0, 1)
        metrics_layout.addWidget(self.mse_label, 1, 0)
        metrics_layout.addWidget(self.threshold_label, 1, 1)
        metrics_box.setLayout(metrics_layout)

        self.cpu_plot = pg.PlotWidget(title="CPU usage")
        self.cpu_plot.setLabel("left", "CPU", units="%")
        self.cpu_plot.setLabel("bottom", "Samples")
        self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen(width=2))

        self.mse_plot = pg.PlotWidget(title="Reconstruction error")
        self.mse_plot.setLabel("left", "MSE")
        self.mse_plot.setLabel("bottom", "Samples")
        self.mse_curve = self.mse_plot.plot(pen=pg.mkPen(width=2))
        self.threshold_curve = self.mse_plot.plot(pen=pg.mkPen(style=Qt.DashLine, width=2))

        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(6)
        self.alerts_table.setHorizontalHeaderLabels([
            "timestamp",
            "old_status",
            "new_status",
            "value",
            "mse",
            "message",
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)

        alerts_box = QGroupBox("Алерты")
        alerts_layout = QVBoxLayout()
        alerts_layout.addWidget(self.alerts_table)
        alerts_box.setLayout(alerts_layout)

        charts_layout = QHBoxLayout()
        charts_layout.addWidget(self.cpu_plot)
        charts_layout.addWidget(self.mse_plot)

        root = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.status_label)
        layout.addWidget(metrics_box)
        layout.addLayout(charts_layout)
        layout.addWidget(alerts_box)
        root.setLayout(layout)

        self.setCentralWidget(root)

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(5000)

        self.refresh()

    def fetch_json(self, path):
        response = requests.get(f"{API_URL}{path}", timeout=5)
        response.raise_for_status()
        return response.json()

    def set_status_style(self, status):
        styles = {
            "normal": ("Нормальное состояние", "#2e7d32"),
            "load_start": ("Появляется нагрузка", "#ef6c00"),
            "anomaly": ("Аномалия", "#c62828"),
            "recovery": ("Разгрузка", "#1565c0"),
            "collecting": ("Сбор начального окна", "#616161"),
        }

        text, color = styles.get(status, (f"Неизвестный статус: {status}", "#444"))

        self.status_label.setText(f"Статус: {text}")
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                padding: 16px;
                border-radius: 8px;
                background-color: {color};
                color: white;
            }}
        """)

    def refresh(self):
        try:
            status = self.fetch_json("/status")
            metrics = self.fetch_json("/metrics?limit=200")
            alerts = self.fetch_json("/alerts")
        except Exception as e:
            self.status_label.setText("Ошибка подключения к API")
            self.status_label.setStyleSheet("""
                QLabel {
                    font-size: 24px;
                    font-weight: bold;
                    padding: 16px;
                    border-radius: 8px;
                    background-color: #c62828;
                    color: white;
                }
            """)
            return

        self.update_status(status)
        self.update_charts(metrics)
        self.update_alerts(alerts)

    def update_status(self, status):
        current_status = status.get("status", "unknown")
        self.set_status_style(current_status)

        value = status.get("value")
        delta = status.get("delta")
        mse = status.get("mse")
        threshold = status.get("threshold")

        self.cpu_label.setText(f"CPU: {value:.2f}%" if value is not None else "CPU: —")
        self.delta_label.setText(f"Delta: {delta:.2f}" if delta is not None else "Delta: —")
        self.mse_label.setText(f"MSE: {mse:.4f}" if mse is not None else "MSE: —")
        self.threshold_label.setText(
            f"Threshold: {threshold:.4f}" if threshold is not None else "Threshold: —"
        )

    def update_charts(self, metrics):
        if not metrics:
            return

        values = [float(item["value"]) for item in metrics if item.get("value") is not None]
        mse_values = [
            float(item["mse"])
            for item in metrics
            if item.get("mse") is not None
        ]
        threshold_values = [
            float(item["threshold"])
            for item in metrics
            if item.get("threshold") is not None and item.get("mse") is not None
        ]

        self.cpu_curve.setData(values)

        if mse_values:
            self.mse_curve.setData(mse_values)

        if threshold_values:
            self.threshold_curve.setData(threshold_values)

    def update_alerts(self, alerts):
        alerts = alerts[-30:]

        self.alerts_table.setRowCount(len(alerts))

        for row, alert in enumerate(alerts):
            values = [
                alert.get("timestamp", ""),
                alert.get("old_status", ""),
                alert.get("new_status", ""),
                f"{alert.get('value', 0):.2f}" if alert.get("value") is not None else "",
                f"{alert.get('mse', 0):.4f}" if alert.get("mse") is not None else "",
                alert.get("message", ""),
            ]

            for col, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() ^ Qt.ItemIsEditable)
                self.alerts_table.setItem(row, col, item)


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MonitoringClient()
    window.show()

    sys.exit(app.exec())