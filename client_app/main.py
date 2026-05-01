import sys
import requests

from PySide6.QtCore import Qt, QTimer, QSettings
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
    QFrame,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QSpinBox,
    QCheckBox,
    QMessageBox,
)
import pyqtgraph as pg


DEFAULT_API_URL = "http://localhost:8000"


APP_STYLE = """
QMainWindow, QWidget {
    background-color: #111827;
    color: #E5E7EB;
    font-family: Arial;
}

QLabel {
    color: #E5E7EB;
}

QFrame#Card {
    background-color: #1F2937;
    border-radius: 14px;
    padding: 12px;
}

QLabel#Title {
    font-size: 28px;
    font-weight: bold;
}

QLabel#Subtitle {
    font-size: 14px;
    color: #9CA3AF;
}

QLabel#MetricTitle {
    font-size: 14px;
    color: #9CA3AF;
}

QLabel#MetricValue {
    font-size: 24px;
    font-weight: bold;
    color: #F9FAFB;
}

QLineEdit, QSpinBox {
    background-color: #1F2937;
    color: #E5E7EB;
    border: 1px solid #374151;
    border-radius: 8px;
    padding: 8px;
}

QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 9px 14px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QPushButton#DangerButton {
    background-color: #B91C1C;
}

QTabWidget::pane {
    border: none;
}

QTabBar::tab {
    background: #1F2937;
    color: #D1D5DB;
    padding: 10px 18px;
    border-radius: 8px;
    margin-right: 6px;
}

QTabBar::tab:selected {
    background: #2563EB;
    color: white;
}

QTableWidget {
    background-color: #1F2937;
    color: #E5E7EB;
    gridline-color: #374151;
    border: none;
    border-radius: 10px;
}

QHeaderView::section {
    background-color: #374151;
    color: #E5E7EB;
    padding: 6px;
    border: none;
}

QTableWidget::item {
    padding: 6px;
}
"""


class MetricCard(QFrame):
    def __init__(self, title, value="—"):
        super().__init__()
        self.setObjectName("Card")

        self.title_label = QLabel(title)
        self.title_label.setObjectName("MetricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")

        layout = QVBoxLayout()
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_value(self, value):
        self.value_label.setText(value)


class MonitoringClient(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QSettings("DiplomProject", "VMAnomalyClient")

        self.api_url = self.settings.value("api_url", DEFAULT_API_URL)
        self.refresh_interval = int(self.settings.value("refresh_interval", 10))
        self.popup_enabled = self.settings.value("popup_enabled", "true") == "true"

        self.last_status = None

        self.setWindowTitle("VM Anomaly Monitoring Client")
        self.resize(1280, 850)

        pg.setConfigOption("background", "#1F2937")
        pg.setConfigOption("foreground", "#E5E7EB")

        self.build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.refresh_interval * 1000)

        self.refresh()

    # =========================
    # UI
    # =========================

    def build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(24, 24, 24, 24)
        root_layout.setSpacing(18)

        title = QLabel("VM Anomaly Monitoring")
        title.setObjectName("Title")

        subtitle = QLabel("Desktop-клиент для наблюдения за виртуальной машиной через FastAPI")
        subtitle.setObjectName("Subtitle")

        root_layout.addWidget(title)
        root_layout.addWidget(subtitle)

        connection_card = self.build_connection_panel()
        root_layout.addWidget(connection_card)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_monitoring_tab(), "Мониторинг")
        self.tabs.addTab(self.build_alerts_tab(), "Алерты")
        self.tabs.addTab(self.build_settings_tab(), "Настройки")

        root_layout.addWidget(self.tabs)

        root.setLayout(root_layout)
        self.setCentralWidget(root)

    def build_connection_panel(self):
        card = QFrame()
        card.setObjectName("Card")

        layout = QHBoxLayout()

        label = QLabel("API URL ВМ:")
        self.api_input = QLineEdit()
        self.api_input.setText(self.api_url)
        self.api_input.setPlaceholderText("http://<VM_IP>:8000")

        connect_button = QPushButton("Подключиться")
        connect_button.clicked.connect(self.apply_api_url)

        self.connection_status_label = QLabel("Статус подключения: неизвестно")

        layout.addWidget(label)
        layout.addWidget(self.api_input, stretch=1)
        layout.addWidget(connect_button)
        layout.addWidget(self.connection_status_label)

        card.setLayout(layout)
        return card

    def build_monitoring_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(18)

        self.status_label = QLabel("Статус: unknown")
        self.status_label.setAlignment(Qt.AlignCenter)
        self.set_status_style("unknown")

        self.cpu_card = MetricCard("CPU, %")
        self.delta_card = MetricCard("Delta")
        self.mse_card = MetricCard("MSE")
        self.threshold_card = MetricCard("Threshold")

        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(16)
        metrics_layout.addWidget(self.cpu_card, 0, 0)
        metrics_layout.addWidget(self.delta_card, 0, 1)
        metrics_layout.addWidget(self.mse_card, 0, 2)
        metrics_layout.addWidget(self.threshold_card, 0, 3)

        self.cpu_plot = pg.PlotWidget(title="CPU usage")
        self.cpu_plot.setLabel("left", "CPU", units="%")
        self.cpu_plot.setLabel("bottom", "Samples")
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.25)
        self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen("#60A5FA", width=3))

        self.mse_plot = pg.PlotWidget(title="Reconstruction error")
        self.mse_plot.setLabel("left", "MSE")
        self.mse_plot.setLabel("bottom", "Samples")
        self.mse_plot.showGrid(x=True, y=True, alpha=0.25)
        self.mse_curve = self.mse_plot.plot(pen=pg.mkPen("#A78BFA", width=3))
        self.threshold_curve = self.mse_plot.plot(
            pen=pg.mkPen("#FBBF24", width=2, style=Qt.DashLine)
        )

        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(16)
        charts_layout.addWidget(self.cpu_plot)
        charts_layout.addWidget(self.mse_plot)

        layout.addWidget(self.status_label)
        layout.addLayout(metrics_layout)
        layout.addLayout(charts_layout)

        tab.setLayout(layout)
        return tab

    def build_alerts_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("История алертов")
        title.setStyleSheet("font-size: 22px; font-weight: bold;")

        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(6)
        self.alerts_table.setHorizontalHeaderLabels([
            "Время",
            "Было",
            "Стало",
            "CPU",
            "MSE",
            "Сообщение",
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        self.alerts_table.verticalHeader().setVisible(False)

        layout.addWidget(title)
        layout.addWidget(self.alerts_table)

        tab.setLayout(layout)
        return tab

    def build_settings_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(16)

        settings_card = QFrame()
        settings_card.setObjectName("Card")
        settings_layout = QGridLayout()

        interval_label = QLabel("Интервал обновления, секунд:")
        self.interval_spin = QSpinBox()
        self.interval_spin.setMinimum(2)
        self.interval_spin.setMaximum(60)
        self.interval_spin.setValue(self.refresh_interval)

        self.popup_checkbox = QCheckBox("Показывать pop-up при аномалии")
        self.popup_checkbox.setChecked(self.popup_enabled)

        save_button = QPushButton("Сохранить настройки")
        save_button.clicked.connect(self.save_settings)

        test_button = QPushButton("Проверить подключение")
        test_button.clicked.connect(self.test_connection)

        settings_layout.addWidget(interval_label, 0, 0)
        settings_layout.addWidget(self.interval_spin, 0, 1)
        settings_layout.addWidget(self.popup_checkbox, 1, 0, 1, 2)
        settings_layout.addWidget(save_button, 2, 0)
        settings_layout.addWidget(test_button, 2, 1)

        settings_card.setLayout(settings_layout)

        info = QLabel(
            "Подсказка: если API запущен на ВМ, укажи адрес вида "
            "http://<публичный_IP_ВМ>:8000"
        )
        info.setObjectName("Subtitle")

        layout.addWidget(settings_card)
        layout.addWidget(info)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    # =========================
    # API
    # =========================

    def normalize_api_url(self, url):
        url = url.strip()

        if not url:
            return DEFAULT_API_URL

        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url

        return url.rstrip("/")

    def apply_api_url(self):
        self.api_url = self.normalize_api_url(self.api_input.text())
        self.api_input.setText(self.api_url)

        self.settings.setValue("api_url", self.api_url)

        self.refresh()

    def fetch_json(self, path):
        response = requests.get(f"{self.api_url}{path}", timeout=5)
        response.raise_for_status()
        return response.json()

    def test_connection(self):
        try:
            result = self.fetch_json("/health")
            QMessageBox.information(self, "Подключение", f"API доступен: {result}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка подключения", str(e))

    # =========================
    # Refresh
    # =========================

    def refresh(self):
        try:
            status = self.fetch_json("/status")
            metrics = self.fetch_json("/metrics?limit=120")
            alerts = self.fetch_json("/alerts")
        except Exception:
            self.connection_status_label.setText("🔴 API недоступен")
            self.set_status_style("connection_error")
            return

        self.connection_status_label.setText("🟢 API подключен")

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

        self.cpu_card.set_value(f"{value:.2f}%" if value is not None else "—")
        self.delta_card.set_value(f"{delta:.2f}" if delta is not None else "—")
        self.mse_card.set_value(f"{mse:.4f}" if mse is not None else "—")
        self.threshold_card.set_value(f"{threshold:.4f}" if threshold is not None else "—")

        if (
            self.popup_enabled
            and current_status == "anomaly"
            and self.last_status != "anomaly"
        ):
            QMessageBox.warning(
                self,
                "Обнаружена аномалия",
                "Система обнаружила аномальное состояние виртуальной машины."
            )

        self.last_status = current_status

    def update_charts(self, metrics):
        if not metrics:
            return

        values = [
            float(item["value"])
            for item in metrics
            if item.get("value") is not None
        ]

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
        alerts = alerts[-50:]

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
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if alert.get("new_status") == "anomaly":
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)

                self.alerts_table.setItem(row, col, item)

    # =========================
    # Settings
    # =========================

    def save_settings(self):
        self.refresh_interval = self.interval_spin.value()
        self.popup_enabled = self.popup_checkbox.isChecked()

        self.settings.setValue("refresh_interval", self.refresh_interval)
        self.settings.setValue("popup_enabled", "true" if self.popup_enabled else "false")

        self.timer.start(self.refresh_interval * 1000)

        QMessageBox.information(self, "Настройки", "Настройки сохранены")

    def set_status_style(self, status):
        styles = {
            "normal": ("🟢 Нормальное состояние", "#166534"),
            "load_start": ("🟠 Появляется нагрузка", "#C2410C"),
            "anomaly": ("🔴 Аномалия", "#B91C1C"),
            "recovery": ("🔵 Разгрузка", "#1D4ED8"),
            "collecting": ("⏳ Сбор начального окна", "#4B5563"),
            "connection_error": ("🔴 Ошибка подключения к API", "#B91C1C"),
            "unknown": ("⚪ Статус неизвестен", "#4B5563"),
        }

        text, color = styles.get(status, (f"Неизвестный статус: {status}", "#4B5563"))

        self.status_label.setText(text)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                padding: 18px;
                border-radius: 16px;
                background-color: {color};
                color: white;
            }}
        """)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    window = MonitoringClient()
    window.show()

    sys.exit(app.exec())