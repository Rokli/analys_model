import sys
import requests
from datetime import datetime

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
    QHeaderView,
)
import pyqtgraph as pg
from PySide6.QtCore import Qt, QTimer, QSettings, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QFileDialog, QGraphicsOpacityEffect

DEFAULT_API_URL = "http://localhost:8000"


APP_STYLE = """
QMainWindow, QWidget {
    background-color: #0B1120;
    color: #E5E7EB;
    font-family: Arial;
}

QLabel {
    color: #E5E7EB;
}

QFrame#Card:hover {
    border: 1px solid #3B82F6;
}

QFrame#RecentAnomaly {
    background-color: #2A1111;
    border: 1px solid #EF4444;
    border-radius: 18px;
    padding: 14px;
}

QFrame#Card {
    background-color: #111827;
    border: 1px solid #1F2937;
    border-radius: 18px;
    padding: 14px;
}

QFrame#ConnectionCard {
    background-color: #111827;
    border: 1px solid #243244;
    border-radius: 18px;
    padding: 12px;
}

QLabel#Title {
    font-size: 30px;
    font-weight: bold;
}

QLabel#Subtitle {
    font-size: 14px;
    color: #9CA3AF;
}

QLabel#SectionTitle {
    font-size: 20px;
    font-weight: bold;
}

QLabel#MetricTitle {
    font-size: 14px;
    color: #9CA3AF;
}

QLabel#MetricValue {
    font-size: 26px;
    font-weight: bold;
    color: #F9FAFB;
}

QLabel#SmallMuted {
    font-size: 12px;
    color: #9CA3AF;
}

QLineEdit, QSpinBox {
    background-color: #0F172A;
    color: #E5E7EB;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 9px;
}

QPushButton {
    background-color: #2563EB;
    color: white;
    border: none;
    border-radius: 10px;
    padding: 10px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #1D4ED8;
}

QTabWidget::pane {
    border: none;
}

QTabBar::tab {
    background: #111827;
    color: #D1D5DB;
    padding: 11px 20px;
    border-radius: 10px;
    margin-right: 6px;
}

QTabBar::tab:selected {
    background: #2563EB;
    color: white;
}

QTableWidget {
    background-color: #111827;
    color: #E5E7EB;
    gridline-color: #1F2937;
    border: none;
    border-radius: 12px;
}

QHeaderView::section {
    background-color: #1F2937;
    color: #E5E7EB;
    padding: 8px;
    border: none;
}

QTableWidget::item {
    padding: 8px;
}
"""


STATUS_META = {
    "normal": ("🟢", "Нормальное состояние", "#166534"),
    "load_start": ("🟠", "Появляется нагрузка", "#C2410C"),
    "anomaly": ("🔴", "Аномалия", "#B91C1C"),
    "recovery": ("🔵", "Разгрузка", "#1D4ED8"),
    "collecting": ("⏳", "Сбор начального окна", "#4B5563"),
    "connection_error": ("🔴", "Ошибка подключения к API", "#B91C1C"),
    "unknown": ("⚪", "Статус неизвестен", "#4B5563"),
}


class MetricCard(QFrame):
    def __init__(self, title, icon, value="—"):
        super().__init__()
        self.setObjectName("Card")

        self.title_label = QLabel(f"{icon} {title}")
        self.title_label.setObjectName("MetricTitle")

        self.value_label = QLabel(value)
        self.value_label.setObjectName("MetricValue")

        layout = QVBoxLayout()
        layout.setSpacing(8)
        layout.addWidget(self.title_label)
        layout.addWidget(self.value_label)
        self.setLayout(layout)

    def set_value(self, value):
        self.value_label.setText(value)


class StatusBadge(QLabel):
    def __init__(self, status="unknown"):
        super().__init__()
        self.setAlignment(Qt.AlignCenter)
        self.set_status(status)

    def set_status(self, status):
        icon, text, color = STATUS_META.get(status, STATUS_META["unknown"])
        self.setText(f"{icon} {text}")
        self.setStyleSheet(f"""
            QLabel {{
                font-size: 24px;
                font-weight: bold;
                padding: 20px;
                border-radius: 20px;
                background-color: {color};
                color: white;
            }}
        """)


class Toast(QLabel):
    def __init__(self, parent, text, color="#1D4ED8"):
        super().__init__(parent)
        self.setText(text)
        self.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: white;
                border-radius: 12px;
                padding: 14px 18px;
                font-size: 14px;
                font-weight: bold;
            }}
        """)
        self.adjustSize()

        x = parent.width() - self.width() - 30
        y = parent.height() - self.height() - 40
        self.move(x, y)
        self.show()

        self.effect = QGraphicsOpacityEffect(self)
        self.setGraphicsEffect(self.effect)

        self.anim = QPropertyAnimation(self.effect, b"opacity")
        self.anim.setDuration(2500)
        self.anim.setStartValue(1.0)
        self.anim.setEndValue(0.0)
        self.anim.setEasingCurve(QEasingCurve.InOutQuad)
        self.anim.finished.connect(self.deleteLater)
        self.anim.start()

class MonitoringClient(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = QSettings("DiplomProject", "VMAnomalyClient")
        self.api_url = self.settings.value("api_url", DEFAULT_API_URL)
        self.refresh_interval = int(self.settings.value("refresh_interval", 10))
        self.popup_enabled = self.settings.value("popup_enabled", "true") == "true"
        self.last_status = None
        self.recent_anomaly_until = None

        self.setWindowTitle("VM Anomaly Monitoring Client")
        self.resize(1320, 880)

        pg.setConfigOption("background", "#111827")
        pg.setConfigOption("foreground", "#E5E7EB")

        self.build_ui()

        self.timer = QTimer()
        self.timer.timeout.connect(self.refresh)
        self.timer.start(self.refresh_interval * 1000)

        self.refresh()

    def build_ui(self):
        root = QWidget()
        root_layout = QVBoxLayout()
        root_layout.setContentsMargins(26, 24, 26, 18)
        root_layout.setSpacing(18)

        top_layout = QHBoxLayout()

        title_block = QVBoxLayout()
        title = QLabel("VM Anomaly Monitoring")
        title.setObjectName("Title")
        subtitle = QLabel("Desktop-клиент для мониторинга виртуальной машины на основе LSTM Autoencoder")
        subtitle.setObjectName("Subtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)

        self.last_update_label = QLabel("Last update: —")
        self.last_update_label.setObjectName("SmallMuted")
        self.last_update_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        top_layout.addLayout(title_block)
        top_layout.addStretch()
        top_layout.addWidget(self.last_update_label)

        root_layout.addLayout(top_layout)
        root_layout.addWidget(self.build_connection_panel())

        self.tabs = QTabWidget()
        self.tabs.addTab(self.build_monitoring_tab(), "Мониторинг")
        self.tabs.addTab(self.build_alerts_tab(), "Алерты")
        self.tabs.addTab(self.build_settings_tab(), "Настройки")

        root_layout.addWidget(self.tabs)

        footer = QLabel("Diplom project · Prometheus · LSTM Autoencoder · FastAPI · PySide6")
        footer.setObjectName("SmallMuted")
        footer.setAlignment(Qt.AlignCenter)
        root_layout.addWidget(footer)

        root.setLayout(root_layout)
        self.setCentralWidget(root)

    def build_connection_panel(self):
        card = QFrame()
        card.setObjectName("ConnectionCard")

        layout = QHBoxLayout()
        layout.setSpacing(12)

        label = QLabel("API URL ВМ:")
        label.setStyleSheet("font-weight: bold;")

        self.api_input = QLineEdit()
        self.api_input.setText(self.api_url)
        self.api_input.setPlaceholderText("http://<VM_IP>:8000")

        connect_button = QPushButton("Подключиться")
        connect_button.clicked.connect(self.apply_api_url)

        self.connection_badge = QLabel("● API: неизвестно")
        self.connection_badge.setStyleSheet("""
            QLabel {
                color: #9CA3AF;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 10px;
                background-color: #1F2937;
            }
        """)

        layout.addWidget(label)
        layout.addWidget(self.api_input, stretch=1)
        layout.addWidget(connect_button)
        layout.addWidget(self.connection_badge)

        card.setLayout(layout)
        return card

    def build_monitoring_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(18)

        self.status_badge = StatusBadge("unknown")

        self.cpu_card = MetricCard("CPU", "🧠")
        self.delta_card = MetricCard("Delta", "📈")
        self.mse_card = MetricCard("MSE", "📉")
        self.threshold_card = MetricCard("Threshold", "🎯")

        self.cpu_card.setToolTip("Текущее значение загрузки CPU виртуальной машины")
        self.delta_card.setToolTip("Изменение CPU относительно предыдущего измерения")
        self.mse_card.setToolTip("Ошибка реконструкции LSTM Autoencoder")
        self.threshold_card.setToolTip("Порог, выше которого состояние считается аномальным")

        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(16)
        metrics_layout.addWidget(self.cpu_card, 0, 0)
        metrics_layout.addWidget(self.delta_card, 0, 1)
        metrics_layout.addWidget(self.mse_card, 0, 2)
        metrics_layout.addWidget(self.threshold_card, 0, 3)

        self.cpu_plot = pg.PlotWidget(title="CPU usage")
        self.cpu_plot.setLabel("left", "CPU", units="%")
        self.cpu_plot.setLabel("bottom", "Samples")
        self.cpu_plot.showGrid(x=True, y=True, alpha=0.18)
        self.cpu_curve = self.cpu_plot.plot(pen=pg.mkPen("#60A5FA", width=3))

        self.mse_plot = pg.PlotWidget(title="Reconstruction error")
        self.mse_plot.setLabel("left", "MSE")
        self.mse_plot.setLabel("bottom", "Samples")
        self.mse_plot.showGrid(x=True, y=True, alpha=0.18)
        self.mse_curve = self.mse_plot.plot(pen=pg.mkPen("#A78BFA", width=3))
        self.threshold_curve = self.mse_plot.plot(
            pen=pg.mkPen("#FBBF24", width=2, style=Qt.DashLine)
        )

        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(16)
        charts_layout.addWidget(self.cpu_plot)
        charts_layout.addWidget(self.mse_plot)

        layout.addWidget(self.status_badge)
        layout.addLayout(metrics_layout)
        layout.addLayout(charts_layout)

        tab.setLayout(layout)
        return tab

    def build_alerts_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()
        layout.setSpacing(14)

        title = QLabel("История алертов")
        title.setObjectName("SectionTitle")

        self.alerts_table = QTableWidget()
        self.alerts_table.setColumnCount(6)
        self.alerts_table.setHorizontalHeaderLabels([
            "Время", "Было", "Стало", "CPU", "MSE", "Сообщение"
        ])
        self.alerts_table.horizontalHeader().setStretchLastSection(True)
        self.alerts_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
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
        settings_layout.setSpacing(14)

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

        export_button = QPushButton("Экспортировать данные CSV")
        export_button.clicked.connect(self.export_metrics_csv)
        settings_layout.addWidget(export_button, 3, 0, 1, 2)

        settings_layout.addWidget(interval_label, 0, 0)
        settings_layout.addWidget(self.interval_spin, 0, 1)
        settings_layout.addWidget(self.popup_checkbox, 1, 0, 1, 2)
        settings_layout.addWidget(save_button, 2, 0)
        settings_layout.addWidget(test_button, 2, 1)

        settings_card.setLayout(settings_layout)

        info = QLabel("Пример API URL: http://158.160.xxx.xxx:8000")
        info.setObjectName("Subtitle")

        layout.addWidget(settings_card)
        layout.addWidget(info)
        layout.addStretch()

        tab.setLayout(layout)
        return tab

    def normalize_api_url(self, url):
        url = url.strip()
        if not url:
            return DEFAULT_API_URL
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
        return url.rstrip("/")

    def export_metrics_csv(self):
        try:
            metrics = self.fetch_json("/metrics?limit=10000")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", f"Не удалось получить данные: {e}")
            return

        if not metrics:
            QMessageBox.information(self, "Экспорт", "Нет данных для экспорта")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить CSV",
            "vm_metrics_export.csv",
            "CSV files (*.csv)"
        )

        if not path:
            return

        import csv

        keys = metrics[0].keys()

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(metrics)

        self.show_toast("✅ Данные экспортированы", "#166534")

    def apply_api_url(self):
        self.api_url = self.normalize_api_url(self.api_input.text())
        self.api_input.setText(self.api_url)
        self.settings.setValue("api_url", self.api_url)
        self.refresh()

    def fetch_json(self, path):
        response = requests.get(f"{self.api_url}{path}", timeout=5)
        response.raise_for_status()
        return response.json()

    def refresh(self):
        try:
            status = self.fetch_json("/status")
            metrics = self.fetch_json("/metrics?limit=120")
            alerts = self.fetch_json("/alerts")
        except Exception:
            self.connection_badge.setText("● API: недоступен")
            self.connection_badge.setStyleSheet("""
                QLabel {
                    color: #FCA5A5;
                    font-weight: bold;
                    padding: 8px 12px;
                    border-radius: 10px;
                    background-color: #7F1D1D;
                }
            """)
            self.status_badge.set_status("connection_error")
            return

        self.connection_badge.setText("● API: подключен")
        self.connection_badge.setStyleSheet("""
            QLabel {
                color: #BBF7D0;
                font-weight: bold;
                padding: 8px 12px;
                border-radius: 10px;
                background-color: #14532D;
            }
        """)

        self.last_update_label.setText(
            "Last update: " + datetime.now().strftime("%H:%M:%S")
        )

        self.update_status(status)
        self.apply_recent_anomaly_effect()
        self.update_charts(metrics)
        self.update_alerts(alerts)

    def apply_recent_anomaly_effect(self):
        if self.recent_anomaly_until and datetime.now().timestamp() < self.recent_anomaly_until:
            self.cpu_card.setObjectName("RecentAnomaly")
            self.mse_card.setObjectName("RecentAnomaly")
        else:
            self.cpu_card.setObjectName("Card")
            self.mse_card.setObjectName("Card")

        self.cpu_card.style().unpolish(self.cpu_card)
        self.cpu_card.style().polish(self.cpu_card)

        self.mse_card.style().unpolish(self.mse_card)
        self.mse_card.style().polish(self.mse_card)

    def update_status(self, status):
        current_status = status.get("status", "unknown")
        if current_status != self.last_status:
            self.animate_status_change()

        self.status_badge.set_status(current_status)

        value = status.get("value")
        delta = status.get("delta")
        mse = status.get("mse")
        threshold = status.get("threshold")

        self.cpu_card.set_value(f"{value:.2f}%" if value is not None else "—")
        self.delta_card.set_value(f"{delta:.2f}" if delta is not None else "—")
        self.mse_card.set_value(f"{mse:.4f}" if mse is not None else "—")
        self.threshold_card.set_value(f"{threshold:.4f}" if threshold is not None else "—")

        if self.popup_enabled and current_status == "anomaly" and self.last_status != "anomaly":
            self.show_toast("🔴 Обнаружена аномалия виртуальной машины", "#B91C1C")
            self.recent_anomaly_until = datetime.now().timestamp() + 10

        self.last_status = current_status

    def show_toast(self, text, color="#1D4ED8"):
        Toast(self, text, color)
    def update_charts(self, metrics):
        if not metrics:
            return

        values = [float(i["value"]) for i in metrics if i.get("value") is not None]
        mse_values = [float(i["mse"]) for i in metrics if i.get("mse") is not None]
        threshold_values = [
            float(i["threshold"])
            for i in metrics
            if i.get("threshold") is not None and i.get("mse") is not None
        ]

        self.cpu_curve.setData(values)

        if mse_values:
            self.mse_curve.setData(mse_values)

        if threshold_values:
            self.threshold_curve.setData(threshold_values)

    def make_status_item(self, status):
        icon, text, color = STATUS_META.get(status, STATUS_META["unknown"])
        item = QTableWidgetItem(f"{icon} {text}")
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        item.setBackground(Qt.GlobalColor.transparent)
        item.setForeground(Qt.white)
        item.setData(Qt.UserRole, color)
        return item

    def update_alerts(self, alerts):
        alerts = alerts[-50:]
        self.alerts_table.setRowCount(len(alerts))

        for row, alert in enumerate(alerts):
            raw_values = [
                alert.get("timestamp", ""),
                alert.get("old_status", ""),
                alert.get("new_status", ""),
                f"{alert.get('value', 0):.2f}" if alert.get("value") is not None else "",
                f"{alert.get('mse', 0):.4f}" if alert.get("mse") is not None else "",
                alert.get("message", ""),
            ]

            for col, value in enumerate(raw_values):
                if col in (1, 2):
                    item = self.make_status_item(str(value))
                    _, _, color = STATUS_META.get(str(value), STATUS_META["unknown"])
                    item.setBackground(Qt.transparent)
                    item.setForeground(Qt.white)
                else:
                    item = QTableWidgetItem(str(value))
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)

                if alert.get("new_status") == "anomaly":
                    item.setBackground(Qt.red)
                    item.setForeground(Qt.white)

                self.alerts_table.setItem(row, col, item)

    def save_settings(self):
        self.refresh_interval = self.interval_spin.value()
        self.popup_enabled = self.popup_checkbox.isChecked()

        self.settings.setValue("refresh_interval", self.refresh_interval)
        self.settings.setValue("popup_enabled", "true" if self.popup_enabled else "false")

        self.timer.start(self.refresh_interval * 1000)
        QMessageBox.information(self, "Настройки", "Настройки сохранены")

    def test_connection(self):
        try:
            result = self.fetch_json("/health")
            QMessageBox.information(self, "Подключение", f"API доступен: {result}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка подключения", str(e))
    def animate_status_change(self):
        effect = QGraphicsOpacityEffect(self.status_badge)
        self.status_badge.setGraphicsEffect(effect)

        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(500)
        anim.setStartValue(0.35)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()

        self.status_anim = anim

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    window = MonitoringClient()
    window.show()

    sys.exit(app.exec())