# config.py
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # корень проекта
DATA_DIR = os.path.join(BASE_DIR, "data")
LABELS_PATH = os.path.join(BASE_DIR, "labels", "combined_windows.json")

# Датасеты: только те, что точно есть в папке data/ и имеют разметку в JSON.
# Ниже список всех CSV из realAWSCloudwatch + несколько из других категорий, 
# которые присутствуют в combined_windows.json (я проверил названия).
DATASETS = [
    "realAWSCloudwatch/ec2_cpu_utilization_24ae8d.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_53ea38.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_5f5533.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_77c1ca.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_ac20cd.csv",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv",
    "realAWSCloudwatch/ec2_disk_write_bytes_1ef3de.csv",
    "realAWSCloudwatch/ec2_disk_write_bytes_c0d644.csv",
    "realAWSCloudwatch/ec2_network_in_257a54.csv",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv",
    "realAWSCloudwatch/elb_request_count_8c0756.csv",
    "realAWSCloudwatch/grok_asg_anomaly.csv",
    "realAWSCloudwatch/iio_us-east-1_i-a2eb1cd9_NetworkIn.csv",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv",
]

# Настройки модели
#SEQ_LEN = 32
HIDDEN_DIM = 32
LATENT_DIM = 4
BATCH_SIZE = 64   # если захотим мини-батчи, пока не используем
EPOCHS = 100
SMOOTH_WINDOW = 10
THRESHOLD_PERCENTILE = 99


PROM_URL = "http://localhost:9090"
PROM_QUERY = '100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[1m])) * 100)'

PROM_MODEL_DIR = "model/prometheus_cpu"
PROM_TRAIN_MINUTES = 30
#PROM_STEP = "15s"

#REALTIME_INTERVAL = 15

SEQ_LEN = 16
REALTIME_INTERVAL = 5
PROM_STEP = "5s"