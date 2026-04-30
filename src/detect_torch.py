import torch
import numpy as np
from utils import generate_data
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

from train_torch import Autoencoder

model = Autoencoder()
model.load_state_dict(torch.load("../model/autoencoder.pt"))
model.eval()

data = generate_data()
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

X = torch.tensor(data_scaled, dtype=torch.float32)

with torch.no_grad():
    reconstructed = model(X)

mse = torch.mean((X - reconstructed) ** 2, dim=1).numpy()

threshold = np.percentile(mse, 95)
anomalies = mse > threshold

plt.figure()
plt.plot(mse)
plt.axhline(threshold)
plt.savefig("../model/anomalies.png")

print("Anomalies:", np.sum(anomalies))