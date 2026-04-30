import torch
import torch.nn as nn
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from utils import generate_data
from fetch_metrics import collect_metrics

class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



if __name__ == "__main__":
    data = collect_metrics()
    
    print(data.head())
    print(data.describe())
    
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X = torch.tensor(data_scaled, dtype=torch.float32)

    model = Autoencoder()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(X)
        loss = criterion(output, X)
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

    torch.save(model.state_dict(), "../model/autoencoder.pt")