import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

data = np.load('vix_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

X_train_tensor = torch.tensor(X_train.reshape(-1, 20), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(-1, 20), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 5), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 5), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class VIXPredictor(nn.Module):
    def __init__(self, hidden1, hidden2):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(20, hidden1),
            nn.ReLU(),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Linear(hidden2, 5)
        )

    def forward(self, x):
        return self.model(x)

def train_model(model, loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
        true = y_test.numpy()
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, mse, mape


results = []

param_grid = [
    {'hidden1': 64, 'hidden2': 128, 'lr': 0.001},
    {'hidden1': 128, 'hidden2': 256, 'lr': 0.001},
    {'hidden1': 32, 'hidden2': 64, 'lr': 0.005},
    {'hidden1': 64, 'hidden2': 128, 'lr': 0.005},
    {'hidden1': 128, 'hidden2': 256, 'lr': 0.005},
    {'hidden1': 32, 'hidden2': 64, 'lr': 0.01},
    {'hidden1': 64, 'hidden2': 128, 'lr': 0.01},
    {'hidden1': 128, 'hidden2': 256, 'lr': 0.01}
]

for params in param_grid:
    model = VIXPredictor(params['hidden1'], params['hidden2'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_model(model, train_loader, criterion, optimizer, epochs=50)
    mae, mse, mape = evaluate(model, X_test_tensor, y_test_tensor)

    print(f"Test Results - hidden1={params['hidden1']}, hidden2={params['hidden2']}, lr={params['lr']}")
    print(f"MAE: {mae:.4f}, MSE: {mse:.4f}, MAPE: {mape:.2f}%\n")

    results.append({
        'hidden1': params['hidden1'],
        'hidden2': params['hidden2'],
        'lr': params['lr'],
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape
    })

df = pd.DataFrame(results)
df.to_excel("mlp_param_results.xlsx", index=False)
print("结果保存完毕 ✅")
