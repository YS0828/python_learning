import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import pandas as pd

# 1️⃣ 加载数据
data = np.load('vix_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 2️⃣ 数据张量转换（适配 LSTM 输入格式）
X_train_tensor = torch.tensor(X_train.reshape(-1, 20, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(-1, 20, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 5), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 5), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3️⃣ 定义 LSTM 模型
class VIXLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 5)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # 取最后时间步
        return self.fc(out)

# 4️⃣ 定义训练函数
def train_model(model, loader, criterion, optimizer, epochs=50):
    model.train()
    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

# 5️⃣ 定义评估函数
def evaluate(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        pred = model(X_test).numpy()
        true = y_test.numpy()
    mae = mean_absolute_error(true, pred)
    mse = mean_squared_error(true, pred)
    mape = np.mean(np.abs((true - pred) / true)) * 100
    return mae, mse, mape

# 6️⃣ 参数组合网格（param_grid）
param_grid = [
    {'hidden_size': 32, 'num_layers': 1, 'lr': 0.001},
    {'hidden_size': 64, 'num_layers': 1, 'lr': 0.001},
    {'hidden_size': 64, 'num_layers': 2, 'lr': 0.005},
    {'hidden_size': 128, 'num_layers': 2, 'lr': 0.005},
    {'hidden_size': 64, 'num_layers': 3, 'lr': 0.01}
]

# 7️⃣ 批量训练 + 记录结果
results_lstm = []

for params in param_grid:
    model = VIXLSTM(params['hidden_size'], params['num_layers'])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])

    train_model(model, train_loader, criterion, optimizer, epochs=50)
    mae, mse, mape = evaluate(model, X_test_tensor, y_test_tensor)

    print(f"LSTM: hidden={params['hidden_size']}, layers={params['num_layers']}, lr={params['lr']}")
    print(f"MAE={mae:.4f}, MSE={mse:.4f}, MAPE={mape:.2f}%\n")

    results_lstm.append({
        'Model': 'LSTM',
        'hidden_size': params['hidden_size'],
        'num_layers': params['num_layers'],
        'lr': params['lr'],
        'MAE': mae,
        'MSE': mse,
        'MAPE': mape
    })

# 8️⃣ 保存结果
df_lstm = pd.DataFrame(results_lstm)
df_lstm.to_excel("lstm_param_results.xlsx", index=False)
print("✅ LSTM 参数对比结果保存成功！")
