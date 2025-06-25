import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1️⃣ 载入数据
data = np.load('vix_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 2️⃣ 转换为张量（确保形状正确）
X_train_tensor = torch.tensor(X_train.reshape(-1, 20), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(-1, 20), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 5), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 5), dtype=torch.float32)

# 3️⃣ 构造 DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 4️⃣ 定义模型结构
class VIXPredictor(nn.Module):
    def __init__(self):
        super(VIXPredictor, self).__init__()
        self.layer1 = nn.Linear(20, 64)
        self.layer2 = nn.Linear(64, 128)
        self.layer3 = nn.Linear(128, 5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)
        return x

model = VIXPredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 5️⃣ 训练函数
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    train_losses = []
    for epoch in range(epochs):
        total_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}')
    return train_losses

# 6️⃣ 测试函数，返回预测结果
def evaluate_model(model, X_test, y_test):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        mse = mean_squared_error(y_test.numpy(), predictions.numpy())
        print(f'Test MSE: {mse:.4f}')
    return predictions

# 7️⃣ 训练模型
loss_history = train_model(model, train_loader, criterion, optimizer, epochs=50)

# 8️⃣ 评估模型并获取预测值
predictions = evaluate_model(model, X_test_tensor, y_test_tensor)

# 9️⃣ 保存模型
torch.save(model.state_dict(), 'vix_predictor.pth')

# 🔟 可视化 Loss 曲线
def plot_loss_curve(loss_history):
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label='Training Loss', marker='o')
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

plot_loss_curve(loss_history)

# 11️⃣ 可视化预测 vs 真实值（整张图，包含所有测试样本）
def plot_predictions_all_samples(predictions, y_test):
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.numpy()
    if isinstance(y_test, torch.Tensor):
        y_test = y_test.numpy()

    plt.figure(figsize=(15, 8))
    for i in range(predictions.shape[1]):
        plt.subplot(predictions.shape[1], 1, i + 1)
        plt.plot(y_test[:, i], label='True', linewidth=1.5)
        plt.plot(predictions[:, i], label='Predicted', linestyle='--')
        plt.title(f'Horizon {i + 1}')
        plt.xlabel("Sample Index")
        plt.ylabel("VIX")
        plt.legend()
        plt.grid(True)
    plt.suptitle("Full Test Set: Predicted vs True Values", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

plot_predictions_all_samples(predictions, y_test_tensor)

# 12️⃣ 计算评估指标：MAE、MSE、MAPE
def evaluate_metrics(y_true, y_pred):
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.numpy()

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    print(f"\nEvaluation Metrics on Test Set:")
    print(f"MAE  : {mae:.4f}")
    print(f"MSE  : {mse:.4f}")
    print(f"MAPE : {mape:.2f}%")

evaluate_metrics(y_test_tensor, predictions)

# 13️⃣ 保存指标到 CSV 文件
import csv
import os

# 设置模型名称（方便区分）
model_name = 'MLP'

# 重新计算以便保存
if isinstance(y_test_tensor, torch.Tensor):
    y_true = y_test_tensor.numpy()
if isinstance(predictions, torch.Tensor):
    y_pred = predictions.numpy()

mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# CSV 文件路径
csv_path = 'vix_model_results.csv'
file_exists = os.path.exists(csv_path)

# 写入 CSV
with open(csv_path, mode='a', newline='') as file:
    writer = csv.writer(file)
    if not file_exists:
        writer.writerow(['Model', 'MAE', 'MSE', 'MAPE (%)'])  # 首次写入表头
    writer.writerow([model_name, f"{mae:.4f}", f"{mse:.4f}", f"{mape:.2f}"])
