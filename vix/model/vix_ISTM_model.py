import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# 1. Load data
data = np.load('vix_data.npz')
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']

# 2. Reshape for LSTM: [batch, seq_len=20, input_size=1]
X_train_tensor = torch.tensor(X_train.reshape(-1, 20, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.reshape(-1, 20, 1), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.reshape(-1, 5), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.reshape(-1, 5), dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 3. Define LSTM model
class VIXLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, 5)

    def forward(self, x):
        out, _ = self.lstm(x)           # out shape: [batch, seq_len, hidden]
        out = out[:, -1, :]             # last time step: [batch, hidden]
        return self.fc(out)             # output: [batch, 5]

model = VIXLSTM()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

# 4. Train
def train_model(model, train_loader, criterion, optimizer, epochs=50):
    model.train()
    losses = []
    for epoch in range(epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg = total_loss / len(train_loader)
        losses.append(avg)
        print(f"Epoch {epoch+1}/50], Loss: {avg:.4f}")
    return losses

losses = train_model(model, train_loader, criterion, optimizer)

# ✅ 5. Save model
torch.save(model.state_dict(), 'vix_predictor_by_ISTM.pth')

# 6. Evaluate
model.eval()
with torch.no_grad():
    preds = model(X_test_tensor)
    preds_np = preds.numpy()
    y_true = y_test_tensor.numpy()

mae = mean_absolute_error(y_true, preds_np)
mse = mean_squared_error(y_true, preds_np)
mape = np.mean(np.abs((y_true - preds_np) / y_true)) * 100

print("\n📊 LSTM Test Set Evaluation:")
print(f"MAE  : {mae:.4f}")
print(f"MSE  : {mse:.4f}")
print(f"MAPE : {mape:.2f}%")

# 7. Plot loss
plt.plot(losses, label='LSTM Loss')
plt.title("LSTM Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

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

plot_predictions_all_samples(preds, y_test_tensor)

# 8️⃣ 保存评估指标到 CSV 文件
import csv
import os

model_name = 'LSTM'  # 设置当前模型名

# 保证预测值与真实值为 numpy 格式
if isinstance(y_test_tensor, torch.Tensor):
    y_true = y_test_tensor.numpy()
if isinstance(preds, torch.Tensor):
    y_pred = preds.numpy()

# 再次计算指标（确保准确性）
mae = mean_absolute_error(y_true, y_pred)
mse = mean_squared_error(y_true, y_pred)
mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# 设置 CSV 文件路径
csv_path = 'vix_model_results.csv'
file_exists = os.path.exists(csv_path)

# 写入或追加到 CSV
with open(csv_path, mode='a', newline='') as f:
    writer = csv.writer(f)
    if not file_exists:
        writer.writerow(['Model', 'MAE', 'MSE', 'MAPE (%)'])  # 首次写入表头
    writer.writerow([model_name, f"{mae:.4f}", f"{mse:.4f}", f"{mape:.2f}"])
