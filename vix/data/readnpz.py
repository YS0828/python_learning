import numpy as np

npz_data = np.load('vix_data.npz')
print("Keys in the npz file:", npz_data.files)
print("X_train shape:", npz_data['X_train'].shape)
print("y_train shape:", npz_data['y_train'].shape)
print("X_test shape:", npz_data['X_test'].shape)
print("y_test shape:", npz_data['y_test'].shape)
# 读取并打印 npz 文件中的数据

print("Hello", "World")