import yfinance as yf
import pandas as pd
import numpy as np

def get_vix(start_date, end_date):
    vix = yf.download('^VIX', start=start_date, end=end_date)
    vix = vix['Close'].dropna()
    return vix

def generate_sliding_window_data(vix, lookback, horizon):
    X, y = [], []
    data = np.array(vix)
    for i in range(len(data) - lookback - horizon + 1):
        X.append(data[i:i+lookback])
        y.append(data[i+lookback:i+lookback+horizon])
    return np.array(X), np.array(y)
    
def train_test_split(X, y, train_ratio):
    split_index = int(len(X) * train_ratio)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    return X_train, X_test, y_train, y_test

def save_data_to_npz(X_train, y_train, X_test, y_test, filename):
    np.savez(filename, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

vix = get_vix('2010-01-01', '2025-06-01')
X, y = generate_sliding_window_data(vix, lookback=20, horizon=5)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_ratio=0.8)
save_data_to_npz(X_train, y_train, X_test, y_test, 'vix_data.npz')