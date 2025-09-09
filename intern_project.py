# stock_lstm.py
# Requirements: pandas, numpy, matplotlib, yfinance, scikit-learn, tensorflow

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -----------------------------
# 1. Load stock data
# -----------------------------
def load_data(ticker="AAPL", start="2015-01-01", end="2025-01-01"):
    df = yf.download(ticker, start=start, end=end)
    df = df[["Close"]]   # only closing price
    return df

# -----------------------------
# 2. Preprocess data
# -----------------------------
def preprocess_data(df, time_steps=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(time_steps, len(scaled)):
        X.append(scaled[i - time_steps:i, 0])
        y.append(scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # [samples, timesteps, features]
    return X, y, scaler

# -----------------------------
# 3. Build LSTM model
# -----------------------------
def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

# -----------------------------
# 4. Train and predict
# -----------------------------
def train_predict(df, time_steps=60, epochs=10, batch_size=32):
    train_size = int(len(df) * 0.8)
    train_data = df[:train_size]
    test_data = df[train_size - time_steps:]

    # Preprocess
    X_train, y_train, scaler = preprocess_data(train_data, time_steps)
    X_test, y_test, _ = preprocess_data(test_data, time_steps)

    # Build model
    model = build_lstm((X_train.shape[1], 1))
    model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)

    # Predict
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

    return y_test_rescaled, predictions, df.index[train_size:]

# -----------------------------
# 5. Plot results
# -----------------------------
def plot_results(df, y_test, predictions, test_dates):
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["Close"], label="Actual Price", color="blue")
    plt.plot(test_dates, predictions, label="Predicted Price", color="red")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.title("Stock Price Prediction with LSTM")
    plt.legend()
    plt.show()

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    ticker = input("Enter stock ticker (default AAPL): ") or "AAPL"
    df = load_data(ticker)

    y_test, predictions, test_dates = train_predict(df, time_steps=60, epochs=15, batch_size=32)

    # (3) Print sample Actual vs Predicted
    results = pd.DataFrame({
        "Date": test_dates,
        "Actual": y_test.flatten(),
        "Predicted": predictions.flatten()
    })
    print("\n📊 Last 10 Predictions:")
    print(results.tail(10))

    # Plot
    plot_results(df, y_test, predictions, test_dates)