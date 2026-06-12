import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import pickle

# ── Load data ──────────────────────────────────────────────────────────────
df = pd.read_csv("crop_data.csv")

le_veg  = LabelEncoder()
le_city = LabelEncoder()

df["vegetable"] = le_veg.fit_transform(df["vegetable"])
df["city"]      = le_city.fit_transform(df["city"])

df["lag1"] = df["current_price"].shift(1)
df["lag2"] = df["current_price"].shift(2)
df = df.dropna()

FEATURES = ["vegetable", "city", "temperature", "rainfall", "demand", "lag1", "lag2"]
X = df[FEATURES].values
y = df["future_price"].values

# ── 1. Random Forest (existing) ────────────────────────────────────────────
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y)
print("✅ Random Forest trained")

# ── 2. LSTM Deep Learning ──────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()

    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    # Reshape for LSTM: (samples, timesteps, features)
    X_lstm = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

    lstm_model = Sequential([
        LSTM(64, input_shape=(1, X_scaled.shape[1]), return_sequences=True),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])

    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(
        X_lstm, y_scaled,
        epochs=30,
        batch_size=32,
        validation_split=0.1,
        verbose=0
    )
    lstm_available = True
    print("✅ LSTM model trained")

except ImportError:
    lstm_model     = None
    scaler_X       = None
    scaler_y       = None
    lstm_available = False
    print("⚠️  TensorFlow not found — LSTM unavailable. Run: pip install tensorflow")

# ── Save everything ────────────────────────────────────────────────────────
pickle.dump(
    (rf_model, le_veg, le_city, lstm_model, scaler_X, scaler_y, lstm_available),
    open("model.pkl", "wb")
)
print("✅ All models saved to model.pkl")
