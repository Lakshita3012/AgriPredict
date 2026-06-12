🌱 AgriPredict — Crop Price Prediction System

This project predicts vegetable prices (Onion, Tomato, Potato) across 5 Indian cities using both Machine Learning and Deep Learning with real-time weather data.

Pages:
  · Dashboard → Select city & vegetable → fetches live weather → 7-day price forecast
  · Market    → Dynamic prices for all city-vegetable combinations
  · Analytics → 4 charts: price trend, scatter, demand bar, volatility

Models:
  · Random Forest Regressor (sklearn) — fast, no extra dependencies
  · LSTM Deep Learning (TensorFlow/Keras) — sequence model, requires tensorflow

To run:

  # 1. Install core libraries
  pip3 install flask scikit-learn pandas numpy requests

  # 2. Install TensorFlow for LSTM (optional but recommended)
  pip3 install tensorflow

  # 3. Add your OpenWeatherMap API key in app.py (line with WEATHER_API_KEY)

  # 4. Generate data & train both models
  python3 generate_data.py
  python3 model.py

  # 5. Start the app
  python3 app.py

  # 6. Open in browser
  http://127.0.0.1:5000

If TensorFlow is not installed, the app runs fine with Random Forest only.
The LSTM button will appear disabled until tensorflow is available.
