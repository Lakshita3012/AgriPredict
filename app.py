from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import requests

app = Flask(__name__)

# ── Load models ──────────────────────────────────────────────────────────────
bundle = pickle.load(open("model.pkl", "rb"))
if len(bundle) == 7:
    rf_model, le_veg, le_city, lstm_model, scaler_X, scaler_y, lstm_available = bundle
else:
    # Legacy pickle (old 3-item format)
    rf_model, le_veg, le_city = bundle
    lstm_model = scaler_X = scaler_y = None
    lstm_available = False

df = pd.read_csv("crop_data.csv")

# ── Config ───────────────────────────────────────────────────────────────────
WEATHER_API_KEY = "54c046548a102dddfa75ce3a0b2fff21"

CITY_COORDS = {
    "Chennai":   {"lat": 13.0827, "lon": 80.2707},
    "Mumbai":    {"lat": 19.0760, "lon": 72.8777},
    "Delhi":     {"lat": 28.6139, "lon": 77.2090},
    "Bangalore": {"lat": 12.9716, "lon": 77.5946},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867},
}

BASE_DEMAND = {
    "Chennai":   {"Onion": 220, "Tomato": 240, "Potato": 200},
    "Mumbai":    {"Onion": 260, "Tomato": 270, "Potato": 210},
    "Delhi":     {"Onion": 230, "Tomato": 250, "Potato": 220},
    "Bangalore": {"Onion": 210, "Tomato": 230, "Potato": 190},
    "Hyderabad": {"Onion": 200, "Tomato": 220, "Potato": 180},
}

BASE_PRICE = {"Onion": 22, "Tomato": 19, "Potato": 26}


# ── Helpers ───────────────────────────────────────────────────────────────────
def get_weather(city):
    coords = CITY_COORDS.get(city, CITY_COORDS["Chennai"])
    url = (
        f"https://api.openweathermap.org/data/2.5/weather"
        f"?lat={coords['lat']}&lon={coords['lon']}"
        f"&appid={WEATHER_API_KEY}&units=metric"
    )
    try:
        r = requests.get(url, timeout=5)
        data = r.json()
        return {
            "temperature":  round(data["main"]["temp"], 1),
            "rainfall":     round(data.get("rain", {}).get("1h", 0), 1),
            "humidity":     data["main"]["humidity"],
            "description":  data["weather"][0]["description"].title(),
            "icon":         f"https://openweathermap.org/img/wn/{data['weather'][0]['icon']}@2x.png",
            "live":         True,
        }
    except Exception as e:
        return {
            "temperature": 30, "rainfall": 5, "humidity": 65,
            "description": "Data unavailable", "icon": "",
            "live": False, "error": str(e),
        }


def predict_rf(veg_enc, city_enc, temp, rain, demand, lag1, lag2):
    inp = np.array([[veg_enc, city_enc, temp, rain, demand, lag1, lag2]])
    forecast = []
    for _ in range(7):
        pred = round(float(rf_model.predict(inp)[0]), 2)
        forecast.append(pred)
        inp[0][5], inp[0][6] = pred, inp[0][5]   # shift lags
    return forecast


def predict_lstm(veg_enc, city_enc, temp, rain, demand, lag1, lag2):
    if not lstm_available or lstm_model is None:
        return None
    row = np.array([[veg_enc, city_enc, temp, rain, demand, lag1, lag2]], dtype=float)
    forecast = []
    l1, l2 = lag1, lag2
    for _ in range(7):
        row[0][5], row[0][6] = l1, l2
        scaled = scaler_X.transform(row).reshape(1, 1, 7)
        pred_s = lstm_model.predict(scaled, verbose=0)[0][0]
        pred   = round(float(scaler_y.inverse_transform([[pred_s]])[0][0]), 2)
        forecast.append(pred)
        l2, l1 = l1, pred
    return forecast


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route("/")
def home():
    return render_template("dashboard.html",
                           cities=list(CITY_COORDS.keys()),
                           lstm_available=lstm_available)


@app.route("/get_data", methods=["POST"])
def get_data():
    data      = request.json
    city      = data["city"]
    vegetable = data["vegetable"]
    weather   = get_weather(city)
    demand    = BASE_DEMAND.get(city, {}).get(vegetable, 220)
    humidity_factor = weather["humidity"] / 100
    demand    = int(demand * (0.9 + humidity_factor * 0.2))
    base      = BASE_PRICE.get(vegetable, 20)
    price     = round(base + (demand * 0.02) - (weather["rainfall"] * 0.15), 2)
    return jsonify({
        "temperature": weather["temperature"],
        "rainfall":    weather["rainfall"],
        "humidity":    weather["humidity"],
        "description": weather["description"],
        "icon":        weather["icon"],
        "demand":      demand,
        "price":       price,
        "live":        weather["live"],
    })


@app.route("/analytics")
def analytics():
    data = df.head(30)
    return render_template(
        "analytics.html",
        labels=list(range(1, len(data) + 1)),
        prices=list(data["current_price"]),
        demand=list(data["demand"]),
        rainfall=list(data["rainfall"]),
        future_prices=list(data["future_price"]),
    )


@app.route("/market")
def market():
    market_data = []
    for city in CITY_COORDS:
        weather = get_weather(city)
        for veg in ["Onion", "Tomato", "Potato"]:
            demand = BASE_DEMAND[city][veg]
            price  = round(BASE_PRICE[veg] + (demand * 0.02) - (weather["rainfall"] * 0.15), 2)
            market_data.append({
                "city": city, "vegetable": veg, "price": price,
                "temperature": weather["temperature"], "rainfall": weather["rainfall"],
            })
    return render_template("market.html", market_data=market_data)


@app.route("/predict", methods=["POST"])
def predict():
    data    = request.json
    model   = data.get("model", "rf")          # "rf" or "lstm"
    veg     = le_veg.transform([data["vegetable"]])[0]
    city    = le_city.transform([data["city"]])[0]
    price   = float(data["price"])
    temp    = float(data["temperature"])
    rain    = float(data["rainfall"])
    demand  = float(data["demand"])
    lag1    = price
    lag2    = price - 2
    history = [round(price - 5, 2), round(price - 2, 2), round(price, 2)]

    if model == "lstm":
        forecast = predict_lstm(veg, city, temp, rain, demand, lag1, lag2)
        if forecast is None:
            return jsonify({"error": "LSTM not available"}), 400
    else:
        forecast = predict_rf(veg, city, temp, rain, demand, lag1, lag2)

    return jsonify({
        "predicted": forecast[0],
        "history":   history,
        "forecast":  forecast,
        "model":     model,
    })


@app.route("/model_status")
def model_status():
    return jsonify({"lstm_available": lstm_available})


if __name__ == "__main__":
    app.run(debug=True)
