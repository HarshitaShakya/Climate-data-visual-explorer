import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from dotenv import load_dotenv
import os

load_dotenv()  # Loads variables from .env into environment

API_KEY = os.getenv("API_KEY")


BASE_URL = 'https://api.openweathermap.org/data/2.5/'

# ---------------- DATA FUNCTIONS ----------------
def get_current_weather(city):
    url = f"{BASE_URL}weather?q={city}&appid={API_KEY}&units=metric"
    response = requests.get(url)
    data = response.json()
    return {
        "city": data['name'],
        'current_temp': round(data['main']['temp']),
        'feels_like': round(data['main']['feels_like']),
        'temp_min': round(data['main']['temp_min']),
        'temp_max': round(data['main']['temp_max']),
        'humidity': round(data['main']['humidity']),
        'description': data['weather'][0]['description'],
        'icon': data['weather'][0]['icon'],   # ✅ Add weather icon
        'country': data['sys']['country'],
        'wind_gust_dir': data['wind']['deg'],
        'pressure': data['main']['pressure'],
        'wind_gust_speed': data['wind']['speed']
    }

def read_historical_data(file_path):
    df = pd.read_csv(file_path).dropna().drop_duplicates()
    if 'Date' not in df.columns:
        df['Date'] = pd.date_range(start="2000-01-01", periods=len(df), freq="D")
    return df

def prepare_data(data):
    le = LabelEncoder()
    data['WindGustDir'] = le.fit_transform(data['WindGustDir'])
    data['RainTomorrow'] = le.fit_transform(data['RainTomorrow'])
    X = data[['MinTemp','MaxTemp','WindGustDir','WindGustSpeed','Humidity','Pressure','Temp']]
    y = data['RainTomorrow']
    return X, y, le

def train_rain_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Rain Model MSE:", mean_squared_error(y_test, y_pred))
    return model

def prepare_regression_data(data, feature):
    X, y = [], []
    for i in range(len(data)-1):
        X.append(data[feature].iloc[i])
        y.append(data[feature].iloc[i+1])
    X = np.array(X).reshape(-1, 1)
    y = np.array(y)
    return X, y

def train_regression_model(X, y):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model

def predict_future(model, current_values):
    predictions = [current_values]
    for i in range(5):
        next_value = model.predict(np.array([[predictions[-1]]]))
        predictions.append(next_value[0])
    return predictions[1:]

def weather_view(city):
    current_weather = get_current_weather(city)
    historical_data = read_historical_data('weather.csv')
    X, y, le_rain = prepare_data(historical_data)
    rain_model = train_rain_model(X, y)

    # Compass direction mapping
    wind_deg = current_weather['wind_gust_dir'] % 360
    compass_points = [
        ("N", 0, 11.25), ("NNE", 11.25, 33.75), ("NE", 33.75, 56.25),
        ("ENE", 56.25, 78.75), ("E", 78.75, 101.25), ("ESE", 101.25, 123.75),
        ("SE", 123.75, 146.25), ("SSE", 146.25, 168.75), ("S", 168.75, 191.25),
        ("SSW", 191.25, 213.75), ("SW", 213.75, 236.25), ("WSW", 236.25, 258.75),
        ("W", 258.75, 281.25), ("WNW", 281.25, 303.75), ("NW", 303.75, 326.25),
        ("NNW", 326.25, 348.75)
    ]
    compass_direction = next(point for point, start, end in compass_points if start <= wind_deg < end)
    compass_direction_encoded = le_rain.transform([compass_direction])[0] if compass_direction in le_rain.classes_ else -1

    current_data = {
        'MinTemp': current_weather['temp_min'],
        'MaxTemp': current_weather['temp_max'],
        'WindGustDir': compass_direction_encoded,
        'WindGustSpeed': current_weather['wind_gust_speed'],
        'Humidity': current_weather['humidity'],
        'Pressure': current_weather['pressure'],
        'Temp': current_weather['current_temp']
    }
    current_df = pd.DataFrame([current_data])
    rain_prediction = rain_model.predict(current_df)[0]

    # Regression models for temp & humidity
    X_temp, y_temp = prepare_regression_data(historical_data, 'Temp')
    X_hum, y_hum = prepare_regression_data(historical_data, 'Humidity')
    temp_model = train_regression_model(X_temp, y_temp)
    hum_model = train_regression_model(X_hum, y_hum)
    future_temp = [round(float(x), 1) for x in predict_future(temp_model, current_weather['current_temp'])]
    future_hum = [round(float(x), 1) for x in predict_future(hum_model, current_weather['humidity'])]

    results = {
        "current_weather": current_weather,
        "rain_prediction": rain_prediction,
        "future_temp": future_temp,
        "future_hum": future_hum,
        "historical_data": historical_data
    }
    return results

# ---------------- VISUALIZATIONS ----------------
def plot_historical_trends(data):
    st.subheader("📈 Historical Trends")
    fig, ax = plt.subplots(figsize=(12,6))  # ✅ Fixed size
    sns.lineplot(data=data, x="Date", y="Temp", label="Temperature (°C)", ax=ax)
    sns.lineplot(data=data, x="Date", y="Humidity", label="Humidity (%)", ax=ax)
    ax.legend()
    st.pyplot(fig, use_container_width=True)

def plot_heatmap(data):
    st.subheader("🌡️ Seasonal Heatmap")
    data['Month'] = pd.to_datetime(data['Date']).dt.month
    pivot = data.pivot_table(index='Month', columns=pd.to_datetime(data['Date']).dt.year, values='Temp', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12,6))  # ✅ Fixed size
    sns.heatmap(pivot, cmap="coolwarm", ax=ax)
    st.pyplot(fig, use_container_width=True)

def detect_anomalies(data, feature="Temp"):
    st.subheader(f"⚠️ Anomaly Detection in {feature}")
    model = IsolationForest(contamination=0.05, random_state=42)
    data['anomaly'] = model.fit_predict(data[[feature]])
    anomalies = data[data['anomaly'] == -1]
    
    fig, ax = plt.subplots(figsize=(12,6))  # ✅ Fixed size
    ax.plot(data['Date'], data[feature], label=feature)
    ax.scatter(anomalies['Date'], anomalies[feature], color='red', label="Anomaly")
    ax.legend()
    st.pyplot(fig, use_container_width=True)
    return anomalies

def generate_summary(data):
    st.subheader("📝 Insights Summary")
    avg_temp = data['Temp'].mean()
    max_temp = data['Temp'].max()
    min_temp = data['Temp'].min()
    avg_hum = data['Humidity'].mean()
    st.write(f"- The **average temperature** is {avg_temp:.1f}°C (range: {min_temp}–{max_temp}°C).")
    st.write(f"- The **average humidity** is {avg_hum:.1f}%.")
    st.write(f"- Seasonal patterns show clear variation in temperature across months.")

# ---------------- STREAMLIT APP ----------------
st.title("Climate Data Visual Explorer 🌍")
city = st.text_input("Enter City Name:")

if st.button("Get Weather"):
    results = weather_view(city)
    cw = results['current_weather']
    historical_data = results['historical_data']

    tabs = st.tabs([
        "🌍 Current Weather",
        "📈 Historical Trends",
        "🔮 Forecasts",
        "⚠️ Anomaly Detection",
        "📝 Insights"
    ])

    with tabs[0]:
        st.subheader("Current Weather")
        col1, col2 = st.columns([1,2])
        with col1:
            icon_url = f"http://openweathermap.org/img/wn/{cw['icon']}@4x.png"  # ✅ High-res icon
            st.image(icon_url, caption=cw['description'].capitalize())
        with col2:
            st.write(f"**City:** {cw['city']}, {cw['country']}")
            st.write(f"**Current Temp:** {cw['current_temp']}°C")
            st.write(f"**Feels Like:** {cw['feels_like']}°C")
            st.write(f"**Min Temp:** {cw['temp_min']}°C | **Max Temp:** {cw['temp_max']}°C")
            st.write(f"**Humidity:** {cw['humidity']}%")
            st.write(f"**Pressure:** {cw['pressure']} hPa")
            st.write(f"**Wind Speed:** {cw['wind_gust_speed']} m/s")
            st.write(f"**Rain Prediction:** {'🌧️ Yes' if results['rain_prediction'] else '☀️ No'}")

    with tabs[1]:
        plot_historical_trends(historical_data)
        plot_heatmap(historical_data)

    # -------- Forecasts --------
    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🔮 Forecasts (Next 5 Hours)")
        forecast_df = pd.DataFrame({
            "Hour Ahead": [f"+{i}h" for i in range(1, 6)],
            "Temp (°C)": results['future_temp'],
            "Humidity (%)": results['future_hum']
        })
        st.dataframe(forecast_df, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tabs[3]:
        anomalies = detect_anomalies(historical_data, "Temp")
        st.write(f"Detected {len(anomalies)} anomalies in temperature data.")

    with tabs[4]:

    # Basic summary
        generate_summary(historical_data)

    # Compute stats
    avg_temp = historical_data['Temp'].mean()
    max_temp = historical_data['Temp'].max()
    min_temp = historical_data['Temp'].min()
    avg_hum = historical_data['Humidity'].mean()

    # Use metrics in a row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🌡️ Avg Temp", f"{avg_temp:.1f}°C")
    col2.metric("🔥 Max Temp", f"{max_temp}°C")
    col3.metric("❄️ Min Temp", f"{min_temp}°C")
    col4.metric("💧 Avg Humidity", f"{avg_hum:.1f}%")

    # Glassmorphism styled box for insights
    st.markdown(f"""
    <div style="
        background: rgba(0, 0, 0, 0.55);
        backdrop-filter: blur(8px);
        border-radius: 15px;
        padding: 20px;
        margin-top: 15px;
        border: 1px solid rgba(255,255,255,0.3);
        color: #fff;
    ">
        <h4>🔍 Key Observations:</h4>
        <ul>
            <li>The <b>average temperature</b> is around {avg_temp:.1f}°C.</li>
            <li>The <b>temperature range</b> goes from {min_temp}°C to {max_temp}°C.</li>
            <li>The <b>average humidity</b> stays around {avg_hum:.1f}%.</li>
            <li>Seasonal patterns show clear variation across months.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
























