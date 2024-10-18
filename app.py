from flask import Flask, render_template, request, redirect, url_for, session
import joblib
import pandas as pd
import numpy as np
import requests
import json
import plotly.graph_objects as go

app = Flask(__name__)

# Load the SVM model
svm_poly_model = joblib.load('svm_poly_model.pkl')

# Label encoding mappings for SVM model
crop_type_mapping = {
    'BANANA': 0, 'BEAN': 1, 'CABBAGE': 2, 'CITRUS': 3, 'COTTON': 4,
    'MAIZE': 5, 'MELON': 6, 'MUSTARD': 7, 'ONION': 8, 'OTHER': 9,
    'POTATO': 10, 'RICE': 11, 'SOYABEAN': 12, 'SUGARCANE': 13,
    'TOMATO': 14, 'WHEAT': 15
}

soil_type_mapping = {'DRY': 0, 'HUMID': 1, 'WET': 2}
weather_condition_mapping = {'NORMAL': 0, 'RAINY': 1, 'SUNNY': 2, 'WINDY': 3}


# Fetch weather data from the OpenWeatherMap API
def get_weather(city):
    api_key = "b3c62ae7f7ad5fc3cb0a7b56cb7cbda6"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for any HTTP errors
    except requests.exceptions.HTTPError as err:
        return None, None, None, None

    try:
        data = json.loads(response.text)
        if data['cod'] != 200:
            return None, None, None, None
    except json.JSONDecodeError as err:
        return None, None, None, None

    # Extract relevant weather information
    weather_description = data['weather'][0]['description']
    temperature = data['main']['temp']
    humidity = data['main']['humidity']
    pressure = data['main']['pressure']
    # Convert temperature from Kelvin to Celsius
    temperature = round(temperature - 273.15, 2)

    return temperature, humidity, weather_description, pressure


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/fetch_weather', methods=['GET'])
def fetch_weather():
    city = request.args.get('city')
    if city:
        temperature, humidity, weather_description, pressure = get_weather(city)
        if temperature is not None:
            return json.dumps({
                'description': weather_description.capitalize(),
                'temperature': temperature,
                'humidity': humidity,
                'pressure': pressure
            })
    return json.dumps(None)  # Return None if weather couldn't be fetched



@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        crop_type = request.form['crop_type']
        soil_type = request.form['soil_type']
        city = request.form['city']
        motor_capacity = float(request.form['motor_capacity'])

        # Get weather data
        temperature, humidity, weather_description, pressure = get_weather(city)

        # Auto-fill weather condition based on description
        if temperature is None:
            auto_weather_condition = 'NORMAL'
            temperature = 32.0
        else:
            auto_weather_condition = ('SUNNY' if 'clear' in weather_description.lower() else
                                       'RAINY' if 'rain' in weather_description.lower() else
                                       'WINDY' if 'wind' in weather_description.lower() else
                                       'NORMAL')

        # Encode inputs for the model
        crop_type_encoded = crop_type_mapping[crop_type]
        soil_type_encoded = soil_type_mapping[soil_type]
        weather_condition_encoded = weather_condition_mapping[auto_weather_condition]

        # Create DataFrame for input
        user_data = pd.DataFrame({
            'CROP TYPE': [crop_type_encoded],
            'SOIL TYPE': [soil_type_encoded],
            'TEMPERATURE': [temperature],
            'WEATHER CONDITION': [weather_condition_encoded]
        })

        # Predict water requirement
        water_requirement = svm_poly_model.predict(user_data)[0]

        # Calculate estimated time duration
        estimated_time_duration = water_requirement / motor_capacity if motor_capacity > 0 else 0
        time_unit = "seconds" if estimated_time_duration < 1 else "minutes"
        estimated_time_duration = estimated_time_duration * 60 if time_unit == "seconds" else estimated_time_duration

        # Create the gauge charts for water requirement and estimated time
        water_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=water_requirement,
            title={"text": "Water Requirement (m³/sq.m)"},
            gauge={
                "axis": {"range": [None, 100]},
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 20], "color": "lightgray"},
                    {"range": [20, 50], "color": "yellow"},
                    {"range": [50, 100], "color": "green"}
                ]
            }
        ))

        time_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=estimated_time_duration,
            title={"text": "Estimated Time (Seconds)"},
            gauge={
                "axis": {"range": [None, 120]},
                "bar": {"color": "blue"},
                "steps": [
                    {"range": [0, 20], "color": "lightgray"},
                    {"range": [20, 60], "color": "yellow"},
                    {"range": [60, 120], "color": "green"}
                ]
            }
        ))

        # Send gauge charts to the prediction page
        water_gauge_path = water_gauge.to_html(full_html=False)
        time_gauge_path = time_gauge.to_html(full_html=False)

        return render_template('predict.html',
                               water_requirement=round(water_requirement, 2),
                               estimated_time_duration=round(estimated_time_duration, 2),
                               time_unit=time_unit,
                               weather_info=f"Weather in {city}: {weather_description.capitalize()}<br>"
                                            f"Temperature: {temperature}°C<br>"
                                            f"Humidity: {humidity}%<br>"
                                            f"Pressure: {pressure} hPa<br>",
                               water_gauge=water_gauge_path,
                               time_gauge=time_gauge_path)

    return redirect(url_for('index'))


def start_motor():
    estimated_time_duration = request.json.get('estimated_time_duration')

    # Assuming this function will trigger the motor start, we will just send the confirmation
    # Frontend will handle the timer and stop functionality
    print(f"Starting motor for {estimated_time_duration} seconds")

    # Optionally, you can trigger hardware or database actions here if needed
    # but for simulation, we are just sending a response.

    return jsonify({"status": "motor_started", "estimated_time_duration": estimated_time_duration})


if __name__ == "__main__":
    app.run(debug=True)
