from flask import Flask, request, render_template
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the trained model
model = joblib.load('flight_delay_model.pkl')


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


# Simulate fetching weather data
def get_weather_data(month, day_of_month, carrier, dest, flight_num):
    # This is a placeholder function. Replace with actual data fetching logic.
    # Simulated weather data
    weather_data = {
        'Temperature': np.random.uniform(20, 30),
        'Dew Point': np.random.uniform(10, 20),
        'Humidity': np.random.uniform(40, 60),
        'Wind Speed': np.random.uniform(5, 15),
        'Wind Gust': np.random.uniform(0, 10),
        'Pressure': np.random.uniform(29.5, 30.5)
    }
    return weather_data


# Define route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.form

    # Extract input flight data
    month = int(data['month'])
    day_of_month = int(data['day_of_month'])
    carrier = data['carrier']
    dest = data['dest']
    flight_num = data['flight_num']

    # Fetch or simulate weather data based on the flight details
    weather_data = get_weather_data(month, day_of_month, carrier, dest, flight_num)

    # Prepare input data for prediction
    input_data = {feature: weather_data[feature] for feature in
                  ['Temperature', 'Dew Point', 'Humidity', 'Wind Speed', 'Wind Gust', 'Pressure']}

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])

    # Make prediction
    prediction = model.predict(input_df)[0]
    delay_status = 'Delayed' if prediction == 1 else 'Not Delayed'

    return render_template('index.html', prediction=delay_status)


if __name__ == '__main__':
    app.run(debug=True)
