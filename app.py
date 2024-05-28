from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('flight_delay_model.pkl')

# Load CSV data
flights_df = pd.read_csv('data/flights.csv')
weather_df = pd.read_csv('data/weather.csv')

# Merge the flights and weather dataframes on the id column
combined_df = pd.merge(flights_df, weather_df, on='id')


# Define route for home page
@app.route('/')
def home():
    return render_template('index.html')


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

    # Filter the combined dataframe based on user input
    flight_data = combined_df[
        (combined_df['month'] == month) &
        (combined_df['day_of_month'] == day_of_month) &
        (combined_df['op_unique_carrier'] == carrier) &
        (combined_df['dest'] == dest) &
        (combined_df['tail_num'] == flight_num)
        ]

    if not flight_data.empty:
        # Use the first matched row for prediction
        flight_data = flight_data.iloc[0]

        # Prepare input data for prediction
        input_data = {
            'Temperature': flight_data['temperature'],
            'Dew Point': flight_data['dew_point'],
            'Humidity': flight_data['humidity'],
            'Wind Speed': flight_data['wind_speed'],
            'Wind Gust': flight_data['wind_gust'],
            'Pressure': flight_data['pressure']
        }

        input_df = pd.DataFrame([input_data])

        # Make prediction
        prediction = model.predict(input_df)[0]
        delay_status = 'Delayed' if prediction == 1 else 'Not Delayed'
    else:
        delay_status = 'No data found for the provided flight details.'

    return render_template('index.html', prediction=delay_status)


if __name__ == '__main__':
    app.run(debug=True)
