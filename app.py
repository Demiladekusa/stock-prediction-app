from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

# Load the model
model = keras.models.load_model('stock_prediction_model.keras')

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    company = request.form['company']
    start_date = request.form['start_date']
    end_date = request.form['end_date']

    # Load the dataset
    data = pd.read_csv('C:/Users/Hp/OneDrive/Documents/PersonalProjects/StockPredictionApp/Data/all_stocks_5yr.csv')
    data['date'] = pd.to_datetime(data['date'])
    data.dropna(inplace=True)
    
    # Filter the data for the selected company and date range
    filtered_data = data[(data['Name'] == company) & 
                         (data['date'] >= start_date) & 
                         (data['date'] <= end_date)]

    # Make sure to sort the data by date
    filtered_data = filtered_data.sort_values(by='date')

    # Prepare the data for prediction
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data[['close']])
    scaled_data = scaler.transform(filtered_data[['close']])
    x_test = []

    # Create the sequence of the past 60 days
    for i in range(60, len(scaled_data)):
        x_test.append(scaled_data[i-60:i, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Predict using the model
    predictions = model.predict(x_test)

    # Inverse transform to get the actual predicted prices
    predictions = scaler.inverse_transform(predictions)

    # Prepare dates for plotting
    dates = filtered_data['date'].iloc[60:].values

    # Plot the actual prices and the predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(filtered_data['date'], filtered_data['close'], color='blue', label='Actual Prices')
    plt.plot(dates, predictions, color='red', label='Predicted Prices')
    plt.title(f'Actual vs Predicted Stock Prices for {company}')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    
    # Save the plot to a file
    plt.savefig('static/predicted_vs_actual.png')
    
    return render_template('index.html', prediction_plot='static/predicted_vs_actual.png')

if __name__ == "__main__":
    app.run(debug=True)

