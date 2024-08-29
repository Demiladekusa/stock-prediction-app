import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import os

# Loading the dataset
data = pd.read_csv('C:/Users/Hp/OneDrive/Documents/PersonalProjects/StockPredictionApp/Data/all_stocks_5yr.csv')

# Convert the date column to datetime type
data['date'] = pd.to_datetime(data['date'])

# Drop rows with missing values
data.dropna(inplace=True)

# List of companies to include in training
companies = ['AAPL', 'GOOGL', 'AMZN', 'MSFT', 'FB', 'NFLX', 'TSLA']

# Filter the data to include only the selected companies
filtered_data = data[data['Name'].isin(companies)]


# Creating an empty list to hold the training sequences
x_train = []
y_train = []

# Set the number of previous days to consider 
time_steps = 60

# Scaling the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(filtered_data[['close']])


# Loop through the data to create sequences
for i in range(time_steps, len(scaled_data)):
    x_train.append(scaled_data[i-time_steps:i, 0])
    y_train.append(scaled_data[i, 0])


# Convert to numpy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape the data to fit the LSTM input requirements
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Building the LSTM model
model = keras.Sequential()

model.add(keras.layers.LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(keras.layers.LSTM(units=50))
model.add(keras.layers.Dense(units=25))
model.add(keras.layers.Dense(units=1))

# Compiling the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Training the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Save the trained model
model.save('stock_prediction_model.keras')


