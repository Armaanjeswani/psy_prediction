import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.models import model_from_json
from keras.models import load_model
import json

# Load data
data = pd.read_csv("D:\Psy\psy_prediction\physiomize data1.csv")

# Separate features and target variable
X = data.drop('time', axis=1)
Y = data['time']

# Normalize the target variable to ensure it's within a reasonable range
max_time = Y.max()
Y = Y / max_time

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Reshape input data for LSTM
X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test.values, (X_test.shape[0], X_test.shape[1], 1))

# Define the LSTM model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(units=100))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=32)

# Save the model architecture as JSON
model_json = model.to_json()
with open("psy_prediction_architecture.json", "w") as json_file:
    json_file.write(model_json)

# Save the trained model weights
model.save("psy_prediction_weights.h5")

# Load the model architecture from JSON
with open('psy_prediction_architecture.json', 'r') as json_file:
    loaded_model_json = json_file.read()

# Load the architecture
loaded_model = model_from_json(loaded_model_json)

# Compile the loaded model with loss and metrics
loaded_model.compile(optimizer='adam', loss='mean_squared_error')

# Load the trained model weights
loaded_model.load_weights('psy_prediction_weights.h5')

num_weeks = int(input("Enter the number of weeks: "))
input_data = []
for i in range(num_weeks):
    input_value = float(input(f"Enter value for week {i+1}: "))
    input_data.append(input_value)

input_array = np.reshape(np.array(input_data), (1, num_weeks, 1))

prediction = loaded_model.predict(input_array)
scaling_factor = 0.8  # Adjust this scaling factor as needed
noise = np.random.normal(loc=0, scale=0.1, size=prediction.shape)  # Adding noise to the prediction
predicted_time = int((prediction[0][0] + noise[0][0]) * max_time * scaling_factor)  # Scale the prediction back to the original range with scaling factor

print("Predicted time:", predicted_time, "Weeks")
