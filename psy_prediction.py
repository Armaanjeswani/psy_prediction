import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense

# Load data
data = pd.read_csv("D:/Psy/psy_prediction/physiomize data2.csv")

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

# Save the entire model to a single HDF5 file
model.save("psy_prediction_model.h5")

# Load the model from the HDF5 file
loaded_model = load_model("psy_prediction_model.h5")

# Define scaling factor
scaling_factor = 0.8  # Adjust this scaling factor as needed

# Prediction using the loaded model
num_weeks = int(input("Enter the number of weeks: "))
input_data = []
for i in range(num_weeks):
    input_value = float(input(f"Enter value for week {i+1}: "))
    input_data.append(input_value)

input_array = np.reshape(np.array(input_data), (1, num_weeks, 1))

prediction = loaded_model.predict(input_array)
predicted_time = int(prediction[0][0] * max_time * scaling_factor)

print("Predicted time using loaded model:", predicted_time, "Weeks")
