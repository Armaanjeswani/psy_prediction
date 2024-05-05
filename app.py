from flask import Flask, render_template, request, jsonify
import numpy as np
from keras.models import load_model

app = Flask(__name__)

# Load the pre-trained model
model = load_model('lstm_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        num_weeks = int(request.form['num_weeks'])
        input_data = []
        for i in range(num_weeks):
            input_value = float(request.form[f'week_{i+1}'])
            input_data.append(input_value)

        # Reshape input data for prediction
        input_array = np.reshape(np.array(input_data), (1, num_weeks, 1))

        # Predict time
        prediction = model.predict(input_array)
        max_time = 100  # Provide the maximum value used for normalization during training
        scaling_factor = 0.8  # Adjust this scaling factor as needed
        noise = np.random.normal(loc=0, scale=0.1, size=prediction.shape)  # Adding noise to the prediction
        predicted_time = int((prediction[0][0] + noise[0][0]) * max_time * scaling_factor)  # Scale the prediction back to the original range with scaling factor

        return render_template('predict.html', prediction=predicted_time)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
