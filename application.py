import numpy as np
import pandas as pd
import pickle
from flask import Flask, render_template, request

# Load model and scaler
ridge_model = pickle.load(open('models/ridge.pkl', 'rb'))
standard_scaler = pickle.load(open('models/scaler.pkl', 'rb'))

# Initialize Flask app
application = Flask(__name__)
app = application

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            # Retrieve input values
            Temperature = float(request.form.get('Temperature', 0))
            RH = float(request.form.get('RH', 0))
            Ws = float(request.form.get('Ws', 0))
            Rain = float(request.form.get('Rain', 0))
            FFMC = float(request.form.get('FFMC', 0))
            DMC = float(request.form.get('DMC', 0))
            ISI = float(request.form.get('ISI', 0))
            Classes = float(request.form.get('Classes', 0))
            Region = float(request.form.get('Region', 0))
        except ValueError:
            return render_template('home.html', error="Please enter valid numeric values.")

        # Create DataFrame with matching feature names (dropping extra columns)
        model_columns = ['Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'ISI', 'Classes', 'Region']
        input_data = pd.DataFrame([[Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region]], columns=model_columns)

        # Scale and predict
        new_data_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(new_data_scaled)

        # Debugging print to check the result
        print("Prediction Result:", result[0])

        # Pass result to the template
        return render_template('home.html', results=round(result[0], 2))

    return render_template('home.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0')
