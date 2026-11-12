import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

# Load model and scaler
with open("models/ridge.pkl", "rb") as model_file:
    ridge_model = pickle.load(model_file)
with open("models/scaler.pkl", "rb") as scaler_file:
    standard_scaler = pickle.load(scaler_file)

FEATURE_COLUMNS = (
    "Temperature",
    "RH",
    "Ws",
    "Rain",
    "FFMC",
    "DMC",
    "ISI",
    "Classes",
    "Region",
)


def build_prediction_frame(form):
    """Validate the submitted form and return features in training-column order."""
    values = {}

    for feature in FEATURE_COLUMNS:
        raw_value = form.get(feature)
        if raw_value is None or not str(raw_value).strip():
            raise ValueError(f"{feature} is required.")

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            raise ValueError(f"{feature} must be a valid number.") from None

        if not np.isfinite(value):
            raise ValueError(f"{feature} must be a finite number.")

        values[feature] = value

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


# Initialize Flask app
application = Flask(__name__)
app = application


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "POST":
        try:
            input_data = build_prediction_frame(request.form)
        except ValueError as exc:
            return render_template("home.html", error=str(exc)), 400

        new_data_scaled = standard_scaler.transform(input_data)
        result = ridge_model.predict(new_data_scaled)

        return render_template("home.html", results=round(float(result[0]), 2))

    return render_template("home.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0")
