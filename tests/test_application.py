import importlib
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from flask import Flask


class RecordingScaler:
    def __init__(self):
        self.frames = []

    def transform(self, frame):
        self.frames.append(frame.copy())
        return frame.to_numpy(dtype=float)


class RecordingModel:
    def __init__(self):
        self.inputs = []

    def predict(self, values):
        self.inputs.append(values.copy())
        return [12.345]


# The route contract is tested independently of the serialized model artifacts.
# The artifacts are loaded by application.py at import time, so replace only
# pickle.load while importing the module.
recording_model = RecordingModel()
recording_scaler = RecordingScaler()
with patch("pickle.load", side_effect=[recording_model, recording_scaler]):
    application_module = importlib.import_module("application")


class PredictionRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        application_module.app.config.update(TESTING=True)
        cls.client = application_module.app.test_client()

    def setUp(self):
        recording_model.inputs.clear()
        recording_scaler.frames.clear()
        self.valid_payload = {
            "Temperature": "31.5",
            "RH": "45",
            "Ws": "2.1",
            "Rain": "0",
            "FFMC": "85.2",
            "DMC": "14.3",
            "ISI": "5.1",
            "Classes": "1",
            "Region": "0",
        }

    def assert_rejected_without_prediction(self, payload, expected_message):
        response = self.client.post("/predictdata", data=payload)

        self.assertEqual(response.status_code, 400)
        self.assertIn(expected_message, response.get_data(as_text=True))
        self.assertEqual(recording_scaler.frames, [])
        self.assertEqual(recording_model.inputs, [])

    def test_valid_payload_preserves_training_feature_order(self):
        response = self.client.post("/predictdata", data=self.valid_payload)

        self.assertEqual(response.status_code, 200)
        self.assertIn("Predicted FWI: 12.35", response.get_data(as_text=True))
        self.assertEqual(len(recording_scaler.frames), 1)
        self.assertEqual(
            list(recording_scaler.frames[0].columns),
            list(application_module.FEATURE_COLUMNS),
        )
        self.assertEqual(
            recording_scaler.frames[0].iloc[0].tolist(),
            [31.5, 45.0, 2.1, 0.0, 85.2, 14.3, 5.1, 1.0, 0.0],
        )
        self.assertEqual(len(recording_model.inputs), 1)

    def test_missing_field_is_rejected(self):
        payload = self.valid_payload.copy()
        payload.pop("Temperature")

        self.assert_rejected_without_prediction(
            payload, "Temperature is required."
        )

    def test_blank_field_is_rejected(self):
        payload = self.valid_payload.copy()
        payload["RH"] = "   "

        self.assert_rejected_without_prediction(payload, "RH is required.")

    def test_non_numeric_field_is_rejected(self):
        payload = self.valid_payload.copy()
        payload["Ws"] = "windy"

        self.assert_rejected_without_prediction(
            payload, "Ws must be a valid number."
        )

    def test_non_finite_values_are_rejected(self):
        for feature, value in (("Rain", "nan"), ("FFMC", "inf"), ("DMC", "-inf")):
            with self.subTest(feature=feature, value=value):
                payload = self.valid_payload.copy()
                payload[feature] = value
                self.assert_rejected_without_prediction(
                    payload, f"{feature} must be a finite number."
                )


if __name__ == "__main__":
    unittest.main()
