# Algerian Forest Fires — Fire Weather Index Predictor

A small Flask application that estimates the numeric **Fire Weather Index (FWI)** from weather and fire-weather inputs. The web app loads a saved scikit-learn Ridge regression model and its fitted StandardScaler from the models directory.

> **Scope:** this is an educational regression demo built around a small historical dataset. It is not a wildfire detection, spread simulation, operational forecasting, emergency alert, or safety decision system.

## What is included

- A Flask form for submitting nine numeric predictors and viewing an FWI estimate.
- A saved Ridge model (models/ridge.pkl) and fitted scaler (models/scaler.pkl).
- The Algerian forest fires CSV files and notebooks used for cleaning, exploration, and model comparison.
- An AWS Elastic Beanstalk WSGI entry-point configuration in .ebextensions/python.config.

## Repository layout

~~~text
.
├── application.py                 # Flask application and prediction route
├── requirements.txt                # Python dependencies
├── models/
│   ├── ridge.pkl                   # Saved Ridge regression estimator
│   └── scaler.pkl                  # Fitted StandardScaler
├── notebooks/
│   ├── Algerian_forest_fire.ipynb  # Data cleaning and exploration
│   ├── model_training.ipynb        # Train/test split and model comparison
│   ├── Algerian_forest_fires_dataset_UPDATE.csv
│   └── Algerian_forest_fires_dataset_cleaned.csv
├── templates/
│   ├── home.html                   # Prediction form
│   └── index.html                  # Welcome page
└── .ebextensions/
    └── python.config               # Elastic Beanstalk WSGI target
~~~

The cleaned CSV currently contains 243 observations. The notebooks are the source of the preprocessing and model-evaluation workflow; the Flask app consumes the already-serialized model and scaler rather than retraining at startup.

## Model workflow

The checked-in training notebook records this workflow:

1. Read the cleaned Algerian forest fires dataset.
2. Remove the calendar fields day, month, and year.
3. Convert the Classes label to a binary value (not fire → 0; other fire labels → 1).
4. Predict FWI using the remaining columns.
5. Split the observations into training and test sets with test_size=0.25 and random_state=42.
6. Identify highly correlated predictors from the training partition using an absolute-correlation threshold of 0.85, then fit StandardScaler on the training partition only.
7. Fit and compare linear and regularized regression estimators. The notebook serializes the fitted scaler and a Ridge estimator for the Flask application.

The form currently sends these nine fields, in this order:

~~~text
Temperature, RH, Ws, Rain, FFMC, DMC, ISI, Classes, Region
~~~

Keep the training feature selection, scaler, model, and form field order aligned whenever the model is retrained. The serialized artifacts are version-sensitive scikit-learn pickles; use artifacts from a trusted source and a compatible Python/scikit-learn environment.

## Recorded notebook evaluation

The notebook's recorded single holdout-split results include:

| Estimator | MAE | R² |
| --- | ---: | ---: |
| Ridge (the estimator saved for the app) | 0.5642 | 0.9843 |
| Linear Regression (comparison only) | 0.5468 | 0.9848 |

These are the notebook's saved outputs for one 75/25 split, not an independently reproduced benchmark or a guarantee of generalization. The plain Linear Regression run has slightly better recorded metrics, but the app currently loads the Ridge artifact. Model selection has not been established through a documented repeated or nested validation study.

## Run locally

Run commands from the repository root.

### Windows PowerShell

~~~powershell
py -3.12 -m venv .venv
.\\.venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python application.py
~~~

### macOS / Linux

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python application.py
~~~

Open:

- http://127.0.0.1:5000/ for the current welcome page.
- http://127.0.0.1:5000/predictdata for the prediction form.

The development server currently binds to 0.0.0.0 in application.py. Use it only on a trusted network. The Elastic Beanstalk configuration points to application:application, but a successful hosted deployment is not asserted here.

## Current limitations and verification notes

- The dataset is small (243 rows in the cleaned file); a high score on one random split should not be treated as evidence of production reliability.
- The notebook records a single holdout evaluation. There is no checked-in automated test suite or CI workflow for this repository at present.
- The prediction handler parses numeric form values but does not enforce domain-specific ranges or missing-field checks server-side; the HTML form's required attributes are not a substitute for API-side validation.
- The app does not provide uncertainty estimates, geographic or temporal forecasting, fire-spread modelling, or a calibrated operational risk category.
- requirements.txt is not version-pinned. The saved scikit-learn artifacts may need to be regenerated if dependency versions are incompatible.
- myenv/ is a legacy Windows virtual environment already tracked in the repository. The setup above creates a fresh environment and does not use it. The accompanying .gitignore prevents new local environment files from being added, but does not untrack or delete the existing directory.

For real-world wildfire monitoring or response, use validated operational products and qualified domain expertise—not this demo.