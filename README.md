# Flight Price Prediction System

A machine learning project that predicts flight ticket prices based on historical flight data and travel details such as airline, route, departure and arrival time, travel class, booking window, and other flight characteristics.

## Problem Statement

Flight ticket prices vary dynamically based on multiple factors, making it difficult for travelers to estimate whether a fare is reasonable. This project builds a regression-based machine learning system to predict airfare prices using historical flight data.

## Features

- Data cleaning and preprocessing
- Feature engineering from date and time columns
- Categorical feature handling
- Multiple regression model comparison
- Hyperparameter tuning
- Model evaluation using MAE, RMSE, and R² score
- Model saving and result tracking

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- CatBoost
- Joblib

## Project Structure

```text
flight_price_prediction_system/
│
├── data/
│   └── cleaned_dataset.csv
│
├── models/
│   ├── linear_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── catboost.pkl
│   └── catboost_without_encoding.pkl
│
├── results/
│   └── model_results.csv
│
├── src/
│   ├── data_processing.py
│   ├── train.py
│   ├── utils.py
│   ├── test_preprocess.py
│   ├── preprocess.py
│   └── evaluate.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd flight_price_prediction_system
```

Create and activate a virtual environment:

```bash
python3 -m venv myenv
source myenv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Dataset

The dataset contains historical flight information with features such as:

- Airline
- Source city
- Destination city
- Journey date
- Departure and arrival time
- Duration
- Number of stops
- Travel class
- Booking window
- Season
- Day of week
- Holiday indicator
- Aircraft type
- Distance
- Airline rating

**Target Variable:** `price_inr`

## Models Trained

- Linear Regression
- Random Forest Regressor
- XGBoost Regressor
- CatBoost Regressor
- CatBoost Regressor (native categorical handling)

## Model Performance

| Model | MAE | RMSE | R² Score |
|------|------:|------:|------:|
| CatBoost | 1292.83 | 3324.41 | **0.68** |
| CatBoost (without encoding + route) | 1296.84 | 3336.20 | 0.68 |
| XGBoost | 1311.14 | 3334.72 | 0.68 |
| Random Forest | 1404.88 | 3405.46 | 0.66 |
| Linear Regression | 1715.02 | 3743.75 | 0.59 |

The CatBoost model achieved the best overall performance with an R² score of **0.68**.

## How to Run

Launch the Streamlit application:

```bash
streamlit run app.py
```

The app will open in your browser and allow users to enter flight details and get a predicted airfare.

### Train the Models

To retrain all models:

```bash
python3 src/train.py
```

### Evaluate Model Performance

To compare model performance:

```bash
python3 src/evaluate.py
```

## Key Learnings

- Gradient boosting models significantly outperformed linear regression.
- CatBoost performed best for this dataset due to its handling of categorical features.
- Additional feature engineering and native categorical handling provided only marginal improvements, indicating that the available features were likely the primary performance constraint.

## Future Improvements

- Incorporate real-time flight pricing data
- Add demand-based features
- Deploy the model using Streamlit or Flask
- Build a REST API for price prediction