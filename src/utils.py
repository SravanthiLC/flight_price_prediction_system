import joblib
import pandas as pd
from pathlib import Path

# Paths

MODELS_DIR = Path("models")
RESULTS_DIR = Path("results")
RESULTS_FILE = RESULTS_DIR / "model_results.csv"

# Save trained model

def save_model(model, filename):
    """
    Save a trained model to the models folder.

    Parameters
    ----------
    model : trained model object
    filename : str
        Example: 'xgboost.pkl'
    """

    # Create models directory if it doesn't exist
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / filename

    joblib.dump(model, model_path)

    print(f"Model saved to: {model_path}")

# Update evaluation results

def update_results(model_name, mae, rmse, r2):
    """
    Add a new model result or update an existing one.

    If the model already exists in the CSV,
    only that row is replaced.
    """

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    new_result = pd.DataFrame({
        "Model": [model_name],
        "MAE": [mae],
        "RMSE": [rmse],
        "R2 Score": [r2],
    })

    # If the file already exists, update the existing row
    if RESULTS_FILE.exists():

        results_df = pd.read_csv(RESULTS_FILE)

        # Remove previous result for this model
        results_df = results_df[results_df["Model"] != model_name]

        # Add the updated result
        results_df = pd.concat(
            [results_df, new_result],
            ignore_index=True
        )

    else:
        # First model being saved
        results_df = new_result

    # Sort by R² Score
    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    results_df.to_csv(RESULTS_FILE, index=False)

    print(f"Results updated: {RESULTS_FILE}")