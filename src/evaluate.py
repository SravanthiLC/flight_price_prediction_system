import pandas as pd
from pathlib import Path

# Path to the results file
RESULTS_PATH = Path("results/model_results.csv")

def display_results():
    """
    Load the saved model evaluation results and display a
    comparison table sorted by R² Score.
    """

    # Check whether the results file exists
    if not RESULTS_PATH.exists():
        print("No model results found.")
        print("Train at least one model first.")
        return

    # Load the results
    results_df = pd.read_csv(RESULTS_PATH)

    # Keep only the expected columns
    expected_columns = ["Model", "MAE", "RMSE", "R2 Score"]
    results_df = results_df[expected_columns]

    # Remove completely empty rows
    results_df = results_df.dropna(how="all")

    # Round values for cleaner display
    results_df["MAE"] = results_df["MAE"].round(2)
    results_df["RMSE"] = results_df["RMSE"].round(2)
    results_df["R2 Score"] = results_df["R2 Score"].round(2)

    # Sort by R² Score (higher is better)
    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    ).reset_index(drop=True)

    # Display the comparison table
    print("Flight Price Prediction - Model Comparison")
    print(results_df.to_string(index=False))

    # Display the best model
    best_model = results_df.iloc[0]

    print("\nBest Performing Model")
    print(f"Model    : {best_model['Model']}")
    print(f"MAE      : {best_model['MAE']:.2f}")
    print(f"RMSE     : {best_model['RMSE']:.2f}")
    print(f"R2 Score : {best_model['R2 Score']:.2f}")

if __name__ == "__main__":
    display_results()