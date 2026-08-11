
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

    # Sort by R² Score (higher is better)
    results_df = results_df.sort_values(
        by="R2 Score",
        ascending=False
    )

    # Reset row numbers after sorting
    results_df = results_df.reset_index(drop=True)

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