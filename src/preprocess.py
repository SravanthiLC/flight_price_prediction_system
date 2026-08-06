import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COLUMN = "price_inr"

DROP_COLUMNS = [
    "price_inr",
    "journey_date",
    "departure_time",
    "arrival_time",
    "route"
]

def load_data(file_path = "data/cleaned_dataset.csv"):
    """
    Load the cleaned dataset.
    """
    return pd.read_csv(file_path)

def get_features_and_target(df):
    """
    Separate input features (X) and target (y).
    X : all columns used for prediction
    y : price_inr
    """
    X = df.drop(columns = DROP_COLUMNS)
    y = df[TARGET_COLUMN]

    return X, y

def create_preprocessor(X):
    """
    Create a peprocessing pipeline.
    Treat categorical and numerical features differently.
    """

    categorical_features = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numerical_features = X.select_dtypes(exclude=["object", "string"]).columns.tolist()

    # one hot encoder for categorical columns
    categorical_transformer = OneHotEncoder(handle_unknown = "ignore")

    # standard scaler for numerical columns
    numerical_transformer = StandardScaler()

    # apply transformations
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_transformer,
                categorical_features,
            ),
            (
                "numerical",
                numerical_transformer,
                numerical_features,
            ),
        ]
    )

    return preprocessor