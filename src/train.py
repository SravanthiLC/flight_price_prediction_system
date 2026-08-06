from preprocess import load_data, get_features_and_target, create_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# load the cleaned dataset
df = load_data()

# separate features (X) and target (y)
X, y = get_features_and_target(df)

# split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size = 0.2,
    random_state = 42
)

# create the preprocessing pipeline
lr_preprocessor = create_preprocessor(X_train)

# build the pipeline
lr_model = Pipeline(
    steps = [
        ("preprocessor", lr_preprocessor),
        ("regressor", LinearRegression())
    ]
)

# fit the model : the model learns the relationship between the flight features and the ticket price
lr_model.fit(X_train, y_train)

# make predictions
y_pred = lr_model.predict(X_test)

# evaluate the model
lr_mae = mean_absolute_error(y_test, y_pred)
lr_rmse = root_mean_squared_error(y_test, y_pred)
lr_r2 = r2_score(y_test, y_pred)

# display results
print("Linear Regression Results:")
print(f"Mean Absolute Error : {lr_mae:.2f}")
print(f"Root Mean Squared Error : {lr_rmse:.2f}")
print(f"R2 Score : {lr_r2:.2f}")