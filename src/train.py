from preprocess import load_data, get_features_and_target, create_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

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
def linear_regression():
    print("\nLINEAR REGRESSION")
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

    # Linear Regression Results:
    # Mean Absolute Error : 1715.02
    # Root Mean Squared Error : 3743.75
    # R2 Score : 0.59


def random_forest():
    print("\nRANDOM FOREST REGRESSOR")
    # create the preprocessing pipeline
    rf_preprocessor = create_preprocessor(X_train, scale_numeric = False)

    # build the pipeline
    rf_pipeline = Pipeline(
        steps = [
            ("preprocessor", rf_preprocessor),
            ("regressor", RandomForestRegressor(random_state = 42))
        ]
    )

    # using GridSearchCV to train multiple models by using different combinations of parameters
    rf_param_grid = {
        "regressor__n_estimators" : [100, 200, 300],
        "regressor__max_depth" : [10, 15, 20],
        "regressor__min_samples_split" : [2, 5, 10],
        "regressor__min_samples_leaf" : [1, 2],
    }

    # perform grid search
    rf_grid_search = GridSearchCV(
        estimator = rf_pipeline,
        param_grid = rf_param_grid,
        scoring = "r2",
        cv = 3,
        n_jobs = -1,
        verbose = 1
    )

    # train all parameter combinations
    rf_grid_search.fit(X_train, y_train)

    # best model
    best_rf_model = rf_grid_search.best_estimator_

    print("\nBest Random Forest Parameters: ")
    print(rf_grid_search.best_params_)

    # make predictions
    y_pred = best_rf_model.predict(X_test)

    # evaluate the model

    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_rmse = root_mean_squared_error(y_test, y_pred)
    rf_r2 = r2_score(y_test, y_pred)

    # display results

    print("\nRandom Forest Results:")
    print(f"Mean Absolute Error : {rf_mae:.2f}")
    print(f"Root Mean Squared Error : {rf_rmse:.2f}")
    print(f"R2 Score : {rf_r2:.2f}")

    # Random Forest Results:
    # Mean Absolute Error : 1404.39
    # Root Mean Squared Error : 3404.88
    # R2 Score : 0.66

def xgboost_regression():
    """
    Train and evaluate an XGBoost Regressor using GridSearchCV to find the best hyperparameters.
    """

    print("\nXGBOOST")

    # preprocessing pipeline
    # xgboost is a tree based algorithm, so it does not need numerical scaling
    xgb_preprocessor = create_preprocessor(X_train, scale_numeric = False)

    xgb_pipeline = Pipeline(
        steps = [
            ("preprocessor", xgb_preprocessor),
            ("regressor", XGBRegressor(
                objective = "reg:squarederror",
                random_state = 42,
                n_jobs = -1
            ))
        ]
    )

    # hyperparameter grid
    xgb_param_grid  = {
        "regressor__n_estimators" : [100, 200],
        "regressor__learning_rate" : [0.05, 0.1],
        "regressor__max_depth" : [3, 5],
        "regressor__subsample" : [0.8, 1.0],
        "regressor__colsample_bytree" : [0.8, 1.0]
    }

    # perform grid search
    xgb_grid_search = GridSearchCV(
        estimator = xgb_pipeline,
        param_grid = xgb_param_grid,
        scoring = "r2",
        cv = 3,
        n_jobs = -1,
        verbose = 1
    )

    xgb_grid_search.fit(X_train, y_train)

    # best model
    best_xgb_model = xgb_grid_search.best_estimator_

    print("\nBest XGBoost parameters: ")
    print(xgb_grid_search.best_params_)

    # make predictions
    y_pred = best_xgb_model.predict(X_test)

    # evaluate the model
    xgb_mae = mean_absolute_error(y_test, y_pred)
    print(f"Mean Absolute Error: {xgb_mae:.2f}")
    xgb_rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Root Mean Squared Error: {xgb_rmse:.2f}")
    xgb_r2 = r2_score(y_test, y_pred)
    print(f"R2 Sore: {xgb_r2:.2f}")
    

# linear_regression()
# random_forest()
xgboost_regression()

