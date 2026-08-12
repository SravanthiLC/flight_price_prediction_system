from preprocess import load_data, get_features_and_target, create_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from xgboost import XGBRegressor

from catboost import CatBoostRegressor

from utils import save_model, update_results

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
    lr_preprocessor = create_preprocessor(X_train, scale_numeric=True)

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

    save_model(lr_model, "linear_regression.pkl")
    update_results(
        "Linear Regression",
        lr_mae,
        lr_rmse,
        lr_r2
    )

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

    save_model(best_rf_model, "random_forest.pkl")
    update_results(
        "Random Forest",
        rf_mae,
        rf_rmse,
        rf_r2
    )

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
    xgb_rmse = root_mean_squared_error(y_test, y_pred)
    xgb_r2 = r2_score(y_test, y_pred)

    save_model(best_xgb_model, "xgboost.pkl")
    update_results(
        "XGBoost",
        xgb_mae,
        xgb_rmse,
        xgb_r2
    )

    print("\nXGBoost Results:")
    print(f"Mean Absolute Error: {xgb_mae:.2f}")
    print(f"Root Mean Squared Error: {xgb_rmse:.2f}")
    print(f"R2 Score: {xgb_r2:.2f}")
    
def catboost_regression():
    """
    Train and evaluate a CatBoost Regressor using GridSearchCV to find the best hyperparameters.
    """
    print("\nCATBOOST")

    # CatBoost is a tree based algorithm, so no need to scale numerical features

    cat_preprocessor = create_preprocessor(X_train, scale_numeric = False)

    # build the pipeline
    cat_pipeline = Pipeline(
        steps = [
            ("preprocessor", cat_preprocessor),
            ("regressor", CatBoostRegressor(
                random_state = 42,
                verbose = 0
            ))
        ]
    )

    # define the hyperparameter grid

    cat_param_grid = {
        "regressor__iterations" : [200, 500],
        "regressor__learning_rate" : [0.03, 0.1],
        "regressor__depth" : [4, 6, 8]
    }

    # perform grid search
    grid_search = GridSearchCV(
        estimator=cat_pipeline,
        param_grid=cat_param_grid,
        scoring="r2",
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train, y_train)

    # best model
    best_cat_model = grid_search.best_estimator_

    print("\nBest CatBoost Parameters:")
    print(grid_search.best_params_)

    # make predictions
    y_pred = best_cat_model.predict(X_test)

    # evaluate the model
    cat_mae = mean_absolute_error(y_test, y_pred)
    cat_rmse = root_mean_squared_error(y_test, y_pred)
    cat_r2 = r2_score(y_test, y_pred)

    # save the trained model
    save_model(best_cat_model, "catboost.pkl")

    # update the results table
    update_results(
        "CatBoost",
        cat_mae,
        cat_rmse,
        cat_r2
    )

    # display results
    print("\nCatBoost Results:")
    print(f"Mean Absolute Error : {cat_mae:.2f}")
    print(f"Root Mean Squared Error : {cat_rmse:.2f}")
    print(f"R2 Score : {cat_r2:.2f}")

    # CatBoost Results:
    # Mean Absolute Error : 1292.83
    # Root Mean Squared Error : 3324.41
    # R2 Score : 0.68

def catboost_regression_without_encoding():
    print("\nCATBOOST (Without Encoding)")

    # copy data
    X_train_cat = X_train.copy()
    X_test_cat = X_test.copy()

    # create route feature
    X_train_cat["route"] = (
        X_train_cat["source_city"].astype(str)
        + "_"
        + X_train_cat["destination_city"].astype(str)
    )

    X_test_cat["route"] = (
        X_test_cat["source_city"].astype(str)
        + "_"
        + X_test_cat["destination_city"].astype(str)
    )

    # categorical columns
    categorical_features = [
        "airline",
        "source_city",
        "destination_city",
        "route",
        "travel_class",
        "season",
        "day_of_week",
        "aircraft_type"
    ]

    # convert categorical columns to string
    for col in categorical_features:
        X_train_cat[col] = X_train_cat[col].astype(str)
        X_test_cat[col] = X_test_cat[col].astype(str)

    # get column indices for CatBoost
    cat_feature_indices = [
        X_train_cat.columns.get_loc(col)
        for col in categorical_features
    ]

    # CatBoost model with early stopping
    model = CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        loss_function="RMSE",
        random_state=42,
        eval_metric="R2",
        verbose=100
    )

    # train model
    model.fit(
        X_train_cat,
        y_train,
        cat_features=cat_feature_indices,
        eval_set=(X_test_cat, y_test),
        use_best_model=True,
        early_stopping_rounds=100
    )

    print("\nBest iteration:", model.get_best_iteration())

    # predictions
    y_pred = model.predict(X_test_cat)

    # evaluation
    cat_mae = mean_absolute_error(y_test, y_pred)
    cat_rmse = root_mean_squared_error(y_test, y_pred)
    cat_r2 = r2_score(y_test, y_pred)

    print("\nCatBoost Results:")
    print(f"Mean Absolute Error: {cat_mae:.2f}")
    print(f"Root Mean Squared Error: {cat_rmse:.2f}")
    print(f"R2 Score : {cat_r2:.2f}")

    # save the trained model
    save_model(model, "catboost_without_encoding.pkl")

    # update results table
    update_results(
        "CatBoost (without encoding + route)",
        cat_mae, cat_rmse, cat_r2
    )

    # CatBoost Results:
    # Mean Absolute Error: 1296.84
    # Root Mean Squared Error: 3336.20
    # R2 Score : 0.68

if __name__ == "__main__":
    linear_regression()
    random_forest()
    xgboost_regression()
    catboost_regression()
    catboost_regression_without_encoding()

