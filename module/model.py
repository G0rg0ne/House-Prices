
import pandas as pd
import numpy as np

from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import optuna


def get_data(use_external_features=False):
    df_train = pd.read_csv('./data/train.csv')
    df_test = pd.read_csv('./data/test.csv')
    import pdb;pdb.set_trace()
    columns_to_remove = [""]
    category_cols = [""]
    df_train = df_train.drop(columns=columns_to_remove, errors="ignore")
    df_test = df_test.drop(columns=columns_to_remove, errors="ignore")
    for cat in category_cols:
        df_train[cat] = df_train[cat].astype("category")
        df_test[cat] = df_test[cat].astype("category")
    X = df_train.drop(columns=["SalePrice", "SalePrice"])
    y = df_train["SalePrice"]
    X_test = df_test
    return X,y,X_test
def xgboost_experimentation(use_external_features=False):
    # Objective function for Optuna to optimize
    def objective(trial):
        # Define the hyperparameter search space
        param = {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'n_estimators': trial.suggest_int('n_estimators', 100, 500),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 0.5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'alpha': trial.suggest_float('alpha', 0, 10),
            'lambda': trial.suggest_float('lambda', 0, 10),
        }
        # Create the model with the current set of hyperparameters
        model = XGBRegressor(**param, enable_categorical=True)

        # Train the model
        model.fit(X_train, y_train)

        # Predict on the test set
        y_pred = model.predict(X_test)

        # Calculate RMSE (Root Mean Squared Error)
        rmse = root_mean_squared_error(y_test, y_pred)
        return rmse

    X, y = get_data(use_external_features)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a study and optimize
    study = optuna.create_study(direction='minimize')  # Minimize RMSE
    study.optimize(objective, n_trials=10)

    # Best parameters
    best_params = study.best_params
    print("Best parameters:", best_params)

    # Train the final model with the best parameters
    best_model = XGBRegressor(**best_params, enable_categorical=True)
    best_model.fit(X, y)

    # Evaluate the model
    y_pred = best_model.predict(X_test)
    rmse = root_mean_squared_error(y_test, y_pred)
    print(f"Final RMSE test set: {rmse}")

    y_pred = best_model.predict(X)
    rmse = root_mean_squared_error(y, y_pred)
    print(f"Final RMSE all : {rmse}")

    X_submission["tow"] = best_model.predict(X_submission.drop(columns="flight_id"))
    X_submission[["flight_id", "tow"]].astype(int).to_csv("team_youthful_xerox_v5_4203b437-6356-43fc-9430-7eb126bbeb55.csv", index=False)

    ax = plot_importance(best_model)
    ax.figure.tight_layout()
    ax.figure.savefig('xgboost_feat_importance.png')

if __name__ == "__main__":
    xgboost_experimentation()