import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.preprocessing.preprocess import load_and_preprocess_regression


def run_tuned_regression():
    X_train_val, X_test, y_train_val, y_test, preprocessor = load_and_preprocess_regression()

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    param_grid = {
        'regressor__n_estimators': [50, 100],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }

    print("Running Hyperparameter Optimization for Random Forest...")
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X_train_val, y_train_val)

    print("\nHyperparameter Optimization Results:")
    print(f"Best Parameters: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)

    mae_test = mean_absolute_error(y_test, y_pred)
    mse_test = mean_squared_error(y_test, y_pred)
    r2_test = r2_score(y_test, y_pred)

    print(f"\nFinal Test Set Performance (Tuned):")
    print(f"Test MAE:  {mae_test:.4f}")
    print(f"Test MSE:  {mse_test:.4f}")
    print(f"Test RMSE: {np.sqrt(mse_test):.4f}")
    print(f"Test R2:   {r2_test:.4f}")

    return best_model


if __name__ == "__main__":
    run_tuned_regression()