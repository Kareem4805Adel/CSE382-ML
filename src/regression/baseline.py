import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from src.preprocessing.preprocess import load_and_preprocess_regression


def run_baseline_regression():
    X_train_val, X_test, y_train_val, y_test, preprocessor = load_and_preprocess_regression()

    baseline_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42, n_jobs=-1))
    ])

    print("Evaluating Baseline Random Forest (Cross-Validation)...")
    cv_results = cross_validate(
        baseline_pipeline, X_train_val, y_train_val,
        cv=3,
        scoring=['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2'],
        n_jobs=-1
    )

    mae_base = -np.mean(cv_results['test_neg_mean_absolute_error'])
    mse_base = -np.mean(cv_results['test_neg_mean_squared_error'])
    r2_base = np.mean(cv_results['test_r2'])

    print(f"Baseline MAE:  {mae_base:.4f}")
    print(f"Baseline MSE:  {mse_base:.4f}")
    print(f"Baseline RMSE: {np.sqrt(mse_base):.4f}")
    print(f"Baseline R2:   {r2_base:.4f}")


if __name__ == "__main__":
    run_baseline_regression()