import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.preprocessing.preprocess import load_and_preprocess_classification

def run_tuned_logistic_regression():
    
    X_train_val, X_test, y_train_val, y_test, preprocessor = load_and_preprocess_classification()
    
    
    log_reg_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    
    
    param_grid = {
        'classifier__C': [0.01, 0.1, 1.0, 10.0],
        'classifier__penalty': ['l2'],
        'classifier__solver': ['lbfgs']
    }
    
   
    print("Running Hyperparameter Optimization for Logistic Regression...")
    grid_search = GridSearchCV(
        estimator=log_reg_pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    
    grid_search.fit(X_train_val, y_train_val)
    
  
    print("\nHyperparameter Optimization Results:")
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation F1-Score: {grid_search.best_score_:.4f}")
    
    # Compute evaluation metrics[cite: 6, 7]
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_train_val)
    y_proba = best_model.predict_proba(X_train_val)[:, 1]
    
    print("\n  Classification Performance Metrics ")
    print(classification_report(y_train_val, y_pred))
    
    print(f"ROC-AUC Score: {roc_auc_score(y_train_val, y_proba):.4f}\n")
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_train_val, y_pred))
    
    return best_model

if __name__ == "__main__":
    run_tuned_logistic_regression()