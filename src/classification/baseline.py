import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate
from src.preprocessing.preprocess import load_and_preprocess_classification

def run_classification_baseline():
    X_train_val, X_test, y_train_val, y_test, preprocessor = load_and_preprocess_classification()
    
    
    baseline_rf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42, n_jobs=-1))
    ])
    
    print("Evaluating Baseline Random Forest (Cross-Validation)...")
    cv_results = cross_validate(
        baseline_rf, X_train_val, y_train_val, 
        cv=5, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    
    print(f"Accuracy:  {np.mean(cv_results['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(cv_results['test_precision']):.4f}")
    print(f"Recall:    {np.mean(cv_results['test_recall']):.4f}")
    print(f"F1-Score:  {np.mean(cv_results['test_f1']):.4f}")
    print(f"ROC-AUC:   {np.mean(cv_results['test_roc_auc']):.4f}")