import sys
from src.classification.baseline import run_classification_baseline
from src.classification.tuned_xgboost import run_tuned_xgboost
from src.classification.tuned_logistic_regression import run_tuned_logistic_regression

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a command.")
        sys.exit(1)
        
    command = " ".join(sys.argv[1:])
    
    
    if "train-task classification-model baseline" in command:
        run_classification_baseline()
    elif "tune-task classification model xgboost" in command:
        run_tuned_xgboost()
    elif "tune-task classification model logistic_regression" in command:
        run_tuned_logistic_regression()
    else:
        print(f"Command '{command}' not recognized.")