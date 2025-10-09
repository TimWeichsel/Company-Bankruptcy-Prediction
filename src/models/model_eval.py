import os
import pandas as pd
import joblib
from src.data.process_data import get_processed_data
from sklearn.metrics import roc_auc_score

folder = "src/models/best_models"

def evaluate_models(data_method: str = "correlation_adjusted", eval_mehod: str = "roc_auc") -> None:
    models = {}
    folder = "src/models/best_models"
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for file in files:
        file_path = os.path.join(folder, file)
        models[file] = joblib.load(file_path)
        
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data(data_method)
    match eval_mehod:
        case "roc_auc":
            for name, model in models.items():
                y_pred = model.predict_proba(X_test)
                auc_score = roc_auc_score(y_test, y_pred[:, 1])
        case _:
            raise ValueError("Evaluation method not known")