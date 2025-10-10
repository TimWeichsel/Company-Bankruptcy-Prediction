import os
import pandas as pd
import joblib
from src.data.process_data import get_processed_data
from sklearn.metrics import roc_curve, roc_auc_score
import matplotlib.pyplot as plt

folder = "src/models/best_models"

def evaluate_models(data_method: str = "correlation_adjusted", eval_method: str = "roc_auc") -> None:
    models = {}
    folder = "src/models/best_models"
    files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    for file in files:
        file_path = os.path.join(folder, file)
        models[file] = joblib.load(file_path)
        
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data(data_method)
    match eval_method:
        case "roc_auc":
            for name, model in models.items():
                y_pred_pos = model.predict_proba(X_test)[:, 1]
                auc_score = roc_auc_score(y_test, y_pred_pos)
                print(f"{name}: ROC-AUC = {auc_score:.3f}")
                fpr, tpr, threshold = roc_curve(y_test, y_pred_pos)
                plot_name = name.removeprefix("best_").removesuffix(f"_{data_method}.pkl").replace("_", " ")
                plt.plot(fpr, tpr, label=f"{plot_name} (AUC = {auc_score:.3f})")
            plt.plot([0, 1], [0, 1], 'k--')
            plt.xlabel("FPR")
            plt.ylabel("TPR")
            plt.legend()
            plt.show()
        case _:
            raise ValueError("Evaluation method not known")
        
if __name__ == "__main__":
    evaluate_models()