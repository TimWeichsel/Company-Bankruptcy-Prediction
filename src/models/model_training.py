import numpy as np
import joblib
import os
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from xgboost import XGBClassifier
from src.data.process_data import get_processed_data

pipe = Pipeline([("model", LogisticRegression())]) #Logistic Regression as placeholder
    
param_grid_log_reg = [
    { #Logistic Regression
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": np.logspace(-3, 3, 7),
    }]
param_grid_rf = [
    { #Random Forest
        "model": [RandomForestClassifier()],
        "model__n_estimators": [50, 100, 200, 500],
        "model__max_depth": [10, 20, 30]
    }]
param_grid_lda = [
    { #Linear Discriminant Analysis
        "model": [LinearDiscriminantAnalysis()],
        "model__solver": ["svd", "lsqr", "eigen"],
    }]

param_grid_xgb = [
    { #XGBoost
        "model": [XGBClassifier()],
        
        
    }
]

grid_log_reg = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_log_reg,
    cv=5,
    scoring='roc_auc',
    n_jobs=4
)

grid_rf = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_rf,
    cv=5,
    scoring='roc_auc',
    n_jobs=4
)

grid_lda = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_lda,
    cv=5,
    scoring='roc_auc',
    n_jobs=4
)

grids = {"log_reg":grid_log_reg, "rf":grid_rf, "lda":grid_lda}

def run_models (data_method: str = "correlation_adjusted", skip_model_grids: str = None) -> None:
    base_dir = os.path.join(os.path.dirname(__file__))
    model_dir = os.path.join(base_dir, "best_models")
    param_dir = os.path.join(base_dir, "best_models_params")
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data(data_method)
    for name, grid in grids.items():
        if skip_model_grids is not None and name in skip_model_grids:
            continue
        grid.fit(X_train_smote, y_train_smote)
        print("all results: ", grid.cv_results_)
        best_params = grid.best_params_
        best_score = grid.best_score_
        print("best params: ", best_params)
        print("best score: ", best_score)
        model = grid.best_estimator_
        model_file_dir = os.path.join(model_dir, f"best_{name}_{data_method}.pkl")
        param_file_dir = os.path.join(param_dir, f"best_{name}_params_{data_method}.pkl")
        joblib.dump(model, model_file_dir)
        joblib.dump(best_params, param_file_dir)
    

if __name__ == "__main__":
    #skip_model_grids = ["log_reg","rf","lda"]
    run_models(data_method="BS_PnL")
