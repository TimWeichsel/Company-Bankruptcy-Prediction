import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from src.data.process_data import get_processed_data


pipe = Pipeline([("model", LogisticRegression())]) #Logistic Regression as placeholder
    

param_grid_list = [
    { #Logistic Regression
        "model": [LogisticRegression(max_iter=1000)],
        "model__C": np.logspace(-3, 3, 7),
    },
    { #Random Forest
        "model": [RandomForestClassifier()],
        "model__n_estimators": [50, 100, 200, 500],
        "model__max_depth": [10, 20, 30]
    },
] 

grid = GridSearchCV(
    estimator=pipe,
    param_grid=param_grid_list,
    cv=5,
    scoring='roc_auc',
    n_jobs=4
)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data()
    grid.fit(X_train_smote, y_train_smote)
    print("all results: ", grid.cv_results_)
    print("best estimator: ", grid.best_estimator_)
    print("best params: ", grid.best_params_)
    print("best score: ", grid.best_score_)