import pingouin as pg
import pandas as pd
import matplotlib.pyplot as plt
from src.data.process_data import get_processed_data
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_curve, roc_auc_score


def test_cov_matrices (X: pd.DataFrame, y: pd.Series) -> None:
    df = X.copy()
    df["Bankrupt?"] = y
    print(pg.box_m(df,dvs=X.columns.tolist(), group="Bankrupt?"))
    
def lda_analysis_svd (X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    lda = LinearDiscriminantAnalysis(solver='svd')
    X_lda = lda.fit_transform(X_train, y_train)
    y_pred = lda.predict_proba(X_test)
    y_pred_pos = y_pred[:, 1]
    auc = roc_auc_score(y_test, y_pred_pos)
    fpr, tpr, threshold = roc_curve(y_test, y_pred_pos)
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.show()
    
    
    

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data()
    lda_analysis_svd (X_train_smote, y_train_smote, X_test, y_test)