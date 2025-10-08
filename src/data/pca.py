from sklearn.decomposition import PCA
from .process_data import get_processed_data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def apply_pca(X: pd.DataFrame, n_components: 2) -> tuple[PCA, np.ndarray]:
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    return pca, X_pca
    
    
def plot_2d_pca (X: pd.DataFrame, y: pd.Series) -> None:
    pca, X_pca = apply_pca(X, 0.95)
    component_x = 1
    component_y = 2
    plt.scatter(X_pca[y==0, component_x], X_pca[y==0, component_y], c="blue", s=5, label="No Bankruptcy")
    plt.scatter(X_pca[y==1, component_x], X_pca[y==1, component_y], c="red", s=5, label="Bankruptcy")
    plt.xlabel(f"Principal Component {component_x} - Variance explained {pca.explained_variance_ratio_[component_x]:.4f}")
    plt.ylabel(f"Principal Component {component_y} - Variance explained {float(pca.explained_variance_ratio_[component_y]):.4f}")
    plt.title(f"2D PCA with a total of {pca.n_components_} components")
    plt.legend()
    plt.show()

    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data()
    plot_2d_pca (X_train, y_train)