import pandas as pd
import numpy as np
import math
from src.data.load_data import load_bankruptcy_data
from src.data.process_data import __train_test_split
from src.data.process_data import get_processed_data
import matplotlib.pyplot as plt

def analyze_features(df: pd.DataFrame, method: str = "info", column: list = None) -> None:
    '''
    Prints specific analyzis
    
    Parameters:
        df: Pandas Dataframe to be analyzed
        method: specific method for analyszing the df
    
    
    '''
    if column is not None:
        df = df[column]
    
    match method:
        case "info":
            print(df.info())
        case "describe":
            print(df.describe())
        case "head":
            print(df.head())
        case "correlation_matrix":
            f = plt.figure(figsize=(10, 10))
            plt.matshow(df.corr(), fignum=f.number)
            plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=2, rotation=45)
            plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=5)
            cb = plt.colorbar()
            cb.ax.tick_params(labelsize=5)
            plt.title('Correlation Matrix', fontsize=10)
            plt.show()
        case "high_correlation_matrix":
                num_df = df.select_dtypes(include=[np.number])
                corr_matrix = num_df.corr()         
                threshold = 0.8

                mask = corr_matrix.abs() > threshold
                np.fill_diagonal(mask.values, False) 
                selected = mask.any(axis=1) 
                filtered_corr = corr_matrix.loc[selected, selected]

                fig, ax = plt.subplots(figsize=(10,10))
                im = ax.matshow(filtered_corr.abs(), vmin=0, vmax=1) 
                
                ax.set_xticks(range(filtered_corr.shape[1]))
                ax.set_yticks(range(filtered_corr.shape[0]))
                ax.set_xticklabels(filtered_corr.columns, rotation=90, fontsize=6)
                ax.set_yticklabels(filtered_corr.index, fontsize=6)

                cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cb.ax.tick_params(labelsize=6)
                ax.set_title(f'Correlation Matrix (nur Features mit |ρ| > {threshold})', pad=12, fontsize=10)
                fig.tight_layout()
                plt.show()
        case "histo":
            plt.hist(df, bins=1000, color='skyblue', edgecolor='black', alpha=0.8)
            plt.show()
        case "box":
            if column is not None:
                plt.boxplot(df, vert=True, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', color='black'),
                            medianprops=dict(color='red'),
                            whiskerprops=dict(color='black'),
                            capprops=dict(color='black'))
                plt.show()
            else:
                num_df = df.select_dtypes(include=[np.number])
                plt.figure(figsize=(max(10, len(num_df.columns) * 0.5), 6))
                plt.boxplot(num_df.values, patch_artist=True,
                            boxprops=dict(facecolor='lightblue', color='black'),
                            medianprops=dict(color='red'),
                            whiskerprops=dict(color='black'),
                            capprops=dict(color='black'),
                            flierprops=dict(marker='o', markersize=2, color='gray'))
                plt.tight_layout()
                plt.show()
        case "scatter_plot":
            if column is not None and len(column) == 3:
                column1 = column[0]
                column2 = column[1]
                label = column[2]
                df = df[[column1, column2, label]]
                print(df.columns)
                plt.scatter(df[df[label] == 0][column1], df[df[label] == 0][column2],
                            c="blue", s=5, label="No Bankruptcy")
                plt.scatter(df[df[label] == 1][column1], df[df[label] == 1][column2],
                            c="red", s=5, label="Bankruptcy")
                plt.xlabel(column1)
                plt.ylabel(column2)
                plt.show()
        case _:
            raise ValueError("Unknown method")

def analyze_label(y_train: pd.Series, method: str = "info") -> None:
    match method:
        case "histo":
            plt.hist(y_train)
            plt.show()
            
    
if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df = get_processed_data(method="BS_PnL")

    #analyze_features(df, method="scatter_plot", column=[" Debt ratio %"," Total expense/Assets", "Bankrupt?"])
    
    #print(X_train_smote.columns)
    #X_train_smote_ln = X_train_smote.apply(np.log1p)
    #analyze_features(X_train_smote, method="histo", column=' Operating Gross Margin')
    #print(X_train_smote.columns)
    #analyze_features(X_train, method="describe", column=' Operating Gross Margin')
    #analyze_features(X_train, method="box")
    #analyze_features(X_train_raw, method="box")
    
    
    
    analyze_features(X_train, method="high_correlation_matrix", column=None) 
    #analyze_label(y_train, method="histo")
    #analyze_label(y_train_smote, method="histo")