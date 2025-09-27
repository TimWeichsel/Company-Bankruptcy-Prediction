import pandas as pd
import numpy as np
from load_data import load_bankruptcy_data
import matplotlib.pyplot as plt

def analyze_data(df: pd.DataFrame, method: str = "info", column: list = None) -> None:
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
                threshold = 0.9

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
        case _:
            raise ValueError("Unknown method")
        
            
    
if __name__ == "__main__":
    analyze_data(load_bankruptcy_data(), method="correlation_matrix", column=None) 
