import pandas as pd
from load_data import load_bankruptcy_data
from sklearn.model_selection import train_test_split
#import sys
#import os
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#from config import get_seed

def __train_test_split (df: pd.DataFrame, test_size: int = 0.2, label = "Bankrupt?") -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split Dataframe to Train and Test -> Features and Labels
    
    Parameters:
        - df: Dataframe to split 
        - test_size: percentage of data to be assigned to test set
        - label: name of label column
        
    Returns:
        - X_train (train features)
        - X_test (test features)
        - y_train (train labels)
        - y_test (test labels)
    '''
    X = df.drop("Bankrupt?", axis=1)
    y = df["Bankrupt?"].squeeze()
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=test_size, 
    stratify=y,
    random_state=42)
    
    return X_train, X_test, y_train, y_test

    
    
def __reduce_to_columns (df, keep_columns: list) -> pd.DataFrame:
    missing_columns = [column for column in keep_columns if column not in df]
    if missing_columns:
        raise ValueError (f"Columns not known: {missing_columns}")
    return df[keep_columns]


def process_data (df) -> pd.DataFrame:
    pass


if __name__ == "__main__":
    __train_test_split(load_bankruptcy_data())
    #Keep: X86, X4, X6, X8 (X6 und X8 nocht stark korreliert!), X16, X19, X20, X26
    #Delete: X1,X2,X3, X5, X89, X7, X10, X17, X18, X21, X43, X42, X27