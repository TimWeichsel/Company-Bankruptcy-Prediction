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

def __drop_columns (X_train, X_test, columns_to_drop: list) -> pd.DataFrame:
    missing_columns = [column for column in columns_to_drop if column not in X_train or column not in X_test]
    if missing_columns:
        raise ValueError (f"Columns not known: {missing_columns}")
    return X_train.drop(columns=columns_to_drop), X_test.drop(columns=columns_to_drop)


def get_processed_data () -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    X_train, X_test, y_train, y_test = __train_test_split(load_bankruptcy_data())
    #Keep: X86, X4, X6 , X16, X19, X20, X26, X75, X22, X25, X26, X37, X91, X64
    #Delete: X1,X2,X3, X5,X8, X89, X7, X10, X17, X18, X21, X43, X42, X27, X9, X73, X23, X31, X38, X40, X77, X66, X78
    columns_to_drop = [' ROA(C) before interest and depreciation before interest',' ROA(A) before interest and % after tax',' ROA(B) before interest and depreciation after tax',' Realized Sales Gross Margin',' After-tax net Interest Rate',' Gross Profit to Sales',' Pre-tax net Interest Rate',' Continuous interest rate (after tax)',' Net Value Per Share (A)',' Net Value Per Share (C)',' Revenue Per Share (Yuan ¥)',' Net profit before tax/Paid-in capital',' Operating profit/Paid-in capital',' Regular Net Profit Growth Rate',' Non-industry income and expenditure/revenue',' Working capitcal Turnover Rate',' Per Share Net profit before tax (Yuan ¥)',' Total Asset Return Growth Rate Ratio',' Net worth/Assets',' Borrowing dependency',' Current Liability to Liability',' Current Liabilities/Equity',' Current Liability to Equity']
    X_train_reduced, X_test_reduced = __drop_columns (X_train, X_test, columns_to_drop)
    return X_train_reduced, X_test_reduced, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = __train_test_split(load_bankruptcy_data())
    #Keep: X86, X4, X6 , X16, X19, X20, X26, X75, X22, X25, X26, X37, X91, X64
    #Delete: X1,X2,X3, X5,X8, X89, X7, X10, X17, X18, X21, X43, X42, X27, X9, X73, X23, X31, X38, X40, X77, X66, X78
    columns_to_drop = [' ROA(C) before interest and depreciation before interest',' ROA(A) before interest and % after tax',' ROA(B) before interest and depreciation after tax',' Realized Sales Gross Margin',' After-tax net Interest Rate',' Gross Profit to Sales',' Pre-tax net Interest Rate',' Continuous interest rate (after tax)',' Net Value Per Share (A)',' Net Value Per Share (C)',' Revenue Per Share (Yuan ¥)',' Net profit before tax/Paid-in capital',' Operating profit/Paid-in capital',' Regular Net Profit Growth Rate',' Non-industry income and expenditure/revenue',' Working capitcal Turnover Rate',' Per Share Net profit before tax (Yuan ¥)',' Total Asset Return Growth Rate Ratio',' Net worth/Assets',' Borrowing dependency',' Current Liability to Liability',' Current Liabilities/Equity',' Current Liability to Equity']
    X_train_reduced, X_test_reduced = __drop_columns (X_train, X_test, columns_to_drop)
    print(X_train.shape)
    print(X_train_reduced.shape)
    print(X_test.shape)
    print(X_test_reduced.shape)
    