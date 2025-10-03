import pandas as pd
from load_data import load_bankruptcy_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

#Keep: X86, X4, X6 , X16, X19, X20, X26, X75, X25, X26, X37, X91, X64
#Delete: X1,X2,X3, X5,X8, X89, X7, X10, X17, X18, X21, X43, X42, X27, X9, X73, X23, X31, X38, X40, X77, X66, X78, X22, X32, X61
columns_to_drop = [' ROA(C) before interest and depreciation before interest',' ROA(A) before interest and % after tax',' ROA(B) before interest and depreciation after tax',' Realized Sales Gross Margin',' After-tax net Interest Rate',' Gross Profit to Sales',' Pre-tax net Interest Rate',' Continuous interest rate (after tax)',' Net Value Per Share (A)',' Net Value Per Share (C)',' Revenue Per Share (Yuan ¥)',' Net profit before tax/Paid-in capital',' Operating profit/Paid-in capital',' Regular Net Profit Growth Rate',' Non-industry income and expenditure/revenue',' Working capitcal Turnover Rate',' Per Share Net profit before tax (Yuan ¥)',' Total Asset Return Growth Rate Ratio',' Net worth/Assets',' Borrowing dependency',' Current Liability to Liability',' Current Liabilities/Equity',' Current Liability to Equity',  ' Cash Flow to Sales', ' Operating Profit Per Share (Yuan ¥)', ' Cash Reinvestment %', ' Operating Funds to Liability', ' Contingent liabilities/Net worth', ' Current Liability to Assets']   
rows_to_drop = [2490, 5015, 2605]
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

def __drop_rows (df: pd.DataFrame, rows_to_drop: list) -> pd.DataFrame:
    missing_rows = [row for row in rows_to_drop if row not in df.index]
    if missing_rows:
        raise ValueError (f"Row indeces {missing_rows} not in DataFrame")
    return df.drop(rows_to_drop, axis=0)

def __drop_columns (X_train, X_test, columns_to_drop: list) -> pd.DataFrame:
    missing_columns = [column for column in columns_to_drop if column not in X_train or column not in X_test]
    if missing_columns:
        raise ValueError (f"Columns not known: {missing_columns}")
    return X_train.drop(columns=columns_to_drop), X_test.drop(columns=columns_to_drop)

def __remove_outliers_IQR (df: pd.DataFrame, factor: float = 1.5) -> pd.DataFrame:
    for column in df.select_dtypes(include=['number']).columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - factor * IQR
        
        

def __smote_oversampling (X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def get_processed_data () -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_bankruptcy_data()
    shortened_df = __drop_rows(df=df,rows_to_drop=rows_to_drop)
    X_train, X_test, y_train, y_test = __train_test_split(shortened_df)
    X_train_reduced, X_test_reduced = __drop_columns (X_train, X_test, columns_to_drop)
    X_train_reduced_smote, y_train_smote = __smote_oversampling (X_train_reduced, y_train)
    return X_train_reduced, X_test_reduced, y_train, y_test, X_train_reduced_smote, y_train_smote, shortened_df


if __name__ == "__main__":
    df = load_bankruptcy_data()
    shortened_df = __drop_rows(df=df,rows_to_drop=rows_to_drop)
    X_train, X_test, y_train, y_test = __train_test_split(load_bankruptcy_data())
    X_train_reduced, X_test_reduced = __drop_columns (X_train, X_test, columns_to_drop)