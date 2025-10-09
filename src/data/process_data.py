import pandas as pd
from .load_data import load_bankruptcy_data
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import IsolationForest

#Keep: X86, X4, X6 , X16, X19, X20, X26, X75, X25, X26, X37, X91, X64
#Delete: X1,X2,X3, X5,X8, X89, X7, X10, X17, X18, X21, X43, X42, X27, X9, X73, X23, X31, X38, X40, X77, X66, X78, X22, X32, X61
columns_to_drop = [' ROA(C) before interest and depreciation before interest',' ROA(A) before interest and % after tax',' ROA(B) before interest and depreciation after tax',' Realized Sales Gross Margin',' After-tax net Interest Rate',' Gross Profit to Sales',' Pre-tax net Interest Rate',' Continuous interest rate (after tax)',' Net Value Per Share (A)',' Net Value Per Share (C)',' Revenue Per Share (Yuan ¥)',' Net profit before tax/Paid-in capital',' Operating profit/Paid-in capital',' Regular Net Profit Growth Rate',' Non-industry income and expenditure/revenue',' Working capitcal Turnover Rate',' Per Share Net profit before tax (Yuan ¥)',' Total Asset Return Growth Rate Ratio',' Net worth/Assets',' Borrowing dependency',' Current Liability to Liability',' Current Liabilities/Equity',' Current Liability to Equity',  ' Cash Flow to Sales', ' Operating Profit Per Share (Yuan ¥)', ' Cash Reinvestment %', ' Operating Funds to Liability', ' Contingent liabilities/Net worth', ' Current Liability to Assets']   
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



def __train_val_split (X_train: pd.DataFrame, y_train: pd.Series, val_size: int = 0.2) -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    '''
    Split Training Data to Train and Val Sets
    
    Parameters:
        - X_train: Dataframe to split 
        - y_train: Series of labels
        
    Returns:
        - X_train (train features)
        - X_val (validation features)
        - y_train (train labels)
        - y_val (validation labels)
    '''
    X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, 
    test_size=val_size, 
    stratify=y_train,
    random_state=42)
    
    return X_train, X_val, y_train, y_val

    
    
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

def __drop_columns (df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    missing_columns = [column for column in columns_to_drop if column not in df]
    if missing_columns:
        raise ValueError (f"Columns not known: {missing_columns}")
    return df.drop(columns=columns_to_drop)

def __remove_outliers_IQR (X_train: pd.DataFrame, y_train: pd.DataFrame, factor: float = 200, outlier_columns_threshold:int = 10, ultimate_factor = 1000) -> tuple[pd.DataFrame, pd.Series]:
    outlier_counter_row = {row:0 for row in X_train.index}
    print ("Number of Rows before outlier removal: ", X_train.shape[0])
    for column in X_train.select_dtypes(include=['number']).columns:
        Q1 = X_train[column].quantile(0.25)
        Q3 = X_train[column].quantile(0.75)
        IQR = Q3 - Q1
        if IQR == 0:
            continue
        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR
        ultimate_lower_bound = Q1 - ultimate_factor * IQR
        ultimate_upper_bound = Q3 + ultimate_factor * IQR
        outlier_counter_row = {row:(outlier_counter_row[row]+1) if (X_train.at[row, column] < lower_bound or X_train.at[row, column] > upper_bound) else outlier_counter_row[row] for row in X_train.index}
        outlier_counter_row = {row:(outlier_counter_row[row]+outlier_columns_threshold) if (X_train.at[row, column] < ultimate_lower_bound or X_train.at[row, column] > ultimate_upper_bound) else outlier_counter_row[row] for row in X_train.index}
    outlier_indices = [row for row, count in outlier_counter_row.items() if count >= outlier_columns_threshold]
    X_train = X_train.drop(outlier_indices, axis=0)
    y_train = y_train.drop(outlier_indices, axis=0)
    print ("Number of Rows after outlier removal: ", X_train.shape[0])
    return X_train, y_train

def isolationForest_outlier_removal (X_train: pd.DataFrame, y_train: pd.DataFrame, contamination: float = 0.1) -> tuple[pd.DataFrame, pd.Series]:
    print ("Number of Rows before outlier removal: ", X_train.shape[0])
    iso = IsolationForest(contamination=contamination, random_state=42)
    mask = iso.fit_predict(X_train) != -1
    X_train, y_train = X_train[mask], y_train[mask] 
    print ("Number of Rows after outlier removal: ", X_train.shape[0])
    return X_train, y_train
        

def __smote_oversampling (X_train: pd.DataFrame, y_train: pd.DataFrame) -> tuple [pd.DataFrame, pd.DataFrame]:
    smote = SMOTE(random_state=42, sampling_strategy='minority')
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def __define_df_columns (df, method: str = "correlation_adjusted") -> pd.DataFrame:
    match method:
        case "all":
            return df
        case "correlation_adjusted":
            return __drop_columns(df, columns_to_drop)
        case default:
            raise ValueError ("Mehod not known")

def get_processed_data (method: str = "correlation_adjusted") -> tuple [pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = load_bankruptcy_data()
    df = __define_df_columns (df, method)
    X_train_raw, X_test, y_train_raw, y_test = __train_test_split(df)
    X_train, y_train = isolationForest_outlier_removal (X_train_raw, y_train_raw)
    X_train_smote, y_train_smote = __smote_oversampling (X_train, y_train)
    return X_train, X_test, y_train, y_test, X_train_smote, y_train_smote, X_train_raw, y_train_raw, df


if __name__ == "__main__":
    get_processed_data ()
