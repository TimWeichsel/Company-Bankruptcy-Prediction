import pandas as pd

def reduce_to_columns (df: pd.DataFrame, keep_columns: list) -> pd.DataFrame:
    missing_columns = [column for column in keep_columns if column not in df]
    if missing_columns:
        raise ValueError (f"Columns not known: {missing_columns}")
    return df[keep_columns]