import pandas as pd
import os

def load_bankruptcy_data():
    '''
    Load the bankruptcy dataset from the local CSV file.
    
    Returns:
        pd.DataFrame: bankruptcy dataset
    '''
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "src/data")
    df = pd.read_csv(data_path + "/bankruptcy_data.csv")
    return df
    
    
if __name__ == "__main__":
    print(load_bankruptcy_data().head())
    