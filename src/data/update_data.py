import kagglehub
import pandas as pd
import os

def update_bankruptcy_data():
    '''
    Update and save the bankruptcy dataset from Kaggle: Bankruptcy data from the Taiwan Economic Journal for the years 1999–2009
    '''
    # Download latest version
    path = kagglehub.dataset_download("fedesoriano/company-bankruptcy-prediction")

    print("Path to dataset files:", path)
    
    df = pd.read_csv(os.path.join(path, "data.csv"))
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, "src/data")
    df.to_csv(f"{data_path}/bankruptcy_data.csv", index=False)
    
if __name__ == "__main__":
    update_bankruptcy_data()