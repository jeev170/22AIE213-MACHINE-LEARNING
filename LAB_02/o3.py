#O3

import numpy as np
import pandas as pd

#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="marketing_campaign")

#function for csc
def csc(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #a4
    print("\nMissing Values:\n", data.isnull().sum())

    #preprocessing
    data = data.fillna(0)

    #a5
    binary_data = data.loc[:, data.nunique() <= 2]
    binary_data = binary_data.drop(columns=["ID"], errors="ignore")
    v1 = binary_data.iloc[0].values
    v2 = binary_data.iloc[1].values

    f11 = np.sum((v1 == 1) & (v2 == 1))
    f10 = np.sum((v1 == 1) & (v2 == 0))
    f01 = np.sum((v1 == 0) & (v2 == 1))
    f00 = np.sum((v1 == 0) & (v2 == 0))

    JC = f11 / (f11 + f10 + f01)
    SMC = (f11 + f00) / (f11 + f10 + f01 + f00)

    #a6
    numeric_data = data.select_dtypes(include=[np.number])
    COS = csc(numeric_data.iloc[0].values, numeric_data.iloc[1].values)

    print("\nJC:", JC)
    print("SMC:", SMC)
    print("CSC:", COS)

if __name__ == "__main__":
    main()
