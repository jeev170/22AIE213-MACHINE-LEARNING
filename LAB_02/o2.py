#O2

import numpy as np
import pandas as pd

#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

def compute_f(v1, v2):
    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1:
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0:
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1:
            f01 += 1
        else:
            f00 += 1
    return f11, f10, f01, f00

def csc(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #preprocessing
    data = data.replace("?", 0)
    data = data.replace({"t": 1, "f": 0})
    data = data.infer_objects(copy=False)

    #random sampling
    sample20 = data.sample(n=20, random_state=10)

    #a4
    print("\nMissing Values:\n", sample20.isnull().sum())

    #a5
    binary_data = sample20.loc[:, sample20.nunique() <= 2]
    binary_data = binary_data.drop(columns=["Record ID"], errors="ignore")

    v1 = binary_data.iloc[0].values
    v2 = binary_data.iloc[1].values
    f11, f10, f01, f00 = compute_f(v1, v2)

    JC = f11 / (f11 + f10 + f01)
    SMC = (f11 + f00) / (f11 + f10 + f01 + f00)

    #a6
    numeric_data = sample20.drop(columns=["Record ID"])
    numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce").fillna(0)

    COS = csc(numeric_data.iloc[0].values, numeric_data.iloc[1].values)

    print("\nJC:", JC)
    print("SMC:", SMC)
    print("CSC:", COS)

if __name__ == "__main__":
    main()
