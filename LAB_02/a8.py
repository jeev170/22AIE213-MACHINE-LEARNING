#A8

import numpy as np
import pandas as pd

#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #replace missing values
    data.replace("?", np.nan, inplace=True)
    data = data.infer_objects(copy=False)

    #mode
    categorical_cols = ["sex", "referral source", "Condition"]

    for col in categorical_cols:
        if col in data.columns:
            mode_value = data[col].mode()[0]
            data[col] = data[col].fillna(mode_value)

    #mean
    data["age"] = data["age"].fillna(data["age"].mean())

    #median
    numeric_outlier_cols = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]

    for col in numeric_outlier_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col] = data[col].fillna(data[col].median())

    #output
    print("Missing values after imputation:\n")
    print(data.isnull().sum())

if __name__ == "__main__":
    main()
