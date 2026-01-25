#a9

import numpy as np
import pandas as pd

#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

#function for min-max normalisation
def min_max_normalize(column):
    return (column - column.min()) / (column.max() - column.min())

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #replace missing data types
    data = data.replace("?", np.nan)
    data = data.infer_objects(copy=False)

    #converting numeric columns
    numeric_cols = ["age", "TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    #applying min-max normalisation as we are scaling to a fixed range(0,1)
    normalized_data = data.copy()
    for col in numeric_cols:
        normalized_data[col] = min_max_normalize(data[col])

    #export normalized data
    output_file = r"C:\Users\jeevi\Downloads\ML_LAB2\thyroid_normalized.xlsx"
    normalized_data.to_excel(output_file, index=False)

    print("\nNormalized data exported successfully.")
    print(output_file)


if __name__ == "__main__":
    main()
