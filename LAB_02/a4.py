#A4

import numpy as np
import pandas as pd

#function to load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)
    
    #replace ? with NaN so that pandas can infer data types
    data.replace("?", np.nan, inplace=True)

    #include numeric columns so panda don't treat them as object types
    numeric_columns = ["TSH", "T3", "TT4", "T4U", "FTI", "TBG"]
    for col in numeric_columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")


    numeric_data = data.select_dtypes(include=[np.number])

    #max-min
    print("\nRanges of Numeric Data:\n")
    for col in numeric_data.columns:
        print(col,
              "Minimum:", numeric_data[col].min(),
              "Maximum:", numeric_data[col].max())

    #mean and variance
    print("\nMean and Variance of Numeric Attributes:\n")
    for col in numeric_data.columns:
        print(col,
              "\n Mean:", numeric_data[col].mean(),
              "\n Variance:", numeric_data[col].var(),
              "\n")


if __name__ == "__main__":
    main()
