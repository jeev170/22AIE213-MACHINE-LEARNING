#A5

import numpy as np
import pandas as pd


#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

#function to compute values of f
def compute_f(v1, v2):
    #setting all to 0 first
    f11 = f10 = f01 = f00 = 0
    for i in range(len(v1)):
        if v1[i] == 1 and v2[i] == 1: #both are true
            f11 += 1
        elif v1[i] == 1 and v2[i] == 0: #1st is true
            f10 += 1
        elif v1[i] == 0 and v2[i] == 1: #2nd is true
            f01 += 1
        else:
            f00 += 1 #none are true
    return f11, f10, f01, f00

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #converting binary values t/f to 1/0
    binary_data = data.replace({"t": 1, "f": 0})

    #selecting only binary columns
    binary_columns = binary_data.select_dtypes(include=[np.number])

    #taking only two observation vectors
    v1 = binary_columns.iloc[0].values
    v2 = binary_columns.iloc[1].values

    #compute f-values
    f11, f10, f01, f00 = compute_f(v1, v2)

    #jc
    JC = f11 / (f01 + f10 + f11)

    #smc
    SMC = (f11 + f00) / (f00 + f01 + f10 + f11)

    #output
    print("f11:", f11)
    print("f10:", f10)
    print("f01:", f01)
    print("f00:", f00)

    print("Jaccard Coefficient:", JC)
    print("Simple Matching Coefficient:", SMC)
    #smc>jc


if __name__ == "__main__":
    main()
