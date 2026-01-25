#A6

import numpy as np
import pandas as pd

#load data
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="thyroid0387_UCI")

#function for csc
def csc(A, B):
    dot_prod = np.sum(A * B)
    norm_A = np.sqrt(np.sum(A ** 2))
    norm_B = np.sqrt(np.sum(B ** 2))
    return dot_prod / (norm_A * norm_B)

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    #fill missing values and convert t/f to 1/0
    data.replace("?", 0, inplace=True)
    data.replace({"t": 1, "f": 0}, inplace=True)

    #no need for record id for csc
    numeric_data = data.drop(columns=["Record ID"])

    #convert to numeric
    numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce").fillna(0)

    #taking only first two observation vectors
    A = numeric_data.iloc[0].values
    B = numeric_data.iloc[1].values

    #csc
    cos_sim = csc(A, B)

    print("Cosine Similarity between first two observations is ", cos_sim)


if __name__ == "__main__":
    main()
