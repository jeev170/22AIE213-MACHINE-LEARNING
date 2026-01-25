#A7

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

#function for jc
def jc(f11, f10, f01):
    if (f11 + f10 + f01) == 0:
        return 0
    return f11 / (f11 + f10 + f01)

#function for smc
def smc(f11, f10, f01, f00):
    total = f11 + f10 + f01 + f00
    if total == 0:
        return 0
    return (f11 + f00) / total

#function for csc
def csc(A, B):
    dot_prod = np.sum(A * B)
    norm_A = np.sqrt(np.sum(A ** 2))
    norm_B = np.sqrt(np.sum(B ** 2))
    return dot_prod / (norm_A * norm_B)

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    # fill missing values and convert t/f to 1/0
    data.replace("?", 0, inplace=True)
    data.replace({"t": 1, "f": 0}, inplace=True)

    first20 = data.iloc[:20]

    # automatically select binary attributes only
    binary_data = first20.loc[:, first20.nunique() <= 2]


    n = len(binary_data)

    JC_matrix = np.zeros((n, n))
    SMC_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            v1 = binary_data.iloc[i].values
            v2 = binary_data.iloc[j].values
            f11, f10, f01, f00 = compute_f(v1, v2)
            JC_matrix[i][j] = jc(f11, f10, f01)
            SMC_matrix[i][j] = smc(f11, f10, f01, f00)

    #for csc
    numeric_data = first20.drop(columns=["Record ID"])
    numeric_data = numeric_data.apply(pd.to_numeric, errors="coerce").fillna(0)

    CSC_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            A = numeric_data.iloc[i].values
            B = numeric_data.iloc[j].values
            CSC_matrix[i][j] = csc(A, B)

    #calculating heatmaps
    plt.figure(figsize=(10, 8))
    sns.heatmap(JC_matrix, annot=True, cmap="coolwarm")
    plt.title("Jaccard Coefficient Heatmap")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(SMC_matrix, annot=True, cmap="coolwarm")
    plt.title("Simple Matching Coefficient Heatmap")
    plt.show()

    plt.figure(figsize=(10, 8))
    sns.heatmap(CSC_matrix, annot=True, cmap="coolwarm")
    plt.title("Cosine Similarity Heatmap")
    plt.show()


if __name__ == "__main__":
    main()
