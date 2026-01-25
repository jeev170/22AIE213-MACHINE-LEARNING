#A1

import numpy as np
import pandas as pd

#to load data and create X and y
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    y = data["Payment (Rs)"].values.reshape(-1, 1)
    return X, y

#to calculate rank
def calculate_rank(matrix):
    return np.linalg.matrix_rank(matrix)

#to calculate cost of each product
def calculate_cost(matrix, price):
    pseudo_inverse = np.linalg.pinv(matrix)
    cost = np.matmul(pseudo_inverse, price)
    return cost


def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    X, y = load_data(file_path)
        
    #Rank calculation
    rank_X = calculate_rank(X)
    
    #Cost estimation
    cost = calculate_cost(X, y)

    #output
    print("Rank:", rank_X)    
    print("Candies:", cost[0][0])
    print("Mangoes:", cost[1][0])
    print("Milk:", cost[2][0])


if __name__ == "__main__":
    main()
