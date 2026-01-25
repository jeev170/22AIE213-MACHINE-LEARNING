#O1

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#load data
def load_purchase_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    payment = data["Payment (Rs)"].values
    return X, payment

def cost_vector(X, y):
    pseudo_inverse = np.linalg.pinv(X)
    cost = np.matmul(pseudo_inverse, y.reshape(-1, 1))
    return cost

def generate_labels(payment):
    return np.array([1 if p > 200 else 0 for p in payment])

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    X, payment = load_purchase_data(file_path)

    #a3
    full_cost = cost_vector(X, payment)
    full_rank = np.linalg.matrix_rank(X)

    #square matrix 1
    X1 = X[:3]
    y1 = payment[:3]
    cost1 = cost_vector(X1, y1)
    rank1 = np.linalg.matrix_rank(X1)

    #square matrix 2
    X2 = X[3:6]
    y2 = payment[3:6]
    cost2 = cost_vector(X2, y2)
    rank2 = np.linalg.matrix_rank(X2)

    #a2
    y_labels = generate_labels(payment)
    clf = LogisticRegression()
    clf.fit(X, y_labels)

    pred_full = clf.predict(X)
    pred_sq1 = clf.predict(X1)
    pred_sq2 = clf.predict(X2)

    print("Rank (Full Data):", full_rank)
    print("Rank (Square Matrix 1):", rank1)
    print("Rank (Square Matrix 2):", rank2)

    print("\nCost Vector (Full Data):\n", full_cost)
    print("\nCost Vector (Square Matrix 1):\n", cost1)
    print("\nCost Vector (Square Matrix 2):\n", cost2)

    print("\nClassifier Predictions (Full):", pred_full)
    print("Classifier Predictions (Square 1):", pred_sq1)
    print("Classifier Predictions (Square 2):", pred_sq2)

if __name__ == "__main__":
    main()
