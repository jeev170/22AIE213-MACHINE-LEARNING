#A2

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

#load data
def load_data(file_path):
    data = pd.read_excel(file_path, sheet_name="Purchase data")
    X = data[["Candies (#)", "Mangoes (Kg)", "Milk Packets (#)"]].values
    payment = data["Payment (Rs)"].values
    return X, payment

#generate labels based on payment
def generate_labels(payment):
    labels = []
    for value in payment:
        if value > 200:
            labels.append(1)   #rich
        else:
            labels.append(0)   #poor
    return np.array(labels)

def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"

    X, payment = load_data(file_path)
    y = generate_labels(payment)

    #train classifier model
    model = LogisticRegression()
    model.fit(X, y)

    #predictions
    predictions = model.predict(X)

    #output
    for i in range(len(predictions)):
        label = "RICH" if predictions[i] == 1 else "POOR"
        print("Customer", i + 1, "classified as", label)


if __name__ == "__main__":
    main()
