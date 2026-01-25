#A3

import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt

#load data first
def load_data(file_path):
    return pd.read_excel(file_path, sheet_name="IRCTC Stock Price")

#custom function for mean
def own_mean(values):
    total = 0
    for v in values:
        total += v
    return total / len(values)

#custom function for variance
def own_variance(values):
    mean_val = own_mean(values)
    total = 0
    for v in values:
        total += (v - mean_val) ** 2
    return total / len(values)

#function for time
def avg_time(func, values):
    times = []
    for _ in range(10):
        start = time.time()
        func(values)
        end = time.time()
        times.append(end - start)
    return sum(times) / 10


def main():
    file_path = r"C:\Users\jeevi\Downloads\ML_LAB2\Lab Session Data.xlsx"
    data = load_data(file_path)

    price = data.iloc[:, 3].values    
    day = data.iloc[:, 2]             
    month = data.iloc[:, 1]           
    change = data.iloc[:, 8].values 

    #python np mean and variance
    np_mean = np.mean(price)
    np_var = np.var(price)

    #custom mean and variance
    man_mean = own_mean(price)
    man_var = own_variance(price)

    #time comparison
    numpy_time = avg_time(np.mean, price)
    manual_time = avg_time(own_mean, price)

    #Wednesday mean
    wed_price = price[day == "Wed"]
    wed_mean = own_mean(wed_price)

    #April mean
    april_price = price[month == "Apr"]
    april_mean = own_mean(april_price)

    #probailities calculation

    #probability of loss of stock
    loss_prob = len(list(filter(lambda x: x < 0, change))) / len(change)

    #probability of making profit on wed
    profit_on_wed = len([i for i in range(len(change)) 
                        if day.iloc[i] == "Wed" and change[i] > 0]) / len(change)

    #conditional probability of profit given wed
    wed_change = change[day == "Wed"]
    cond_prob = len([x for x in wed_change if x > 0]) / len(wed_change)


    #output
    print("Population Mean:", np_mean)
    print("Own Mean:", man_mean)

    print("Population Variance:", np_var)
    print("Own Variance:", man_var)

    print("Average Time using Python numpy:", numpy_time)
    print("Average Time using Custom Functions:", manual_time)

    print("Wednesday Mean:", wed_mean)
    print("April Mean:", april_mean)

    print("Probability of Loss:", loss_prob)
    print("Probability of Profit on Wednesday:", profit_on_wed)
    print("Conditional Probability of Profit on Wednesday:", cond_prob)

    #scatter plot
    plt.scatter(day, change)
    plt.xlabel("Day")
    plt.ylabel("Chg%")
    plt.title("Chg% vs Day of Week")
    plt.show()


if __name__ == "__main__":
    main()
