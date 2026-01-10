#q3
import numpy as np

#function to find a matrix raised to a power
def matrix_power(A, m):
    #checking if m is positive
    if m <= 0:
        return "Power should be positive integer"
    return np.linalg.matrix_power(A, m)

#user input
n = int(input("Enter order of matrix: "))
A = np.array([[int(input()) for _ in range(n)] for _ in range(n)])
m = int(input("Enter power: "))

#output: A^m
result = matrix_power(A, m)
print(result)
