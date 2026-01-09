def matrix_power(A, m):
    result = A
    for _ in range(m - 1):
        temp = [[0, 0], [0, 0]]
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    temp[i][j] += result[i][k] * A[k][j]
        result = temp
    return result


A = []
print("Enter 2x2 matrix:")
for i in range(2):
    row = []
    for j in range(2):
        row.append(int(input()))
    A.append(row)

m = int(input("Enter power: "))
print("Result:", matrix_power(A, m))
