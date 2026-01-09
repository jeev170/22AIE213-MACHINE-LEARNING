def count_sum_with_pairs(arr, sum):
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == sum:
                count = count + 1
    return count


n = int(input("Enter number of elements: "))
arr = []

for i in range(n):
    arr.append(int(input("Enter element: ")))

sum = int(input("Enter target sum: "))
print("Number of pairs:", count_pairs(arr, sum))

