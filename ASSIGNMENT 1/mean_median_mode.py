def mean(arr):
    return sum(arr) / len(arr)


def median(arr):
    arr.sort()
    return arr[len(arr)//2]


def mode(arr):
    return max(arr, key=arr.count)


n = int(input("Enter number of elements: "))
arr = []

for i in range(n):
    arr.append(int(input("Enter element: ")))

print("Mean:", mean(arr))
print("Median:", median(arr))
print("Mode:", mode(arr))
