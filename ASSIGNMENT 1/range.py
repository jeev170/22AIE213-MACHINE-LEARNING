def find_range(arr):
    if len(arr) < 3:
        return "Not possible"
    return max(arr) - min(arr)


n = int(input("Enter number of elements: "))
arr = []

for i in range(n):
    arr.append(int(input("Enter element: ")))

print("Range:", find_range(arr))
