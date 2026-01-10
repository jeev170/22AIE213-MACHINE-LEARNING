#q2

#function to find range (max-min)
def find_range(arr):
    #making sure length is 3+
    if len(arr) < 3:
        return "Range determination not possible"
    arr_range = max(arr) - min(arr)
    return arr_range

#user input
n = int(input("Enter number of elements: "))
arr = []

#checking for only real numbers
for i in range(n):
    while True:
        try:
            val = float(input(f"Enter a real number {i+1}: "))
            arr.append(val)
            break
        except ValueError:
            print("Invalid input")

#output
print("Range: ", find_range(arr))
