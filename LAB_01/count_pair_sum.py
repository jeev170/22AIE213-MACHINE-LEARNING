#q1

#function that counts pairs with required sum
def count_pair_sum(arr, r_sum):
    count = 0
    for i in range(len(arr)):
        for j in range(i + 1, len(arr)):
            if arr[i] + arr[j] == r_sum:
                count += 1
    return count

#predefined input
arr = [2, 7, 4, 1, 3, 6]
r_sum = 10

#output
print("Number of pairs:", count_pair_sum(arr, r_sum))
