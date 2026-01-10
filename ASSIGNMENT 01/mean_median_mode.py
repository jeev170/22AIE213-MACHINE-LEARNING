#q5

import random
import numpy as np

#separate functions for generation of random number, mean , median and mode
def generate_numbers():
    return [random.randint(1, 10) for _ in range(25)]

def find_mean(nums):
    return np.mean(nums)

def find_median(nums):
    return np.median(nums)

def find_mode(nums):
    return max(nums, key=nums.count)

#generating numbers
nums = generate_numbers()

#output
print(nums)
print("Mean:", find_mean(nums))
print("Median:", find_median(nums))
print("Mode:", find_mode(nums))
