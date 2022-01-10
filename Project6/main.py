import numpy as np
from numpy import random
from scipy import stats


print("Exercise 1: \n")
arr = np.random.randint(20, size=(2, 3))
print(arr)
print("Sum of row-wise : ", np.sum(arr, axis=1))
arr2 = np.sum(arr, axis=1)
print(arr2)


print("Exercise 2: \n")
arr = np.array([[1, 55, 22, 4], [np.nan, np.nan, 99, 9]])
print(arr)
x = np.argwhere(np.isnan(arr))
print("Positions of missing values are: ", x)
x = np.nan_to_num(arr)
print("Replacing the missing values with zeros: ", x)

x = arr[~np.isnan(arr)]
print("Removing the missing values: ", x)

x = arr[~np.isnan(arr).any(axis = 1),:]
print("Removing rows with missing values: ", x)



print("Exercise 3: \n")
arr = np.random.randint(20, size=(2, 4))
print(arr)
(x, y) = np.unique(arr, return_counts=True)
frequencies = np.asarray((x, y)).T

print("Unique values count: ", frequencies)


print("Exercise 4: \n")
arr = np.random.randint(20, size=(1, 9))
print(arr)
mean = np.mean(arr)
median = np.median(arr)

modus = stats.mode(arr)

print("Mean = ", mean,
      "\n Median =  ", median,
      "\n Moving average = ",
      "\n Modus = ", modus)


