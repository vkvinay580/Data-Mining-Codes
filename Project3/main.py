import numpy as np

temperature = np.array([['Mo', 10, 21, 17, 11],
                        ['Tu', 9, 17, 20, 13],
                        ['We', 8, 24, 20, 13],
                        ['Th', 12, 28, 25, 9],
                        ['Fr', 18, 17, 23, 22],
                        ['Sa', 12, 22, 20, 18],
                        ['Su', 13, 15, 19, 16]])

np.reshape(temperature, (7, 5))
new_arr = np.delete(temperature, 0, 1)
print(new_arr)

max_temp = np.amax(new_arr.astype(int))
min_temp = np.amin(new_arr.astype(int))
print(f"Maximum temperature of the week: {max_temp}")
print(f"Minimum temperature of the week: {min_temp}")
