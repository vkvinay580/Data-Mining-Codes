import pandas as pd
import seaborn as sns
import numpy as np
import scipy.stats as stats
from matplotlib import pyplot as plt


print("Part 01")

missing_values = ["-", " "]
df = pd.read_csv('book.csv', na_values=missing_values)
print("The original df: ", df)
print("1- Detecting Duplicates:")
print("Number of duplicates: ", df.duplicated().sum())

print("2- Removing Duplicates:")
df.drop_duplicates(inplace=True)
print("Duplicates removed.")

print("3- Detecting Missing Fields:")
print("Number of missing fields: ", df.isnull().sum())

print("4- Replacing the missing & incorrect values:")
df.loc[0, 'price'] = 3500
df.loc[1, 'title'] = '12 Rules of life'
df.loc[2, 'price'] = 0
df.loc[3, 'price'] = 4000
print("The new df: ", df)


df.to_csv("new_book.csv")
_csv("new_book.csv")


print("Part 02")

alpha = 5
loc = 100.5
beta = 22
data = stats.gamma.rvs(alpha, loc=loc, scale=beta, size=10000)
print(data)

fit_alpha, fit_loc, fit_beta = stats.gamma.fit(data)
print(fit_alpha, fit_loc, fit_beta)

print(alpha, loc, beta)

x = np.linspace(0, 10, 10)
y = stats.gamma.pdf(x, a=5, scale=0.5)
plt.hist(y)
plt.show()
plt.plot(x, y, "ro-", label=(r'$\alpha=0, \beta=3$'))
plt.legend(loc='upper right')

plt.show()


print("Part 03")
df = pd.read_csv('data.csv', na_values=missing_values)
print("The original df: ", df)
print("1- Detecting Duplicates:")
print("Number of duplicates: ", df.duplicated().sum())

print("2- Removing Duplicates:")
df.drop_duplicates(inplace=True)
print("Duplicates removed.")

print("3- Detecting Missing Fields:")
print("Number of missing fields: ", df.isnull().sum())
print("4- Replacing the missing & incorrect values:")
df.dropna(subset=['Calories'], inplace=True)
print("The new df: ", df)

df.to_csv("data_cleaned.csv")



print("Part 04")
q1 = 500
q2 = 1200
df = pd.read_csv('data_cleaned.csv')
df["Category"] = ""
for row in df.index:
    if df.loc[row, "Calories"] <= q1:
        df.loc[row, "Category"] = "Few"
    elif q1 < df.loc[row, "Calories"] < q2:
        df.loc[row, "Category"] = "Normal"
    elif df.loc[row, "Calories"] > q2:
        df.loc[row, "Category"] = "High"


print("Save to data_cleaned.csv !")
df.to_csv("data_cleaned.csv")

grouped_df = df.groupby("Calories")
mean_group = grouped_df.mean()
print(mean_group)
mean_group = mean_group.reset_index()
print(mean_group)
