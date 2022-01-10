import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
import scipy.stats as stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


print("Part 1")
df = pd.read_csv('data_cleaned.csv')
print(df.describe())
print(df.info)
print("Check if there is still empty cells:")
print(df.isnull().sum())
print("Check if there is duplicates")
print(df.duplicated().sum())


df.plot(kind='box')
plt.show()

corr = df.corr()
print(corr)

sb.heatmap(corr, annot=True,
           xticklabels=corr.columns.values,
           yticklabels=corr.columns.values)
plt.show()

sb.jointplot(x="Calories", y="Category", data=df)
plt.show()

sb.jointplot(x="Calories", y="Duration", data=df)
plt.show()

corr = np.corrcoef(df["Calories"], df["Duration"])[0,1]
print("Correlation between Calories and Duration:",round(corr,2))
ttest, pval = stats.ttest_ind(df["Calories"], df["Duration"])
print("Independent t-test:", ttest, pval)


sb.jointplot(x="Calories", y="Pulse", data=df)
plt.show()

corr = np.corrcoef(df["Calories"], df["Pulse"])[0,1]
print("Correlation between Calories and Pulse:",round(corr,2))
ttest, pval = stats.ttest_ind(df["Calories"], df["Pulse"])
print("Independent t-test:", ttest, pval)


sb.jointplot(x="Calories", y="Maxpulse", data=df)
plt.show()

corr = np.corrcoef(df["Calories"], df["Maxpulse"])[0,1]
print("Correlation between Calories and Maxpulse:",round(corr,2))
ttest, pval = stats.ttest_ind(df["Calories"], df["Maxpulse"])
print("Independent t-test:", ttest, pval)

#Correlation - Pulse and Maxpulse
sb.jointplot(x="Pulse", y="Maxpulse", data=df)
plt.show()

corr = np.corrcoef(df["Pulse"], df["Maxpulse"])[0,1]
print("Correlation between Pulse and Maxpulse:",round(corr,2))
ttest, pval = stats.ttest_ind(df["Pulse"], df["Maxpulse"])
print("Independent t-test:", ttest, pval)




X = df['Calories'].values.reshape(-1,1)
y = df['Duration'].values.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
regressor=LinearRegression()
regressor.fit(X_train, y_train)
print(regressor.intercept_)
print(regressor.coef_)
y_pred = regressor.predict(X_test)

plt.scatter(X_test, y_test,  color='gray')
plt.plot(X_test, y_pred, color='red', linewidth=2)
plt.show()



print("Part 2")
print("Label encoding")

df = pd.read_csv("data_cleaned.csv")
print(df)

labelEncoder = LabelEncoder()
df["Category_N"] = labelEncoder.fit_transform(df["Category"])
df.to_csv("labelEncoding.csv")
print(df)

df = pd.read_csv("data_cleaned.csv")
print("One Hot encoding")
dummies = pd.get_dummies(df["Category"])
print(dummies)

merged_data = pd.concat([df, dummies], axis=1)
print(merged_data.head())
merged_data.to_csv("OneHotEncoding.csv")

data = merged_data.drop(columns='Category')
print(data.head())

print("Correlation between vars")

df = pd.read_csv("data_cleaned.csv")

onehotencoder = OneHotEncoder(sparse=False, handle_unknown='error')
onehotencoder_df = pd.DataFrame(onehotencoder.fit_transform(df[["Category"]]))
df = df.join(onehotencoder_df)
df.drop(columns=['Category'], inplace=True)


corr = df.corr()
sb.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)
plt.show()

print("Ordinal Encoding")

data = pd.read_csv("data_cleaned.csv")
dataOrdinal = pd.DataFrame(data)
category_dict = {'Few':0, 'Normal':1, 'High':2}
dataOrdinal["Category"] = dataOrdinal.Category.map(category_dict)

print(dataOrdinal)





