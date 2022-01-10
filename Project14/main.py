import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pandas.plotting import parallel_coordinates
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn import metrics
from sklearn.metrics import make_scorer, accuracy_score, precision_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import cross_val_score

df = pd.read_csv('penguins.csv')
df

df.shape

df.info()

df.describe()

df.isnull().sum()

df.dropna(subset=['sex'], inplace=True)

median = df['bill_length_mm'].median()
df['bill_length_mm'].fillna(median, inplace=True)

median = df['bill_depth_mm'].median()
df['bill_depth_mm'].fillna(median, inplace=True)

median = df['flipper_length_mm'].median()
df['flipper_length_mm'].fillna(median, inplace=True)

median = df['body_mass_g'].median()
df['body_mass_g'].fillna(median, inplace=True)

df.isnull().sum()

print(df.duplicated)

df.drop_duplicates(inplace=True)

df.describe()

df.reset_index(inplace=True, drop=True)

label_encoder = LabelEncoder()
column_encoded = label_encoder.fit_transform(df['sex'])
column_reshaped = column_encoded.reshape(len(column_encoded), 1)

onehot_encoder = OneHotEncoder(sparse=False, handle_unknown='error', drop='first', categories='auto')
column_onehot_encoded = onehot_encoder.fit_transform(column_reshaped)

ohe_df = pd.DataFrame(column_onehot_encoded)
df = df.rename({0: "sex"}, axis=1)
df.drop(columns=['sex'], inplace=True)
df = df.join(ohe_df)

df.head()

islands_dict = {'Biscoe': 1, 'Dream': 2, 'Torgersen': 3}

df['island_ordinal'] = df.island.map(islands_dict).astype('int64')
df.drop(columns=['island'], inplace=True)
df

df = df.rename({"island_ordinal": "island"}, axis=1)
df

pd.crosstab(index=df["species"], columns="count")

n_bins = 10
fig, axs = plt.subplots(3, 2, figsize=(10, 10))

axs[0, 0].hist(df.iloc[:, 1].values, bins=n_bins, rwidth=0.95)
axs[0, 0].set_title('bill_length_mm')

axs[0, 1].hist(df.iloc[:, 2].values, bins=n_bins, rwidth=0.95)
axs[0, 1].set_title('bill_depth_mm')

axs[1, 0].hist(df.iloc[:, 3].values, bins=n_bins, rwidth=0.95)
axs[1, 0].set_title('flipper_length_mm')

axs[1, 1].hist(df.iloc[:, 4].values, bins=n_bins, rwidth=0.95)
axs[1, 1].set_title('body_mass_g')

axs[2, 0].hist(df.iloc[:, 5].values, bins=n_bins, rwidth=0.95)
axs[2, 0].set_title('sex')

axs[2, 1].hist(df.iloc[:, 6].values, bins=n_bins, rwidth=0.95)
axs[2, 1].set_title('island')

# Adding some white spacing between subplots
fig.tight_layout(pad=2)

x = df.iloc[:, 1:].values
y = df.iloc[:, 0].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

gaussian = GaussianNB()
gaussian.fit(x_train, y_train)
Y_pred = gaussian.predict(x_test)
accuracy_nb = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
acc_gaussian = round(gaussian.score(x_train, y_train) * 100, 2)

cm = metrics.confusion_matrix(y_test, Y_pred)
accuracy = metrics.accuracy_score(y_test, Y_pred)
precision = metrics.precision_score(y_test, Y_pred, average='micro')
recall = metrics.recall_score(y_test, Y_pred, average='micro')
f1 = metrics.f1_score(y_test, Y_pred, average='micro')
print('Confusion matrix for Naive Bayes\n', cm)
print('accuracy_Naive Bayes: %.3f' % accuracy)
print('precision_Naive Bayes: %.3f' % precision)
print('recall_Naive Bayes: %.3f' % recall)
print('f1-score_Naive Bayes : %.3f' % f1)

error = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train, y_train)
    pred_i = knn.predict(x_test)
    error.append(np.mean(pred_i != y_test))

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o', markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

k_list = list(range(1, 120, 2))

cv_scores = []
for k in k_list:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, x_train, y_train, cv=10, scoring='accuracy')
    cv_scores.append(scores.mean())

MSE = [1 - x for x in cv_scores]

plt.figure()
plt.figure(figsize=(15, 10))
plt.title('The optimal number of neighbors', fontsize=20)
plt.xlabel('Number of Neighbors K', fontsize=15)
plt.ylabel('Misclassification Error', fontsize=15)
sns.set_style("whitegrid")
plt.plot(k_list, MSE)

plt.show()

best_k = k_list[MSE.index(min(MSE))]
print("The optimal number of neighbors is %d." % best_k)

knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(x_train, y_train)
Y_pred = knn.predict(x_test)
accuracy_knn = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
acc_knn = round(knn.score(x_train, y_train) * 100, 2)

cm = metrics.confusion_matrix(y_test, Y_pred)
accuracy = metrics.accuracy_score(y_test, Y_pred)
precision = metrics.precision_score(y_test, Y_pred, average='micro')
recall = metrics.recall_score(y_test, Y_pred, average='micro')
f1 = metrics.f1_score(y_test, Y_pred, average='micro')

print('Confusion matrix for KNN\n', cm)
print('accuracy_KNN : %.3f' % accuracy)
print('precision_KNN : %.3f' % precision)
print('recall_KNN: %.3f' % recall)
print('f1-score_KNN : %.3f' % f1)

linear_svc = LinearSVC(dual=False)
linear_svc.fit(x_train, y_train)
Y_pred = linear_svc.predict(x_test)
accuracy_svc = round(metrics.accuracy_score(y_test, Y_pred) * 100, 2)
acc_linear_svc = round(linear_svc.score(x_train, y_train) * 100, 2)

cm = metrics.confusion_matrix(y_test, Y_pred)
accuracy = metrics.accuracy_score(y_test, Y_pred)
precision = metrics.precision_score(y_test, Y_pred, average='micro')
recall = metrics.recall_score(y_test, Y_pred, average='micro')
f1 = metrics.f1_score(y_test, Y_pred, average='micro')

print('Confusion matrix for SVC\n', cm)
print('accuracy_SVC: %.3f' % accuracy)
print('precision_SVC: %.3f' % precision)
print('recall_SVC: %.3f' % recall)
print('f1-score_SVC : %.3f' % f1)

results = pd.DataFrame({
    'Model': ['KNN',
              'Naive Bayes',
              'Support Vector Machine'],
    'Score': [acc_knn,
              acc_gaussian,
              acc_linear_svc],
    "Accuracy_score": [accuracy_knn,
                       accuracy_nb,
                       accuracy_svc
                       ]})
result_df = results.sort_values(by='Accuracy_score', ascending=False)
result_df = result_df.reset_index(drop=True)
result_df