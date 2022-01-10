import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

shopping = pd.read_csv("shopping-data.csv")
x = shopping.iloc[:, [2, 3, 4]].values
shopping.info()
shopping[0:10]

from sklearn.preprocessing import LabelEncoder

stringCol = shopping.iloc[:, 1]
encoder = LabelEncoder()
encoder.fit(stringCol)
encoder.transform(stringCol)

shopping["Genre"].replace(to_replace=shopping["Genre"].tolist(),
                          value=encoder.transform(stringCol),
                          inplace=True)
shopping.head()

iris_outcome = pd.crosstab(index=shopping["Genre"], columns="count")
iris_outcome

sns.FacetGrid(shopping, hue="Genre", height=3).map(sns.distplot, "Age").add_legend()
sns.FacetGrid(shopping, hue="Genre", height=3).map(sns.distplot, "Annual Income (k$)").add_legend()
sns.FacetGrid(shopping, hue="Genre", height=3).map(sns.distplot, "Spending Score (1-100)").add_legend()
plt.show()

sns.set_style("whitegrid")
sns.pairplot(shopping, hue="Genre", height=3);
plt.show()

from sklearn.preprocessing import LabelEncoder

floatCol = shopping.iloc[:, 1]
encoder = LabelEncoder()
encoder.fit(floatCol)
encoder.transform(floatCol)

print(x)

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++',
                    max_iter=300, n_init=10, random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

from yellowbrick.cluster import KElbowVisualizer

model = KMeans()
visualizer = KElbowVisualizer(model, k=(1, 10))

visualizer.fit(x)
visualizer.show()

from yellowbrick.cluster import KElbowVisualizer

model = KMeans()

visualizer = KElbowVisualizer(model, k=(2, 10), metric='silhouette', timings=True)

visualizer.fit(x)
visualizer.show()

from yellowbrick.cluster import SilhouetteVisualizer

fig, ax = plt.subplots(2, 2, figsize=(15, 8))
for i in [2, 3, 4, 5]:
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=42)
    q, mod = divmod(i, 2)

    visualizer = SilhouetteVisualizer(km, colors='yellowbrick', ax=ax[q - 1][mod])
    visualizer.fit(x)

kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

plt.legend()

kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans == 0, 0], x[y_kmeans == 0, 1], s=100, c='purple', label='Cluster 1')
plt.scatter(x[y_kmeans == 1, 0], x[y_kmeans == 1, 1], s=100, c='orange', label='Cluster 2')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='green', label='Cluster 3')
plt.scatter(x[y_kmeans == 2, 0], x[y_kmeans == 2, 1], s=100, c='yellow', label='Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=100, c='red', label='Centroids')

plt.legend()