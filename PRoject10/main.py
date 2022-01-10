
iris = pd.read_csv("iris.csv")
x = iris.iloc[:, [0, 1, 2, 3]].values
iris.info()
iris[0:10]
iris_outcome = pd.crosstab(index=iris["species"], columns="count")
iris_outcome
sns.FacetGrid(iris,hue="species",height=3).map(sns.distplot,"petal_length").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.distplot,"petal_width").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.distplot,"sepal_length").add_legend()
sns.FacetGrid(iris,hue="species",height=3).map(sns.distplot,"sepal_width").add_legend()
plt.show()
sns.set_style("whitegrid")
sns.pairplot(iris, hue="species", height=3);
plt.show()
from scipy.cluster.hierarchy import dendrogram, linkage

X = iris.loc[:,["sepal_length","sepal_width","petal_length","petal_width"]]

dist_sin = linkage(X, method="single")
plt.figure(figsize=(18,6))
dendrogram(dist_sin, leaf_rotation=90)
plt.xlabel('Index')
plt.ylabel('Distance')
plt.suptitle("DENDROGRAM",fontsize=18)

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=2, affinity='euclidean',
                                  linkage='single')
cluster.fit_predict(X)
print(cluster.labels_)
data = X.iloc[:, 0:2].values
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()

data = X.iloc[:, 2:4].values
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()

from sklearn.cluster import AgglomerativeClustering

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean',
                                  linkage='single')
cluster.fit_predict(X)
print(cluster.labels_)



data = X.iloc[:, 0:2].values
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()

data = X.iloc[:, 2:4].values
plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow')
plt.show()



from scipy.cluster.hierarchy import fcluster
iris_HAC = iris.copy()

iris_HAC['K=2'] = fcluster(dist_sin, 2, criterion='maxclust')
iris_HAC['K=3'] = fcluster(dist_sin, 3, criterion='maxclust')
iris_HAC.head()


plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering Single Method - Petal",fontsize=18)

plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_HAC, hue="K=2")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_HAC, hue="K=3")

plt.subplot(1,3,3)
plt.title("Species",fontsize=14)
sns.scatterplot(x="petal_length",y="petal_width", data=iris_HAC, hue="species")



plt.figure(figsize=(24,4))

plt.suptitle("Hierarchical Clustering Single Method - Sepal",fontsize=18)

plt.subplot(1,3,1)
plt.title("K = 2",fontsize=14)
sns.scatterplot(x="sepal_length",y="sepal_width", data=iris_HAC, hue="K=2")

plt.subplot(1,3,2)
plt.title("K = 3",fontsize=14)
sns.scatterplot(x="sepal_length",y="sepal_width", data=iris_HAC, hue="K=3")

plt.subplot(1,3,3)
plt.title("Species",fontsize=14)
sns.scatterplot(x="sepal_length",y="sepal_width", data=iris_HAC, hue="species")
