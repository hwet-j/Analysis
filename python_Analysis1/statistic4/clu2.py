# 계층적 클러스터링 : iris
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head(3))
print(iris_df.loc[0:4, ['sepal length (cm)',  'sepal width (cm)']])

from scipy.spatial.distance import pdist, squareform
# dist_vec = pdist(iris_df.loc[0:4, ['sepal length (cm)',  'sepal width (cm)']], metric = 'euclidean')
dist_vec = pdist(iris_df.loc[:, ['sepal length (cm)',  'sepal width (cm)']], metric = 'euclidean')
print(dist_vec)
row_dist = pd.DataFrame(squareform(dist_vec))
print('row_dist : \n',row_dist)

print()
from scipy.cluster.hierarchy import linkage, dendrogram
row_clusters = linkage(dist_vec, method = 'complete')
print('row_clusters : ', row_clusters)

df = pd.DataFrame(row_clusters, columns = ['id1','id2','dist','count'])
print(df)

row_dend = dendrogram(row_clusters,)
plt.ylabel('dist test')
plt.show()

from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean',linkage = 'ward')
x = iris_df.loc[0:4, ['sepal length (cm)',  'sepal width (cm)']]
labels = ac.fit_predict(x)
print('클러스터 결과 : ', labels)

plt.hist(labels)
plt.grid(True)
plt.show()


