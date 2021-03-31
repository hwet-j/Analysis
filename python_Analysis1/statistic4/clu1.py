# 비지도학습은 label이 존재하지않음
# 비지도학습의 일종 : 클러스터링(Clustering)
# -계층적 군집 분석
# 응집형 : 자료 하나하나를 군집으로 간주하고, 가까운 군집끼리 연결하는 방법. 군집의 크기를 점점 늘려가는 알고리즘. 상향식
# 분리형 : 전체 자료를 하나의 큰 군집으로 간주하고, 유의미한 부부을 분리해 나가는 방법. 군집의 크기를 점점 줄여가는 알고리즘. 하향식

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family = 'malgun gothic')

np.random.seed(123)
var = ['x','y']
labels = ['점0', '점1', '점2', '점3', '점4']
x = np.random.random_sample([5,2]) * 10
df = pd.DataFrame(x, columns = var, index = labels)
print(df)

# plt.scatter(x[:, 0], x[:, 1], c='blue', marker='o')
# plt.grid(True)
# plt.show()

from scipy.spatial.distance import pdist, squareform
dist_vec = pdist(df, metric = 'euclidean')
print('dist_vec : ', dist_vec)

row_dist = pd.DataFrame(squareform(dist_vec))
print(row_dist)

print()
from scipy.cluster.hierarchy import linkage # 응집형 계층적 군집 분석
row_cluster = linkage(dist_vec, method='ward')  # complete, single, average,.....작성가능
print(row_cluster)
print()
df = pd.DataFrame(row_cluster, columns = ['클러스터1','클러스터2','거리','멤버수'])
print(df)

from scipy.cluster.hierarchy import dendrogram
row_dend = dendrogram(row_cluster, labels=labels)
plt.tight_layout()
plt.ylabel('유클리드 거리')
plt.show()

# 계층적 클러스터 분류 결과 시각화
from sklearn.cluster import AgglomerativeClustering
ac = AgglomerativeClustering(n_clusters = 3, affinity = 'euclidean',linkage = 'ward')
labels = ac.fit_predict(x)
print('결과 : ', labels)

a = labels.reshape(-1, 1)
print(a)
x1 = np.hstack([x, a])
print('x1 : ', x1)
x_0 = x1[x1[:,2] == 0, :]
x_1 = x1[x1[:,2] == 1, :]
x_2 = x1[x1[:,2] == 2, :]

plt.scatter(x_0[:, 0], x_0[:, 1])
plt.scatter(x_1[:, 0], x_1[:, 1])
plt.scatter(x_2[:, 0], x_2[:, 1])
plt.legend(['cluster0', 'cluster1', 'cluster2'])
plt.show()







