# iris dataset으로 지도/비지도 학습 - KNN, KMeans

from sklearn.datasets import load_iris

iris_dataset = load_iris()
print(iris_dataset.keys())

# 지도학습 : KNN
print(iris_dataset['data'][:3])
print(iris_dataset['feature_names'])
print(iris_dataset['target'][:3])
print(iris_dataset['target_names'])

# train / test
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(iris_dataset['data'], iris_dataset['target'],\
                                                    test_size = 0.25, random_state = 42)
print(train_x.shape, test_x.shape)

print('KNN --------------------------------')
# 지도학습 : KNN
from sklearn.neighbors import KNeighborsClassifier

knnModel = KNeighborsClassifier(n_neighbors = 3, weights = 'distance', metric='euclidean')
print(knnModel)
knnModel.fit(train_x, train_y)  # feature, label 참여

# 모델 성능 
import numpy as np
predict_label = knnModel.predict(test_x)                    
print('예측값 :', predict_label)
print('실제값 : ', test_y)
print('test acc : {:.3f}'.format(np.mean(predict_label == test_y)))

from sklearn import metrics
print('test acc : ', metrics.accuracy_score(test_y, predict_label))

print()
# 새로운 값을 분류
new_input = np.array([[6.6, 5.5, 4.4, 1.1]])
print(knnModel.predict(new_input))
print(knnModel.predict_proba(new_input))

dist, index = knnModel.kneighbors(new_input)
print(dist, index)

print()
print('KMeans------------------')
# 비지도학습 : KMeans
from sklearn.cluster import KMeans
kmeansModel = KMeans(n_clusters = 3, init = 'k-means++', random_state = 0)
kmeansModel.fit(train_x)    # feature만 참여
print(kmeansModel.labels_)

print('0 cluster : ', train_y[kmeansModel.labels_ == 0])
print('1 cluster : ', train_y[kmeansModel.labels_ == 1])
print('2 cluster : ', train_y[kmeansModel.labels_ == 2])

new_input = np.array([[6.6, 5.5, 4.4, 1.1]])
predict_cluster = kmeansModel.predict(new_input)
print(predict_cluster)

# 성능 측정
print()
predict_test_x = kmeansModel.predict(test_x)
print(predict_test_x)

np_arr = np.array(predict_test_x)
np_arr[np_arr == 0], np_arr[np_arr == 1], np_arr[np_arr == 2] = 3, 4, 5 # 임시 저장용
print(np_arr)
np_arr[np_arr == 3] = 1 # 군집 3을 1 versicolor로 변경
np_arr[np_arr == 4] = 0 # 군집 4을 0 setosa로 변경
np_arr[np_arr == 5] = 2 # 군집 5을 2 verginica로 변경
print()
print(np_arr)

print()
predict_label = np_arr.tolist()
print(predict_label)

print('test acc : {:.3f}'.format(np.mean(predict_label == test_y))) # test acc : 0.947







