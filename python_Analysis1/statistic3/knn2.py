# KNN - 최근접 이웃 알고리즘 
from sklearn.neighbors import KNeighborsClassifier

kmodel = KNeighborsClassifier(n_neighbors = 3, weights = 'distance')
train = [
    [5, 3, 2],
    [1, 3, 5],
    [4, 5, 7],
]

label = [0, 1, 1]

import matplotlib.pyplot as plt
plt.plot(train, 'o')
plt.xlim([-1, 5])
plt.ylim([0, 10])
plt.show()

kmodel.fit(train, label)
pred = kmodel.predict(train)
print('pred : ', pred)
print('acc : ', kmodel.score(train, label))

new_data = [[1,2,8],[6,4,1]]
new_pred = kmodel.predict(new_data)
print('new_pred : ', new_pred)