# setosa + versicolor, verginica로 분리해 구분 결정간격 시각화
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = datasets.load_iris()
print(iris.data[:3])
print(iris.keys())
print(iris.target)

x = iris['data'][:, 3:]  # petal width으로 실습
print(x[:5])
y = (iris['target'] == 2).astype(np.int)
print(y[:5])

print()
log_reg = LogisticRegression().fit(x, y)
print(log_reg)

x_new = np.linspace(0, 3, 1000).reshape(-1, 1)
print(x_new.shape)  # (1000, 1)
y_proba = log_reg.predict_proba(x_new)
print(y_proba)

import matplotlib.pyplot as plt
plt.plot(x_new, y_proba[:, 1], 'r-', label='verginica')
plt.plot(x_new, y_proba[:, 0], 'b--', label='setosa + versicolor')
plt.xlabel('petal width')
plt.legend()
plt.show()

print(log_reg.predict([[1.5],[1.7]]))  # [0 1]
print(log_reg.predict([[2.5],[0.7]]))

# softmax 함수가 각 클래스의 예측 확률을 찾는 데 사용되었음을 알 수 있다.
print(log_reg.predict_proba([[2.5],[0.7]]))  # [[0.02563061 0.97436939] [0.98465572 0.01534428]]

