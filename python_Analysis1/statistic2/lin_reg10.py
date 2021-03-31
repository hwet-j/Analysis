# 선형회귀모델을 다항회귀로 변환

import numpy as np
import matplotlib.pyplot as plt 

x = np.array([1,2,3,4,5])
y = np.array([4,2,1,3,7])
# plt.scatter(x, y)
# plt.show()

# 선형회귀모델
from sklearn.linear_model import LinearRegression
x = x[:, np.newaxis]
print(x)
model = LinearRegression().fit(x, y)
y_pred = model.predict(x)
print(y_pred)

plt.scatter(x, y)
plt.plot(x, y_pred, c="red")
plt.show()

# 비선형인 경우 다항식 특징을 추가해서 작업한다.
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=3, include_bias = False)   # degree = 열개수, include_bias = False 편향
x2 = poly.fit_transform(x)  # 특징행렬을 만듦
print(x2)

model2 = LinearRegression().fit(x2, y)
y_pred2 = model2.predict(x2)
print(y_pred2)

plt.scatter(x, y)
plt.plot(x, y_pred2, c="red")
plt.show()










