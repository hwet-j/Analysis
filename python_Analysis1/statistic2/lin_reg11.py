# 선형회귀모델을 다항회귀로 변환 : 다항식 추가. 특정행렬을 만듦

import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics._regression import mean_squared_error, r2_score

x = np.array([258, 270, 294, 320, 342, 368, 396, 446, 480, 586])[:, np.newaxis]
print(x)
y = np.array([236, 234, 253, 298, 314, 342, 360, 368, 391, 390])
print(y)

# 비교목적으로 일반 회귀모델 클래스와 
lr = LinearRegression()
pr = LinearRegression()
polyf = PolynomialFeatures(degree=2)
x_quad = polyf.fit_transform(x)
# print(x_quad)

lr.fit(x,y)
x_fit = np.arange(250, 600, 10)[:, np.newaxis]
y_lin_fit = lr.predict(x_fit)
print(y_lin_fit)

pr.fit(x_quad, y)
y_quad_fit = pr.predict(polyf.fit_transform(x_fit))
print(y_quad_fit)

# 시각화
plt.scatter(x, y, label="train points")
plt.plot(x_fit, y_lin_fit, label='linear fit', linestyle='--', c='red')
plt.plot(x_fit, y_quad_fit, label='quardratic fit', linestyle='-', c='blue')
plt.legend()
plt.show()

print()
# MSE(평균제곱오차)와 R2(결정계수) 값 확인
y_lin_pred = lr.predict(x)
print('y_lin_pred : ', y_lin_pred)
y_quad_pred = pr.predict(x_quad)
print('y_quad_pred : ', y_quad_pred)

print('train MSE 비교 : 선형모델은 %.3f, 다항모델은 %.3f'%(mean_squared_error(y, y_lin_pred)), mean_squared_error(y, y_quad_pred))
print('train 결정계수 비교 : 선형모델은 %.3f, 다항모델은 %.3f'%(r2_score(y, y_lin_pred), r2_score(y, y_quad_pred)))



