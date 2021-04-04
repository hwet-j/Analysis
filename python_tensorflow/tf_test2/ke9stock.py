# 주식 데이터 회귀분석
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler


xy = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/stockdaily.csv",\
                   delimiter=',', skiprows=1)
print(xy[:2], len(xy))

# 정규화
scaler = MinMaxScaler(feature_range=(0, 1))
xy = scaler.fit_transform(xy)
print(xy[:3])

print()
x_data = xy[:, 0:-1]
y_data = xy[:, -1]
print(x_data[0], y_data[0])
print(x_data[1], y_data[1])

print()
# 하루전 데이터로 다음날 종가 예측
x_data = np.delete(x_data, -1, 0)   # 마지막 행 삭제
y_data = np.delete(y_data, 0)   # 0행 삭제

print('predict tomorrow')
print(x_data[0], '=>', y_data[0])

model=Sequential()
model.add(Dense(input_dim = 4, units=1))

model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model.fit(x_data, y_data, epochs = 100, verbose=0)

print(x_data[10])
test = x_data[10].reshape(-1, 4)
print(test)
print('실제값 : ', y_data[10], '예측값 : ', model.predict(test).flatten())

pred = model.predict(x_data)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, pred))

print('\n데이터를 분리---------------')
train_size = int(len(x_data) * 0.7)
test_size = len(x_data) - train_size
# print(train_size, test_size)    # 511 220
x_train, x_test = x_data[0:train_size], x_data[train_size:len(x_data)]
# print(x_train[:2], x_train.shape)   # (511, 4)
# print(x_test[:2], x_test.shape)     # (220, 4)
y_train, y_test = y_data[0:train_size], y_data[train_size:len(y_data)]
# print(y_train[:2], y_train.shape)   # (511,)
# print(y_test[:2], y_test.shape)     # (220,)

model2=Sequential()
model2.add(Dense(input_dim = 4, units=1))

model2.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
model2.fit(x_train, y_train, epochs = 100, verbose=0)

result = model2.evaluate(x_test, y_test)
print('result : ', result)

pred2 = model2.predict(x_test)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_test, pred2))

plt.plot(y_test, 'b')
plt.plot(pred2, 'r--')
plt.show()

# 머신러닝의 이슈는 최적화와 일반화의 줄다리기
# 최적화 : 성능 좋은 모델 생성. 과적합 발생
# 일반화 : 모델이 새로운 데이터에 대한 분류/예측을 잘함
