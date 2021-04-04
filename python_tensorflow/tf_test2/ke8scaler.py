# 정규화/표준화 : 데이터 간에 단위에 차이가 큰 경우
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler, RobustScaler
from tensorflow.python.keras.layers.core import Activation

# 1    StandardScaler    기본 스케일. 평균과 표준편차 사용. 이상치가 있으면 좋지 않음
# 2    MinMaxScaler    최대/최소값이 각각 1, 0이 되도록 스케일링
# 3    MaxAbsScaler    최대절대값과 0이 각각 1, 0이 되도록 스케일링
# 4    RobustScaler    중앙값(median)과 IQR(interquartile range) 사용. 아웃라이어의 영향을 최소화

data = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/Advertising.csv")
print(data.head(3))
del data['no']
print(data.head(3))

# 정규화
xy = minmax_scale(data, axis = 0, copy = True)
print(xy[:2])

# train / test : 과적합 방지
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xy[:, 0:-1], xy[:, -1], \
                                                    test_size = 0.3, random_state=123)
print(x_train[:2], x_train.shape)   # tv, radio, newspaper
print(x_test[:2], x_test.shape)
print(y_train[:2], y_train.shape)   # sales

model = Sequential()
# model.add(Dense(1, input_dim = 3))  # 레이어 1개
# model.add(Activation('linear'))
model.add(Dense(1, input_dim = 3, activation = 'linear'))

model.add(Dense(20, input_dim = 3, activation='linear'))
model.add(Dense(10, activation='linear'))
model.add(Dense(1, activation='linear'))

model.summary()
tf.keras.utils.plot_model(model, 'abc.png')

model.compile(optimizer=Adam(0.01), loss='mse', metrics=['mse'])
history = model.fit(x_train, y_train, epochs=100, batch_size=32, verbose=1,\
          validation_split=0.2) # train data를 8:2로 분리해서 학습 도중 검증도함

# print('history : ', history.history)
loss = model.evaluate(x_test, y_test)
print('loss :', loss)

pred = model.predict(x_test)
print('real : ', y_test[:3])
print('pred : ', pred[:3].flatten())
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_test, pred))



