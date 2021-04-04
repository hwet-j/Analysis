# 보스턴 집 값from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import boston_housing

# print(boston_housing.load_data())
(x_train, y_train), (x_test, y_test) = boston_housing.load_data()
print(x_train[:2], x_train.shape)   # (404, 13)
print(x_test[:2], x_test.shape)     # (102, 13)
print(y_train[:2], y_train.shape)   # (404,)

# 속성                내용
# CRIM      범죄율
# ZN        25,000평방피트당 주거지역의 비율
# INDUS     비소매 상업지구 비율(단위: 에이커)
# CHAS      찰스강에 인접해 있으면 1, 그렇지 않으면 0
# NOX       일산화질소 농도(단위:0.1ppm)
# RM        주택당 방의 수
# AGE       1940년 이전에 건설된 주택의 비율
# DIS       5개의 보스턴 직업고용센터와의 거리(가중 평균)
# RAD       고속도로 접근성
# TAX       재산세율
# PTRATIO   학생/교사비율
# B         흑인비율
# LSTAT     하위계층비율
# MEDV      타운의 주택 가격 중앙값(단위: 1,000달러)

from sklearn.preprocessing import MinMaxScaler, minmax_scale, StandardScaler
# 표준화 : (요소값 - 평균) / 표준편차
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)
print(x_train[:2])

def build_model():
    model = Sequential()
    model.add(Dense(64, activation='linear', input_shape = (x_train.shape[1], )))
    model.add(Dense(32, activation='linear'))
    model.add(Dense(1, activation='linear'))
    
    model.compile(loss='mse', optimizer = 'adam', metrics=['mse'])
    return model

model = build_model()
print(model.summary())

# 연습1 : train / test로 학습. validation dataset X
# history = model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0)
# mse_history = history.history['mse']    # loss, mse중 mse만 봄
# print('mse_history : ',mse_history)

# 연습2 : train / test로 학습. validation dataset O
history = model.fit(x_train, y_train, epochs=50, batch_size=10, verbose=0,\
                    validation_split = 0.3)
mse_history = history.history['mse']    # # loss, mse, val_loss, val_mse중 mse만 봄
print('mse_history : ',mse_history)
val_history = history.history['val_mse']    # # loss, mse, val_loss, val_mse중 val_mse만 봄
print('val_history : ',val_history)

plt.plot(mse_history, 'r')
plt.plot(val_history, 'b--')
plt.xlabel('epoch')
plt.ylabel('mse, val_mse')
plt.show()

from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_test, model.predict(x_test)))    # 설명력 :  0.735884





