# 문제) 21세 이상의 피마 인디언 여성의 당뇨병 발병 여부에 대한 dataset을 이용하여 당뇨 판정을 위한 분류 모델을 작성한다.
# 당뇨 판정 칼럼은 outcome 이다.   1이면 당뇨 환자로 판정
# train / test 분류 실시
# 모델 작성은 Sequential API, Function API 두 가지를 사용한다.
# loss에 대한 시각화도 실시한다.
# 출력결과는 Django framework를 사용하시오.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split


data = pd.read_csv('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv')
print(data.head(3))
print(data.info())
dataset = data.values
print(dataset)
x = dataset[:, 0:8]
y = dataset[:, -1]
print(x[0])
print(y[0])

# 과적합 방지 train / test 분류 
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape)   # (530, 8) (228, 8) (530,)

# Model     Sequential API
print('Sequentail API ----------------')
model = Sequential()
model.add(Dense(30, input_dim=8, activation='relu'))
model.add(tf.keras.layers.BatchNormalization()) # 배치 정규화. 그레디언트 손실과 폭주 문제
print(model.summary)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(x_train, y_train, epochs = 100, verbose=0)





# Model     function API
print('function API -------------')
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

inputs = Input(shape=(8,))
outputs = Dense(1, activation='sigmoid')(inputs)
model2 = Model(inputs, outputs)

model2.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model2.fit(x, y, epochs = 500, batch_size = 1, verbose = 0)
meval2 = model2.evaluate(x, y)
print(meval2)

model2 = Model(inputs, outputs)
print(model2.summary())
