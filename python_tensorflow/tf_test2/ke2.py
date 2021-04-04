# Keras 모듈로 논리회로 처리 모델 (분류)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import numpy as np

x = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0,1,1,0]) # xor

model = Sequential()
# model.add(Dense(units=5, input_dim = 2))
# model.add(Activation('relu'))
# model.add(Dense(units=5))
# model.add(Activation('relu'))
# model.add(Dense(units=1))
# model.add(Activation('sigmoid'))

model.add(Dense(units=5, input_dim = 2, activation = 'relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


# 모델 파라미터 확인
print(model.summary())

# 분류일 경우 entropy , 회귀일 경우 mse와같은 것을 사용함
model.compile(optimizer=Adam(0.01), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(x, y, epochs = 100, batch_size = 1, verbose = 1)

loss_metrics = model.evaluate(x, y)
print('loss_metrics : ', loss_metrics)

pred = (model.predict(x) > 0.5).astype('int32')
print('pred : ', pred.flatten())

print('--------------')
print(model.input)
print(model.output)
print(model.weights)    # kernel(가중치), bias 값 확인

print('~~~~~~~~~~~~~~~~~~~~~~~~')
print(history.history['loss'])
print(history.history['accuracy'])

# 모델 학습 시 발생하는 loss 값 시각화
import matplotlib.pyplot as plt
plt.plot(history.history['loss'], label='train loss')
plt.xlabel('epochs')
plt.show()

print()
import pandas as pd 
plt.plot()







