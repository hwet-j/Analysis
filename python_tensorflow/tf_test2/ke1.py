# Keras 모듈로 논리회로 처리 모델 (분류)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import SGD, RMSprop, Adam
import numpy as np

print(tf.keras.__version__)

# 1. 데이터 수집 및 가공
x = np.array([[0,0],[0,1],[1,0],[1,1]])
# y = np.array([0,1,1,1]) # or
y = np.array([0,0,0,1]) # and
# y = np.array([0,1,1,0]) # xor : Node가 1인 경우 처리 불가

print(x)
print(y)

# # 2. 모델 생성(네트워크 구성)
# model = Sequential([
#     Dense(input_dim = 2, units = 1),
#     Activation('sigmoid')
# ])

model = Sequential()
model.add(Dense(units=1, input_dim = 2))
model.add(Activation('sigmoid'))


print(model)

# 3. 모델 학습과정 설정
# model.compile(optimizer='sgb', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=SGD(lr=0.01, momentum=0.9), loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer=RMSprop(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['accuracy'])

# 4. 모델 학습 
model.fit(x, y, epochs=1000, batch_size = 1, verbose=2)

# 5. 모델 평가
loss_metrics = model.evaluate(x, y)
print('loss_metrics : ', loss_metrics)

# 6. 예측값
# pred = model.predict(x)
# print('pred : ', pred)
# pred = (model.predict(x) > 0.5).astype('int32')
# print('pred : ', pred.flatten())

print('^^^^^^^^^^^^^^^^^^^^^^^^^^')
# 완벽한 모델이라 판단이 되면 모델을 저장
model.save('test.hdf5')
# del model   # 모델 삭제

from tensorflow.keras.models import load_model
model2 = load_model('test.hdf5')
pred2 = (model2.predict(x) > 0.5).astype('int32')
print('pred2 : ', pred2.flatten())




