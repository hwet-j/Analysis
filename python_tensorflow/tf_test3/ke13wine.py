# 와인의 등급, 맛, 산도 등을 측정해 얻은 자료로 레드 와인과 화이트 와인 분류
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split

wdf = pd.read_csv("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/wine.csv")
print(wdf.head(3))
print(wdf.info())
print(wdf.iloc[:, 12].unique()) # [1 0]

dataset = wdf.values
print(dataset)
x = dataset[:, 0:12]
y = dataset[:, -1]
print(x[0])
print(y[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=12)
print(x_train.shape, x_test.shape, y_train.shape)   # (4547, 12) (1949, 12) (4547,)

# Model 
model = Sequential()
model.add(Dense(30, input_dim=12, activation='relu'))
model.add(tf.keras.layers.BatchNormalization()) # 배치 정규화. 그레디언트 손실과 폭주 문제
model.add(Dense(15, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(8, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
print(model.summary)

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

loss, acc = model.evaluate(x_train, y_train, verbose=2)
print('훈련되지 않은 모델의 평가 : {:5.2f}%'.format(100 * acc))

# 모델 저장 및 폴더 설정
import os
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)


# 모델 저장 조건 설정 
modelpath = 'model/{epoch:02d}-{loss:4f}.hdf5'

# 모델 학습시 모니터링의 결과를 파일로 저장
# chkpoint = ModelCheckpoint(filepath='./model/abc.hdf5', monitor='loss', save_best_only=True)
chkpoint = ModelCheckpoint(filepath=modelpath,monitor='loss', save_best_only=True)

early_stop = EarlyStopping(monitor='loss', patience=5)

history = model.fit(x_train, y_train, epochs=10000, batch_size=64,\
                    validation_split = 0.3, callbacks = [early_stop, chkpoint])

# model.load_weights('./model/abc.hdf5')

loss, acc = model.evaluate(x_test, y_test, verbose=2, batch_size = 64)
print('훈련되지 않은 모델의 평가 : {:5.2f}%'.format(100 * acc))    

# loss, val_loss  ==> validation_split = 0.3 history에서 이값이 없으면 loss, val_loss 값은 존재하지않음
vloss = history.history['val_loss']
print('vloss :', vloss, len(vloss))

loss = history.history['loss']
print('loss :', loss, len(loss))
acc = history.history['accuracy']
print('acc :', acc)

epoch_len = np.arange(len(acc))
plt.plot(epoch_len, vloss, c = 'red', label='val_loss')
plt.plot(epoch_len, loss, c = 'blue', label='loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend(loc = 'best')    # 알맞은위치에 (best) 범례지정
plt.show()

plt.plot(epoch_len, acc, c = 'black', label='acc')
plt.xlabel('epochs')
plt.ylabel('acc')
plt.legend(loc = 'best')  
plt.show()

# 예측 
np.set_printoptions(suppress = True)    # 과학적 표기 형식 해제
new_data = x_test[:5, :]
print(new_data)
pred = model.predict(new_data)
print('예측 결과 : ', np.where(pred > 0.5, 1, 0).flatten())

