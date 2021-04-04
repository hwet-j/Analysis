'''
문제) BMI 식으로 작성한 bmi.csv 파일을 이용하여 분류모델 작성 후 분류 작업을 진행한다.
train/test 분리 작업을 수행.
평가 및 정확도 확인이 끝나면 모델을 저장하여, 저장된 모델로 새로운 데이터에 대한 분류작업을 실시한다.
새로운 데이터, 즉 키와 몸무게는 키보드를 통해 입력하기로 한다.
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import numpy as np
import pandas as pd
import tensorflow as tf

"""
xy = pd.read_csv('./bmi.csv')
print(xy.head(3))

x = xy[['weight', 'height']].values

bclass = {'thin': [1, 0, 0], 'normal': [0, 1, 0], 'fat': [0, 0, 1]}

y = np.empty((50000, 3))
for i, v in enumerate(xy['label']):
    y[i] = bclass[v]

# train/test로 분리 작업을 수행
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state = 12)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)  # (35000, 2) (15000, 2) (35000,) (15000,)

# Model
model = Sequential()
model.add(Dense(32, input_shape=(2, ), activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))

opti = 'adam'  # sgd, rmsprop ...
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['acc'])
"""
"""
# 모델 저장 및 폴더 설정
import os
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

# 모델 저장 조건 설정
modelpath = "mymodel.hdf5"

# 모델 학습 시 모니터링의 결과를 파일로 저장
chkpoint = ModelCheckpoint(filepath=modelpath, monitor='loss', save_best_only=True)

early_stop = EarlyStopping(monitor='loss', patience=5)  # 필요없는 학습을 막기위해
history = model.fit(x_train, y_train, epochs=100, batch_size=32, validation_split=0.3, \
                    verbose=2, callbacks = [early_stop, chkpoint])
print(model.evaluate(x,y))
"""

from tensorflow.keras.models import load_model
model = load_model('mymodel.hdf5')

"""
# history.history를 pickle 모듈을 이용해서 저장 후 읽어 사용
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']  # 41번줄에 validation_split이 있기 때문에 val_loss가 있는거임
acc = history_dict['acc']
val_acc = history_dict['val_acc']
"""

"""
print()
pred_datas = x[:5]  # 여러 개
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값 : ', preds)
print('실제값 : ', y[:5].flatten())
"""
# 새로운 데이터
weight_input = int(input("몸무게 :"))
height_input = int(input("키 :"))

new_data = [[weight_input, height_input]]
new_pred = np.argmax(model.predict(new_data))
print(new_pred)
# thin:0  normal:1  fat:2
if new_pred == 0:
    result = 'thin'
elif new_pred == 1:
    result = 'normal'
else:
    result = 'fat'
print('분류결과 : ', result)

