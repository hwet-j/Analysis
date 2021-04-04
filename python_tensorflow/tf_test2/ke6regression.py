# 단순선형모델 작성 : 공부시간에 따른 성적 결과 예측
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.array([1,2,3,4,5], dtype=np.float32)        # feature
y_data = np.array([11,32,53,64,70], dtype=np.float32)   # label

print(np.corrcoef(x_data, y_data))  # 0.974 인과관계가 있다고 가정

# 모델 작성 방법 3가지
# 방법1 : Sequential API 사용 - 여러 개의 층을 순서대로 쌓아올린 완전 연결 모델
model = Sequential()
model.add(Dense(units=1, input_dim=1, activation='linear'))
model.add(Dense(units=1, activation='linear'))
print(model.summary())

opti = optimizers.Adam(lr=0.01)
model.compile(optimizer = opti, loss='mse', metrics=['mse'])
model.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=1)
loss_metrics = model.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model.predict(x_data)))

print('실제값 : ', y_data)
print('예측값 : ', model.predict(x_data).flatten())
print('예상점수 : ', model.predict([0.5, 3.45, 6.7]).flatten())

# plt.plot(x_data, model.predict(x_data), 'b', x_data, y_data, 'ko')
# plt.show()

# 방법2 : function API 사용 - Sequential API 보다 유연한 모델을 작성
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
inputs = Input(shape=(1,))   # input layer
output1 = Dense(2, activation='linear')(inputs)
output2 = Dense(1, activation='linear')(output1)
model2 = Model(inputs, output2)

opti = optimizers.Adam(lr=0.01)
model2.compile(optimizer = opti, loss='mse', metrics=['mse'])
model2.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model2.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model2.predict(x_data)))

print('@@@@@@@@@@@@@@@@@@@@@@')
# 방법2 : Model subclassing 사용 - 동적인 모델을 작성
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.d1 = Dense(2, activation='linear')
        self.d2 = Dense(1, activation='linear')
        
    def call(self, x):  # 모델.fit()이 호출
        x = self.d1(x)
        return self.d2(x)
    
model3 = MyModel()  # init 호출

opti = optimizers.Adam(lr=0.01)
model3.compile(optimizer = opti, loss='mse', metrics=['mse'])
model3.fit(x=x_data, y=y_data, batch_size=1, epochs=100, verbose=0)
loss_metrics = model3.evaluate(x=x_data, y=y_data)
print('loss_metrics : ', loss_metrics)
from sklearn.metrics import r2_score
print('설명력 : ', r2_score(y_data, model3.predict(x_data)))



