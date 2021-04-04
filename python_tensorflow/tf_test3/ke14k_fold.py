# 모델 학습 시 k-fold 교차 검증 : train data에 대해 k겹으로 나눠, 모든 데이터가 최소 1번은 test data로 학습에 사용되도록 하는 방법
# k-fold 교차 검증을 할 때는 validation_split은 사용하지 않는다.
# 데이터의 양이 적을 경우 많이 사용되는 방법

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics._scorer import accuracy_scorer

data = np.loadtxt('https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/diabetes.csv', \
                  dtype=np.float32, delimiter=',')
print(data[:2], data.shape)  # (759, 9)

x = data[:, 0:-1]
y = data[:, -1]
print(x[:2])
print(y[:2])

# 일반적인 모델 네트워크 작성방법1
model = Sequential([
    Dense(units = 64, input_dim = 8, activation = 'relu'),
    Dense(units = 32, activation = 'relu'),
    Dense(units = 1, activation = 'sigmoid')    
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit(x,y, epochs = 200, batch_size = 32, verbose = 2)
print(model.evaluate(x, y))  # [0.25278204679489136, 0.8893280625343323]

pred1 = model.predict(x[:3, :])
print('예측값 : ', pred1.flatten())
print('실제값 : ', y[:3])

print()
# 일반적인 모델 네트워크 작성방법2
def build_model():
    model = Sequential()
    model.add(Dense(units = 64, input_dim = 8, activation = 'relu'))
    model.add(Dense(units = 32, activation = 'relu'))
    model.add(Dense(units = 1, activation = 'sigmoid'))
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    return model
    
# K-겹 교차검증을 사용한 모델 네트워크
estimatorModel = KerasClassifier(build_fn = build_model, epochs = 200, batch_size = 32, verbose = 2)
kfold = KFold(n_splits = 5, shuffle = True, random_state = 12)
print(cross_val_score(estimatorModel, x, y, cv = kfold))
estimatorModel.fit(x,y, epochs = 200, batch_size = 32, verbose = 2)
#print(estimatorModel.evaluate(x, y)) # AttributeError: 'KerasClassifier' object has no attribute 'evaluate'

pred2 = estimatorModel.predict(x[:3, :])
print('예측값 : ', pred2.flatten())
print('실제값 : ', y[:3])

print()
from sklearn.metrics import accuracy_score

print('분류정확도(estimatorModel)', accuracy_score(y, estimatorModel.predict(x)))










