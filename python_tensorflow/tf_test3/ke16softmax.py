# 활성화 함수를 softmax를 사용하여 다항분류

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np

x_data = np.array([[1,2,1,4,],
                [1,3,1,6],  
                [1,4,1,8],  
                [2,1,2,1],  
                [3,1,3,1],  
                [5,1,5,1],  
                [1,2,3,4],  
                [5,6,7,8]], dtype=np.float32)
y_data = to_categorical([2,2,2,1,1,1,0,0])  # One_hot encoding
print(x_data)
print(y_data)

model = Sequential()
model.add(Dense(50, input_shape=(4,)))
model.add(Activation('relu'))
model.add(Dense(50))
model.add(Activation('relu'))
model.add(Dense(3))
model.add(Activation('softmax'))
print(model.summary())

opti = 'adam'   # sgd, rmsprop....
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_data, y_data, epochs=100)
print(model.evaluate(x_data, y_data))
print(np.argmax(model.predict(np.array([[1,8,1,8]]))))
print(np.argmax(model.predict(np.array([[10,8,5,1]]))))













