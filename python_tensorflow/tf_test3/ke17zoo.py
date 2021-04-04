# 다항분류 : 동물 type

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.utils import to_categorical
import numpy as np

xy = np.loadtxt("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/zoo.csv", delimiter=',')

print(xy[:2], xy.shape) # (101, 17)
x_data = xy[:, 0:-1]    # feature
y_data = xy[:, [-1]]    # label(class), type열
print(x_data[:3])
print(y_data[:3])
print(set(y_data.ravel()))  # {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0}

nb_classes = 7
y_one_hot = to_categorical(y_data, num_classes=nb_classes)  # label에 one-hot처리
print(y_one_hot[:3])

model = Sequential()
model.add(Dense(32, input_shape=(16, ), activation = 'relu'))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(nb_classes, activation = 'softmax'))

opti = 'adam'
model.compile(optimizer=opti, loss='categorical_crossentropy', metrics=['acc'])

history = model.fit(x_data, y_one_hot, epochs=100, batch_size = 32, validation_split=0.3, verbose=0)
print(model.evaluate(x_data, y_one_hot))

history_dist = history.history
loss = history_dist['loss']
val_loss = history_dist['val_loss']
acc = history_dist['acc']
val_acc = history_dist['val_acc']

# 시각화
import matplotlib.pyplot as plt
plt.plot(loss, 'b-', label='train loss')
plt.plot(val_loss, 'r--', label='train val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()


plt.plot(acc, 'b-', label='train acc')
plt.plot(val_acc, 'r--', label='train val_acc')
plt.xlabel('epoch')
plt.ylabel('acc')
plt.legend()
plt.show()

# predict
pred_data = x_data[:1]  # 한개만
pred = np.argmax(model.predict(pred_data))
print('예측값 : ', pred)

print()
pred_datas = x_data[:5] # 여러 개
preds = [np.argmax(i) for i in model.predict(pred_datas)]
print('예측값 : ', preds)
print('실제값 : ', y_data[:5].flatten())

# 새로운 data
# print(x_data[:1])
new_data = [[1., 0., 1., 1., 0., 0., 1., 1., 0., 0., 0., 1., 8., 0., 1., 1.]]
new_pred = np.argmax(model.predict(new_data))
print('예측값 : ', new_pred)

