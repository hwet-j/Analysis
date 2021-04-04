# 여기에서는 인터넷 영화 데이터베이스(Internet Movie Database)에서 수집한 50,000개의 영화 리뷰 텍스트를 
# 담은 IMDB 데이터셋을 사용하겠습니다. 25,000개 리뷰는 훈련용으로, 25,000개는 테스트용으로 나뉘어져 있습니다. 
# 훈련 세트와 테스트 세트의 클래스는 균형이 잡혀 있습니다. 즉 긍정적인 리뷰와 부정적인 리뷰의 개수가 동일합니다.

from tensorflow.keras.datasets import imdb

# 매개변수 num_words=10000은 훈련 데이터에서 가장 많이 등장하는 상위 10,000개의 단어를 선택합니다. 
# 데이터 크기를 적당하게 유지하기 위해 드물에 등장하는 단어는 제외하겠습니다.
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])    # [1, 14, 22, ..... 각 숫자는 사전에 있는 전체 문서에 나타난 모든 단어에 고유한 번호를 부여한 어휘 사전
print(train_labels) # [1 0 0 ... 0 1 0] 긍정1 부정 0

aa = []
for seq in train_data:
    # print(max(seq))
    aa.append(max(seq))

print(max(aa), len(aa)) # 9999 25000

word_index = imdb.get_word_index()  # 단어와 정수 인덱스를 매핑한 딕셔너리
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
decord_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print(decord_review)

# 데이터 준비 : list -> tensor로 변환 Onehot vector
import numpy as np
def vector_seq(sequences, dim=10000):
    results = np.zeros((len(sequences), dim))
    for i, seq in enumerate(sequences):
        results[i, seq] = 1
    return results

x_train = vector_seq(train_data)
x_test = vector_seq(test_data)
print(x_train, ' ', x_train.shape)
y_train = train_labels
y_test = test_labels
print(y_train)

# 신경망 모델
from tensorflow.keras import models, layers, regularizers

model = models.Sequential()
# 뉴런의 개수를 정해주는것은 분석가의 재량(능력)임 수정해가면서 좋은 데이터를 뽑아내야함
# kernel_regularizer=regularizers.l2(0.001) : 가중치 행령의 모든 원소를 제곱하고 0.001을 곱하여 네트워크의 전체손실에 더해진다는 의미, 이 규제(패널티)는 훈련할때만 추가함
model.add(layers.Dense(32, activation='relu', input_shape = (10000,), kernel_regularizer=regularizers.l2(0.001))) # 16이란 숫자는 알아서설정(뉴런의 개수로 생각)
model.add(layers.Dropout(0.3))  # 과적합 방지를 목적으로 노드 일부는 학습에 참여하지 않음
model.add(layers.Dense(16, activation='relu'))  
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1, activation='sigmoid'))  # 마지막은 0,1로 나가게 설정해야함 

# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])    #  그래프그릴때 acc로 불러와 오류
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

#  훈련시 검증 데이터(validation data)
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
print(len(x_val), len(partial_x_train))
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train, partial_y_train, epochs = 30,\
                    batch_size=512, validation_data=(x_val, y_val))

print(model.evaluate(x_test, y_test))

# 시각화
import matplotlib.pyplot as plt
history_dict = history.history
loss = history_dict['loss']
val_loss = history_dict['val_loss']
epochs = range(1, len(loss) + 1)

# "bo"는 "파란색 점"입니다
plt.plot(epochs, loss, 'bo', label='Training loss')
# b는 "파란 실선"입니다
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


acc = history_dict['acc']   #comfile 시 metrics의 이름과 같아야함
val_acc = history_dict['val_acc']

# "bo"는 "파란색 점"입니다
plt.plot(epochs, acc, 'bo', label='Training acc')
# b는 "파란 실선"입니다
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation acc')
plt.xlabel('Epochs')
plt.ylabel('acc')
plt.legend()
plt.show()

# print('예측값 : ', model.predict(x_test[:5]))
import numpy as np
pred = model.predict(x_test[:5])
print('예측값 : ', np.where(pred > 0.5, 1, 0).flatten())
print('실제값 : ', y_test[:5])

