import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from konlpy.tag import Okt
from tensorflow.keras.layers import Embedding, Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

train_data = pd.read_table("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_train.txt")
test_data = pd.read_table("https://raw.githubusercontent.com/pykwon/python/master/testdata_utf8/ratings_test.txt")
print(train_data[:3], len(train_data))   # 150000
print(test_data[:3], len(test_data))    # 50000
print(train_data.columns)   # ['id', 'document', 'label']

#  데이터 전처리
print(train_data['document'].nunique(), test_data['document'].nunique())  # 146182 49157 중복자료가 존재한다.

train_data.drop_duplicates(subset=['document'], inplace=True)
print(len(train_data['document']))
print(set(train_data['label']))   # {0 부정, 1 긍정}

train_data['label'].value_counts().plot(kind='bar')
plt.show()

print(train_data.groupby('label').size())

# Null 값 확인 
print(train_data.isnull().values.any())
print(train_data.isnull().sum())    # document에 null  1개
print(train_data.loc[train_data.document.isnull()])

train_data = train_data.dropna(how='any')
print(train_data.isnull().values.any())
print(len(train_data))  # 146182

# 순수 한글 관련 자료 이외의 구둣점 등은 제거
print(train_data[:3])
train_data['document'] = train_data['document'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]","")
print(train_data[:3])

train_data['document'].replace('', np.nan, inplace=True)
print(train_data.isnull().sum())

train_data = train_data.dropna(how='any')
print(train_data.isnull().values.any())
print(len(train_data))  # 146182개에서 145791로

# test
test_data.drop_duplicates(subset=['document'], inplace = True)
test_data['document'] = test_data['document'].str.replace("[^ㄱ-ㅎ ㅏ-ㅣ 가-힣]","")
test_data['document'].replace('', np.nan, inplace=True)
test_data = test_data.dropna(how='any')
print(test_data.isnull().values.any())
print(len(test_data))   # 48995

# 불용어 제거 
stopwords = ['아','휴','아이구','아이쿠','아이고','어','나','우리','저희','따라','의해','을','를','에','의','가','으로','로','에게','뿐이다','의거하여']

# 형태소 분류
okt = Okt()
x_train = []
for sen in train_data['document']:
  imsi = []
  imsi = okt.morphs(sen, stem=True)
  imsi = [word for word in imsi if not word in stopwords]
  x_train.append(imsi)

print(x_train[:3])

x_test = []
for sen in test_data['document']:
  imsi = []
  imsi = okt.morphs(sen, stem=True)
  imsi = [word for word in imsi if not word in stopwords]
  x_test.append(imsi)

print(x_test[:3])


# 워드 임베딩
tok = Tokenizer()
tok.fit_on_texts(x_train)
print(tok.word_index)
# 등장 빈도수를 확인해서 비중이 적은 자료는 배제
threshold = 3
total_cnt = len(tok.word_index)
rare_cnt = 0
total_freq = 0
rare_freq = 0

for k, v in tok.word_counts.items():
  total_freq = total_freq + v
  if v < threshold:
    rare_cnt = rare_cnt + 1
    rare_freq = rare_freq + v

print('total_cnt : ', total_cnt)
print('rare_cnt : ', rare_cnt)
print('rare_freq : ', (rare_cnt / total_cnt) * 100)
print('total_cnt : ', (rare_freq / total_freq) * 100) # 2회 이하인 단어 전체 비중 1.7% 이므로 2회 이하인 단어들은 배제해도 무관할 것 같다

# OOV(Out Of Vocabulary) : 단어 사전에 없으면 index자체를 할 수 없게 되는데 이런 문제를 OOV
vocab_size = total_cnt - rare_cnt + 2
print('vocab_size 크기 : ', vocab_size) # 19423

tok = Tokenizer(vocab_size, oov_token='OOV')
tok.fit_on_texts(x_train)
x_train = tok.texts_to_sequences(x_train)
x_test = tok.texts_to_sequences(x_test)
print(x_train[:3])

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])

# 비어있는 샘플은 제거
drop_train = [index for index, sen in enumerate(x_train) if len(sen) < 1]

x_train = np.delete(x_train, drop_train, axis = 0)
y_train = np.delete(y_train, drop_train, axis = 0)
print(len(x_train), ' ', len(y_train))

print('리뷰 최대 길이 : ', max(len(i) for i in x_train))
print('리뷰 평균 길이 : ', sum(map(len, x_train)) / len(x_train))

plt.hist([len(s) for s in x_train], bins = 50)
plt.show()

# 전체 샘플 중에서 길이가 max_len 이하인 샘플 비율이 몇 %인지 확인 함수 작성
def below_threshold_len(max_len, nested_list):
  cnt = 0
  for s in nested_list:
    if len(s) <= max_len:
      cnt = cnt + 1
    print('전체 샘플 중에서 길이가 %s 이하인 샘플 비율 : %s'%(max_len, (cnt / len(nested_list)) * 100))

max_len = 30
below_threshold_len(max_len, x_train) # 92%정도가 30이하의 길이를 가짐

x_train = pad_sequences(x_train, maxlen=max_len)
x_test = pad_sequences(x_test, maxlen=max_len)
print(x_train[:10])


'''
# 모델
model = Sequential()
model.add(Embedding(vocab_size, 100))
model.add(LSTM(128, activation='tanh'))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(1, activation='sigmoid'))
print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
mc = ModelCheckpoint('tfrnn13.h5', monitor='val_acc', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, epochs=10, callbacks=[es, mc], batch_size=64, validation_split=0.2)
'''
# 저장된 모델로 나머지 작업
loaded_model = load_model('tfrnn13.h5')
print('acc : ', loaded_model.evaluate(x_test, y_test)[1])
print('loss : ', loaded_model.evaluate(x_test, y_test)[0])

# 예측
def new_pred(new_sentence):
    new_sentence = okt.morphs(new_sentence, stem = True)
    new_sentence = [word for word in new_sentence if not word in stopwords]
    encoded = tok.texts_to_sequences([new_sentence])
    pad_new = pad_sequences(encoded, maxlen=max_len)
    pred = float(loaded_model.predict(pad_new))
    if pred > 0.5:
        print("{:.2f}% 확률로 긍정!".format(pred * 100))
    else:
        print("{:.2f}% 확률로 부정 ㅠㅠ".format((1 - pred) * 100))


with open("./data/Twitter_2020-12-30.txt", encoding='utf-8') as text: 
    data = text.readlines() 
    
    
new_pred('영화가 재밌네요')
new_pred(data)
