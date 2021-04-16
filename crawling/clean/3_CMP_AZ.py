import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
import numpy as np
import re
import urllib.request
from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

data = pd.read_excel("./data/result(AZ).xlsx")

# print(type(data))
# print(data.columns)
data = data.set_index('Unnamed: 0')
# print(data)

###################################
############ 데이터 정제하기 ############
###################################
train_data, test_data = train_test_split(data, test_size=0.30, random_state=123)
# print(train_data.shape, test_data.shape)    # (2354, 2) (1009, 2)

# 중복되는 데이터 제거작업
# print(train_data['title'].nunique(), train_data['label'].nunique())
train_data.drop_duplicates(subset = ['title'], inplace=True)
# print(train_data.shape)

# 빈도수
# print(train_data.groupby('label').size().reset_index(name = 'count'))

# 한글빼고 그냥 다 제거
train_data['title'] = train_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# print(train_data)


# 데이터 정제 과정에서 영어만 존재하거나 한글이없는 데이터는 공백이므로 Null값이나 다름없는 데이터라 판단 
train_data['title'] = train_data['title'].str.replace('^ +', "") # white space 데이터를 empty value로 변경(긴 공백이나 특수문자들로만 이루어진데이터를 공백하나로 바꿔줌)
train_data['title'].replace('', np.nan, inplace=True)   # 공백으로 바꿔준 데이터를 Null값으로 
# print(train_data.isnull().sum())   # Null값 존재확인 
# train_data = train_data.dropna(how = 'any') # Null값 제거
# print(len(train_data)) 


# test 동일작업
test_data.drop_duplicates(subset = ['title'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
test_data['title'] = test_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
test_data['title'] = test_data['title'].str.replace('^ +', "") # 공백은 empty 값으로 변경
test_data['title'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
test_data = test_data.dropna(how='any') # Null 값 제거
# print('전처리 후 테스트용 샘플의 개수 :',len(test_data)) 

#############################
############ 토큰화 ############
#############################
# 불용어 정의
stopwords_read = pd.read_csv("./data/stopwords_self.txt", sep="\n", header=None)
# print(stopwords_read.columns)
stopwords = list(stopwords_read[0]) 
print(stopwords, type(stopwords))
# 형태소 분석
okt = Okt()
# 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장
X_train = []
for sentence in train_data['title']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_train.append(temp_X)

print(X_train[:3])

# 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장(test)
X_test = []
for sentence in test_data['title']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    X_test.append(temp_X)

print(X_test[:3])


'''
import csv
with open('train_ex.txt', 'w', newline='\n') as f: 
    writer = csv.writer(f) 
    writer.writerow(X_train) 
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)
print(tokenizer.word_index)

threshold = 2
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


tokenizer2 = Tokenizer()
tokenizer2.fit_on_texts(X_test)
print(tokenizer2.word_index)

threshold = 2
total_cnt = len(tokenizer2.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer2.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


vocab_size = total_cnt - rare_cnt + 1
print('단어 집합의 크기 :',vocab_size)

y_train = np.array(train_data['label'])
y_test = np.array(test_data['label'])
'''

max_words = 35000 
tokenizer = Tokenizer(num_words = max_words) 
tokenizer.fit_on_texts(X_train) 
X_train = tokenizer.texts_to_sequences(X_train) 
X_test = tokenizer.texts_to_sequences(X_test)
print(X_train[:3])
print(X_test[:3])
print(train_data['label'][:40])
#################################################
#################################################
# train_use = pd.DataFrame(data=None, index=None, columns=None, dtype=None, copy=False)
# train_use['x_train'] = X_train
# train_use['y_train'] = train_data['label']
# 
# train_use.to_csv("./data/train_d.csv", mode='w', index = False, header = False)



# 다음으로는 y값으로 들어갈 label -1, 0, 1을 컴퓨터가 보고 알수 있도록 one-hot encoding

y_train = [] 
y_test = []

for i in range(len(train_data['label'])):
    if train_data['label'].iloc[i] == 1:
        y_train.append([0, 0, 1])
    elif train_data['label'].iloc[i] == 0:
        y_train.append([0, 1, 0])
    elif train_data['label'].iloc[i] == -1:
        y_train.append([1, 0, 0])
    
for i in range(len(test_data['label'])):
    if test_data['label'].iloc[i] == 1:
        y_test.append([0, 0, 1])
    elif test_data['label'].iloc[i] == 0:
        y_test.append([0, 1, 0])
    elif test_data['label'].iloc[i] == -1:
        y_test.append([1, 0, 0])

y_train = np.array(y_train)
y_test = np.array(y_test)

print(y_train[:3])
print(y_test[:3])



#############################
############ 모델링 ############
#############################

# 긍정, 부정, 중립 3가지로 분류해야하니 LSTM, softmax, categorical_crossentropy를 사용
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

X_train = pad_sequences(X_train)
X_test = pad_sequences(X_test)
'''
model = Sequential()
model.add(Embedding(max_words, 100))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

model.save('model1(AZ).h5')
'''
# 위에 작업을 한번만 하기위해 저장

from tensorflow.keras.models import load_model
loaded_model1 = load_model('model1(AZ).h5')  


print("\n 테스트 정확도 : {:.2f}%".format(loaded_model1.evaluate(X_test, y_test)[1] * 100))
print('rmsprop\nacc : ', loaded_model1.evaluate(X_test, y_test)[1])
print('loss : ', loaded_model1.evaluate(X_test, y_test)[0])


'''
model2 = Sequential()
model2.add(Embedding(max_words, 100))
model2.add(LSTM(128))
model2.add(Dense(3, activation='softmax'))

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = model2.fit(X_train, y_train, epochs=10, batch_size=10, validation_split=0.1)

model2.save('model2(AZ).h5')     # 위에 작업을 한번만 하기위해 저장
'''
loaded_model2 = load_model('model2(AZ).h5') 




print("\n 테스트 정확도 : {:.2f}%".format(loaded_model2.evaluate(X_test, y_test)[1] * 100))
print('adam\nacc : ', loaded_model2.evaluate(X_test, y_test)[1])
print('loss : ', loaded_model2.evaluate(X_test, y_test)[0])

predict = loaded_model2.predict(X_test)
print("X_test ", X_test[:3])
import numpy as np 
predict_labels = np.argmax(predict, axis=1) 
original_labels = np.argmax(y_test, axis=1)


# 어떤내용 어떻게 예측했는지
for i in range(30): 
    print("내용 : ", test_data['title'].iloc[i], "/\t 원래 라벨 : ", original_labels[i], "/\t예측한 라벨 : ", predict_labels[i])






# train_data['label'].value_counts().plot(kind='bar')
# test_data['label'].value_counts().plot(kind='bar')
# print(train_data.groupby('label').size().reset_index(name = 'count'))
# print(test_data.groupby('label').size().reset_index(name = 'count'))
# plt.show()

