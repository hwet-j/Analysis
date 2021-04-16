from django.shortcuts import render
from django.http import request
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import matplotlib.pyplot as plt
plt.rc('font',family='malgun gothic')
plt.rcParams['axes.unicode_minus'] = False 
from tensorflow.keras import optimizers
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from konlpy.tag import Okt
from tensorflow.keras.layers import Embedding, Dense, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model


# Create your views here.
def MainFunc(request):
    Analysis_AZ()
    return render(request, 'main.html')


def Analysis_AZ():
    # 데이터 읽어오기
    #data = pd.read_excel("django_pj1/Sentiment/templates/data/result(AZ).xlsx")
    data = pd.read_excel("/User/ghlckd/Python/django_pj1/Sentiment/templates/data/result(AZ).xlsx")
    
    data = data.set_index('Unnamed: 0')
    
    train_data, test_data = train_test_split(data, test_size=0.30, random_state=123)
    print(train_data.shape, test_data.shape)  
    
    # 데이터 정제하기
    train_data.drop_duplicates(subset = ['title'], inplace=True)
    print(train_data.groupby('label').size().reset_index(name = 'count'))
    train_data['title'] = train_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
    train_data['title'] = train_data['title'].str.replace('^ +', "") # white space 데이터를 empty value로 변경(긴 공백이나 특수문자들로만 이루어진데이터를 공백하나로 바꿔줌)
    train_data['title'].replace('', np.nan, inplace=True)   # 공백으로 바꿔준 데이터를 Null값으로 
    #print(train_data.isnull().sum())
    train_data = train_data.dropna(how = 'any') # Null값 제거
    #print(len(train_data)) 
    # test 동일작업
    test_data.drop_duplicates(subset = ['title'], inplace=True) # document 열에서 중복인 내용이 있다면 중복 제거
    test_data['title'] = test_data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","") # 정규 표현식 수행
    test_data['title'] = test_data['title'].str.replace('^ +', "") # 공백은 empty 값으로 변경
    test_data['title'].replace('', np.nan, inplace=True) # 공백은 Null 값으로 변경
    test_data = test_data.dropna(how='any') # Null 값 제거
    #print('전처리 후 테스트용 샘플의 개수 :',len(test_data)) 
    
    # 불용어 정의
    stopwords_read = pd.read_csv("django_pj1/Sentiment/templates/data/stopwords_self.txt", sep="\n", header=None)
    # print(stopwords_read.columns)
    stopwords = list(stopwords_read[0]) 
    #print(stopwords, type(stopwords))
    # 형태소 분석
    okt = Okt()
    
    # 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장
    X_train = []
    for sentence in train_data['title']:
        temp_X = []
        temp_X = okt.morphs(sentence, stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_train.append(temp_X)
    
    #print(X_train[:3])
    
    # 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장(test)
    X_test = []
    for sentence in test_data['title']:
        temp_X = []
        temp_X = okt.morphs(sentence, stem=True) # 토큰화
        temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
        X_test.append(temp_X)
    
    #print(X_test[:3])
    
    max_words = 35000 
    tokenizer = Tokenizer(num_words = max_words) 
    tokenizer.fit_on_texts(X_train) 
    X_train = tokenizer.texts_to_sequences(X_train) 
    X_test = tokenizer.texts_to_sequences(X_test)
    #print(X_train[:3])
    #print(X_test[:3])
    #print(train_data['label'][:40])
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
    
    #print(y_train[:3])
    #print(y_test[:3])
    
    X_train = pad_sequences(X_train)
    X_test = pad_sequences(X_test)
    
    loaded_model = load_model('django_pj1/Sentiment/templates/data/model(AZ).h5') 

    print("\n 테스트 정확도 : {:.2f}%".format(loaded_model.evaluate(X_test, y_test)[1] * 100))
    print('adam\nacc : ',loaded_model.evaluate(X_test, y_test)[1])
    print('loss : ',loaded_model.evaluate(X_test, y_test)[0])
    
    predict =loaded_model.predict(X_test)
    print("X_test ", X_test[:3])
    import numpy as np 
    predict_labels = np.argmax(predict, axis=1) 
    original_labels = np.argmax(y_test, axis=1)
    
    for i in range(30): 
        print("내용 : ", test_data['title'].iloc[i], "/\t 원래 라벨 : ", original_labels[i], "/\t예측한 라벨 : ", predict_labels[i])

    return loaded_model.evaluate(X_test, y_test)[0], loaded_model.evaluate(X_test, y_test)[1]
   #return loss, acc, df, history
    



