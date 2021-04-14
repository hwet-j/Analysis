# 잘못됨 ... 수정 필요

import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split



with open("./data/freq_AZ.txt", encoding='utf-8') as text: 
    data = text.readlines() 

list = []
wordlist = {}
for i in data:  # \n가 출력되어 이를 제거함
    data_list = i.replace("\n", "")
    list.append(data_list)

wordCount = {}
for word in data:
    for i in word.split():
        wordCount[i] = wordCount.get(i, 0) + 1 
        keys = sorted(wordCount.keys())

for word in keys:
    #print(word + ':' + str(wordCount[word])) 
    #wordlist.append(word + ':' + str(wordCount[word]))
    wordlist[word] = wordCount[word]
    
print(wordlist)   
wordlist = sorted(wordlist.items(), key=lambda x: x[1], reverse=True)
print(wordlist) 

def write_txt(list, fname, sep) :
    file = open(fname, 'w', encoding='utf8')
    vstr = ''
    
    for a in list :
        vstr = vstr + str(a) + sep
    vstr = vstr.rstrip(sep)  # 마지막에도 추가되는  sep을 삭제 
    
    file.writelines(vstr)      # 한 라인씩 저장 
        
    file.close()
    print('[ 파일 저장 완료 ]')

write_txt(wordlist, "./data/freq(AZ).txt", "\n")


