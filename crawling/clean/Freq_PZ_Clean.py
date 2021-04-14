# 빈도수를 확인하기위해 데이터를 정리

import pandas as pd
import numpy as np
from konlpy.tag import Okt

data = pd.read_excel("./data/result(PZ).xlsx")

data = data.set_index('Unnamed: 0')
print(data)

# 한글빼고 제거
data['title'] = data['title'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

data['title'] = data['title'].str.replace('^ +', "") # white space 데이터를 empty value로 변경(긴 공백이나 특수문자들로만 이루어진데이터를 공백하나로 바꿔줌)
data['title'].replace('', np.nan, inplace=True)   # 공백으로 바꿔준 데이터를 Null값으로 
print(data.isnull().sum())   # Null값 존재확인 
data = data.dropna(how = 'any') # Null값 제거
print(len(data)) 

# 불용어 정의
stopwords_read = pd.read_csv("./data/stopwords_self.txt", sep="\n", header=None)
# print(stopwords_read.columns)
stopwords = list(stopwords_read[0]) 
print(stopwords, type(stopwords))
# 형태소 분석
okt = Okt()
# 형태소 분석을 하여 토큰화를 한후 불용어를 제거하여 저장
datas = []
for sentence in data['title']:
    temp = []
    temp = okt.morphs(sentence, stem=True) # 토큰화
    temp = [word for word in temp if not word in stopwords] # 불용어 제거
    datas.append(temp)
    
print(datas[:3])
df = []
for i in datas:
    df += i


from konlpy.tag import Okt
import collections
from collections import Counter, OrderedDict
counter = Counter(df)
# 한글자 제거
counter_re = Counter({x: counter[x] for x in counter if len(x) > 1})

# value값을 기준으로 정렬
sorted_by_value = sorted(counter_re.items(), key=lambda x: x[1], reverse=True)
# list타입으로 바뀌어서 dict로 변환 (저장할때 오류남 - key,value를 사용할수없음)
sorted_dict = collections.OrderedDict(sorted_by_value)

# 저장
with open("./data/freq_PZ.txt", 'w', encoding="UTF-8") as f:
    for key,value in sorted_dict.items():
        f.write(key+':'+str(value)+'\n')




