# 크롤링한 데이터를 읽어와 형태소 분석해보기..
# https://somjang.tistory.com/entry/Keras기사-제목을-가지고-긍정-부정-중립-분류하는-모델-만들어보기 [솜씨좋은장씨]

# 파일에서 단어를 불러와 posneg리스트를 만드는 코드
# 크롤링한 기사 제목과 기사 제목과 posneg를 활용하여 만든
# 긍정(1), 부정(-1), 중립(0)라벨 정보를 가지는 dataframe을 만드는 코드

import codecs 
positive = [] 
negative = [] 
posneg = [] 

pos = codecs.open("./data/positive_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = pos.readline() 
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 
    
    if not line: break 
pos.close() 

neg = codecs.open("./data/negative_words_self.txt", 'rb', encoding='UTF-8') 

while True: 
    line = neg.readline() 
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 
    
    if not line: break 
neg.close()


with open("./data/Twitter_all_data.txt", encoding='utf-8') as text: 
    data = text.readlines() 
    
data = [text.replace("\n", "") for text in data]
data = [text.replace(",", "") for text in data]

print(data)
print(len(data))


import requests 
from bs4 import BeautifulSoup 
import re 
import pandas as pd 

label = [0] * 4000 

my_title_dic = {"title":[], "label":label} 

j = 0 

for i in range(400): 
    num = i * 10 + 1 
    # 버거킹
    for title in data: 
        title_data = title.text 
        title_data = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title_data) 
        my_title_dic['title'].append(title_data) 
        
        for i in range(len(posneg)): 
            posflag = False 
            negflag = False 
            if i < (len(positive)-1): 
                # print(title_data.find(posneg[i])) 
                if title_data.find(posneg[i]) != -1: 
                    posflag = True 
                    print(i, "positive?","테스트 : ",title_data.find(posneg[i]),"비교단어 : ", posneg[i], "인덱스 : ", i, title_data) 
                    break 
            if i > (len(positive)-2): 
                if title_data.find(posneg[i]) != -1: 
                    negflag = True 
                    print(i, "negative?","테스트 : ",title_data.find(posneg[i]),"비교단어 : ", posneg[i], "인덱스 : ", i, title_data) 
                    break 
        if posflag == True: 
            label[j] = 1 
            # print("positive", j) 
        elif negflag == True: 
            label[j] = -1 
            # print("negative", j) 
        elif negflag == False and posflag == False: 
            label[j] = 0 
            # print("objective", j) 
        j = j + 1 
        
    my_title_dic['label'] = label 
    my_title_df = pd.DataFrame(my_title_dic)

