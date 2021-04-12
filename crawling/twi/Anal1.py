# 크롤링한 데이터를 읽어와 형태소 분석해보기..
# https://somjang.tistory.com/entry/Keras%EA%B8%B0%EC%82%AC-%EC%A0%9C%EB%AA%A9%EC%9D%84-%EA%B0%80%EC%A7%80%EA%B3%A0-%EA%B8%8D%EC%A0%95-%EB%B6%80%EC%A0%95-%EC%A4%91%EB%A6%BD-%EB%B6%84%EB%A5%98%ED%95%98%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0

# 파일에서 단어를 불러와 posneg리스트를 만드는 코드
# 크롤링한 기사 제목과 기사 제목과 posneg를 활용하여 만든
# 긍정(1), 부정(-1), 중립(0)라벨 정보를 가지는 dataframe을 만드는 코드

import codecs 
positive = [] 
negative = [] 
posneg = [] 
 
pos = codecs.open("./data/positive_words_twit.txt", 'rb', encoding='UTF-8') 
 
while True: 
    line = pos.readline() 
    line = line.replace('\n', '') 
    positive.append(line) 
    posneg.append(line) 
     
    if not line: break 
pos.close() 
 
neg = codecs.open("./data/negative_words_twit.txt", 'rb', encoding='UTF-8') 
 
while True: 
    line = neg.readline() 
    line = line.replace('\n', '') 
    negative.append(line) 
    posneg.append(line) 
     
    if not line: break 
neg.close()

print(posneg)

import requests 
from bs4 import BeautifulSoup 
import re 
import pandas as pd 

label = [0] * 4000 
my_title_dic = {"title":[], "label":label} 
j = 0 


titles=1
# for i in range(400): 
#     num = i * 10 + 1 
#     url3 = "https://search.naver.com/search.naver?&where=news&query=%EB%B2%84%EA%B1%B0%ED%82%B9&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=23&start=" + str(num) 
#     
#     req = requests.get(url3) 
#     
#     soup = BeautifulSoup(req.text, 'lxml') 
#     
#     titles = soup.select("a._sp_each_title") 
    
for title in titles: 
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



