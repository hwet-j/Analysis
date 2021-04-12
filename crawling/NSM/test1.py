# 크롤링한 데이터를 읽어와 형태소 분석해보기..
# https://somjang.tistory.com/entry/Keras%EA%B8%B0%EC%82%AC-%EC%A0%9C%EB%AA%A9%EC%9D%84-%EA%B0%80%EC%A7%80%EA%B3%A0-%EA%B8%8D%EC%A0%95-%EB%B6%80%EC%A0%95-%EC%A4%91%EB%A6%BD-%EB%B6%84%EB%A5%98%ED%95%98%EB%8A%94-%EB%AA%A8%EB%8D%B8-%EB%A7%8C%EB%93%A4%EC%96%B4%EB%B3%B4%EA%B8%B0

# 파일에서 단어를 불러와 posneg리스트를 만드는 코드
# 크롤링한 기사 제목과 기사 제목과 posneg를 활용하여 만든
# 긍정(1), 부정(-1), 중립(0)라벨 정보를 가지는 dataframe을 만드는 코드

with open("./negative_words_self.txt", encoding='utf-8') as neg: 
    negative = neg.readlines()

negative = [neg.replace("\n", "") for neg in negative] 

with open("./positive_words_self.txt", encoding='utf-8') as pos: 
    positive = pos.readlines() 
    
negative = [neg.replace("\n", "") for neg in negative] 
positive = [pos.replace("\n", "") for pos in positive]


import requests 
from bs4 import BeautifulSoup 
import re 
import pandas as pd 
from tqdm import tqdm 

labels = [] 
titles = [] 
j = 0 

for k in tqdm(range(400)): 
    num = k * 10 + 1 
    # 버거킹 
    url = "https://search.naver.com/search.naver?&where=news&query=%EB%B2%84%EA%B1%B0%ED%82%B9&sm=tab_pge&sort=0&photo=0&field=0&reporter_article=&pd=0&ds=&de=&docid=&nso=so:r,p:all,a:all&mynews=0&cluster_rank=23&start=" + str(num) 
    
    req = requests.get(url) 
    
    soup = BeautifulSoup(req.text, 'lxml') 
    
    titles = soup.select("a._sp_each_title") 
    
    for title in titles: 
        title_data = title.text 
        clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title_data) 
        negative_flag = False 
        
        label = 0 
        for i in range(len(negative)): 
            if negative[i] in clean_title: 
                label = -1 
                negative_flag = True 
                print("negative 비교단어 : ", negative[i], "clean_title : ", clean_title) 
                break 
        if negative_flag == False: 
            for i in range(len(positive)): 
                if positive[i] in clean_title: 
                    label = 1 
                    print("positive 비교단어 : ", positive[i], "clean_title : ", clean_title) 
                    break 
        titles.append(clean_title) 
        labels.append(label) 
        
my_title_df = pd.DataFrame({"title":titles, "label":labels})

print(my_title_df)

my_title_df.to_csv("datas.csv", mode='w', index = False, header = False)
my_title_df.to_csv("datas.txt", mode='w', index = False, header = False)









