# 분석에서 사용할 데이터를 만들어줌
# 두가지를 분석할것이기 때문에 데이터를 읽어올때와 저장할때 파일명을 설정해주어 실행

# 긍정 1 부정 -1 그외 0

# 설정해 놓은 긍정 부정 단어를 가져옴
with open("./data/negative_words_twit.txt", encoding='utf-8') as neg: 
    negative = neg.readlines()

negative = [neg.replace("\n", "") for neg in negative] 

with open("./data/positive_words_twit.txt", encoding='utf-8') as pos: 
    positive = pos.readlines() 
    
negative = [neg.replace("\n", "") for neg in negative] 
positive = [pos.replace("\n", "") for pos in positive]


from tqdm import tqdm 
import re 
import pandas as pd


# 트위터 데이터 가져옴 한 줄마다 한 게시글 
datas = pd.read_csv("./data/Twitter_all_data(AZ).txt", sep="\n", header=None)


# print(datas.head(4))
# print(datas.columns)
# 칼럼명 지정
datas = datas.rename(columns={0:"title"})
# 확인
print(datas.columns)

# 0, -1, 1의값을 넣어줄 라벨 준비
labels = [] 

# 데이터를 리스트화
title_data = list(datas['title']) 

for title in tqdm(title_data): 
    clean_title = re.sub('[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…\"\“》]', '', title) 
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
        
    labels.append(label) 
    
datas['label'] = labels
   
print(datas)

print(datas['label'].value_counts())

datas.to_excel(excel_writer="./data/result(AZ).xlsx")
