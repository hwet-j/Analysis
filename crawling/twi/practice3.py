# 패키지 읽기 
import twitter
import numpy as np
import pandas as pd
import string
import re
import warnings
import os
import snscrape
import snscrape.modules.twitter as sntwitter
from time import sleep
import datetime
 
warnings.filterwarnings(action='ignore')

# 인자 : 트위터 계정명,시작날짜(yyyy-mm-dd), 종료날짜(yyyy-mm-dd)
# 리턴되는 결과 : 해당 계정의 트위터 결과 데이터 프레임 (url,time,id,content,text,username)
# 계정 이름이 누락되는 경우 데이터 프레임이 반환되지 않고 대신 String 이 반환됨
def read_tweet_list(twitterName,startDay,endDay):
    
    tweets_list1 = []
    tweets_df2 = pd.DataFrame(columns=['URL','Datetime', 'Tweet Id','Content', 'Username'])
    
    # 계정 정보가 잘못 들어온 경우 빈 데이터프레임을 반환 함
    if pd.isnull(twitterName) or twitterName == "":
        return tweets_df2;
        
    # Using TwitterSearchScraper to scrape data and append tweets to list
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+twitterName+' since:'+startDay+' until:'+endDay+'').get_items()):
        tweets_list1.append([tweet.url, tweet.date, tweet.id, tweet.content, tweet.username])
        #print(tweets_list1)
        # Creating a dataframe from the tweets list above 
        tweets_df2 = pd.DataFrame(tweets_list1, columns=['URL','Datetime', 'Tweet Id','Content', 'Username'])
    
    return tweets_df2;


# 인자 : 데이터 프레임 / 내용에 해당하는 컬럼의 이름
# 리턴되는 결과 : RT 및 리플라이를 제외한 결과 트윗 데이터 프레임
def remove_rt_reply(df,contentCol):
    # content 의 가장 앞에 '@' 라는 문자열이  있는 경우 = Reply 로 판단.
    # content 의 가장 앞에 'RT @' 라는 문자열이 있는 경우 = retweet 으로 판단.
    # 결과적으로 내용 앞에 @ 가 있는 경우를 제거함으로서 리플라이 및 리트윗을 제거하고 남은 데이터 프레잉을 리턴함
    rs = df.copy(deep=True)
    row = -1
    target = rs[contentCol]
    
    
    rs['retflag'] = False
    for i in target:
        row = row + 1
        
        if(i[0:1] == "@" or i[0:2] == "RT"):
            rs['retflag'][row] = True
        else:
            rs['retflag'][row] = False
        
    rs_L1 = rs[rs['retflag'] == False]
    
    del rs_L1['retflag']
    rs_L1 = rs_L1.reset_index(drop=True)
    
    return rs_L1;
 
 
# 인자 : 데이터 프레임 / 키워드 리스트 / 내용에 해당하는 컬럼 이름 / 헤시태그만 찾을 것인지 여부 / 키워드에 해당하는 내용이 없는 트윗 삭제 여부
    # 리턴되는 결과 : 찾고자 하는 키워드가 있는 데이터가 존재하는 데이터 프레임
        # 조건 1: isOnlyHashtag 가 True 인 경우 키워드 앞에 # 를 붙여서 헤시태그에 해당하는 내용만 찾음 (False 인 경우 순수하게 키워드 존재 여부로 찾아주)
        # 조건 2 : isremove 가 True 인 경우 키워드를 찾지 못한 내용은 삭제한 후 리턴 (False 인경우 flag 만 붙여준 후 리턴)
def search_keyword(df,keyword,contentCol,isOnlyHashtag,isremove):
    
    rs = df.copy(deep=True)
    target = rs[contentCol]
    keyword_low = []
    # 오로지 헤시태그만 찾고자 하는 경우 키워드 앞에 # 을 붙이는 과정을 진행한다.
    if(isOnlyHashtag == True):
        for k in range(0,len(keyword),1):
            keyword[k] = '#' + keyword[k]
    else:
        keyword = keyword
        
    for k in range(0,len(keyword),1):
        keyword_low.append(keyword[k].lower())
            
    rs['findKeywordFlag'] = False
    rs['findKeyword'] = ''
    
    row = -1
    for i in target: # 콘텐츠의 내용
        i_low = i.lower()
        row = row + 1
        for k in keyword_low: # 키워드 (대소문자는 구분하지 않음)
            
            if(i_low.find(k) >= 0): 
                rs['findKeywordFlag'][row] = True
                key = rs['findKeyword'][row]
                rs['findKeyword'][row] = rs['findKeyword'][row] +  k + '|'
                
    if(isremove == True):
        rs_L1 = rs[rs['findKeywordFlag'] == True]
        rs_L1 = rs_L1.reset_index(drop=True)
    else:
        rs_L1 = rs
        
    return rs_L1;        


st_day = "2021-01-01" # 시작날짜 지정
ed_day = "2021-01-31" # 마지막 날짜 지정

my_keyword = ['마감'] # 찾고자 하는 키워드 지정

output_file_name = "./out/output" # 출력할 파일 이름과 장소
log_file_path = "./log.txt" # 로그 파일의 이름과 장소

target_tweet_name = ['twittlions']

append_mode = False 

for sid in target_tweet_name:
    
    # 트위터 수집 관련 예외처리 구분
    try:
        if pd.isnull(sid) or sid == "":
            with open(log_file_path, "a") as file:
                file.write("잘못된 닉네임 입니다. 로그 저장 시각 : " + str(datetime.datetime.now()) + "\n")
            file.close()
            continue;
        
            
        result = read_tweet_list(sid,st_day,ed_day)
        print(len(result))
        with open(log_file_path, "a") as file:
            file.write(sid + " 아이디의 트위터 검색 완료! 총 " + str(len(result)) + "개의 트위터를 찾았습니다. 로그 저장 시각 : " + str(datetime.datetime.now()) + "\n")
        file.close()
    except:
        with open(log_file_path, "a") as file:
            file.write(sid + " 아이디의 트위터 검색 중 오류가 발생하였습니다. 해당 아이디를 건너뜁니다. 로그 저장 시각 : " + str(datetime.datetime.now()) + "\n")
        file.close()
        continue;


    # 리트윗 / 리플라이 제거
    # 인자 : 데이터 프레임 / 내용에 해당하는 컬럼 이름
    ## 리트윗을 삭제 하고자 하는 경우 아래의 코드 주석을 헤제하고 result_L2 변수 할당 하는 부분의 첫번재 인자를 result -> result_L1 로 변경하면 됨.
    # result_L1 = remove_rt_reply(result,'Content') 
    
    # 키워드가 존재하는 행 찾기
    # 인자 : 데이터 프레임 / 키워드 리스트 / 내용에 해당하는 컬럼 이름 / 헤시태그만 찾을 것인지 여부 / 키워드에 해당하는 내용이 없는 트윗 삭제 여부
    result_L2 = search_keyword(result,my_keyword,'Content',False,True)
    
    sleep(10) # 쿨타임을 줘서 과부하 방지
    
    # 데이터가 존재하는 경우 쓰기 (라인단위)
    if(len(result_L2) > 0):
                
        if append_mode == False:
            append_mode = True
            result_L2.to_csv(output_file_name + ".csv",index=False,header=True)
            
        elif append_mode == True:
            for i in range(len(result_L2)):
                result_L2.loc[[i]].to_csv(output_file_name,index=False,header=False,mode='a')