from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time

# 11년 8월 ~ 12년 5월
a = ['2011', '2012','2013','2014','2015','2016','2018','2019','2020','2021']
b = ['1','2','3','4','5','6','8','9','10','11','12']
for i in a:
    for j in b:
        try:
            url = "https://sports.news.naver.com/wfootball/schedule/index.nhn?year="+ i +"&month="+j.zfill(2)+"&category=epl"
            print(url)
            browser = webdriver.Chrome('C:/work/chromedriver')  # 파일이 존재하는 경로로
            browser.implicitly_wait(time_to_wait=5)
            # Implicit Waits(암묵적 대기) 찾으려는 element가 로드될 때까지 지정한 시간만큼 대기할 수 있도록 설정
            # Explicit Waits(명시적 대기) 함수를 사용하여 무조건 몇 초간 대기하는 방법이다 편리하긴 하지만, 형편없는 효율 == Ex)time.sleep(3)
            
            browser.get(url)
            
            # 경기 시작시간
            time = browser.find_elements_by_class_name("time")
            # 경기위치
            place = browser.find_elements_by_class_name("place")
            # 팀명
            name = browser.find_elements_by_class_name("name")
            # 홀수번째로 출력된게 home, 짝수가 away 점수
            score = browser.find_elements_by_class_name("score")
            
            ndatas = []
            sdatas = []
            for n in name:
                ndatas.append(n.text.strip("\n"))
                # print(n.text.split("\n"))
            for n in score:
                sdatas.append(n.text.strip("\n"))
                # print(n.text.split("\n"))
            
            print(ndatas)    
            print(len(ndatas))
            print(sdatas)    
            print(len(sdatas))
            
            score_data = pd.DataFrame(sdatas)
            score_data.to_csv('score'+i+'_'+j.zfill(2)+'.csv', index=False, header=False)
        
            
            browser.quit()  # 브라우저 종료
            print('성공')
        except Exception:
            print('에러')    