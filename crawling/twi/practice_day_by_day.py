from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


try:
    end = datetime.today().strftime("%Y-%m-%d")
    for i in range(5):  # range 날짜 범위 -> 오늘 부터
        # 사람이 아닌것으로 판단되어 차단될 수 있으니 사람이라는 것을 인지 시켜주기 위함.
        # http://www.useragentstring.com/ 에서 개인의 User_Agent확인가능
        headers = {
            "User-Agent":"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.114 Safari/537.36",
            "Accept-Language":"ko-KR,ko"
            }
        
        browser = webdriver.Chrome('C:/work/chromedriver')  # 각자 컴퓨터에 파일이 존재하는 경로로
        browser.implicitly_wait(time_to_wait=5)
        # Implicit Waits(암묵적 대기) 찾으려는 element가 로드될 때까지 지정한 시간만큼 대기할 수 있도록 설정
        # Explicit Waits(명시적 대기) 함수를 사용하여 무조건 몇 초간 대기하는 방법이다 편리하긴 하지만, 형편없는 효율 == Ex)time.sleep(3)
        # browser.maximize_window()
         
        start = (datetime.today() - timedelta(i+1)).strftime("%Y-%m-%d")
        end = (datetime.today() - timedelta(i)).strftime("%Y-%m-%d")
        url = "https://twitter.com/search?q={0}%20until%3A{2}%20since%3A{1}&src=typed_query"
        # 검색어 변경은 여기서
        search = url.format("코로나", start ,end)
        
        browser.get(search)
        
        # 화면 가장 아래로 스크롤 내리기
        # browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        
        prev_height = browser.execute_script("return document.body.scrollHeight")
        
        # 웹페이지 맨 아래까지 무한 스크롤
        while True:
       
            element = browser.find_elements_by_class_name("css-901oao.r-18jsvk2.r-1qd0xha.r-a023e6.r-16dba41.r-rjixqe.r-bcqeeo.r-bnwqim.r-qvutc0")
            text = []   # 작성글 하나씩 분류 저장하기 위함 - while문 돌때마다 초기화
            for n in element:
                # name1 = n.text.split("\n")    # 확인
                text.append(n.text.split("\n"))
            print(text)
            
            # 스크롤을 화면 가장 아래로 내린다
            browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        
            # 페이지 로딩 대기
            # browser.implicitly_wait(time_to_wait=5)
            time.sleep(2)
            
            # 현재 문서 높이를 가져와서 저장 (if문 사용하기 위해)
            curr_height = browser.execute_script("return document.body.scrollHeight")
    
            if(curr_height == prev_height):
                break
            else:
                prev_height = browser.execute_script("return document.body.scrollHeight")
        
        browser.quit()  # 브라우저 종료 - 완성되면
        print('성공')
    print('끝')
except Exception:
    print('에러')    


# 홈페이지가 코로나검색해서 최대한스크롤을 내린후에 이전날 다시 검색 기간 정해서 스크롤내려주는것까지 오나료
# 스크롤 다내린후에 크롤링완성되면 그에대한 데이터 저장 및 그 페이지 종료까지 해볼것...



