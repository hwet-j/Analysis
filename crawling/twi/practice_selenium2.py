# 트위터에서 코로나 라는 단어를 검색하면 데이터가 한정된 양만 검색이됨 (기간을 길게 잡아도 로딩의 양이 정해진듯함)
# 데이터 양을 늘려주기 위해서 하루하루 날짜를 지정하여 코로나를 검색함 
# 동적데이터 이므로 selenium을 사용하며, 스크롤을 내려줘야 데이터를 읽어오기 때문에 자동으로 스크롤을 내려주어 데이터를 읽어오면 그 후에 데이터 추출
 

from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta


try:
    end = datetime.today().strftime("%Y-%m-%d")
    # datetime.today()은 지금 실행했을때의 시간데이터를 가져와줌 // strftime("%Y-%m-%d")을 사용해 원하는데이터 및 원하는 형태로 만들어준다.
    for i in range(5):  # range 날짜 범위 -> 오늘 부터 시작되게 만들어놓음
        
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
        
        # end의 시작은 오늘 start는 end의 전날로 설정하여 하루의 데이터를 가져오기위해 설정  
        start = (datetime.today() - timedelta(i+1)).strftime("%Y-%m-%d")
        end = (datetime.today() - timedelta(i)).strftime("%Y-%m-%d")
        url = "https://twitter.com/search?q={0}%20until%3A{2}%20since%3A{1}&src=typed_query"
        # 검색어 변경은 여기서 -> 코드로 입력 말고 input 사용해도됨. 현재로써 필요성을 느끼지 못함
        search = url.format("코로나", start ,end)
        
        browser.get(search)
        
        # 화면 가장 아래로 스크롤 내리기 
        # browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        
        # 현재 스크롤 높이를 변수로 저장함
        prev_height = browser.execute_script("return document.body.scrollHeight")
        
        # 웹페이지 맨 아래까지 무한 스크롤
        while True:
            # 데이터 추출
            element = browser.find_elements_by_class_name("css-901oao.r-18jsvk2.r-1qd0xha.r-a023e6.r-16dba41.r-rjixqe.r-bcqeeo.r-bnwqim.r-qvutc0")
            text = []   # 작성글 하나씩 분류 저장하기 위함 - while문 돌때마다 초기화
            for n in element:
                # name1 = n.text.split("\n")    # 올바른 데이터가 나오는지 확인
                text.append(n.text.split("\n")) # 변수 저장
            print(text) # 확인
            
            # 스크롤을 화면 가장 아래로 내린다
            browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        
            # 페이지 로딩 대기 (스크롤 내리고 데이터 로딩을 기다림)
            # browser.implicitly_wait(time_to_wait=5)    # 이것을 사용하면 제대로 작동되지 않음 브라우저를 처음 실행할때만 유용한듯 싶음
            time.sleep(2)
            
            # 현재 문서 높이를 가져와서 저장 (if문 사용하기 위해)
            curr_height = browser.execute_script("return document.body.scrollHeight")
    
            if(curr_height == prev_height): # 더이상 로딩이 없으면 while문을 나감
                break   
            else:   # 로딩이 존재하면 현재 높이를 다시 저장
                prev_height = browser.execute_script("return document.body.scrollHeight")
        
        browser.quit()  # 브라우저 종료 - 완성되면
        print('성공')
    print('끝')
except Exception:
    print('에러')    





