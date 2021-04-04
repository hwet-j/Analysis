from selenium import webdriver
import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
# https://book.coalastudy.com/data-crawling/week-6/stage-2


try:
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

    url = "https://twitter.com/search?q={0}%20until%3A{2}%20since%3A{1}&src=typed_query"
    # 검색어, 시작, 끝
    search = url.format("코로나","2021-01-01","2021-03-01")  # 입력으로 바꿔줄수도있기 때문에 따로 만듦
    # print(search)   # 확인
    
    browser.get(search)
    # 화면 가장 아래로 스크롤 내리기
    time.sleep(2)
    browser.execute_script("window.scrollTo(0,document.body.scrollHeight)")
    time.sleep(2)
    # prev_height = browser.execute_script("return document.body.scrollHeight")
    
    
    #react-root > div > div > div.css-1dbjc4n.r-18u37iz.r-13qz1uu.r-417010 > main > div > div > div > div > div > div:nth-child(2) > div > div > section > div > div > div:nth-child(8) > div > div > article > div > div > div > div.css-1dbjc4n.r-18u37iz > div.css-1dbjc4n.r-1iusvr4.r-16y2uox.r-1777fci.r-kzbkwu > div:nth-child(2) > div:nth-child(1) > div
    test = browser.find_elements_by_class_name("css-901oao.r-18jsvk2.r-1qd0xha.r-a023e6.r-16dba41.r-rjixqe.r-bcqeeo.r-bnwqim.r-qvutc0")
    print("test:", test)
    tests = []
    for n in test:
        name1 = n.text.split("\n")
        tests.append(n.text.split("\n"))
        print(name1)
    print(tests)
    browser.quit()  # 브라우저 종료 - 완성되면
    print('성공')
except Exception:
    print('에러')    


