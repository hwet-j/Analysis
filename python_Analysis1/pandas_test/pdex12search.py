# 구글 검색 기능 이용 : 검색 결과의 개수 만큼 브라우저 창을 띄움 

import requests
from bs4 import BeautifulSoup
import webbrowser

def searchFunc():
    base_url="https://www.google.com/search?q={0}"
    sword = base_url.format("파이썬")
    print(sword)
    
    headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'}
    plain_text = requests.get(sword, headers = headers)
    #print(plain_text.text)
    
    soup = BeautifulSoup(plain_text.text, 'lxml')
    #print(soup)
    
    # #rso > div:nth-child(1) > div:nth-child(1) > div.tF2Cxc > div.yuRUbf > a
    #link_data = soup.select('a')
    link_data = soup.select('div.tF2Cxc > div.yuRUbf > a')
    print(link_data)
    
    for link in link_data[:3]:
        print(link)
        #print(type(link), ' ', type(str(link)))     # <class 'bs4.element.Tag'>   <class 'str'>
        print(str(link).find('https'), ' ', str(link).find('ping') - 2) # https://....." ping 에서 데이터를 잘라 읽어내야하기 때문에 ping에서 앞으로 2자리 까지만 데이터를 읽어줘야해 -2를 작성해준다.
        urls = str(link)[str(link).find('https'):str(link).find('ping') - 2]
        print(urls)
        
        #webbrowser.open(urls)   # for문에서 정해준 개수 만큼 실행됨 
        
    
searchFunc()


