# BeautifulSoup : XML, HTML 문서의 일부 자료 추출

import requests     # 웹에서 읽어오기 위해서 requests
from bs4 import BeautifulSoup

def go():
    base_url = "http://www.naver.com/index.html"

    #storing all the information including headers in the variable source code
    source_code = requests.get(base_url)

    #sort source code and store only the plaintext
    plain_text = source_code.text   # 일반적인 텍스트로 읽어옴
    #print(plain_text)

    #converting plain_text to Beautiful Soup object so the library can sort thru it
    convert_data = BeautifulSoup(plain_text, 'lxml')    #텍스트로 읽어온값을 BeautifulSoup을 이용해 읽음

    for link in convert_data.findAll('a'):  # 읽어온 데이터에서 a태그 전부를 가져온다.
        href = base_url + link.get('href')  #Building a clickable url
        print(href)                          #displaying href

go()