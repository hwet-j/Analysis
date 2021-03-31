# 멀티 프로세싱을 통한 웹 크롤링 - web1은 멀티프로세싱을 사용하지 않음 (비교하기위함)

import requests
from bs4 import BeautifulSoup as bs
import time

def get_links(): # 해당 컨텐츠의 a 태그 얻기
    data = requests.get("https://beomi.github.io/beomi.github.io_old/").text
    soup = bs(data, 'html.parser')
    my_titles = soup.select('h3 > a')   # h3태그 안의 a태그
    data = []
    
    for title in my_titles:
        data.append(title.get('href'))  # a tag의 속성중 href값 반환
    
    return data

def get_content(link):
    abs_link = "https://beomi.github.io/" + link
    req = requests.get(abs_link)
    html = req.text
    soup = bs(html, 'html.parser')
    print(soup.select('h1')[0].text)    # 첫번째 h1 태그의 문자열만 출력
    

if __name__ == '__main__':
    # print(get_links())
    # print(len(get_links()))
    
    start_time = time.time()
    
    for link in get_links():
        get_content(link)
        
    
    print('소요시간 : %s 초'%(time.time() - start_time))