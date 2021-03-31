from bs4 import BeautifulSoup
# urllib.request, requests 모듈로 웹 자료 읽기
# naver의 영화 랭킹 정보 

# 방법 1
import urllib.request

url = "https://movie.naver.com/movie/sdb/rank/rmovie.nhn"
data = urllib.request.urlopen(url).read()
print(data)
soup = BeautifulSoup(data, 'lxml')
#print(soup)
# #old_content > table > tbody > tr:nth-child(2) > td.title > div > a
#print(soup.select('div.tit3'))
print(soup.select('div[class=tit3]'))   # 위와 동일

for tag in soup.select('div[class=tit3]'):
    #print(tag.text)
    print(tag.text.strip()) # 빈칸(?)제거
    
print('***' * 20)
# 방법 2
import requests
data = requests.get(url)
print(data)
print(data.status_code, ' ', data.encoding) # 몇 가지 정보를 반환  200  MS949
datas = data.text   # 문서 읽기
#print(datas)

datas = requests.get(url).text  # 명령을 연속적으로 주고 읽음
soup2 = BeautifulSoup(datas, 'lxml')
#print(soup2)

#m_list = soup.findAll("div", "tit3")    # 태그명, 속성명
#print(m_list)
m_list = soup.findAll("div", {'class':'tit3'})  # 태그명, {속성}

count = 1
for i in m_list:
    title = i.find('a')
    #print(title)
    print(str(count) + "위 :" + title.string)
    count += 1
    
    
print('~~~~~~~~~~~~네이버 실시간 검색어~~~~~~~~~~~~~~~~~~')
# import requests
from bs4 import BeautifulSoup  # html 분석 라이브러리

# 유저 설정
headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.122 Safari/537.36'}

url = 'https://datalab.naver.com/keyword/realtimeList.naver?where=main'
res = requests.get(url, headers = headers)
soup = BeautifulSoup(res.content, 'html.parser')

# span.item_title 정보를 선택
data = soup.select('span.item_title')
i = 1
for item in data:
    print(str(i) + ')' + item.get_text())
    i += 1


    







