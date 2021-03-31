# 웹문서 읽기 연습 - web scraping
import urllib.request as req
from bs4 import BeautifulSoup

url = "https://ko.wikipedia.org/wiki/%EC%9D%B4%EC%88%9C%EC%8B%A0"
wiki = req.urlopen(url)
print(wiki) # <http.client.HTTPResponse object at 0x000001D88C0B1550>

soup = BeautifulSoup(wiki, 'html.parser')
# 원하는 자료 소스에서 우클릭 Copy - Copyselector
# #mw-content-text > div.mw-parser-output > p:nth-child(5)
# print(soup.select("#mw-content-text > div.mw-parser-output > p"))


print('------------------')
import bs4
# #kakaoIndex > a:nth-child(1)
url = "https://news.daum.net/society"
daum = req.urlopen(url)
print(daum)
soup = bs4.BeautifulSoup(daum, 'lxml')
print(soup.select_one('div#kakaoIndex > a').string)
datas = soup.select('div#kakaoIndex > a')
print(datas)

for i in datas:
    href = i.attrs['href']
    text = i.string
    print('href:%s, text:%s'%(href, text))
    
print()
datas = soup.findAll('a')
for i in datas[:2]:
    href = i.attrs['href']
    text = i.string
    print('href:%s, text:%s'%(href, text))
    
