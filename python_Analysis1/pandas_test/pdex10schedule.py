# 웹 문서를  다운받아 파일로 저장하기 - 스케줄러
from bs4 import BeautifulSoup
import urllib.request as req
import datetime

url = "https://finance.naver.com/marketindex/"
data = req.urlopen(url)

soup = BeautifulSoup(data, 'html.parser')

price = soup.select_one("div.head_info > span.value").string
print("미국 USD : ",price)

t = datetime.datetime.now()
print(t)
fname = "./usd/" + t.strftime('%Y-%m-%d-%H-%M-%S') + '.txt'
print(fname)    # 2021-02-25-14-54-03.txt

with open(fname, 'w') as f:
    f.write(price)
    




