# 문제 1번
# 메뉴와 가격 : DataFrame
# 전체가격에 대한 평균
# 전체가격에 대한 표준편차

import urllib.request as req
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np

url = "https://www.bbq.co.kr/menu/menuList.asp"
data = req.urlopen(url)
soup = BeautifulSoup(data, 'html.parser')

name = soup.select("div.info > p.name")
price = soup.select("div.info > p.pay")
ndatas = []
pdatas = []
for p in price:
    #print(p.text.strip()) # 빈칸(?)제거
    pdatas.append(p.text.strip())
for n in name:
    ndatas.append(n.text.strip())

print(ndatas)
print(pdatas)

# 메뉴와 가격 DataFrame
df = pd.DataFrame(columns=['name','price'])
df['name'] = ndatas
df['price'] = pdatas
print(df)
#print(type(df))

tpdatas = []
for p in price:
    tpdatas.append(int(p.text.strip()[0:-1].replace(",","")))   # 맨뒤에글자 (원)을 빼고 읽어주고, ','값을 빈칸으로대체하면서 타입을 int로 변경

# 전체가격
print('총합 : ',np.sum(tpdatas),'원')  # 1383600 원
# 전체가격에 대한 평균,표준편차 2째 자리까지
print('평균 : ',round(np.mean(tpdatas),2),'원')    # 20347.06 원
print('표준편차 : ',round(np.std(tpdatas),2),'원')   # 표준편차 :  3406.2 원



    



