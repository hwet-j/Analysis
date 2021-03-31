# 위키백과 사이트에서 원하는 단어 검색후 형태소 분석. 단어 출현 빈도수 출력

import urllib
from bs4 import BeautifulSoup
from konlpy.tag import Okt
from urllib import parse # 한글 인코딩용

okt = Okt()
#para = input("검색단어 입력 :")
para = "이순신"
para = parse.quote(para)    # 웹에서 읽을수 있도록
#print(para)

url = "https://ko.wikipedia.org/wiki/" + para
#print(url)

page = urllib.request.urlopen(url)
#print(page)

soup = BeautifulSoup(page.read(), 'lxml')
#print(soup)

wordlist = [] # 형태소 분석으로 명사만 추출해 기억

for item in soup.select("#mw-content-text > div > p"):
    #print(item)
    if item.string != None:
        #print(item.string)
        ss = item.string
        wordlist += okt.nouns(ss)
        
print("wordlist : ", wordlist)        
print("발견된 단어 수 : ", len(wordlist))

# 단어의 발생횟수를 dict type     당시:2 조산:5
word_dict = {}

for i in wordlist:
    if i in word_dict:
        word_dict[i] += 1
    else:
        word_dict[i] = 1
        
print("word_dict : ", word_dict)

setdata = set(wordlist)
print(setdata)
print("발견된 단어수 (중복 제거후) : ", len(setdata))

print("\n판다스의 Series type으로 처리")
import pandas as pd

woList = pd.Series(wordlist)
print(woList[:5])
print(woList.value_counts()[:5])    # 단어별 횟수 총 개수 top 5

print()
woDict = pd.Series(word_dict)
print(woDict[:5])
print(woDict.value_counts())

print("\nDataFrame으로 처리")
df1 = pd.DataFrame(wordlist, columns=['단어'])
print(df1.head(5))

print()  # 단어/빈도수
df2 = pd.DataFrame([word_dict.keys(), word_dict.values()])
#print(df2)
df = df2.T
print(df2)
        
df2.to_csv('./이순신.csv', sep=',', index=False)

df3 = pd.read_csv('./이순신.csv')
print(df3.head(3)) 
        




