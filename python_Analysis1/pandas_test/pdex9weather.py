# 날씨 정보 예보 출력

import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd

url = "http://www.weather.go.kr/weather/forecast/mid-term-rss3.jsp"
data = urllib.request.urlopen(url).read()
#print(data.decode('utf8'))

soup = BeautifulSoup(urllib.request.urlopen(url), 'lxml')
#print(soup)

title = soup.find('title').string
#print(title)    # 기상청 육상 중기예보

wf = soup.find('wf')
#wf = soup.select_one('description > header > wf')
print(wf)

city = soup.find_all('city')
#print(city)
cityDatas = []
for c in city:
    #print(c.string)
    cityDatas.append(c.string)
    
df = pd.DataFrame()
df['city'] = cityDatas
print(df.head(3))

tmEfs = soup.select_one('location > data > tmEf')
#tmefs = soup.select_one('location > data > tmef')    # 실제 기입은 tmEf로 되어있지만 읽어온데이터는 소문자 tmef로 되어있기 때문에 실제 읽어온값을 작성해줘야함
tmefs = soup.select_one('location > province + city + data > tmef') # +는 형제태그일때 사용

print(tmEfs)
print(tmefs)

tempMins = soup.select('location > province + city + data > tmn')
tempDatas = []
for t in tempMins:
    tempDatas.append(t.string)

df['temp_min'] = tempDatas
print(df.head(3))

df.columns = ['지역', '최저기온'] # 칼럼명 지정
print(df.head(3), len(df))
print(df.describe())
print(df.info())    # 구조

# 파일로 저장
df.to_csv('날씨정보.csv', index=False)
df2 = pd.read_csv('날씨정보.csv')
#print(df2)

print('****' * 10)
# 앞에서 2개
print(df.head(2))
print(df[0:2])
# 뒤에서 2개
print(df.tail(2))
print(df[-2:len(df)])

print('****' * 10)
# 하나만 가져와 Series
print(df.iloc[0])
print(type(df.iloc[0]))
print()
# 여러개를 가져와 DataFrame
print(df.iloc[0:2, :])
print(type(df.iloc[0:2, :]))

print()
print(df.iloc[0:2, 0:1])

print()
print(df['지역'])     # 지역 전체
print(df['지역'][0:2])    # 2개만
print(df['지역'][:2])     # 2개만


print()
print(df.loc[1:3])  
print()
print(df.loc[[1,3]])    # 1행과 3행
#print(df.loc[:, ['지역']])    # 모든 행의 지역

print('-----------------')
df = df.astype({'최저기온':'int'}) # 최저기온의 타입을 int로
print(df.info())
print(df['최저기온'].mean())
print(df['최저기온'].std()) # 편차
print(df['최저기온'].describe())
print(df['최저기온'] >= 5)  # 5보다 크면 True 아니면 False
print(df.loc[df['최저기온'] >= 5])  # 최저기온이 5 이상인 지역
print(df.sort_values(['최저기온'], ascending = True))



