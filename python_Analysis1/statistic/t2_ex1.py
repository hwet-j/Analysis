import pandas as pd
import scipy.stats as stats
import numpy as np
print(' 1번문제')
# [one-sample t 검정 : 문제1]  
# 영사기에 사용되는 구형 백열전구의 수명은 250시간이라고 알려졌다. 
# 한국연구소에서 수명이 50시간 더 긴 새로운 백열전구를 개발하였다고 발표하였다. 
# 연구소의 발표결과가 맞는지 새로 개발된 백열전구를 임의로 수집하여 수명시간을 수집하여 다음의 자료를 얻었다. 
# 한국연구소의 발표가 맞는지 새로운 백열전구의 수명을 분석하라.

one_sample = [305, 280, 296, 313, 287, 240, 259, 266, 318, 280, 325, 295, 315, 278]
print(np.array(one_sample).mean())
result1 = stats.ttest_1samp(one_sample, popmean=300)  # sample의 개수가 하나일때
print('result1 : ', result1)    
# pvalue=0.1436062  > 0.05 귀무가설 채택 : 백열전구의 수명시간의 평균은 300시간이다.

print('\n 2번문제')
# [one-sample t 검정 : 문제2] 
# 국내에서 생산된 대다수의 노트북 평균 사용 시간이 5.2 시간으로 파악되었다. A회사에서 생산된 노트북 평균시간과 차이가 있는지를 검정하기 위해서 A회사 노트북 150대를 랜덤하게 선정하여 검정을 실시한다.
# 실습 파일 : one_sample.csv

data = pd.read_csv("../testdata/one_sample.csv")
#print(data)

# 빈칸존재 NaN으로 대체
df = data.replace({'time':'     '},{'time':np.nan})
# NaN 가 있는 행 제외
df = df.dropna()
# 타입이 안맞아 숫자형으로 변환
df['time'] = df['time'].apply(pd.to_numeric)

# 정규성 확인 함수    pvalue=0.7242600 > 0.05 정규성을 만족
print(stats.shapiro(df['time']))
print(np.array(df['time']).mean())

result2 = stats.ttest_1samp(df.time, popmean=5.2)
print('result2 : ', result2)    
# pvalue=0.0001416 < 0.05 귀무가설  : A회사노트북의 평균시간 5.2시간이 아니다.  

print('\n 3번문제')
# [one-sample t 검정 : 문제3] 
# http://www.price.go.kr에서 메뉴 중  가격동향 -> 개인서비스요금 -> 조회유형:지역별, 품목:미용 자료를 파일로 받아 미용요금을 얻도록 하자. 
# 정부에서는 전국평균 미용요금이 15000원이라고 발표하였다. 이 발표가 맞는지 검정하시오

loc = ['서울', '부산', '대구', '인천', '광주',  '대전',' 울산', '경기', 
       '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주']
price = [18308, 16000, 19500, 19167, 15400, 15600, 14400, 15346, 
         17222, 15571, 16200, 12700, 14889, 15831, 15500, 17000]

df2 = pd.DataFrame(loc,columns={"위치":loc})
df2['가격'] = price
# print(df2)
print(df2['가격'])

# 정규성 확인 함수    pvalue=0.00286391 < 0.05 정규성을 만족하지않음
print(stats.shapiro(df2['가격']))

print(np.array(df2['가격']).mean())   # 15539.625

result3 = stats.ttest_1samp(df2.가격, popmean=15000)
print('result3 : ', result3)  
# pvalue=0.01761996 < 0.05 대립 : 전국평균 미용실 요금은 15000원이 아니다.


# [one-sample t 검정 : 문제3] - 2
# 가설설정
# 귀무 : 전국평균 미용실 요금은 15000원이다.
# 대립 : 전국평균 미용실 요금은 15000원이 아니다.
data3 = pd.ExcelFile('service.xls')
data3_xl = data3.parse('개인서비스지역동향2021-02')
#print(data3_xl, type(data3_xl))

# 결측값 제거
data3_xl = data3_xl.dropna(axis=1)
data3_xl = data3_xl.drop(['번호','품목'], axis=1)
print(data3_xl.loc[0])
print(np.array(data3_xl.loc[0]).mean())

result3 = stats.ttest_1samp(data3_xl.loc[0], popmean=15000)
print(result3) # pvalue=0.01761996
# 결론 p-value < 0.05이므로 귀무가설이 기각
# => 전국평균 미용실 요금은 15000원이 아니다.

