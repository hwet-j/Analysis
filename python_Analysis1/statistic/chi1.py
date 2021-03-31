# 교차분석 (chi2, 카이제곱) 가설 검정
# 데이터나 집단의 분산을 추정하고 검정할 때 사용
# 독립변수, 종속변수 : 범주형(건수, 개수)
# 일원 카이제곱 : (변인 단수) - 적합성(선호도) - 교차분할표 X
# 이원 카이제곱 : (변인 복수) - 독립성(동질성) - 교차분할표 O
# 절차 : 가설을 설정 -> 유의수준 결정(신뢰도를 언급하지않으면 95%) --> 검정통계량 계산 -> 귀무가설 채택여부 판단 -> 검정결과를 진술 
# 수식 : sum((관찰빈도 - 기대빈도)^2) / 기대빈도

import pandas as pd

data = pd.read_csv("../testdata/pass_cross.csv", encoding='euc-kr')
print(data.head(3))
print(data.shape)

# 귀무 가설 : 벼락치기 공부하는 것과 합격여부는 관계가 없다.
# 대립 가설 : 벼락치기 공부하는 것과 합격여부는 관계가 있다.

# 공부했을때 합격 18, 불합격 7
print(data[(data['공부함'] == 1) & (data['합격'] == 1)].shape[0])    
print(data[(data['공부함'] == 1) & (data['불합격'] == 1)].shape[0])

# 빈도표
data2 = pd.crosstab(index=data['공부안함'], columns=data['불합격'], margins=True)
data2.columns = ['합격', '불합격', '행합']
data2.index = ['공부함', '공부안함', '열합']
print(data2)

ch2 = (18 - 15) ** 2 / 15 + (7 - 10) ** 2 / 10 + (12 - 15) ** 2 / 15 + (13 - 10) ** 2 / 10
print('ch2 :',ch2)
# 자유도(df) : (행개수 - 1) * (열개수 - 1)  : 1
# 카이제곱 분포표로 임계치 : 3.84
# 결론 : ch2 3 < 3.84 이므로 귀무가설 채택역 내에 있어 귀무가설을 채택
# ====> 벼락치기 공부하는 것과 합격여부는 관계가 없다.

print('\n전문가가 제공하는 모듈을 사용')
import scipy.stats as stats
chi2, p, ddof, expected = stats.chi2_contingency(data2)
print('chi2 :', chi2)   # 위에서 직접작성해준값과 동일함 
print('p :', p)
# 두값(ddof, expected)은 여기서(귀무가설 채택) 크게 상관없는 값이라 주석처리 - p값이 중요 
#print('ddof :', ddof)
#print('expected :', expected)
# 결론 : 유의수준 0.05 < 유의확률(p-value) 0.5578 이므로 귀무가설 채택역 내에 있어 귀무가설을 채택 
# ====> 벼락치기 공부하는 것과 합격여부는 관계가 없다.



